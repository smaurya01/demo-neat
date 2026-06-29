#!/usr/bin/env bash
# Reproducible model build (canonical einsum path):
#   yolo26n.pt -> ONNX -> C2PSA einsum graph surgery -> BF16 quantize + MLA
#   tessellation -> Modalix MLA model pack (plc_yolo26n_mpk.tar.gz).
#
# Run inside the SiMa Model SDK container (the `afe` toolchain + ultralytics +
# model_to_pipeline must be importable), e.g.:
#
#   sima-cli sdk model "bash /workspace/NEAT/demo-neat/plc-defect-detection-yolo26n/scripts/compile_model.sh"
#
# All artifacts are written into assets/models/. The final pack is staged to
# assets/models/plc_yolo26n_mpk.tar.gz (the path config/default.conf points to).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$PROJ/assets/models"
MTP="${MTP:-/workspace/tool-model-to-pipeline}"                 # model_to_pipeline (onnx_helpers + surgeon_yolo11)
ULTRA="${ULTRA:-$MTP/.model_venv/lib/python3.10/site-packages}" # ultralytics for export
PY="${PY:-python}"

cd "$PROJ"
mkdir -p "$MODELS"

echo "[1/4] export yolo26n.pt -> yolo26n.onnx"
PYTHONPATH="$ULTRA" "$PY" -c \
  "from ultralytics import YOLO; YOLO('$MODELS/yolo26n.pt').export(format='onnx', imgsz=640, opset=17, simplify=True, dynamic=False)"

echo "[2/4] C2PSA einsum surgery + raw heads -> yolo26n_einsum_raw.onnx"
PYTHONPATH="$MTP" "$PY" compile/surgery_einsum_attention.py \
  --model_path "$MODELS/yolo26n.onnx" \
  --out "$MODELS/yolo26n_einsum_raw.onnx"

echo "[3/4] BF16 quantize + MLA tessellation compile (modalix)"
"$PY" compile/compile_yolo26_modelsdk.py \
  --model "$MODELS/yolo26n_einsum_raw.onnx" \
  --build-dir compile/build \
  --strict-one-mla \
  --json-output compile/compile_report.json

echo "[4/4] stage the compiled MLA pack -> assets/models/plc_yolo26n_mpk.tar.gz"
PACK="$(find compile/build -name '*_mpk.tar.gz' | head -n1)"
if [[ -n "${PACK}" ]]; then
  cp -f "${PACK}" "$MODELS/plc_yolo26n_mpk.tar.gz"
  echo "    pack -> assets/models/plc_yolo26n_mpk.tar.gz  (from ${PACK})"
else
  echo "    WARNING: no *_mpk.tar.gz found under compile/build" >&2
  exit 1
fi
echo "[done] compile complete. Deploy with: bash scripts/run_dk.sh"
