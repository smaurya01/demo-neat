#!/usr/bin/env bash
# Compile every model in models.yaml, ONE AT A TIME, and collect the artifacts.
#
# Strictly serial: the compiler is memory-hungry and concurrent compiles OOM.
#
# For each model:
#   1. convert_to_onnx.py   -> work/<id>/onnx/<id>.onnx      (+ downloads <id>.pt for ultralytics)
#   2. graph_surgery.py     -> work/<id>/surgery/<id>.compile_ready.onnx
#   3. compiler.py          -> work/<id>/compile_int8/<...>_mpk.tar.gz
#   4. test_model.py --validate-only  -> contract check (1 .elf, 0 .so)
#   5. collect into assets/models/<id>/  (original + ONNX + compiled archive)
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
OUT="$ROOT/assets/models"
LOG="$ROOT/compile_all.log"

MODELS=(resnet50 densenet169 convnext_tiny efficientnet_v2_s
        yolo11n yolo11s yolo26n yolo11s-seg yolo26s-pose yolox_s)

mkdir -p "$OUT"
: > "$LOG"

say() { echo "$*" | tee -a "$LOG"; }

say "=== compile_all started $(date -u '+%F %T') UTC ==="
say "models: ${MODELS[*]}"
say ""

for id in "${MODELS[@]}"; do
  t0=$(date +%s)
  say "############################################################"
  say "### $id — start $(date -u '+%T') UTC"
  say "############################################################"

  ok=1
  for step in convert_to_onnx graph_surgery compiler; do
    say "--- [$id] $step"
    if ! python compile/$step.py --model-id "$id" >>"$LOG" 2>&1; then
      say "!!! [$id] $step FAILED"
      ok=0; break
    fi
  done

  if [ "$ok" -eq 1 ]; then
    say "--- [$id] validate contract"
    python compile/test_model.py --model-id "$id" --validate-only >>"$LOG" 2>&1 || say "!!! [$id] validate FAILED"

    # ---- collect artifacts ------------------------------------------------
    dst="$OUT/$id"; mkdir -p "$dst"
    # compiled SiMa archive (the thing people actually want)
    arc=$(find "work/$id/compile_int8" -name '*_mpk.tar.gz' 2>/dev/null | head -1)
    [ -n "$arc" ] && cp -f "$arc" "$dst/" && say "    archive : $(basename "$arc") ($(du -h "$arc" | cut -f1))"
    # the ONNX export (the graph that was compiled)
    [ -f "work/$id/onnx/$id.onnx" ] && cp -f "work/$id/onnx/$id.onnx" "$dst/" \
      && say "    onnx    : $id.onnx ($(du -h "work/$id/onnx/$id.onnx" | cut -f1))"
    # the original weights — ultralytics downloads <arch>.pt into the CWD, and for
    # every ultralytics entry in models.yaml, arch == "<id>.pt". Torchvision models
    # have no .pt here (weights come from torchvision's pretrained API); megvii ships
    # a pre-exported ONNX. In both of those cases the ONNX above IS the original.
    if [ -f "$ROOT/$id.pt" ]; then
      cp -f "$ROOT/$id.pt" "$dst/"
      say "    weights : $id.pt ($(du -h "$ROOT/$id.pt" | cut -f1))"
    fi
  fi

  t1=$(date +%s)
  say "### $id — done in $(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s   [$( [ "$ok" -eq 1 ] && echo OK || echo FAILED )]"
  say ""
done

say "=== compile_all finished $(date -u '+%F %T') UTC ==="
say ""
say "=== collected artifacts ==="
du -sh "$OUT"/* 2>/dev/null | tee -a "$LOG"
