#!/usr/bin/env python3
"""T7: quantize + compile a transformer / difficult model to INT8 MPK for modalix.

Generalized from 04_quantize_compile.py and 15_compile_t5_int8.py. Unlike the
YOLO raw-head flow, these models keep their natural single/dual output tensors
(classification logits, DINOv2 features, DETR pred_logits/pred_boxes); the
CPU-side postprocess (softmax/top-k, nearest-label, Hungarian matching) lives in
the reference pipeline, not the graph.

Reads input/output contract from models.yaml so it never drifts from the graph.
Picks, in order of preference:
    work/<id>/surgery/<id>.compile_ready.onnx   (T7 attention/shape surgery)
    work/<id>/surgery/<id>.surgery.onnx         (generic simplify+gather fix)
    work/<id>/onnx/<id>.onnx                     (raw export)
unless --onnx overrides.

MUST be launched through the global compile slot wrapper, in the BACKGROUND:
    compile_slot.sh "G:<id>-int8" python scripts/18_compile_transformer_int8.py --model-id <id>
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
QC = Path("/home/surajmaurya/.codex/skills/sima-model-quantize-compile/scripts/quantize_compile.py")


def model_cfg(model_id):
    reg = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    for model in reg["models"]:
        if model["id"] == model_id:
            return model, reg.get("project", {})
    raise KeyError(model_id)


def pick_onnx(base: Path, model_id: str, override: str | None) -> Path:
    if override:
        return Path(override)
    candidates = [
        base / "surgery" / f"{model_id}.compile_ready.onnx",
        base / "surgery" / f"{model_id}.surgery.onnx",
        base / "onnx" / f"{model_id}.onnx",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"no ONNX found for {model_id}: tried {candidates}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--onnx", default=None, help="override ONNX path")
    parser.add_argument("--calib-dir", type=Path, default=Path("/workspace/calibration_images"))
    parser.add_argument("--num-calib-samples", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--any-shape-on-mla", action="store_true",
                        help="keep non-4D ops (LayerNorm reshapes, attention) on the MLA")
    parser.add_argument("--mla-tesselation", action="store_true")
    parser.add_argument("--input-names", nargs="+", default=None, help="override input names")
    parser.add_argument("--input-shapes", nargs="+", default=None, help="override input shapes, e.g. 1,3,224,224")
    parser.add_argument("--output-names", nargs="+", default=None, help="override output names")
    args = parser.parse_args()

    model, project = model_cfg(args.model_id)
    base = ROOT / "work" / model["id"]
    onnx_path = pick_onnx(base, model["id"], args.onnx)
    report_dir = base / "reports"
    compile_dir = base / "compile_int8"
    report_dir.mkdir(parents=True, exist_ok=True)
    compile_dir.mkdir(parents=True, exist_ok=True)

    input_names = args.input_names or [model["input_name"]]
    if args.input_shapes:
        input_shapes = args.input_shapes
    else:
        input_shapes = [",".join(str(x) for x in model["input_shape"])]
    output_names = args.output_names or model.get("output_names") or []

    cmd = [
        "python", str(QC),
        "--model_path", str(onnx_path),
        "--model_format", "onnx",
        "--model_layout", project.get("default_layout", "NCHW"),
        "--input_names", *input_names,
        "--input_shapes", *input_shapes,
        "--device", args.device or project.get("default_device", "modalix"),
        "--build_dir", str(compile_dir),
        "--calib_method", project.get("default_calib_method", "mse"),
        "--requant_mode", project.get("default_requant_mode", "sima"),
        "--mean", *[str(x) for x in model.get("mean", [0, 0, 0])],
        "--std", *[str(x) for x in model.get("std", [1, 1, 1])],
        "--real_data",
        "--dataset_images", str(args.calib_dir),
        "--num_calib_samples", str(args.num_calib_samples),
    ]
    if output_names:
        cmd.extend(["--output_names", *output_names])
    if args.any_shape_on_mla:
        cmd.append("--any_shape_on_mla")
    if args.mla_tesselation:
        cmd.append("--mla-tesselation")

    (report_dir / "compile_int8.command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (report_dir / "compile_int8.log").write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout[-4000:])
    print(f"[18_compile] {model['id']} rc={proc.returncode} onnx={onnx_path}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
