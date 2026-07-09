#!/usr/bin/env python3
"""Compile fresh YOLO compile-ready ONNX models as INT8 MPK artifacts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QC = Path("/home/surajmaurya/.codex/skills/sima-model-quantize-compile/scripts/quantize_compile.py")
OUTPUT_NAMES = ["bbox_0", "bbox_1", "bbox_2", "class_logit_0", "class_logit_1", "class_logit_2"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=["yolo11n", "yolo26n"], required=True)
    parser.add_argument("--num-calib-samples", type=int, default=20)
    parser.add_argument("--calib-dir", type=Path, default=ROOT / "assets" / "yolo_calibration")
    parser.add_argument("--no-mla-tesselation", action="store_true")
    parser.add_argument("--no-any-shape-on-mla", action="store_true")
    args = parser.parse_args()

    base = ROOT / "work" / args.model_id
    report_dir = base / "reports"
    build_dir = base / "compile_int8"
    model_path = base / "surgery" / f"{args.model_id}.compile_ready.onnx"
    report_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(QC),
        "--model_path",
        str(model_path),
        "--model_format",
        "onnx",
        "--model_layout",
        "NCHW",
        "--input_names",
        "images",
        "--input_shapes",
        "1,3,640,640",
        "--output_names",
        *OUTPUT_NAMES,
        "--device",
        "modalix",
        "--build_dir",
        str(build_dir),
        "--calib_method",
        "mse",
        "--requant_mode",
        "sima",
        "--mean",
        "0.0",
        "0.0",
        "0.0",
        "--std",
        "1.0",
        "1.0",
        "1.0",
        "--real_data",
        "--dataset_images",
        str(args.calib_dir),
        "--num_calib_samples",
        str(args.num_calib_samples),
    ]
    if not args.no_mla_tesselation:
        cmd.append("--mla-tesselation")
    if not args.no_any_shape_on_mla:
        cmd.append("--any_shape_on_mla")

    (report_dir / "compile_ready_int8.command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (report_dir / "compile_ready_int8.log").write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
