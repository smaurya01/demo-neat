#!/usr/bin/env python3
"""T5 phase 1: compile compile_ready ONNX to INT8 MPK for modalix.

Copied and generalized from 12_compile_yolo_int8.py (frozen 01-12). The only
difference is that the exposed output-name set varies per task (detection 6,
pose 9, segmentation 10), and yolox_s uses a plain 3-scale raw head. Output
names are read from the surgery report so this never drifts from the graph.

MUST be launched through the global compile slot wrapper:
    compile_slot.sh "C:<model>-int8" python scripts/15_compile_t5_int8.py --model-id <id>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QC = Path("/home/surajmaurya/.codex/skills/sima-model-quantize-compile/scripts/quantize_compile.py")

MODEL_IDS = ["yolo11s", "yolo11s-seg", "yolo26s-pose", "yolox_s"]


def output_names_from_report(base: Path, model_id: str) -> list[str]:
    report = base / "reports" / "compile_ready_surgery.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        names = data.get("outputs")
        if names:
            return names
    raise SystemExit(f"no surgery report / outputs for {model_id}: {report}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
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

    output_names = output_names_from_report(base, args.model_id)

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
        *output_names,
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
