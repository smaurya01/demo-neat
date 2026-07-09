#!/usr/bin/env python3
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--mla-tesselation", action="store_true")
    parser.add_argument("--raw-heads", action="store_true")
    parser.add_argument("--num-calib-samples", type=int, default=None)
    args = parser.parse_args()

    model, project = model_cfg(args.model_id)
    base = ROOT / "work" / model["id"]
    if args.raw_heads:
        onnx_path = base / "surgery" / f"{model['id']}.raw_heads.onnx"
    else:
        onnx_path = base / "surgery" / f"{model['id']}.surgery.onnx"
    if not onnx_path.exists():
        onnx_path = base / "onnx" / f"{model['id']}.onnx"

    compile_dir = base / "compile"
    report_dir = base / "reports"
    compile_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(QC),
        "--model_path",
        str(onnx_path),
        "--model_format",
        "onnx",
        "--model_layout",
        project.get("default_layout", "NCHW"),
        "--input_names",
        model["input_name"],
        "--input_shapes",
        ",".join(str(x) for x in model["input_shape"]),
        "--device",
        args.device or project.get("default_device", "modalix"),
        "--build_dir",
        str(compile_dir),
        "--calib_method",
        project.get("default_calib_method", "mse"),
        "--requant_mode",
        project.get("default_requant_mode", "sima"),
        "--mean",
        *[str(x) for x in model.get("mean", [0, 0, 0])],
        "--std",
        *[str(x) for x in model.get("std", [1, 1, 1])],
    ]
    if model.get("output_names"):
        cmd.extend(["--output_names", *model["output_names"]])
    if args.raw_heads:
        cmd.extend([
            "--output_names",
            "bbox_0",
            "bbox_1",
            "bbox_2",
            "class_prob_0",
            "class_prob_1",
            "class_prob_2",
        ])
    if args.bf16:
        cmd.extend(["--bf16-weights", "--bf16-activations"])
    if args.mla_tesselation:
        cmd.append("--mla-tesselation")
    if args.no_compile:
        cmd.append("--no-compile")
    if args.verify:
        cmd.append("--verify")
    if args.real_data:
        cmd.extend([
            "--real_data",
            "--dataset_images",
            str(ROOT / "assets" / "calibration"),
            "--num_calib_samples",
            str(args.num_calib_samples or project.get("default_num_calib_samples", 50)),
        ])

    (report_dir / "quantize_compile.command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (report_dir / "quantize_compile.log").write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
