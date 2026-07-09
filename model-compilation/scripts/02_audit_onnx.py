#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GUARD = Path("/home/surajmaurya/.codex/skills/sima-model-surgery/scripts/model_surgery_guard.py")


def model_cfg(model_id):
    reg = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    for model in reg["models"]:
        if model["id"] == model_id:
            return model
    raise KeyError(model_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dtype", default="int8")
    args = parser.parse_args()

    model = model_cfg(args.model_id)
    base = ROOT / "work" / model["id"]
    onnx_path = base / "surgery" / f"{model['id']}.surgery.onnx"
    if not onnx_path.exists():
        onnx_path = base / "onnx" / f"{model['id']}.onnx"
    report_dir = base / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", str(GUARD), "audit-model", "--model", str(onnx_path), "--dtype", args.dtype, "--json"]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (report_dir / f"audit_{args.dtype}.log").write_text(proc.stdout, encoding="utf-8")

    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        parsed = {"parse_error": True, "raw": proc.stdout}
    parsed["returncode"] = proc.returncode
    (report_dir / f"audit_{args.dtype}.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(proc.stdout)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
