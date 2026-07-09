#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd, log_path):
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--real-data", action="store_true")
    args = parser.parse_args()

    registry = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    models = [m for m in registry["models"] if m.get("enabled")]
    if args.models:
        wanted = set(args.models)
        models = [m for m in registry["models"] if m["id"] in wanted]
    if args.limit:
        models = models[: args.limit]

    rows = []
    for model in models:
        model_id = model["id"]
        base = ROOT / "work" / model_id
        reports = base / "reports"
        row = {"model_id": model_id, "export": "", "audit": "", "surgery": "", "quantize_compile": ""}

        row["export"] = run_step(["python", "scripts/01_download_or_export.py", "--model-id", model_id], reports / "01_export.wrapper.log")
        if row["export"] == 0:
            row["audit"] = run_step(["python", "scripts/02_audit_onnx.py", "--model-id", model_id], reports / "02_audit.wrapper.log")
        if row["export"] == 0:
            row["surgery"] = run_step(["python", "scripts/03_surgery.py", "--model-id", model_id], reports / "03_surgery.wrapper.log")
        if args.compile and row["export"] == 0 and row["surgery"] == 0:
            cmd = ["python", "scripts/04_quantize_compile.py", "--model-id", model_id]
            if args.real_data:
                cmd.append("--real-data")
            row["quantize_compile"] = run_step(cmd, reports / "04_quantize_compile.wrapper.log")
        rows.append(row)
        print(row)

    out = ROOT / "results" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "export", "audit", "surgery", "quantize_compile"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
