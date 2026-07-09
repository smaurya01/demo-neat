#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def status_for(model):
    model_id = model["id"]
    base = ROOT / "work" / model_id
    reports = base / "reports"
    onnx_path = base / "onnx" / f"{model_id}.onnx"
    surgery_path = base / "surgery" / f"{model_id}.surgery.onnx"
    audit = read_json(reports / "audit_int8.json") or {}
    archive = read_json(reports / "archive_validation.json") or {}
    compile_status = read_json(reports / "compile_status.json") or {}
    raw_head_status = read_json(reports / "raw_head_compile_status.json") or {}
    mpks = sorted(base.glob("compile/**/*_mpk.tar.gz"))
    qlog = reports / "quantize_compile.log"
    partial_sima = sorted(base.glob("compile/**/*.sima"))
    mpk_status = "pass" if mpks else ("running_or_missing" if qlog.exists() else "not_started")
    if not mpks and compile_status.get("status"):
        mpk_status = compile_status["status"]
    elif not mpks and partial_sima:
        mpk_status = "partial_sima"

    notes = model.get("blocked_reason", "")
    if compile_status.get("notes"):
        notes = compile_status["notes"]
    if raw_head_status.get("status"):
        raw_note = f"raw_heads={raw_head_status['status']}"
        notes = f"{notes}; {raw_note}" if notes else raw_note

    return {
        "model_id": model_id,
        "task": model.get("task", ""),
        "family": model.get("family", ""),
        "enabled": model.get("enabled", False),
        "onnx": "pass" if onnx_path.exists() else "missing",
        "surgery": "pass" if surgery_path.exists() else "missing",
        "audit_unsupported": audit.get("unsupported_count", ""),
        "audit_unknown": audit.get("unknown_count", ""),
        "mpk": mpk_status,
        "archive_contract": archive.get("status", ""),
        "single_elf": archive.get("single_elf", ""),
        "no_so": archive.get("no_so", ""),
        "mpk_path": str(mpks[-1]) if mpks else "",
        "notes": notes,
    }


def main():
    registry = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    rows = [status_for(model) for model in registry["models"]]

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "summary.csv"
    md_path = results / "summary.md"

    fields = [
        "model_id",
        "task",
        "family",
        "enabled",
        "onnx",
        "surgery",
        "audit_unsupported",
        "audit_unknown",
        "mpk",
        "archive_contract",
        "single_elf",
        "no_so",
        "mpk_path",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Model Compilation Status", ""]
    lines.append("| Model | ONNX | Unsupported | Unknown | MPK | Archive | Notes |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| `{row['model_id']}` | {row['onnx']} | {row['audit_unsupported']} | "
            f"{row['audit_unknown']} | {row['mpk']} | {row['archive_contract']} | {row['notes']} |"
        )
    lines.append("")
    lines.append("Archive `pass` means the `.tar.gz` has exactly one ELF member and no `.so` members.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
