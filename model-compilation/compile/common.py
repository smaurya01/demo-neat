#!/usr/bin/env python3
"""Shared helpers for the model-compilation flow.

One registry (`models.yaml`) drives all four steps, so a model's input name, shapes, normalization
and surgery kind can never drift between export, surgery, compile and test.

Layout produced per model:

    work/<id>/
      onnx/<id>.onnx                     # step 1: convert_to_onnx.py
      surgery/<id>.compile_ready.onnx    # step 2: graph_surgery.py  (only if surgery != none)
      compile_int8/<...>_mpk.tar.gz      # step 3: compiler.py (precision int8, the default)
      compile_bf16/<...>_mpk.tar.gz      # step 3: compiler.py --precision bf16
      reports/                           # logs, audit, validation, surgery notes
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
ASSETS = ROOT / "assets"
CALIB_DIR = ASSETS / "calibration"      # REAL images — never synthetic
INFER_DIR = ASSETS / "inference"        # REAL images for smoke tests
REGISTRY = ROOT / "models.yaml"


def load_registry() -> tuple[dict, list[dict]]:
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return reg.get("project", {}), reg["models"]


def model_cfg(model_id: str) -> tuple[dict, dict]:
    project, models = load_registry()
    for m in models:
        if m["id"] == model_id:
            return m, project
    known = ", ".join(m["id"] for m in models)
    raise SystemExit(f"unknown model '{model_id}'. Known: {known}")


def all_model_ids(enabled_only: bool = True) -> list[str]:
    _, models = load_registry()
    return [m["id"] for m in models if m.get("enabled", True) or not enabled_only]


PRECISIONS = ("int8", "bf16")


def model_precision(model_id: str, override: str | None = None) -> str:
    """The precision to build/test: `--precision` if given, else the model's `precision:` in models.yaml."""
    if override:
        prec = override
    else:
        cfg, _ = model_cfg(model_id)
        prec = cfg.get("precision")
        if prec is None:
            raise SystemExit(f"{model_id}: models.yaml has no `precision:` for this model. "
                             f"Add `precision: int8` or `precision: bf16` to its entry.")
    if prec not in PRECISIONS:
        raise SystemExit(f"{model_id}: unknown precision '{prec}' (use one of {', '.join(PRECISIONS)})")
    return prec


def paths(model_id: str, precision: str = "int8") -> dict[str, Path]:
    base = WORK / model_id
    return {
        "base": base,
        "onnx": base / "onnx" / f"{model_id}.onnx",
        "onnx_dir": base / "onnx",
        "surgery": base / "surgery" / f"{model_id}.compile_ready.onnx",
        "surgery_dir": base / "surgery",
        "compile_dir": base / f"compile_{precision}",
        "reports": base / "reports",
    }


def ensure_dirs(model_id: str, precision: str = "int8") -> dict[str, Path]:
    p = paths(model_id, precision)
    for k in ("onnx_dir", "surgery_dir", "compile_dir", "reports"):
        p[k].mkdir(parents=True, exist_ok=True)
    return p


def compile_input_onnx(model_id: str) -> Path:
    """The ONNX the compiler should consume: the surgery output if present, else the raw export."""
    p = paths(model_id)
    return p["surgery"] if p["surgery"].exists() else p["onnx"]


def archive_path(model_id: str, precision: str | None = None) -> Path | None:
    """Find the produced _mpk.tar.gz for a model at `precision` (default: the registry's), if any."""
    prec = model_precision(model_id, precision)
    int8_dir = paths(model_id, "int8")["compile_dir"]
    # Archives built before compile_bf16/ existed were all written to compile_int8/, bf16 ones
    # included; their command record (reports/compile.command.txt) says which precision they are.
    legacy_cmd = paths(model_id)["reports"] / "compile.command.txt"
    legacy_bf16 = legacy_cmd.exists() and "--bf16-weights" in legacy_cmd.read_text(encoding="utf-8")
    if prec == "int8" and legacy_bf16:
        return None                       # compile_int8/ holds a bf16 build, not an int8 one
    hits = sorted(paths(model_id, prec)["compile_dir"].rglob("*_mpk.tar.gz"))
    if not hits and prec == "bf16" and legacy_bf16:
        hits = sorted(int8_dir.rglob("*_mpk.tar.gz"))
    return hits[0] if hits else None
