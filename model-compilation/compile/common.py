#!/usr/bin/env python3
"""Shared helpers for the model-compilation flow.

One registry (`models.yaml`) drives all four steps, so a model's input name, shapes, normalization
and surgery kind can never drift between export, surgery, compile and test.

Layout produced per model:

    work/<id>/
      onnx/<id>.onnx                     # step 1: convert_to_onnx.py
      surgery/<id>.compile_ready.onnx    # step 2: graph_surgery.py  (only if surgery != none)
      compile_int8/<...>_mpk.tar.gz      # step 3: compiler.py
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


def paths(model_id: str) -> dict[str, Path]:
    base = WORK / model_id
    return {
        "base": base,
        "onnx": base / "onnx" / f"{model_id}.onnx",
        "onnx_dir": base / "onnx",
        "surgery": base / "surgery" / f"{model_id}.compile_ready.onnx",
        "surgery_dir": base / "surgery",
        "compile_dir": base / "compile_int8",
        "reports": base / "reports",
    }


def ensure_dirs(model_id: str) -> dict[str, Path]:
    p = paths(model_id)
    for k in ("onnx_dir", "surgery_dir", "compile_dir", "reports"):
        p[k].mkdir(parents=True, exist_ok=True)
    return p


def compile_input_onnx(model_id: str) -> Path:
    """The ONNX the compiler should consume: the surgery output if present, else the raw export."""
    p = paths(model_id)
    return p["surgery"] if p["surgery"].exists() else p["onnx"]


def archive_path(model_id: str) -> Path | None:
    """Find the produced _mpk.tar.gz for a model, if any."""
    hits = sorted(paths(model_id)["compile_dir"].rglob("*_mpk.tar.gz"))
    return hits[0] if hits else None
