#!/usr/bin/env python3
"""STEP 3 - quantize (INT8) + compile to a Modalix MPK archive. Calibration lives here.

    python compile/compiler.py --model-id yolo11n
    python compile/compiler.py --all

CALIBRATION (the part people get wrong):
  Quantization needs REAL images from the target domain to learn activation ranges. Synthetic /
  gradient images produce an archive that *looks* fine (rc=0, one ELF) but whose ranges are
  meaningless -- the model is quietly wrong. This script therefore:
    * defaults to assets/calibration (real COCO images), and
    * REFUSES to run if the calibration set looks synthetic.

TARGET CONTRACT: exactly one `.elf` and zero `.so`.
  A `.so` means part of the graph fell back to the host CPU (A65) -- i.e. surgery did not remove
  everything the MLA cannot place. The compile log's "Plugin distribution ... A65: 0" is the
  signal to check. Validate with test_model.py, which fails on any `.so`.

Input : work/<id>/surgery/<id>.compile_ready.onnx  (or the raw ONNX when surgery: none)
Output: work/<id>/compile_int8/<...>_mpk.tar.gz    (+ reports/compile.log, compile.command.txt)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import (ROOT, all_model_ids, compile_input_onnx, ensure_dirs, load_registry,
                    model_cfg, paths)

# The SDK's quantize+compile driver (from the sima-model-quantize-compile skill).
QC = Path.home() / ".codex/skills/sima-model-quantize-compile/scripts/quantize_compile.py"

SYNTHETIC_MARKERS = ("synthetic", "dummy", "gradient", "random", "noise")


def assert_real_calibration(calib_dir: Path, min_images: int = 8) -> list[Path]:
    """Refuse to calibrate on synthetic data -- it silently produces a wrong model."""
    if not calib_dir.is_dir():
        raise SystemExit(f"[compile] calibration dir not found: {calib_dir}")
    imgs = sorted(p for p in calib_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(imgs) < min_images:
        raise SystemExit(f"[compile] only {len(imgs)} calibration images in {calib_dir}; "
                         f"need >= {min_images} real images")
    bad = [p.name for p in imgs if any(m in p.name.lower() for m in SYNTHETIC_MARKERS)]
    if bad:
        raise SystemExit(
            f"[compile] REFUSING: calibration set looks SYNTHETIC ({', '.join(bad[:3])}...).\n"
            f"          Quantization must use real images from the target domain, or the archive\n"
            f"          will compile cleanly but be quietly wrong. Point --calib-dir at real images."
        )
    return imgs


def run(model_id: str, calib_dir: Path | None, num_calib: int | None, extra: list[str]) -> int:
    cfg, project = model_cfg(model_id)
    p = ensure_dirs(model_id)

    onnx_path = compile_input_onnx(model_id)
    if not onnx_path.exists():
        raise SystemExit(f"[compile] {model_id}: no ONNX at {onnx_path} -- run steps 1/2 first")

    calib = calib_dir or (ROOT / project.get("calib_dir", "assets/calibration"))
    imgs = assert_real_calibration(calib)
    n = num_calib or project.get("default_num_calib_samples", 20)

    cmd = [
        sys.executable, str(QC),
        "--model_path", str(onnx_path),
        "--model_format", "onnx",
        "--model_layout", project.get("default_layout", "NCHW"),
        "--input_names", cfg["input_name"],
        "--input_shapes", ",".join(str(x) for x in cfg["input_shape"]),
        "--device", project.get("default_device", "modalix"),
        "--build_dir", str(p["compile_dir"]),
        "--calib_method", project.get("default_calib_method", "mse"),
        "--requant_mode", project.get("default_requant_mode", "sima"),
        "--mean", *[str(x) for x in cfg.get("mean", [0, 0, 0])],
        "--std", *[str(x) for x in cfg.get("std", [1, 1, 1])],
        "--real_data",
        "--dataset_images", str(calib),
        "--num_calib_samples", str(n),
    ]
    if cfg.get("output_names"):
        cmd += ["--output_names", *cfg["output_names"]]
    cmd += extra

    (p["reports"] / "compile.command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    print(f"[compile] {model_id}: onnx={onnx_path.name} calib={calib.name} ({len(imgs)} real imgs, "
          f"using {n})")

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (p["reports"] / "compile.log").write_text(proc.stdout, encoding="utf-8")

    dist = [l for l in proc.stdout.splitlines() if "A65" in l or "Plugin distribution" in l]
    for l in dist[-3:]:
        print("   ", l.strip())
    print(f"[compile] {model_id}: rc={proc.returncode}")
    if proc.returncode != 0:
        print(f"   see {p['reports'] / 'compile.log'}")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="override the calibration image dir (must be REAL images)")
    ap.add_argument("--num-calib-samples", type=int, default=None)
    # unknown flags (e.g. --calib_method min_max) pass straight through to quantize_compile.py
    args, extra = ap.parse_known_args()

    ids = all_model_ids() if args.all else [args.model_id]
    if not ids or ids == [None]:
        raise SystemExit("give --model-id <id> or --all")

    rc = 0
    for mid in ids:                       # strictly serial: one compile at a time
        rc |= run(mid, args.calib_dir, args.num_calib_samples, extra)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
