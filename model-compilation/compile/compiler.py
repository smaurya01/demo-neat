#!/usr/bin/env python3
"""STEP 3 - quantize (INT8 or bf16) + compile to a Modalix MPK archive. Calibration lives here.

    python compile/compiler.py --model-id yolo11n                     # models.yaml precision: int8
    python compile/compiler.py --model-id yolo26m                     # models.yaml precision: bf16
    python compile/compiler.py --model-id yolo26s --precision bf16    # override the registry
    python compile/compiler.py --model-id yolo26m --precision int8 --batch-size 6   # multi-batch
    python compile/compiler.py --all

BATCH SIZE:
  `--batch-size N` (or `batch_size: N` in models.yaml; default 1) compiles the archive to run N
  images per MLA call. Quantization still sees one image at a time (the ONNX input stays
  [1,3,H,W]); the compiler builds the batch-N program and the archive's input becomes [N,H,W,3].
  A batch-N build goes to its own dir, compile_<precision>_b<N>/, next to the batch-1 one. The
  script fails unless the compiler reports "Achieved batch size: N". See MULTI-BATCH.md
  for what Neat 0.5.0 can and cannot do with a batched archive at runtime.

PRECISION:
  int8 quantizes weights and activations to 8 bits using the calibration images below.
  bf16 keeps weights and activations in bfloat16 (`--bf16-weights --bf16-activations`): no
  calibration ranges to get wrong, usually closer to FP32 accuracy, but a larger and slower model.
  The precision comes from the model's `precision:` in models.yaml (every model states it), unless
  --precision overrides it for this one build. Each precision gets its own output dir, so one model
  can be built both ways.

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
Output: work/<id>/compile_<precision>[_b<N>]/<...>_mpk.tar.gz
        (+ reports/compile_<precision>[_b<N>].log, compile_<precision>[_b<N>].command.txt)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from common import (ROOT, all_model_ids, compile_dir_name, compile_input_onnx, ensure_dirs,
                    load_registry, model_batch_size, model_cfg, model_precision, paths)

# The SDK's quantize+compile driver, from the sima-model-quantize-compile skill that the Neat SDK
# installs for its coding agents (~/.codex/skills and ~/.claude/skills). SIMA_QUANTIZE_COMPILE
# overrides it, e.g. inside a container that has neither.
_SKILL = "skills/sima-model-quantize-compile/scripts/quantize_compile.py"
QC = Path(os.environ.get("SIMA_QUANTIZE_COMPILE") or
          next((p for p in (Path.home() / f".{a}" / _SKILL for a in ("codex", "claude")) if p.exists()),
               Path.home() / ".codex" / _SKILL))

SYNTHETIC_MARKERS = ("synthetic", "dummy", "gradient", "random", "noise")
BF16_FLAGS = ["--bf16-weights", "--bf16-activations"]


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


def run(model_id: str, calib_dir: Path | None, num_calib: int | None, precision: str | None,
        batch_size: int | None, extra: list[str]) -> int:
    cfg, project = model_cfg(model_id)
    # Passing `--bf16-weights` by hand (the older way) still selects bf16 and its output dir.
    # `--bf16-activations` alone (bf16 activations, int8 weights) is passed through untouched.
    if "--bf16-weights" in extra:
        if precision == "int8":
            raise SystemExit(f"[compile] {model_id}: --precision int8 conflicts with --bf16-weights")
        precision = "bf16"
        extra = [f for f in extra if f not in BF16_FLAGS]
    prec = model_precision(model_id, precision)
    if prec == "int8" and "--bf16-activations" in extra:
        raise SystemExit(f"[compile] {model_id}: --bf16-activations with int8 weights is neither int8 "
                         f"nor bf16 and would land in compile_int8/; use --precision bf16")
    batch = model_batch_size(model_id, batch_size)
    p = ensure_dirs(model_id, prec, batch)
    tag = compile_dir_name(prec, batch).removeprefix("compile_")   # int8, bf16, int8_b6, ...

    onnx_path = compile_input_onnx(model_id)
    if not onnx_path.exists():
        raise SystemExit(f"[compile] {model_id}: no ONNX at {onnx_path} -- run steps 1/2 first")
    if not QC.is_file():
        raise SystemExit(f"[compile] {model_id}: the SDK's quantize_compile.py driver was not found at "
                         f"{QC}.\n          Install the sima-model-quantize-compile skill, or point "
                         f"SIMA_QUANTIZE_COMPILE at quantize_compile.py (see README > Setup).")

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
    if prec == "bf16":
        cmd += BF16_FLAGS
    if batch > 1:
        cmd += ["--batch_size", str(batch)]
    cmd += extra

    (p["reports"] / f"compile_{tag}.command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    print(f"[compile] {model_id}: precision={prec} batch={batch} onnx={onnx_path.name} "
          f"calib={calib.name} ({len(imgs)} real imgs, using {n})")

    # Clear the previous archive first, so a failed build cannot leave an old one looking current.
    for old in p["compile_dir"].rglob("*_mpk.tar.gz"):
        old.unlink()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = p["reports"] / f"compile_{tag}.log"
    log.write_text(proc.stdout, encoding="utf-8")

    dist = [l for l in proc.stdout.splitlines() if "A65" in l or "Plugin distribution" in l]
    for l in dist[-3:]:
        print("   ", l.strip())
    rc = proc.returncode
    if rc == 0 and batch > 1:
        # The compiler may build a smaller batch than asked for (e.g. when it does not fit in
        # memory) and still return 0. A batch-N dir holding another batch would be wrong.
        achieved = re.findall(r"Achieved batch size: (\d+)", proc.stdout)
        print(f"    desired batch {batch}, achieved {achieved[-1] if achieved else '?'}")
        if not achieved or int(achieved[-1]) != batch:
            print(f"[compile] {model_id}: the compiler did not build batch {batch}; see {log}")
            for a in p["compile_dir"].rglob("*_mpk.tar.gz"):   # keep it, but not as a batch-N archive
                a.rename(a.with_name(a.name + ".wrong_batch"))
            rc = 1
    if rc != 0 and batch > 1 and "Cannot be segmented due to batch size" in proc.stdout:
        # A layer too big for MLA memory is split spatially ("large tensor support"); the compiler
        # cannot also split it by batch. Seen on yolo26x (INT8) and on yolo26m bf16's input.
        layer = re.search(r"CannotSplitLargeTensorError: (\S+?):", proc.stdout)
        print(f"[compile] {model_id}: batch {batch} is not possible for this model at {prec}: layer "
              f"{layer.group(1) if layer else '?'} needs large-tensor splitting, which the compiler "
              f"cannot combine with batching. Use batch 1, or try --precision int8 (half-size "
              f"tensors) if this was bf16.")
    print(f"[compile] {model_id} ({tag}): rc={rc}  -> {p['compile_dir']}")
    if rc != 0:
        print(f"   see {log}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--calib-dir", type=Path, default=None,
                    help="override the calibration image dir (must be REAL images)")
    ap.add_argument("--num-calib-samples", type=int, default=None)
    ap.add_argument("--precision", choices=["int8", "bf16"], default=None,
                    help="override the model's `precision` in models.yaml for this build")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="images per MLA call (default: `batch_size` in models.yaml, else 1); "
                         "batch N > 1 builds into compile_<precision>_b<N>/")
    # unknown flags (e.g. --calib_method min_max) pass straight through to quantize_compile.py
    args, extra = ap.parse_known_args()
    if any(f == "--batch_size" or f.startswith("--batch_size=") for f in extra):
        raise SystemExit("use --batch-size N (it also picks the output dir), not --batch_size")

    ids = all_model_ids() if args.all else [args.model_id]
    if not ids or ids == [None]:
        raise SystemExit("give --model-id <id> or --all")

    rc = 0
    for mid in ids:                       # strictly serial: one compile at a time
        try:
            rc |= run(mid, args.calib_dir, args.num_calib_samples, args.precision, args.batch_size,
                      extra)
        except SystemExit as e:           # with --all, report and go on to the next model
            if len(ids) == 1:
                raise
            print(e); rc |= 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
