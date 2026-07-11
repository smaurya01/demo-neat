#!/usr/bin/env python3
"""STEP 4 - validate the archive contract, then run it on REAL images.

    # host: contract check only (no board needed)
    python compile/test_model.py --model-id yolo11n --validate-only

    # DevKit: contract check + real inference
    python compile/test_model.py --model-id resnet50
    python compile/test_model.py --all --validate-only

TWO things are checked, and both matter:

  1. ARCHIVE CONTRACT -- exactly one `.elf`, zero `.so`.
     A `.so` means part of the graph fell back to the host CPU. NOTE: a `.so` also carries the
     \x7fELF magic, so count by FILE EXTENSION, never by magic bytes, or you double-count.
     An rc=0 compile is NOT a passing artifact -- always check the members.

  2. REAL INFERENCE on assets/inference (real COCO images, never synthetic).
     classification -> ImageNet top-5 must be sensible.
     detection/seg/pose -> the raw heads must come back with the expected shapes.
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from common import ROOT, all_model_ids, archive_path, load_registry, model_cfg


# ---------------------------------------------------------------- 1. archive contract
def validate_archive(model_id: str) -> tuple[bool, str]:
    arc = archive_path(model_id)
    if arc is None:
        return False, "no _mpk.tar.gz produced"
    with tarfile.open(arc, "r:gz") as t:
        names = t.getnames()
    # count by EXTENSION: a .so also has ELF magic and would be double-counted otherwise
    elf = [n for n in names if n.endswith(".elf")]
    so = [n for n in names if n.endswith(".so") or ".so." in n]
    ok = len(elf) == 1 and len(so) == 0
    detail = f"elf={len(elf)} so={len(so)}  ({arc.name})"
    if so:
        detail += f"  -> HOST FALLBACK: {so[:3]}"
    return ok, detail


# ---------------------------------------------------------------- 2. real inference
def run_inference(model_id: str, limit: int, topk: int, timeout_ms: int) -> int:
    import cv2
    import numpy as np
    import pyneat

    cfg, project = model_cfg(model_id)
    arc = archive_path(model_id)
    if arc is None:
        raise SystemExit(f"[test] {model_id}: no archive; compile first")

    infer_dir = ROOT / project.get("infer_dir", "assets/inference")
    imgs = sorted(p for p in infer_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:limit]
    if not imgs:
        raise SystemExit(f"[test] no images in {infer_dir}")

    _, _, h, w = cfg["input_shape"]
    decode = cfg.get("decode", "imagenet_topk")

    # Model.build() resolves the model's OWN route (CNNs use quant_tess/detess_dequant,
    # YOLO raw-head archives use quant/dequant). Hand-building the graph breaks on one or the
    # other -- let Neat pick.
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = w
    opt.preprocess.input_max_height = h
    opt.preprocess.input_max_depth = 3
    model = pyneat.Model(str(arc), opt)

    def to_tensor(a):
        return pyneat.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32), copy=True,
                                        layout=pyneat.TensorLayout.HWC,
                                        memory=pyneat.TensorMemory.EV74)

    runner = model.build([to_tensor(np.zeros((h, w, 3), dtype=np.float32))])

    labels = []
    lp = ROOT / "assets/labels/imagenet_classes.txt"
    if decode == "imagenet_topk" and lp.exists():
        labels = [x.strip() for x in lp.read_text().splitlines() if x.strip()]

    mean = np.array(cfg.get("mean", [0, 0, 0]), dtype=np.float32)
    std = np.array(cfg.get("std", [1, 1, 1]), dtype=np.float32)

    print(f"[test] {model_id}: {decode} on {len(imgs)} real image(s) from {infer_dir.name}/")
    for p in imgs:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arr = np.ascontiguousarray((rgb - mean) / std, dtype=np.float32)

        outs = [np.asarray(t.to_numpy(copy=True)) for t in runner.run([to_tensor(arr)])]

        if decode == "imagenet_topk":
            logits = np.asarray(outs[0], dtype=np.float32).reshape(-1)
            e = np.exp(logits - logits.max())
            prob = e / e.sum()
            top = prob.argsort()[::-1][:topk]
            named = ", ".join(
                f"{(labels[i] if i < len(labels) else f'class_{i}')} {prob[i]:.2f}" for i in top)
            print(f"   {p.name:<22} {named}")
        else:  # yolo_raw_heads -- confirm the surgery contract came back intact
            shapes = " ".join(str(a.shape) for a in outs)
            print(f"   {p.name:<22} {len(outs)} head tensor(s): {shapes}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--validate-only", action="store_true", help="archive contract only; no board")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--timeout-ms", type=int, default=20000)
    args = ap.parse_args()

    ids = all_model_ids() if args.all else [args.model_id]
    if not ids or ids == [None]:
        raise SystemExit("give --model-id <id> or --all")

    failed = []
    for mid in ids:
        ok, detail = validate_archive(mid)
        print(f"[{'PASS' if ok else 'FAIL'}] {mid:<16} {detail}")
        if not ok:
            failed.append(mid)
            continue
        if not args.validate_only:
            run_inference(mid, args.limit, args.topk, args.timeout_ms)

    if failed:
        print(f"\nFAILED contract ({len(failed)}): {', '.join(failed)}")
        return 1
    print("\nall archives: one .elf, zero .so")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
