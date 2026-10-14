#!/usr/bin/env python3
"""STEP 4 - validate the archive contract, then run it on REAL images.

    # host: contract check only (no board needed)
    python compile/test_model.py --model-id yolo11n --validate-only
    python compile/test_model.py --all --validate-only

    # DevKit: contract check + real inference
    dk compile/test_model.py --model-id resnet50

    # a batch-N build (compiler.py --batch-size N)
    python compile/test_model.py --model-id yolo26m --precision int8 --batch-size 6 --validate-only
    dk compile/test_model.py --model-id yolo26m --precision int8 --batch-size 6         # DevKit

TWO things are checked, and both matter:

  1. ARCHIVE CONTRACT -- exactly one `.elf`, zero `.so`.
     A `.so` means part of the graph fell back to the host CPU. NOTE: a `.so` also carries the
     \\x7fELF magic, so count by FILE EXTENSION, never by magic bytes, or you double-count.
     An rc=0 compile is NOT a passing artifact -- always check the members.

     For a batch-N build the archive must also declare batch N (its plugins' actual_batch_size).

  2. REAL INFERENCE on assets/inference (real COCO images, never synthetic).
     classification -> 1000 finite class scores; the top-k (--topk, default 3) is printed.
     detection/seg/pose -> one finite tensor per `output_names` entry; their shapes are printed.
     batch N -> N images go in as ONE [N,H,W,3] tensor. When the batch-1 build of the same model
     and precision exists, every batch item's outputs are compared with that build's output for
     the same image and the worst cosine is reported. Both builds share their quantization
     parameters, so a cosine below 0.999 means the batch-N program computes different numbers:
     compare the accuracy and latency of the two builds before choosing one (MULTI-BATCH.md).
     yolo26m INT8 batch 6 reaches a worst cosine of 0.92 and finds fewer objects than batch 1.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from common import ROOT, all_model_ids, archive_path, load_registry, model_batch_size, model_cfg

# int8 / bf16: which build of the model to test (None = the model's precision in models.yaml).
PRECISION: str | None = None
# Which batch-size build to test (None = the model's batch_size in models.yaml, else 1).
BATCH: int | None = None


def archive_batch(arc: Path) -> int:
    """The batch size the archive was compiled for: the largest actual_batch_size its plugins declare."""
    with tarfile.open(arc, "r:gz") as t:
        member = next((m for m in t.getmembers() if m.name.endswith("_mpk.json")), None)
        if member is None:
            return 1
        doc = json.load(t.extractfile(member))
    found = [1]

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "actual_batch_size" and isinstance(v, int):
                    found.append(v)
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(doc)
    return max(found)


# ---------------------------------------------------------------- 1. archive contract
def validate_archive(model_id: str) -> tuple[bool, str]:
    arc = archive_path(model_id, PRECISION, model_batch_size(model_id, BATCH))
    if arc is None:
        return False, "no _mpk.tar.gz produced"
    want = model_batch_size(model_id, BATCH)
    got = archive_batch(arc)
    with tarfile.open(arc, "r:gz") as t:
        names = t.getnames()
    # count by EXTENSION: a .so also has ELF magic and would be double-counted otherwise
    elf = [n for n in names if n.endswith(".elf")]
    so = [n for n in names if n.endswith(".so") or ".so." in n]
    ok = len(elf) == 1 and len(so) == 0 and got == want
    detail = f"elf={len(elf)} so={len(so)} batch={got}  ({arc.parent.parent.name}/{arc.name})"
    if so:
        detail += f"  -> HOST FALLBACK: {so[:3]}"
    if got != want:
        detail += f"  -> archive is batch {got}, expected {want}"
    return ok, detail


# ---------------------------------------------------------------- 2. real inference
def run_inference(model_id: str, limit: int, topk: int, timeout_ms: int) -> int:
    import cv2
    import numpy as np
    import pyneat

    cfg, project = model_cfg(model_id)
    arc = archive_path(model_id, PRECISION, model_batch_size(model_id, BATCH))
    if arc is None:
        print(f"   FAIL: {model_id}: no archive; compile first")
        return 1
    batch = model_batch_size(model_id, BATCH)
    if batch > 1:
        return run_batched(model_id, arc, batch, limit, timeout_ms)

    infer_dir = ROOT / project.get("infer_dir", "assets/inference")
    imgs = sorted(p for p in infer_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:limit]
    if not imgs:
        print(f"   FAIL: no images in {infer_dir}")
        return 1

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
    rc, done = 0, 0
    try:
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"   {p.name:<22} FAIL: cannot read the image"); rc = 1
                continue
            img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arr = np.ascontiguousarray((rgb - mean) / std, dtype=np.float32)

            outs = [np.asarray(t.to_numpy(copy=True))
                    for t in runner.run([to_tensor(arr)], timeout_ms=timeout_ms)]
            done += 1
            if not all(np.isfinite(np.asarray(a, dtype=np.float32)).all() for a in outs):
                print(f"   {p.name:<22} FAIL: non-finite values in the output"); rc = 1

            if decode == "imagenet_topk":
                logits = np.asarray(outs[0], dtype=np.float32).reshape(-1)
                if logits.size != 1000:
                    print(f"   {p.name:<22} FAIL: {logits.size} class scores, expected 1000"); rc = 1
                    continue
                e = np.exp(logits - logits.max())
                prob = e / e.sum()
                top = prob.argsort()[::-1][:topk]
                named = ", ".join(
                    f"{(labels[i] if i < len(labels) else f'class_{i}')} {prob[i]:.2f}" for i in top)
                print(f"   {p.name:<22} {named}")
            else:  # yolo_raw_heads -- confirm the surgery contract came back intact
                shapes = " ".join(str(a.shape) for a in outs)
                print(f"   {p.name:<22} {len(outs)} head tensor(s): {shapes}")
                want = len(cfg.get("output_names") or [])
                if want and len(outs) != want:
                    print(f"   FAIL: {len(outs)} head tensors, models.yaml lists {want}"); rc = 1
    finally:
        runner.close()
    if done == 0:
        print("   FAIL: no image was run"); rc = 1
    return rc


def preprocess(cfg: dict, path: Path):
    """Resize + RGB + /255 + mean/std, as the batch-1 path does: one [H,W,3] float32 image."""
    import cv2
    import numpy as np

    _, _, h, w = cfg["input_shape"]
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"cannot read the image {path.name}")
    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(cfg.get("mean", [0, 0, 0]), dtype=np.float32)
    std = np.array(cfg.get("std", [1, 1, 1]), dtype=np.float32)
    return np.ascontiguousarray((rgb - mean) / std, dtype=np.float32)


def raw_runner(arc: Path, cfg: dict, seed):
    """Model on the tensor route (raw outputs, no box decode), built for `seed`'s shape."""
    import pyneat

    _, _, h, w = cfg["input_shape"]
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = w
    opt.preprocess.input_max_height = h
    opt.preprocess.input_max_depth = 3
    model = pyneat.Model(str(arc), opt)
    t = batch_tensor(seed)
    return model, model.build([t])


def batch_tensor(a):
    """[H,W,3] or [N,H,W,3] float32 -> EV74 tensor. A batch must be tagged N/H/W/C for Neat."""
    import numpy as np
    import pyneat

    t = pyneat.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32), copy=True,
                                 layout=pyneat.TensorLayout.HWC, memory=pyneat.TensorMemory.EV74)
    if a.ndim == 4:
        A = pyneat.TensorAxisSemantic
        t.axis_semantics = [A.N, A.H, A.W, A.C]
    return t


def run_batched(model_id: str, arc: Path, batch: int, limit: int, timeout_ms: int = 20000) -> int:
    """Run `batch` real images as one [N,H,W,3] tensor; compare each item with the batch-1 build."""
    import numpy as np

    cfg, project = model_cfg(model_id)
    infer_dir = ROOT / project.get("infer_dir", "assets/inference")
    imgs = sorted(p for p in infer_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        print(f"   FAIL: no images in {infer_dir}")
        return 1
    picked = [imgs[i % len(imgs)] for i in range(batch)]      # repeat images to fill the batch
    stack = np.stack([preprocess(cfg, p) for p in picked])

    model, runner = raw_runner(arc, cfg, stack)
    try:
        print(f"[test] {model_id}: batch {model.compiled_batch_size()} archive, "
              f"{batch} image(s) as one {list(stack.shape)} tensor")
        outs = [np.asarray(t.to_numpy(copy=True))
                for t in runner.run([batch_tensor(stack)], timeout_ms=timeout_ms)]
        print(f"   {len(outs)} output tensor(s): {' '.join(str(a.shape) for a in outs)}")
    finally:
        runner.close()

    if any(a.shape[:1] != (batch,) for a in outs):
        print(f"   FAIL: outputs do not start with the batch dimension {batch}")
        return 1
    ref_arc = archive_path(model_id, PRECISION, 1)
    if ref_arc is None:
        print("   no batch-1 build to compare with; compile it to check the items one by one")
        return 0
    _, ref = raw_runner(ref_arc, cfg, stack[0])
    worst = 1.0
    try:
        for i, p in enumerate(picked):
            ref_outs = [np.asarray(t.to_numpy(copy=True))
                        for t in ref.run([batch_tensor(stack[i])], timeout_ms=timeout_ms)]
            cos = []
            for a, r in zip(outs, ref_outs):
                x, y = a[i].astype(np.float64).ravel(), r.astype(np.float64).ravel()
                if x.size != y.size:
                    print(f"   FAIL: item {i} head size {x.size} vs batch-1 {y.size}")
                    return 1
                den = np.linalg.norm(x) * np.linalg.norm(y)
                cos.append(float(x @ y / den) if den else float(np.allclose(x, y)))
            worst = min(worst, min(cos))
            print(f"   item {i} {p.name:<22} vs batch-1: cosine min {min(cos):.5f} over {len(cos)} heads")
    finally:
        ref.close()
    if worst >= 0.999:
        print(f"   OK: batch items match the batch-1 build (worst cosine {worst:.5f})")
    else:
        print(f"   NOTE: batch items differ from the batch-1 build (worst cosine {worst:.5f}; the two "
              f"builds share their quantization, so the batch-{batch} program computes different "
              f"numbers). Compare the accuracy and latency of batch 1 and batch {batch} before "
              f"choosing one: MULTI-BATCH.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--precision", choices=["int8", "bf16"], default=None,
                    help="test the int8 or bf16 build (default: the model's precision in models.yaml)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="test the batch-N build (default: the model's batch_size in models.yaml, else 1)")
    ap.add_argument("--validate-only", action="store_true", help="archive contract only; no board")
    ap.add_argument("--limit", type=int, default=3,
                    help="images to run (batch 1); a batch-N build always runs one batch of N images")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--timeout-ms", type=int, default=20000)
    args = ap.parse_args()
    global PRECISION, BATCH
    PRECISION = args.precision
    BATCH = args.batch_size

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
        if args.validate_only:
            continue
        try:
            if run_inference(mid, args.limit, args.topk, args.timeout_ms):
                failed.append(mid)
        except Exception as e:            # report it and go on to the next model
            print(f"   FAIL: {type(e).__name__}: {e}")
            failed.append(mid)

    if failed:
        print(f"\nFAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("\nall archives: one .elf, zero .so")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
