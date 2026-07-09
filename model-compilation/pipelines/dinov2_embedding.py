#!/usr/bin/env python3
"""Reference pipeline: DINOv2 embedding extraction + nearest-neighbour sanity.

The compiled INT8 dinov2_vits14 archive exposes a single `features` [1,384]
tensor — the CLS-token embedding. DINOv2 has NO classifier head, so the "label"
of an image is decided by nearest-neighbour in embedding space against a small
labelled reference set. That retrieval step is entirely CPU-side.

Two modes:
  * --gallery DIR --query IMG : embed every image in DIR (label = file stem),
    embed the query, print the top-k most similar gallery images (cosine sim).
  * --images A B C ...        : embed each and print the pairwise cosine-similarity
    matrix (sanity: an image is most similar to itself ~1.0, near-duplicates high).

This proves (a) the embedding shape is right and (b) the embedding is
discriminative — the two checks the T7 validation asks for. Model.run API is the
same house pattern as scripts/06_neat_smoke_test.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: str, size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def embed(model, image_path: str, size: int, timeout_ms: int) -> np.ndarray:
    import pyneat

    tensor = pyneat.Tensor.from_numpy(preprocess(image_path, size), copy=True)
    out = model.run([tensor], timeout_ms=timeout_ms)
    v = np.asarray(out[0]).reshape(-1).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--timeout-ms", type=int, default=15000)
    ap.add_argument("--gallery", default=None, help="directory of labelled reference images")
    ap.add_argument("--query", default=None, help="query image for --gallery mode")
    ap.add_argument("--images", nargs="+", default=None, help="images for pairwise similarity mode")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    import pyneat

    model = pyneat.Model(args.archive)

    if args.gallery and args.query:
        gdir = Path(args.gallery)
        gpaths = sorted(p for p in gdir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        gemb = {p.stem: embed(model, str(p), args.input_size, args.timeout_ms) for p in gpaths}
        q = embed(model, args.query, args.input_size, args.timeout_ms)
        print(f"embedding dim: {q.shape[0]}  (expected 384)")
        sims = sorted(((float(np.dot(q, v)), name) for name, v in gemb.items()), reverse=True)
        print(f"query: {args.query}")
        print(f"nearest {args.topk} gallery labels (cosine sim):")
        for sim, name in sims[: args.topk]:
            print(f"  {name:<30s} sim={sim:.4f}")
        print(f"nearest-label => {sims[0][1]}")
        return 0

    if args.images:
        embs = [embed(model, im, args.input_size, args.timeout_ms) for im in args.images]
        print(f"embedding dim: {embs[0].shape[0]}  (expected 384)")
        print("pairwise cosine similarity:")
        names = [Path(im).name for im in args.images]
        print("        " + " ".join(f"{n[:8]:>8s}" for n in names))
        for i, a in enumerate(embs):
            row = " ".join(f"{float(np.dot(a, b)):8.3f}" for b in embs)
            print(f"{names[i][:8]:>8s} {row}")
        return 0

    ap.error("provide either --gallery DIR --query IMG, or --images A B C ...")


if __name__ == "__main__":
    raise SystemExit(main())
