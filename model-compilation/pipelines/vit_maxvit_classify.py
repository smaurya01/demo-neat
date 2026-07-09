#!/usr/bin/env python3
"""Reference pipeline: ImageNet top-k classification for a compiled ViT / MaxViT.

Minimal, board-runnable. The compiled INT8 archive exposes a single `logits`
[1,1000] tensor (the graph's classifier head); ALL postprocess (softmax + top-k
+ label lookup) is CPU-side here — nothing task-specific was baked into the MLA
graph. This is the transformer analogue of a CNN classifier: after the attention
surgery (see work/<id>/reports/compile_ready_surgery.json) a ViT runs the same
"one image in, logits out" contract as resnet50.

Run on the DevKit (ssh; dk needs a TTY):
    timeout 180 ssh -o BatchMode=yes sima@192.168.135.203 \
      'source $HOME/pyneat/bin/activate; \
       python /workspace/demo-neat/model-compilation/pipelines/vit_maxvit_classify.py \
         --archive /workspace/.../vit_b_16.compile_ready_mpk.tar.gz \
         --image /workspace/demo-neat/model-compilation/assets/yolo_inference/000000000139.jpg'

API source: pyneat.Model + Model.run([Tensor]) — the same route as
scripts/06_neat_smoke_test.py (verified house pattern for NCHW torchvision models).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_labels(path: Path):
    if path.exists():
        return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]
    return None


def preprocess(image_path: str, size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)  # [1,3,H,W] NCHW
    return chw


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--input-size", type=int, default=224, help="224 for vit_b_16 / maxvit_t")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--labels", default=str(ROOT / "assets" / "labels" / "imagenet_classes.txt"))
    ap.add_argument("--timeout-ms", type=int, default=15000)
    args = ap.parse_args()

    import pyneat

    labels = load_labels(Path(args.labels))
    chw = preprocess(args.image, args.input_size)
    tensor = pyneat.Tensor.from_numpy(chw, copy=True)

    model = pyneat.Model(args.archive)
    outputs = model.run([tensor], timeout_ms=args.timeout_ms)

    logits = np.asarray(outputs[0]).reshape(-1)
    if logits.shape[0] != 1000:
        print(f"[warn] expected 1000 logits, got {logits.shape[0]}")
    probs = softmax(logits.astype(np.float32))
    topk = np.argsort(-probs)[: args.topk]

    print(f"image: {args.image}")
    print(f"logits shape: {list(np.asarray(outputs[0]).shape)}")
    print(f"top-{args.topk}:")
    for rank, idx in enumerate(topk, 1):
        name = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
        print(f"  {rank}. [{idx:4d}] {name:<30s} p={probs[idx]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
