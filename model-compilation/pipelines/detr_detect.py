#!/usr/bin/env python3
"""Reference pipeline: DETR (detr_resnet50) detection with CPU postprocess.

The compiled INT8 archive exposes DETR's two raw transformer heads:
    pred_logits [1, 100, 92]   (100 object queries x 91 COCO classes + no-object)
    pred_boxes  [1, 100, 4]    (cxcywh, normalised to [0,1])
Everything after that is CPU-side and lives here:
    1. softmax over the 92-way class axis, per query;
    2. drop the last "no-object" class, take argmax over the 91 real classes;
    3. keep queries whose score exceeds --threshold;
    4. convert cxcywh(normalised) -> xyxy(pixels) using the ORIGINAL image size.
DETR is NMS-free by construction (set prediction), so there is no NMS step; and
Hungarian matching is a TRAINING-only loss component, absent at inference. This
is why DETR needs no box-decode unit on the MLA — the graph ends at the two heads.

Run on the DevKit (ssh):
    timeout 200 ssh -o BatchMode=yes sima@192.168.135.203 \
      'source $HOME/pyneat/bin/activate; \
       python /workspace/demo-neat/model-compilation/pipelines/detr_detect.py \
         --archive /workspace/.../detr_resnet50.compile_ready_mpk.tar.gz \
         --image /workspace/demo-neat/model-compilation/assets/yolo_inference/000000000139.jpg'
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


def preprocess(image_path: str, size: int):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    arr = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return chw, (orig_w, orig_h)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--input-size", type=int, default=800, help="detr_resnet50 export uses 800x800")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--labels", default=str(ROOT / "assets" / "labels" / "coco91_detr.txt"))
    ap.add_argument("--timeout-ms", type=int, default=20000)
    args = ap.parse_args()

    import pyneat

    labels = load_labels(Path(args.labels))
    chw, (orig_w, orig_h) = preprocess(args.image, args.input_size)
    tensor = pyneat.Tensor.from_numpy(chw, copy=True)

    model = pyneat.Model(args.archive)
    outputs = model.run([tensor], timeout_ms=args.timeout_ms)

    # Route BY SHAPE, not by index order: the MLA delivers the raw heads through a
    # single multi-tensor sample and may permute axes (it emits NHWC, not NCHW).
    # pred_logits has a size-92 axis, pred_boxes a size-4 axis; move that axis last.
    def pick(arrs, feat):
        for a in arrs:
            if feat in a.shape:
                ax = list(a.shape).index(feat)
                return np.moveaxis(a, ax, -1).reshape(-1, feat)
        raise SystemExit(f"no output tensor with an axis of size {feat}; shapes={[x.shape for x in arrs]}")

    arrs = [np.asarray(o) for o in outputs]
    logits = pick(arrs, 92)
    boxes = pick(arrs, 4)

    prob = softmax(logits, axis=-1)[:, :-1]      # drop no-object class
    scores = prob.max(axis=-1)
    classes = prob.argmax(axis=-1)
    keep = scores > args.threshold

    xyxy = cxcywh_to_xyxy(boxes)
    scale = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
    xyxy = xyxy * scale

    print(f"image: {args.image}  ({orig_w}x{orig_h})")
    print(f"heads: pred_logits {logits.shape}  pred_boxes {boxes.shape}")
    print(f"detections over threshold {args.threshold}: {int(keep.sum())}")
    order = np.argsort(-scores[keep])
    kept_idx = np.nonzero(keep)[0][order]
    for i in kept_idx:
        c = int(classes[i])
        name = labels[c] if labels and c < len(labels) else f"class_{c}"
        x1, y1, x2, y2 = xyxy[i]
        print(f"  {name:<15s} score={scores[i]:.3f}  box=[{x1:6.1f},{y1:6.1f},{x2:6.1f},{y2:6.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
