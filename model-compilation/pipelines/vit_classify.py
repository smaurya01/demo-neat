#!/usr/bin/env python3
"""ViT (DINOv2 ViT-S/14) ImageNet classification on Modalix — reference pipeline.

Runs the SiMa-validated `vits14` archive (1 .elf / 0 .so), downloaded from the model zoo:
  sima-cli download .../model_zoo/gen2/image_classification/vits14/vits14_mpk.tar.gz

IMPORTANT: this is SiMa's PREPARED ViT ONNX. A stock DINOv2/ViT export compiled with our own ONNX
surgery fragments badly (99 .elf / 195 .so) — see ../work/dinov2_vits14/reports/surgery.md. The
lever is a source-level prepared model, NOT ONNX surgery or compile flags.

Preprocess per SiMa's vits14.yaml recipe: resize 256 -> center-crop 224 -> ImageNet normalize.

Run on the DevKit:
  python pipelines/vit_classify.py --archive work/dinov2_vits14/official/vits14_mpk.tar.gz \
      --images assets/yolo_inference --limit 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyneat

SIZE = 224
RESIZE = 256


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """resize shorter side to 256, center-crop 224, RGB, ImageNet-normalise -> HWC f32."""
    h, w = bgr.shape[:2]
    scale = RESIZE / min(h, w)
    img = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_LINEAR)
    h2, w2 = img.shape[:2]
    top, left = (h2 - SIZE) // 2, (w2 - SIZE) // 2
    img = img[top:top + SIZE, left:left + SIZE]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray((rgb - mean) / std, dtype=np.float32)


def tensor_from_hwc(arr: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(arr, dtype=np.float32), copy=True,
        layout=pyneat.TensorLayout.HWC, memory=pyneat.TensorMemory.EV74,
    )


def iter_tensors(sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    elif sample.kind == pyneat.SampleKind.TensorSet:
        yield from sample.tensors
    for f in sample.fields:
        yield from iter_tensors(f)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default="assets/labels/imagenet_classes.txt")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--timeout-ms", type=int, default=20000)
    args = ap.parse_args()

    labels = []
    lp = Path(args.labels)
    if lp.exists():
        labels = [l.strip() for l in lp.read_text().splitlines() if l.strip()]

    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = SIZE
    opt.preprocess.input_max_height = SIZE
    opt.preprocess.input_max_depth = 3
    model = pyneat.Model(args.archive, opt)

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    graph.add(pyneat.groups.mla(model))
    graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
    graph.add(pyneat.nodes.output())
    runner = graph.build([tensor_from_hwc(np.zeros((SIZE, SIZE, 3), dtype=np.float32))])
    print(f"model built: {args.archive}")

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[: args.limit]
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        if not runner.push([tensor_from_hwc(preprocess(bgr))]):
            raise RuntimeError("push failed")
        sample = runner.pull(timeout_ms=args.timeout_ms)
        if sample is None:
            raise RuntimeError("pull returned no sample")

        arrays = [np.asarray(t.to_numpy(copy=True)) for t in iter_tensors(sample)]
        logits = np.asarray(arrays[0], dtype=np.float32).reshape(-1)
        probs = softmax(logits)
        top = probs.argsort()[::-1][: args.topk]

        print(f"\n{p.name}  out_shape={arrays[0].shape}")
        for i in top:
            name = labels[i] if i < len(labels) else f"class_{i}"
            print(f"   {name:<28} {probs[i]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
