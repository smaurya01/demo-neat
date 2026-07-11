#!/usr/bin/env python3
"""DETR object detection on Modalix — reference pipeline.

Runs the SiMa-validated DETR archive
(`detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz`, 1 .elf / 0 .so) over a folder of
images and prints/saves decoded detections.

IMPORTANT — how this model was obtained:
  This uses SiMa's **pre-modified** DETR (`..._modified_class_embed_bbox_embed`), downloaded with
  `sima-cli download`. DETR compiles CLEANLY (single ELF) when the class_embed/bbox_embed heads are
  rewritten at the SOURCE level before export. Compiling a stock DETR export with ONNX-level surgery
  fragments badly — see ../work/detr_resnet50/reports/surgery.md.

DETR emits two raw tensors and has NO built-in Neat box decode (BoxDecodeType.Detr is an enum token
only), so the decode below runs on the host:
  logits [100,92] -> softmax, drop the last "no-object" class, take max/argmax
  boxes  [100,4]  -> sigmoid, cxcywh (normalised) -> xyxy, scale, map back to the original image

Preprocess and decode are ported from the validated example
apps/examples/object-detection/detr-object-detector.

Run on the DevKit (/workspace is NFS-mounted, so no copying):
  python pipelines/detr_detect.py --archive work/detr_resnet50/official/<archive>.tar.gz \
      --images assets/yolo_inference --limit 3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyneat

FRAME_WIDTH = 1333          # model input W
FRAME_HEIGHT = 800          # model input H
PERSON_CLASS_ID = 1

DETR_COCO_LABELS = [

    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass
class PreprocMeta:
    orig_h: int
    orig_w: int
    pad_top: int
    pad_left: int
    scale: float


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def class_name(cid: int) -> str:
    if 0 <= cid < len(DETR_COCO_LABELS) and DETR_COCO_LABELS[cid] != "N/A":
        return DETR_COCO_LABELS[cid]
    return f"class_{cid}"


def preprocess_bgr(image_bgr: np.ndarray) -> tuple[np.ndarray, PreprocMeta]:
    """Aspect-preserving resize + centre-pad to the model frame, RGB, ImageNet-normalised HWC f32."""
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(FRAME_WIDTH / float(orig_w), FRAME_HEIGHT / float(orig_h))
    rw, rh = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    pad_left, pad_top = (FRAME_WIDTH - rw) // 2, (FRAME_HEIGHT - rh) // 2

    resized = cv2.resize(image_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    canvas[pad_top:pad_top + rh, pad_left:pad_left + rw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    chw = np.ascontiguousarray((rgb - mean) / std, dtype=np.float32)
    return chw, PreprocMeta(orig_h, orig_w, pad_top, pad_left, scale)


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


def split_logits_boxes(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Route the two DETR outputs BY SHAPE (never by index order)."""
    logits = boxes = None
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim < 2:
            continue
        flat = a.reshape(-1, a.shape[-1])
        if flat.shape[-1] == 4:
            boxes = flat
        elif flat.shape[-1] > 4:
            logits = flat
    if logits is None or boxes is None:
        raise RuntimeError("expected DETR logits + boxes in model output")
    return logits, boxes


def decode(logits, boxes, meta: PreprocMeta, conf: float, person_only: bool) -> list[dict]:
    boxes = sigmoid(boxes)
    prob = softmax(logits, axis=-1)
    scores = prob[..., :-1].max(axis=-1)          # drop the trailing "no-object" class
    class_ids = prob[..., :-1].argmax(axis=-1)

    keep = scores > conf
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    if person_only:
        m = class_ids == PERSON_CLASS_ID
        boxes, scores, class_ids = boxes[m], scores[m], class_ids[m]
    if len(boxes) == 0:
        return []

    x_c, y_c, w, h = boxes.T                       # cxcywh, normalised to the padded frame
    x1 = (x_c - 0.5 * w) * FRAME_WIDTH
    y1 = (y_c - 0.5 * h) * FRAME_HEIGHT
    x2 = (x_c + 0.5 * w) * FRAME_WIDTH
    y2 = (y_c + 0.5 * h) * FRAME_HEIGHT

    # undo the centre-pad + resize, back to original image pixels
    out = []
    for i in range(len(scores)):
        ox1 = float(np.clip((x1[i] - meta.pad_left) / meta.scale, 0, meta.orig_w))
        oy1 = float(np.clip((y1[i] - meta.pad_top) / meta.scale, 0, meta.orig_h))
        ox2 = float(np.clip((x2[i] - meta.pad_left) / meta.scale, 0, meta.orig_w))
        oy2 = float(np.clip((y2[i] - meta.pad_top) / meta.scale, 0, meta.orig_h))
        out.append({"bbox": (ox1, oy1, ox2, oy2), "score": float(scores[i]),
                    "class_id": int(class_ids[i]), "label": class_name(int(class_ids[i]))})
    return sorted(out, key=lambda d: -d["score"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="DETR .tar.gz model package")
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.70)
    ap.add_argument("--person-only", action="store_true")
    ap.add_argument("--timeout-ms", type=int, default=20000)
    ap.add_argument("--save-dir", default=None, help="write annotated images here")
    args = ap.parse_args()

    # tensor-input route: we preprocess on the host, the model takes a ready tensor
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = FRAME_WIDTH
    opt.preprocess.input_max_height = FRAME_HEIGHT
    opt.preprocess.input_max_depth = 3
    model = pyneat.Model(args.archive, opt)

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    graph.add(pyneat.groups.mla(model))
    graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
    graph.add(pyneat.nodes.output())
    dummy = tensor_from_hwc(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32))
    runner = graph.build([dummy])
    print(f"model built: {args.archive}")

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[: args.limit]
    if not paths:
        raise SystemExit(f"no images in {args.images}")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    total = 0
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"  {p.name}: unreadable, skipped")
            continue
        pre, meta = preprocess_bgr(bgr)

        if not runner.push([tensor_from_hwc(pre)]):
            raise RuntimeError("Run.push() failed")
        sample = runner.pull(timeout_ms=args.timeout_ms)
        if sample is None:
            raise RuntimeError("Run.pull() returned no sample")

        arrays = [np.asarray(t.to_numpy(copy=True)) for t in iter_tensors(sample)]
        logits, boxes = split_logits_boxes(arrays)
        dets = decode(logits, boxes, meta, args.conf, args.person_only)
        total += len(dets)

        shapes = " ".join(str(a.shape) for a in arrays)
        print(f"\n{p.name}  ({bgr.shape[1]}x{bgr.shape[0]})  raw_out={shapes}")
        if not dets:
            print("   no detections above threshold")
        for d in dets[:10]:
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            print(f"   {d['label']:<16} {d['score']:.2f}  [{x1},{y1},{x2},{y2}]")

        if args.save_dir:
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d["bbox"])
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(bgr, f"{d['label']} {d['score']:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(Path(args.save_dir) / p.name), bgr)

    print(f"\ntotal detections: {total} across {len(paths)} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
