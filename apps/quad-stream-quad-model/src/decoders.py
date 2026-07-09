"""Host-side (A65/CPU) task decoders for the quad-stream-quad-model pipeline.

WHY THIS FILE EXISTS — the load-bearing design lesson of this app:

The four compiled archives all expose *raw per-scale head tensors* (the surgery
in model-compilation/work/<model>/reports/SURGERY.md deliberately cuts the
data-dependent decode/NMS tail so it stays MLA-clean, A65:0). Neat's built-in
``BoxDecodeType`` gives a fused on-device decode for the plain *detection*
family, and that is what stream 0 (yolo11s) uses. For **segmentation, pose and
YOLOX** this app decodes the raw heads here, on the host, in NumPy:

  * the MLA hands you calibrated FP32 NCHW head tensors (detess+dequant done for
    you),
  * *you* own the anchor-grid + stride geometry, the sigmoid/exp activations,
    score thresholding, NMS, the letterbox-inverse back to frame pixels, and —
    for seg — the prototype-mask assembly.

Every tensor is routed by SHAPE, not by output name, so the decoders do not
depend on the compiler's output ordering. Shapes are unambiguous:

  bbox (post-DFL / yolo26)    C == 4        class(det/seg) C == 80
  class(pose)                 C == 1        mask_coeff     C == 32, H in {80,40,20}
  keypoints (pose)            C == 51       proto          C == 32, H == 160
  yolox head                  C == 85

Grid conventions (Ultralytics / YOLOX):
  * detection & seg (yolo11): anchor point = (i+0.5, j+0.5), stride = 640/H,
    dist2bbox l/t/r/b.
  * pose (yolo26): same box decode; keypoint xy = (kxy*2 + i) * stride.
  * yolox: integer grid (no +0.5), xy = (reg_xy + i) * stride,
    wh = exp(reg_wh) * stride.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── COCO labels ──────────────────────────────────────────────────────────────
COCO_LABELS = [
    "PERSON", "BICYCLE", "CAR", "MOTORCYCLE", "AIRPLANE", "BUS", "TRAIN", "TRUCK",
    "BOAT", "TRAFFIC LIGHT", "FIRE HYDRANT", "STOP SIGN", "PARKING METER", "BENCH",
    "BIRD", "CAT", "DOG", "HORSE", "SHEEP", "COW", "ELEPHANT", "BEAR", "ZEBRA",
    "GIRAFFE", "BACKPACK", "UMBRELLA", "HANDBAG", "TIE", "SUITCASE", "FRISBEE",
    "SKIS", "SNOWBOARD", "SPORTS BALL", "KITE", "BASEBALL BAT", "BASEBALL GLOVE",
    "SKATEBOARD", "SURFBOARD", "TENNIS RACKET", "BOTTLE", "WINE GLASS", "CUP",
    "FORK", "KNIFE", "SPOON", "BOWL", "BANANA", "APPLE", "SANDWICH", "ORANGE",
    "BROCCOLI", "CARROT", "HOT DOG", "PIZZA", "DONUT", "CAKE", "CHAIR", "COUCH",
    "POTTED PLANT", "BED", "DINING TABLE", "TOILET", "TV", "LAPTOP", "MOUSE",
    "REMOTE", "KEYBOARD", "CELL PHONE", "MICROWAVE", "OVEN", "TOASTER", "SINK",
    "REFRIGERATOR", "BOOK", "CLOCK", "VASE", "SCISSORS", "TEDDY BEAR", "HAIR DRIER",
    "TOOTHBRUSH",
]

# COCO-17 keypoint skeleton (Ultralytics pose order), 0-based indices.
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


def class_label(class_id: int) -> str:
    if 0 <= class_id < len(COCO_LABELS):
        return COCO_LABELS[class_id]
    return f"CLASS {class_id}"


# ── result containers ────────────────────────────────────────────────────────
@dataclass
class LetterboxGeom:
    """Inverse mapping from 640x640 letterboxed model space back to frame px."""
    scale: float
    pad_x: float
    pad_y: float

    @classmethod
    def compute(cls, frame_w: int, frame_h: int, model_w: int, model_h: int) -> "LetterboxGeom":
        scale = min(model_w / frame_w, model_h / frame_h)
        new_w, new_h = frame_w * scale, frame_h * scale
        return cls(scale=scale, pad_x=(model_w - new_w) / 2.0, pad_y=(model_h - new_h) / 2.0)

    def to_frame_xy(self, x_model, y_model):
        return (x_model - self.pad_x) / self.scale, (y_model - self.pad_y) / self.scale


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    keypoints: Optional[np.ndarray] = None   # [17,3] (x,y,vis) in frame px, pose only
    mask: Optional[np.ndarray] = None        # [Hbox,Wbox] uint8 crop, seg only


@dataclass
class DecodeResult:
    detections: list = field(default_factory=list)


# ── generic helpers ──────────────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _squeeze_batch(arr):
    """Return a [C,H,W] float32 array. The MLA dequantized heads arrive NHWC
    (channel-last, e.g. (1,80,80,4)); transpose HWC->CHW so the rest of the
    decoders can treat channel as axis 0."""
    if hasattr(arr, "to_numpy"):
        arr = arr.to_numpy(copy=True)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]                                  # NHWC -> HWC
    if a.ndim == 3:
        a = np.ascontiguousarray(np.transpose(a, (2, 0, 1)))   # HWC -> CHW
    return a


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, top_k: int) -> np.ndarray:
    """Plain global NMS. boxes [N,4] xyxy. Returns kept indices (score-desc)."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < top_k:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-9
        order = rest[(inter / union) <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def _anchor_grid(h: int, w: int, offset: float):
    """Return gx, gy each [h*w] in cell units (col+offset, row+offset)."""
    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    return (gx.reshape(-1) + offset), (gy.reshape(-1) + offset)


def _dist2bbox(dist_lt_rb, gx, gy, stride):
    """dist_lt_rb [4,N] (l,t,r,b) in cell units -> xyxy in model px."""
    l, t, r, b = dist_lt_rb
    x1 = (gx - l) * stride
    y1 = (gy - t) * stride
    x2 = (gx + r) * stride
    y2 = (gy + b) * stride
    return np.stack([x1, y1, x2, y2], axis=1)


def _classify(tensors, model_w: int):
    """Route raw model tensors by shape into named per-scale buckets."""
    buckets = {"bbox": {}, "cls": {}, "mask_coeff": {}, "kpt": {}, "yolox": {}, "proto": None}
    for t in tensors:
        a = _squeeze_batch(t)          # [C,H,W]
        if a.ndim != 3:
            continue
        c, h, w = a.shape
        if c == 32 and h == 160:
            buckets["proto"] = a
        elif c == 32:
            buckets["mask_coeff"][h] = a
        elif c == 4:
            buckets["bbox"][h] = a
        elif c == 80:
            buckets["cls"][h] = a
        elif c == 1:
            buckets["cls"][h] = a
        elif c == 51:
            buckets["kpt"][h] = a
        elif c == 85:
            buckets["yolox"][h] = a
    return buckets


# ── detection (yolo11s), used when built-in decode is bypassed ────────────────
def decode_detection(tensors, frame_w, frame_h, model_w=640, model_h=640,
                     score_thr=0.25, iou_thr=0.5, top_k=100) -> DecodeResult:
    b = _classify(tensors, model_w)
    geom = LetterboxGeom.compute(frame_w, frame_h, model_w, model_h)
    all_boxes, all_scores, all_cls = [], [], []
    for h in sorted(b["bbox"], reverse=True):
        if h not in b["cls"]:
            continue
        stride = model_h / h
        w = b["bbox"][h].shape[2]
        bbox = b["bbox"][h].reshape(4, -1)
        cls = _sigmoid(b["cls"][h].reshape(b["cls"][h].shape[0], -1))  # [80,N]
        gx, gy = _anchor_grid(h, w, 0.5)
        boxes = _dist2bbox(bbox, gx, gy, stride)
        scores = cls.max(axis=0)
        classes = cls.argmax(axis=0)
        all_boxes.append(boxes); all_scores.append(scores); all_cls.append(classes)
    return _finish_boxes(all_boxes, all_scores, all_cls, geom, frame_w, frame_h,
                         score_thr, iou_thr, top_k)


def _finish_boxes(all_boxes, all_scores, all_cls, geom, frame_w, frame_h,
                  score_thr, iou_thr, top_k, extra=None):
    if not all_boxes:
        return DecodeResult([])
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_cls, axis=0)
    m = scores >= score_thr
    boxes, scores, classes = boxes[m], scores[m], classes[m]
    idx_map = np.nonzero(m)[0]
    if boxes.shape[0] == 0:
        return DecodeResult([])
    keep = _nms(boxes, scores, iou_thr, top_k)
    dets = []
    for k in keep:
        x1, y1 = geom.to_frame_xy(boxes[k, 0], boxes[k, 1])
        x2, y2 = geom.to_frame_xy(boxes[k, 2], boxes[k, 3])
        det = Detection(
            x1=float(np.clip(x1, 0, frame_w - 1)), y1=float(np.clip(y1, 0, frame_h - 1)),
            x2=float(np.clip(x2, 0, frame_w - 1)), y2=float(np.clip(y2, 0, frame_h - 1)),
            score=float(scores[k]), class_id=int(classes[k]),
        )
        if extra is not None:
            extra(det, idx_map[k], geom)
        dets.append(det)
    return DecodeResult(dets)


# ── YOLOX ─────────────────────────────────────────────────────────────────────
def decode_yolox(tensors, frame_w, frame_h, model_w=640, model_h=640,
                 score_thr=0.25, iou_thr=0.5, top_k=100) -> DecodeResult:
    b = _classify(tensors, model_w)
    geom = LetterboxGeom.compute(frame_w, frame_h, model_w, model_h)
    all_boxes, all_scores, all_cls = [], [], []
    for h in sorted(b["yolox"], reverse=True):
        head = b["yolox"][h]                 # [85,H,W]
        w = head.shape[2]
        stride = model_h / h
        flat = head.reshape(85, -1)
        gx, gy = _anchor_grid(h, w, 0.0)     # YOLOX: integer grid, no +0.5
        cx = (flat[0] + gx) * stride
        cy = (flat[1] + gy) * stride
        bw = np.exp(np.clip(flat[2], -10, 10)) * stride
        bh = np.exp(np.clip(flat[3], -10, 10)) * stride
        # Some YOLOX exports emit obj/cls already sigmoid-activated; others expose
        # the pre-sigmoid logits at the concat. Auto-adapt: only sigmoid if the
        # values fall outside [0,1].
        def _maybe_sig(x):
            return x if (float(x.min()) >= 0.0 and float(x.max()) <= 1.0) else _sigmoid(x)
        obj = _maybe_sig(flat[4])
        cls = _maybe_sig(flat[5:85])
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        scores = obj * cls.max(axis=0)
        classes = cls.argmax(axis=0)
        all_boxes.append(boxes); all_scores.append(scores); all_cls.append(classes)
    return _finish_boxes(all_boxes, all_scores, all_cls, geom, frame_w, frame_h,
                         score_thr, iou_thr, top_k)


# ── pose (yolo26s-pose) ───────────────────────────────────────────────────────
def decode_pose(tensors, frame_w, frame_h, model_w=640, model_h=640,
                score_thr=0.25, iou_thr=0.5, top_k=100) -> DecodeResult:
    b = _classify(tensors, model_w)
    geom = LetterboxGeom.compute(frame_w, frame_h, model_w, model_h)
    all_boxes, all_scores, all_cls, per_scale = [], [], [], []
    for h in sorted(b["bbox"], reverse=True):
        if h not in b["cls"] or h not in b["kpt"]:
            continue
        w = b["bbox"][h].shape[2]
        stride = model_h / h
        bbox = b["bbox"][h].reshape(4, -1)   # yolo26: already l/t/r/b (no DFL)
        cls = _sigmoid(b["cls"][h].reshape(-1))     # [N] single person class
        kpt = b["kpt"][h].reshape(51, -1)           # [51,N]
        gx05, gy05 = _anchor_grid(h, w, 0.5)
        gxi, gyi = _anchor_grid(h, w, 0.0)
        boxes = _dist2bbox(bbox, gx05, gy05, stride)
        all_boxes.append(boxes)
        all_scores.append(cls)
        all_cls.append(np.zeros_like(cls, dtype=np.int64))
        per_scale.append((kpt, gxi, gyi, stride, boxes.shape[0]))
    if not all_boxes:
        return DecodeResult([])
    offsets = np.cumsum([0] + [p[4] for p in per_scale])

    def attach(det, global_idx, g):
        # find scale + local index for this anchor
        s = int(np.searchsorted(offsets, global_idx, side="right") - 1)
        kpt, gxi, gyi, stride, _n = per_scale[s]
        li = global_idx - offsets[s]
        kx = (kpt[0::3, li] * 2.0 + gxi[li]) * stride
        ky = (kpt[1::3, li] * 2.0 + gyi[li]) * stride
        kv = _sigmoid(kpt[2::3, li])
        fx, fy = g.to_frame_xy(kx, ky)
        det.keypoints = np.stack([fx, fy, kv], axis=1).astype(np.float32)  # [17,3]

    return _finish_boxes(all_boxes, all_scores, all_cls, geom, frame_w, frame_h,
                         score_thr, iou_thr, top_k, extra=attach)


# ── segmentation (yolo11s-seg) ────────────────────────────────────────────────
def decode_segmentation(tensors, frame_w, frame_h, model_w=640, model_h=640,
                        score_thr=0.25, iou_thr=0.5, top_k=100,
                        max_masks=12) -> DecodeResult:
    b = _classify(tensors, model_w)
    geom = LetterboxGeom.compute(frame_w, frame_h, model_w, model_h)
    proto = b["proto"]                        # [32,160,160] or None
    all_boxes, all_scores, all_cls, coeff_scales = [], [], [], []
    for h in sorted(b["bbox"], reverse=True):
        if h not in b["cls"] or h not in b["mask_coeff"]:
            continue
        w = b["bbox"][h].shape[2]
        stride = model_h / h
        bbox = b["bbox"][h].reshape(4, -1)
        cls = _sigmoid(b["cls"][h].reshape(b["cls"][h].shape[0], -1))
        coeff = b["mask_coeff"][h].reshape(32, -1)         # [32,N]
        gx, gy = _anchor_grid(h, w, 0.5)
        boxes = _dist2bbox(bbox, gx, gy, stride)
        all_boxes.append(boxes)
        all_scores.append(cls.max(axis=0))
        all_cls.append(cls.argmax(axis=0))
        coeff_scales.append(coeff)
    if not all_boxes:
        return DecodeResult([])
    coeff_all = np.concatenate(coeff_scales, axis=1)       # [32, total]
    made = {"n": 0}

    def attach(det, global_idx, g):
        if proto is None or made["n"] >= max_masks:
            return
        c = coeff_all[:, global_idx]                       # [32]
        pm = proto.reshape(32, -1)                         # [32,160*160]
        m = _sigmoid(c @ pm).reshape(proto.shape[1], proto.shape[2])   # [160,160]
        # proto is at 1/4 of model input (160 = 640/4); map box (frame px) -> proto px
        px1 = int(np.clip((det.x1 * g.scale + g.pad_x) / 4.0, 0, 159))
        py1 = int(np.clip((det.y1 * g.scale + g.pad_y) / 4.0, 0, 159))
        px2 = int(np.clip((det.x2 * g.scale + g.pad_x) / 4.0, 0, 160))
        py2 = int(np.clip((det.y2 * g.scale + g.pad_y) / 4.0, 0, 160))
        if px2 > px1 and py2 > py1:
            det.mask = (m[py1:py2, px1:px2] > 0.5).astype(np.uint8)
            made["n"] += 1

    return _finish_boxes(all_boxes, all_scores, all_cls, geom, frame_w, frame_h,
                         score_thr, iou_thr, top_k, extra=attach)


# task name -> decoder function
HOST_DECODERS = {
    "segmentation": decode_segmentation,
    "pose": decode_pose,
    "yolox": decode_yolox,
    "detection_host": decode_detection,
}
