"""Presentation types for the quad-stream-quad-model overlay: COCO labels, the
COCO-17 skeleton, and the small containers `main.py` draws from.

There is deliberately NO decode logic in this file.

This module replaced a ~380-line NumPy decoder (`src/decoders.py`) that rebuilt
anchor grids, applied sigmoid/exp activations, ran NMS and assembled prototype
masks on the A65 for the segmentation, pose and YOLOX streams. All of that now
runs on-device in Neat's fused BoxDecode stage, selected per task with
`ModelOptions.decode_type` (see DECODE_FAMILY in main.py). `pyneat.decode_bbox` /
`decode_pose` / `decode_segmentation` just read the resulting BBOX payload.

What survives here is only what an on-device decoder cannot give you: the human
names for the class ids, and the skeleton edges used to draw a pose.
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


@dataclass
class Detection:
    """One decoded instance, in FRAME pixels. Neat already clamped and NMS'd it."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    keypoints: Optional[np.ndarray] = None   # [17,3] (x, y, visibility), pose only
    mask: Optional[np.ndarray] = None        # [Hbox,Wbox] uint8 crop, segmentation only


@dataclass
class DecodeResult:
    detections: list = field(default_factory=list)
