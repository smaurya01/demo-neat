"""Shared helpers for the PCB defect detection app."""

import logging
import sys
from pathlib import Path
from typing import Iterable, List


PLC_DEFECT_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

DEFECT_COLOR_PALETTE = [
    (56, 56, 255),
    (29, 178, 255),
    (10, 249, 72),
    (255, 194, 0),
    (255, 0, 200),
    (49, 210, 207),
]


def init_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("plc")


def is_image(path: Path, exts: Iterable[str]) -> bool:
    return path.suffix.lower() in {ext.lower() for ext in exts}


def discover_images(input_dir: Path, exts: Iterable[str]) -> List[Path]:
    if not input_dir.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and is_image(path, exts))


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def class_color(class_id: int):
    if class_id < 0:
        class_id = 0
    return DEFECT_COLOR_PALETTE[class_id % len(DEFECT_COLOR_PALETTE)]


def class_name(class_id: int, names) -> str:
    if names and 0 <= class_id < len(names):
        return names[class_id]
    if 0 <= class_id < len(PLC_DEFECT_NAMES):
        return PLC_DEFECT_NAMES[class_id]
    return f"class_{class_id}"


def draw_detections(bgr, dets: List[dict], names,
                    draw_labels: bool = True, thickness: int = 2) -> None:
    """Draw class-colored detection boxes on a BGR image in-place."""
    import cv2

    for det in dets:
        x1 = max(0, int(round(det["x1"])))
        y1 = max(0, int(round(det["y1"])))
        x2 = min(bgr.shape[1] - 1, int(round(det["x2"])))
        y2 = min(bgr.shape[0] - 1, int(round(det["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue

        color = class_color(det["class_id"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
        if not draw_labels:
            continue

        label = f"{class_name(det['class_id'], names)} {det['score']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(0, y1 - text_h - 4)
        cv2.rectangle(bgr, (x1, label_y), (x1 + text_w + 2, y1), color, -1)
        cv2.putText(
            bgr,
            label,
            (x1 + 1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
