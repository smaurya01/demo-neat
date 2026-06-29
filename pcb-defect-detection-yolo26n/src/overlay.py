"""Draw detection boxes + labels on the original BGR image (class-colored)."""

from typing import List

import cv2
import numpy as np

from .labels import class_color, class_name


def draw_detections(bgr: np.ndarray, dets: List[dict], names,
                    draw_labels: bool = True, thickness: int = 2) -> None:
    """Draw each detection in its class color. Coordinates are original-image space."""
    for d in dets:
        x1 = max(0, int(round(d["x1"])))
        y1 = max(0, int(round(d["y1"])))
        x2 = min(bgr.shape[1] - 1, int(round(d["x2"])))
        y2 = min(bgr.shape[0] - 1, int(round(d["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue
        col = class_color(d["class_id"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), col, thickness)
        if draw_labels:
            label = f"{class_name(d['class_id'], names)} {d['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytop = max(0, y1 - th - 4)
            cv2.rectangle(bgr, (x1, ytop), (x1 + tw + 2, y1), col, -1)
            cv2.putText(bgr, label, (x1 + 1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
