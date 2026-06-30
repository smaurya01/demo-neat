"""PLC / PCB defect class names and a per-class BGR color palette.

The six class names are index-aligned with the trained YOLO26n checkpoint
(`yolo26n.pt` -> `model.names`). Boxes and labels for a class share one color.
"""

# Default labels (overridden by the YAML `labels:` block when present).
PLC_DEFECT_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# Distinct, high-contrast BGR colors — one per defect class.
DEFECT_COLOR_PALETTE = [
    (56, 56, 255),    # missing_hole    - red
    (29, 178, 255),   # mouse_bite      - orange
    (10, 249, 72),    # open_circuit    - green
    (255, 194, 0),    # short           - cyan/blue
    (255, 0, 200),    # spur            - magenta
    (49, 210, 207),   # spurious_copper - yellow-green
]


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
