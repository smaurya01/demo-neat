"""Runtime config loader for the PLC defect detector.

Parses a simple key=value `.conf` (single-stream-yolo26n style) — no third-party
deps so it runs as-is inside the on-device pyneat environment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    model: str = "assets/models/plc_yolo26n_mpk.tar.gz"
    input_dir: str = "input_images"
    output_dir: str = "output_images"
    output_suffix: str = "_detected"
    infer_size: int = 640
    score: float = 0.25
    nms: float = 0.45
    top_k: int = 300
    timeout_ms: int = 8000
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"]
    )
    labels: List[str] = field(default_factory=lambda: [
        "missing_hole", "mouse_bite", "open_circuit",
        "short", "spur", "spurious_copper",
    ])


def load_config(path: str) -> Config:
    cfg = Config()
    text = Path(path).read_text()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key in ("model", "input_dir", "output_dir", "output_suffix"):
            setattr(cfg, key, val)
        elif key in ("infer_size", "top_k", "timeout_ms"):
            setattr(cfg, key, int(val))
        elif key in ("score", "nms"):
            setattr(cfg, key, float(val))
        elif key == "labels":
            cfg.labels = [s.strip() for s in val.split(",") if s.strip()]
    return cfg


def resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)
