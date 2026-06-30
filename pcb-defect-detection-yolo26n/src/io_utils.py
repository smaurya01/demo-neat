"""Image IO helpers."""

from pathlib import Path
from typing import Iterable, List


def is_image(path: Path, exts: Iterable[str]) -> bool:
    return path.suffix.lower() in {e.lower() for e in exts}


def discover_images(input_dir: Path, exts: Iterable[str]) -> List[Path]:
    if not input_dir.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_dir}")
    return sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p, exts))


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
