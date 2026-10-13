#!/usr/bin/env python3
"""Download a fixed COCO val2017 subset for the accuracy study.

    python accuracy/prepare_coco_subset.py              # all of val2017 (default; ~800 MB)
    python accuracy/prepare_coco_subset.py --num 500    # a quick 500-image subset

Writes (all git-ignored):
    accuracy/data/annotations/instances_val2017.json   full val2017 annotations (5000 images)
    accuracy/data/val2017_subset/<file_name>            the N images (COCO names, e.g. 000000000285.jpg)
    accuracy/data/instances_val2017_subset.json         annotations for just those N images

The subset is the first N image ids in sorted order, so every run and every precision is scored
on exactly the same images. Images used for INT8 calibration (assets/calibration, also COCO
val2017) are skipped, so a model is never scored on the images it was calibrated with.

Runs on the SDK host (needs internet); the DevKit reads the result over the shared workspace.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CALIB_DIR = HERE.parent / "assets" / "calibration"
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=5000,
                    help="number of val2017 images (default and max 5000: the full set, minus the "
                         "calibration images)")
    args = ap.parse_args()

    ann_dir = DATA / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    full = ann_dir / "instances_val2017.json"
    if not full.exists():
        zpath = DATA / "annotations_trainval2017.zip"
        print(f"[coco] downloading annotations (~241 MB) -> {zpath}")
        fetch(ANN_ZIP_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            full.write_bytes(z.read("annotations/instances_val2017.json"))
        zpath.unlink()

    coco = json.loads(full.read_text())
    calib = {p.name for p in CALIB_DIR.glob("*.jpg")} if CALIB_DIR.is_dir() else set()
    images = [im for im in sorted(coco["images"], key=lambda im: im["id"])
              if im["file_name"] not in calib][: args.num]
    keep = {im["id"] for im in images}
    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
        "categories": coco["categories"],
    }
    (DATA / "instances_val2017_subset.json").write_text(json.dumps(subset))

    img_dir = DATA / "val2017_subset"
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"[coco] downloading {len(images)} images -> {img_dir}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda im: fetch(IMG_URL.format(file_name=im["file_name"]),
                                       img_dir / im["file_name"]), images))

    n_ann = sum(1 for a in subset["annotations"] if not a.get("iscrowd", 0))
    print(f"[coco] OK: {len(images)} images ({len(calib)} calibration images skipped), "
          f"{n_ann} objects -> {DATA / 'instances_val2017_subset.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
