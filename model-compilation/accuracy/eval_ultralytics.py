#!/usr/bin/env python3
"""Independent FP32 reference: the original Ultralytics .pt, through Ultralytics' own pipeline.

    python accuracy/eval_ultralytics.py --model-id yolo26s      # SDK host, PyTorch on CPU

This does not use this repository's letterbox or decoder at all: Ultralytics loads the .pt,
preprocesses, runs PyTorch and decodes the boxes. It writes accuracy/results/<model>_pytorch.json in
the same format as eval_coco.py, so `eval_coco.py score` lists it next to the other rows.

If this row and the `fp32` row (the compile-ready ONNX through this repository's decoder) agree,
the ONNX export, the graph surgery and the decoder did not change the model's accuracy, and the
INT8 / bf16 rows measure the compiler alone.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from eval_coco import ANN, IMG_DIR, ROOT_DIR, artifact_name, result_path


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="an Ultralytics model, e.g. yolo26s")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    args = ap.parse_args()

    pt = ROOT_DIR / f"{args.model_id}.pt"          # downloaded by compile/convert_to_onnx.py
    model = YOLO(str(pt) if pt.exists() else f"{args.model_id}.pt")
    coco = json.loads(ANN.read_text())
    cat_ids = [c["id"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    images = coco["images"][: args.limit] if args.limit else coco["images"]
    print(f"[ultralytics] {args.model_id}: {pt.name}, {len(images)} images")

    dets, infer_ms = [], []
    for i, im in enumerate(images, 1):
        t0 = time.perf_counter()
        r = model.predict(str(IMG_DIR / im["file_name"]), imgsz=640, conf=args.conf,
                          max_det=args.max_det, device="cpu", verbose=False)[0]
        infer_ms.append((time.perf_counter() - t0) * 1000)
        b = r.boxes
        for (x1, y1, x2, y2), s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.int().tolist()):
            dets.append({"image_id": im["id"], "category_id": cat_ids[c],
                         "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                         "score": round(s, 5)})
        if i % 500 == 0 or i == len(images):
            print(f"   {i}/{len(images)}  median {np.median(infer_ms):.1f} ms/image", flush=True)

    out = result_path(args.model_id, "pytorch", args.limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model_id, "precision": "pytorch",
                               "artifact": artifact_name(pt), "images": len(images),
                               "image_ids": [im["id"] for im in images],
                               "median_infer_ms": round(float(np.median(infer_ms)), 2),
                               "detections": dets}))
    print(f"[ultralytics] wrote {len(dets)} detections -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
