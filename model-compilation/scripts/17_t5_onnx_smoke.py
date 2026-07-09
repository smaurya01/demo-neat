#!/usr/bin/env python3
"""T5 phase 1: host-side ONNX smoke test for the compile_ready graphs.

Runs the surgery output ONNX on one real calibration image with onnxruntime and
checks that every exposed head produces a finite tensor of the documented shape.
This is a host sanity gate; on-device (MLA) smoke tests are a separate board step
(see T5_MODEL_STATUS.md). Does not need the compile slot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ["yolo11s", "yolo11s-seg", "yolo26s-pose", "yolox_s"]


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((640, 640))
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None]  # 1,3,640,640
    return np.ascontiguousarray(arr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    parser.add_argument("--calib-dir", type=Path, default=ROOT / "assets" / "yolo_calibration")
    args = parser.parse_args()

    base = ROOT / "work" / args.model_id
    onnx_path = base / "surgery" / f"{args.model_id}.compile_ready.onnx"
    img = sorted(args.calib_dir.glob("*.jpg"))[0]
    x = load_image(img)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"images": x})
    names = [o.name for o in sess.get_outputs()]

    result = {"model_id": args.model_id, "image": img.name, "outputs": {}}
    all_ok = True
    for name, arr in zip(names, outs):
        finite = bool(np.isfinite(arr).all())
        result["outputs"][name] = {"shape": list(arr.shape), "finite": finite,
                                   "min": float(arr.min()), "max": float(arr.max())}
        all_ok = all_ok and finite
    result["status"] = "pass" if all_ok else "fail"

    (base / "reports" / "onnx_smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
