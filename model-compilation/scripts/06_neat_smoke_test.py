#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def model_cfg(model_id):
    reg = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    for model in reg["models"]:
        if model["id"] == model_id:
            return model
    raise KeyError(model_id)


def load_image_tensor(path, model):
    import pyneat

    _, c, h, w = model["input_shape"]
    image = Image.open(path).convert("RGB").resize((w, h))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(model.get("mean", [0, 0, 0]), dtype=np.float32)
    std = np.asarray(model.get("std", [1, 1, 1]), dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    if c != 3:
        raise ValueError(f"unsupported channel count for smoke test: {c}")
    return pyneat.Tensor.from_numpy(chw, copy=True)


def sample_summary(outputs):
    summary = {"type": type(outputs).__name__}
    try:
        summary["count"] = len(outputs)
        summary["shapes"] = [list(t.shape) for t in outputs]
    except Exception:
        summary["repr"] = repr(outputs)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--image", default=str(ROOT / "assets" / "sample_images" / "synthetic_rgb_gradient.jpg"))
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    import pyneat

    model = model_cfg(args.model_id)
    tensor = load_image_tensor(args.image, model)
    neat_model = pyneat.Model(args.archive)
    outputs = neat_model.run([tensor], timeout_ms=args.timeout_ms)

    result = {
        "model_id": args.model_id,
        "archive": args.archive,
        "image": args.image,
        "status": "pass",
        "outputs": sample_summary(outputs),
    }
    text = json.dumps(result, indent=2)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
