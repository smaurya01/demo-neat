#!/usr/bin/env python3
"""Run a compiled YOLO MPK on sample images with Neat box decode."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVES = {
    "yolo11n": ROOT / "work" / "yolo11n" / "compile_int8" / "yolo11n.compile_ready" / "yolo11n.compile_ready_mpk.tar.gz",
    "yolo26n": ROOT / "work" / "yolo26n" / "compile_int8" / "yolo26n.compile_ready" / "yolo26n.compile_ready_mpk.tar.gz",
}
DEFAULT_INPUT_DIR = ROOT / "assets" / "yolo_inference"
DEFAULT_IMAGE = DEFAULT_INPUT_DIR / "000000000139.jpg"
DEFAULT_LABELS = Path("/workspace/apps/examples/object-detection/yolo26-object-detector/src/common/coco_label.txt")
DEFAULT_OUTPUT_DIR = ROOT / "work" / "sample_runs"
MODEL_SIZE = 640
BOX_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"labels file is empty: {path}")
    return labels


def extract_bbox_payload(tensors) -> bytes | None:
    if len(tensors) != 1:
        return None
    try:
        payload = tensors[0].copy_payload_bytes()
    except Exception:
        return None
    return payload or None


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int, min_score: float) -> list[dict]:
    if len(payload) < 4:
        return []
    max_count = (len(payload) - 4) // 24
    count = min(struct.unpack_from("<I", payload, 0)[0], max_count)
    boxes = []
    off = 4
    for _ in range(count):
        x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, off)
        off += 24
        if float(score) < min_score:
            continue
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": float(score),
                "class_id": int(cls_id),
            }
        )
    return boxes


def draw_boxes(frame, boxes: list[dict], labels: list[str]):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    for box in boxes:
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        cls_id = int(box["class_id"])
        score = float(box["score"])
        b, g, r = BOX_COLORS[cls_id % len(BOX_COLORS)]
        color = (r, g, b)
        label = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        text = f"{label} {score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        left, top_text, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top_text
        top = max(0, y1 - th - 4)
        draw.rectangle((x1, top, x1 + tw + 2, y1), fill=color)
        draw.text((x1 + 1, top), text, fill=(0, 0, 0), font=font)
    return frame


def image_to_model_tensor(pyneat, bgr):
    import numpy as np

    rgb = np.ascontiguousarray(bgr[:, :, ::-1].astype(np.float32) / 255.0)
    return pyneat.Tensor.from_numpy(
        rgb,
        copy=True,
        layout=pyneat.TensorLayout.HWC,
        memory=pyneat.TensorMemory.EV74,
    )


def build_runner(pyneat, model_path: Path, seed_bgr, score_threshold: float, nms_iou: float, top_k: int):
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.enable = pyneat.AutoFlag.Off
    options.preprocess.input_max_width = MODEL_SIZE
    options.preprocess.input_max_height = MODEL_SIZE
    options.preprocess.input_max_depth = 3
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.score_threshold = score_threshold
    options.nms_iou_threshold = nms_iou
    options.top_k = top_k
    options.num_classes = 80
    options.boxdecode_original_width = 640
    options.boxdecode_original_height = 640

    seed_tensor = image_to_model_tensor(pyneat, seed_bgr)

    run_options = pyneat.RunOptions()
    run_options.queue_depth = 4
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.preset = pyneat.RunPreset.Balanced

    model = pyneat.Model(str(model_path), options)
    runner = model.build(
        [seed_tensor],
        route_options=pyneat.ModelRouteOptions(),
        run_options=run_options,
    )
    return runner, seed_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run yolo11n/yolo26n compile-ready int8 MPK on sample images")
    parser.add_argument("--model-id", choices=sorted(DEFAULT_ARCHIVES), default="yolo11n")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    archive = args.archive or DEFAULT_ARCHIVES[args.model_id]
    report_path = args.report or args.output_dir / f"{args.model_id}_sample_run.json"

    for path, label in [(archive, "archive"), (args.labels, "labels")]:
        if not path.is_file():
            print(f"Missing {label}: {path}", file=sys.stderr)
            return 2
    if args.image is not None:
        images = [args.image]
    else:
        if not args.input_dir.is_dir():
            print(f"Missing input dir: {args.input_dir}", file=sys.stderr)
            return 2
        images = sorted(path for path in args.input_dir.iterdir() if path.is_file() and is_image(path))
    if not images:
        print("No input images found", file=sys.stderr)
        return 2
    for image in images:
        if not image.is_file():
            print(f"Missing image: {image}", file=sys.stderr)
            return 2
    if not 0.0 <= args.score_threshold <= 1.0:
        print("--score-threshold must be in [0.0, 1.0]", file=sys.stderr)
        return 2
    if not 0.0 <= args.nms_iou <= 1.0:
        print("--nms-iou must be in [0.0, 1.0]", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k must be >= 1", file=sys.stderr)
        return 2
    if args.timeout_ms <= 0:
        print("--timeout-ms must be positive", file=sys.stderr)
        return 2

    try:
        import numpy as np
        from PIL import Image
        import pyneat
    except Exception as exc:
        print(f"Missing runtime dependency: {exc}", file=sys.stderr)
        return 4

    try:
        seed_rgb = Image.open(images[0]).convert("RGB").resize((MODEL_SIZE, MODEL_SIZE))
    except Exception:
        print(f"Failed to read image: {images[0]}", file=sys.stderr)
        return 2
    seed_bgr = np.asarray(seed_rgb, dtype=np.uint8)[:, :, ::-1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    runner = None
    try:
        runner, seed_tensor = build_runner(
            pyneat,
            archive,
            seed_bgr,
            score_threshold=args.score_threshold,
            nms_iou=args.nms_iou,
            top_k=args.top_k,
        )
        runner.run([seed_tensor], timeout_ms=args.timeout_ms)

        results = []
        for image_path in images:
            image_rgb = Image.open(image_path).convert("RGB").resize((MODEL_SIZE, MODEL_SIZE))
            frame_bgr = np.asarray(image_rgb, dtype=np.uint8)[:, :, ::-1]
            tensor = image_to_model_tensor(pyneat, frame_bgr)
            start = time.perf_counter()
            outputs = runner.run([tensor], timeout_ms=args.timeout_ms)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            payload = extract_bbox_payload(outputs)
            boxes = parse_bbox_payload(payload, MODEL_SIZE, MODEL_SIZE, args.score_threshold) if payload else []
            overlay = draw_boxes(image_rgb.copy(), boxes, labels)
            overlay_path = args.output_dir / f"{args.model_id}_{image_path.stem}_overlay.png"
            overlay.save(overlay_path)
            results.append(
                {
                    "image": str(image_path),
                    "overlay": str(overlay_path),
                    "detections": boxes,
                    "elapsed_ms": elapsed_ms,
                }
            )

        result = {
            "status": "pass",
            "model_id": args.model_id,
            "archive": str(archive),
            "input_count": len(images),
            "results": results,
        }
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        result = {
            "status": "fail",
            "model_id": args.model_id,
            "archive": str(archive),
            "images": [str(image) for image in images],
            "error": str(exc),
        }
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 4
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
