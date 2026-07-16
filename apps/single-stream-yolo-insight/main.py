#!/usr/bin/env python3
"""RTSP single-stream Neat demo publishing clean video via VideoSender and
detections as JSON via MetadataSender — no boxes are drawn on the frames.
Neat Insight (or any metadata-aware viewer) overlays the boxes client-side."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
from pathlib import Path
import sys
import time

cv2 = None
np = None
pyneat = None

COCO_LABELS = [
    "PERSON", "BICYCLE", "CAR", "MOTORCYCLE", "AIRPLANE", "BUS", "TRAIN", "TRUCK",
    "BOAT", "TRAFFIC LIGHT", "FIRE HYDRANT", "STOP SIGN", "PARKING METER", "BENCH",
    "BIRD", "CAT", "DOG", "HORSE", "SHEEP", "COW", "ELEPHANT", "BEAR", "ZEBRA",
    "GIRAFFE", "BACKPACK", "UMBRELLA", "HANDBAG", "TIE", "SUITCASE", "FRISBEE",
    "SKIS", "SNOWBOARD", "SPORTS BALL", "KITE", "BASEBALL BAT", "BASEBALL GLOVE",
    "SKATEBOARD", "SURFBOARD", "TENNIS RACKET", "BOTTLE", "WINE GLASS", "CUP",
    "FORK", "KNIFE", "SPOON", "BOWL", "BANANA", "APPLE", "SANDWICH", "ORANGE",
    "BROCCOLI", "CARROT", "HOT DOG", "PIZZA", "DONUT", "CAKE", "CHAIR", "COUCH",
    "POTTED PLANT", "BED", "DINING TABLE", "TOILET", "TV", "LAPTOP", "MOUSE",
    "REMOTE", "KEYBOARD", "CELL PHONE", "MICROWAVE", "OVEN", "TOASTER", "SINK",
    "REFRIGERATOR", "BOOK", "CLOCK", "VASE", "SCISSORS", "TEDDY BEAR", "HAIR DRIER",
    "TOOTHBRUSH",
]


@dataclass
class Config:
    rtsp_url: str = "rtsp://<rtsp-server-ip>:8555/stream"
    model_path: str = ""
    insight_host: str = "<insight-host-ip>"
    channel: int = 0
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    fallback_width: int = 1280
    fallback_height: int = 720
    fallback_fps: int = 60
    model_width: int = 640
    model_height: int = 640
    latency_ms: int = 200
    score_threshold: float = 0.25
    nms_iou: float = 0.50
    top_k: int = 100
    num_classes: int = 80
    frames: int = 0
    bitrate_kbps: int = 4000
    tcp: bool = True
    print_backend: bool = False


def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    if pyneat is not None:
        return
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module
    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module


def project_name() -> str:
    return Path(__file__).resolve().parent.name


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "default.conf"


def resolve_app_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def apply_config_value(cfg: Config, key: str, value: str) -> None:
    if key == "rtsp_url":
        cfg.rtsp_url = value
    elif key == "model_path":
        cfg.model_path = value
    elif key == "insight_host":
        cfg.insight_host = value
    elif key == "channel":
        cfg.channel = int(value)
    elif key == "video_port_base":
        cfg.video_port_base = int(value)
    elif key == "metadata_port_base":
        cfg.metadata_port_base = int(value)
    elif key == "fallback_width":
        cfg.fallback_width = int(value)
    elif key == "fallback_height":
        cfg.fallback_height = int(value)
    elif key == "fallback_fps":
        cfg.fallback_fps = int(value)
    elif key == "model_width":
        cfg.model_width = int(value)
    elif key == "model_height":
        cfg.model_height = int(value)
    elif key == "latency_ms":
        cfg.latency_ms = int(value)
    elif key == "score_threshold":
        cfg.score_threshold = float(value)
    elif key == "nms_iou":
        cfg.nms_iou = float(value)
    elif key == "top_k":
        cfg.top_k = int(value)
    elif key == "num_classes":
        cfg.num_classes = int(value)
    elif key == "frames":
        cfg.frames = int(value)
    elif key == "bitrate_kbps":
        cfg.bitrate_kbps = int(value)
    elif key == "rtsp_transport":
        cfg.tcp = value.strip().lower() == "tcp"
    elif key == "print_backend":
        cfg.print_backend = parse_bool(value)
    else:
        raise ValueError(f"unknown config key: {key}")


def load_config_file(cfg: Config, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_no}: expected key=value")
        key, value = [part.strip() for part in line.split("=", 1)]
        apply_config_value(cfg, key, value)


def parse_args(argv: list[str] | None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--rtsp")
    parser.add_argument("--model")
    parser.add_argument("--insight-host")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--score", type=float)
    parser.add_argument("--nms", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--bitrate", type=int)
    parser.add_argument("--print-backend", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config()
    load_config_file(cfg, args.config)
    if args.rtsp is not None:
        cfg.rtsp_url = args.rtsp
    if args.model is not None:
        cfg.model_path = args.model
    if args.insight_host is not None:
        cfg.insight_host = args.insight_host
    if args.channel is not None:
        cfg.channel = args.channel
    if args.score is not None:
        cfg.score_threshold = args.score
    if args.nms is not None:
        cfg.nms_iou = args.nms
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.frames is not None:
        cfg.frames = args.frames
    if args.bitrate is not None:
        cfg.bitrate_kbps = args.bitrate
    if args.print_backend:
        cfg.print_backend = True
    return cfg


def resolve_model_path(cfg: Config) -> str:
    if cfg.model_path:
        return str(resolve_app_path(cfg.model_path))
    return str(Path(__file__).resolve().parent / "assets" / "models" / "yolo_11n_mpk.tar.gz")


def validate_config(cfg: Config) -> None:
    if not cfg.rtsp_url:
        raise ValueError("RTSP URL must not be empty")
    if not cfg.insight_host or cfg.insight_host.startswith("<"):
        raise ValueError("insight_host must be set to the host running Neat Insight")
    if cfg.channel < 0:
        raise ValueError("channel must be >= 0")
    if not 0 < cfg.video_port_base + cfg.channel <= 65535:
        raise ValueError("video port must be in 1..65535")
    if not 0 < cfg.metadata_port_base + cfg.channel <= 65535:
        raise ValueError("metadata port must be in 1..65535")
    model_path = Path(resolve_model_path(cfg))
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")


def probe_rtsp(cfg: Config) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(cfg.rtsp_url)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
        cap.release()
        if width > 0 and height > 0:
            return width, height, fps if fps > 0 else cfg.fallback_fps
    return cfg.fallback_width, cfg.fallback_height, cfg.fallback_fps


def make_source_options(cfg: Config, width: int, height: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp_url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.fallback_h264_fps = fps
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return opt


def make_model(cfg: Config, width: int, height: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.input_max_width = width
    opt.preprocess.input_max_height = height
    opt.preprocess.input_max_depth = 1
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = cfg.model_width
    opt.preprocess.resize.height = cfg.model_height
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # Zoo yolo_11n exposes raw 64-channel DFL heads -> YoloV8 decode family.
    # A self-compiled YOLO11 archive needs BoxDecodeType.YoloV26 instead.
    opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.top_k
    opt.num_classes = cfg.num_classes
    return pyneat.Model(resolve_model_path(cfg), opt)


def build_pipeline(cfg: Config, width: int, height: int, fps: int):
    """One in-graph pipeline: the decoded stream branches to the VideoSender
    (clean frames, never touched by the CPU) and to the model, whose decoded
    detections surface at the named "detections" output."""
    source = pyneat.groups.rtsp_decoded_input(make_source_options(cfg, width, height, fps))
    branch = pyneat.graphs.branch("source", ["video", "model"])

    video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    video_options.host = cfg.insight_host
    video_options.channel = cfg.channel
    video_options.video_port_base = cfg.video_port_base
    video_options.encoder.bitrate_kbps = cfg.bitrate_kbps
    video_graph = pyneat.Graph("video")
    video_graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(video_options))

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), make_model(cfg, width, height))

    detections_graph = pyneat.Graph("detections")
    detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    graph = pyneat.Graph(project_name() + "_python")
    live_link = pyneat.GraphLinkOptions()
    live_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    graph.connect(source, branch)
    graph.connect(branch, video_graph, live_link)
    graph.connect(branch, model_graph, live_link)
    graph.connect(model_graph, detections_graph)
    return graph, video_options.video_port


def extract_tensors(sample) -> list:
    if sample is None or not hasattr(sample, "kind"):
        return []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    tensors = []
    for field in getattr(sample, "fields", []):
        tensors.extend(extract_tensors(field))
    return tensors


def decode_boxes(tensors: list, width: int, height: int, top_k: int) -> list[dict]:
    decoded = pyneat.decode_bbox(tensors, clamp_to=(width, height), top_k=top_k)
    boxes = []
    for tensor in decoded:
        arr = np.asarray(tensor.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
        for x1, y1, x2, y2, score, class_id in arr:
            boxes.append({
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "score": float(score), "class_id": int(class_id),
            })
    return boxes[:top_k]


def class_label(class_id: int) -> str:
    if 0 <= class_id < len(COCO_LABELS):
        return COCO_LABELS[class_id]
    return f"CLASS {class_id}"


def build_metadata_objects(boxes: list[dict], width: int, height: int,
                           min_score: float) -> list[dict]:
    """Insight expects object-detection bbox as [x, y, w, h] top-left in
    encoded-frame pixels; decode_bbox returns corners, so convert here."""
    objects = []
    for index, box in enumerate(boxes, start=1):
        if box["score"] < min_score:
            continue
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        w = max(0, min(width - x, int(box["x2"] - box["x1"])))
        h = max(0, min(height - y, int(box["y2"] - box["y1"])))
        if w <= 0 or h <= 0:
            continue
        objects.append({
            "id": f"obj_{index}",
            "label": class_label(box["class_id"]),
            "confidence": float(box["score"]),
            "bbox": [float(x), float(y), float(w), float(h)],
        })
    return objects


def should_log_frame(processed: int, target_frames: int) -> bool:
    return processed == 1 or processed % 30 == 0 or (target_frames > 0 and processed == target_frames)


def run(cfg: Config) -> int:
    load_runtime_dependencies()
    validate_config(cfg)
    width, height, fps = probe_rtsp(cfg)

    graph, video_port = build_pipeline(cfg, width, height, fps)
    if cfg.print_backend:
        print("Backend:")
        print(graph.describe_backend())

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run_handle = graph.build(run_options)

    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = cfg.channel
    metadata_options.metadata_port_base = cfg.metadata_port_base
    metadata_sender = pyneat.MetadataSender(metadata_options)

    print(f"RTSP input:  {cfg.rtsp_url} ({width}x{height}@{fps})")
    print(f"Model:       {resolve_model_path(cfg)}")
    print(f"Video out:   udp://{cfg.insight_host}:{video_port} (clean frames, no overlay)")
    print(f"Metadata out: udp://{cfg.insight_host}:{metadata_sender.metadata_port()} "
          f"(object-detection JSON, channel={cfg.channel})")
    print(f"Viewer:      Neat Insight Video Viewer channel {cfg.channel} draws boxes from metadata")

    processed = 0
    pull_ms_sum = 0.0
    send_ms_sum = 0.0
    run_start = time.perf_counter()
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            pull_start = time.perf_counter()
            sample = run_handle.pull("detections", 20000)
            pull_end = time.perf_counter()
            if sample is None:
                print("[warn] timed out waiting for detections", file=sys.stderr)
                continue

            tensors = extract_tensors(sample)
            boxes = decode_boxes(tensors, width, height, cfg.top_k)
            objects = build_metadata_objects(boxes, width, height, cfg.score_threshold)

            frame_id = getattr(sample, "frame_id", -1)
            if frame_id is None or frame_id < 0:
                frame_id = processed
            send_start = time.perf_counter()
            # Send every frame, including an empty list, so stale boxes never
            # linger in the viewer. UDP is fire-and-forget; True only proves
            # the datagram left this host.
            metadata_sender.send_metadata(
                "object-detection",
                json.dumps({"objects": objects}, separators=(",", ":")),
                int(time.time() * 1000),
                str(frame_id),
            )
            send_end = time.perf_counter()

            processed += 1
            pull_ms_sum += (pull_end - pull_start) * 1000.0
            send_ms_sum += (send_end - send_start) * 1000.0
            if should_log_frame(processed, cfg.frames):
                elapsed = time.perf_counter() - run_start
                fps_now = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"frame={processed} detections={len(boxes)} published={len(objects)} "
                    f"fps={fps_now:.2f} "
                    f"avg_ms(pull={pull_ms_sum / processed:.2f}, "
                    f"metadata_send={send_ms_sum / processed:.2f})",
                    flush=True,
                )
    finally:
        run_handle.close()
    print(f"processed={processed} video={cfg.insight_host}:{video_port} "
          f"metadata={cfg.insight_host}:{metadata_sender.metadata_port()}")
    return processed


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
