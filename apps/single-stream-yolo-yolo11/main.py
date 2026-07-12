#!/usr/bin/env python3
"""RTSP single-stream Neat demo with Python overlay and H.264/RTP UDP output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import os
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

PALETTE = [
    (82, 90, 240), (170, 60, 180), (210, 40, 40), (180, 40, 150),
    (200, 100, 30), (145, 54, 34), (160, 210, 190), (120, 220, 90),
    (225, 110, 170), (190, 80, 210), (235, 128, 128), (95, 200, 220),
]


@dataclass
class Config:
    rtsp_url: str = "rtsp://<rtsp-server-ip>:8555/stream"
    model_path: str = ""
    models_dir: str = ""
    fallback_width: int = 1280
    fallback_height: int = 720
    fallback_fps: int = 25
    model_width: int = 640
    model_height: int = 640
    latency_ms: int = 200
    score_threshold: float = 0.25
    nms_iou: float = 0.50
    top_k: int = 100
    num_classes: int = 80
    frames: int = 0
    udp_host: str = "<host-ip-that-receives-video>"
    udp_port: int = 0
    udp_port_base: int = 0
    bitrate_kbps: int = 4000
    tcp: bool = True
    print_backend: bool = False


@dataclass
class StageProfile:
    decoder_ms: float = 0.0
    inference_ms: float = 0.0
    overlay_ms: float = 0.0
    encoder_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.decoder_ms + self.inference_ms + self.overlay_ms + self.encoder_ms

    def add(self, other: "StageProfile") -> None:
        self.decoder_ms += other.decoder_ms
        self.inference_ms += other.inference_ms
        self.overlay_ms += other.overlay_ms
        self.encoder_ms += other.encoder_ms

    def average(self, count: int) -> "StageProfile":
        if count <= 0:
            return StageProfile()
        return StageProfile(
            self.decoder_ms / count,
            self.inference_ms / count,
            self.overlay_ms / count,
            self.encoder_ms / count,
        )


def should_log_frame(processed: int, target_frames: int) -> bool:
    return processed == 1 or processed % 30 == 0 or (target_frames > 0 and processed == target_frames)


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


def model_spec() -> tuple[str, str]:
    return "yolo11", "yolo_11n_mpk.tar.gz"


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
    elif key == "models_dir":
        cfg.models_dir = value
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
    elif key == "udp_host":
        cfg.udp_host = value
    elif key == "udp_port":
        cfg.udp_port = int(value)
    elif key == "udp_port_base":
        cfg.udp_port_base = int(value)
    elif key == "bitrate_kbps":
        cfg.bitrate_kbps = int(value)
    elif key == "rtsp_transport":
        cfg.tcp = value.strip().lower() == "tcp"
    elif key == "print_backend":
        cfg.print_backend = parse_bool(value)
    elif key in {"only", "allow_missing", "load_only"}:
        return
    else:
        raise ValueError(f"unknown config key: {key}")


def load_config_file(cfg: Config, path: Path, required: bool) -> None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"config file not found: {path}")
        return
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
    parser.add_argument("--models-dir")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--model-width", type=int)
    parser.add_argument("--model-height", type=int)
    parser.add_argument("--score", type=float)
    parser.add_argument("--nms", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--classes", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--udp-host")
    parser.add_argument("--udp-port", type=int)
    parser.add_argument("--udp-port-base", type=int)
    parser.add_argument("--bitrate", type=int)
    parser.add_argument("--rtsp-udp", action="store_true")
    parser.add_argument("--print-backend", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config()
    load_config_file(cfg, args.config, required=args.config.exists())
    if args.rtsp is not None:
        cfg.rtsp_url = args.rtsp
    if args.model is not None:
        cfg.model_path = args.model
    if args.models_dir is not None:
        cfg.models_dir = args.models_dir
    if args.width is not None:
        cfg.fallback_width = args.width
    if args.height is not None:
        cfg.fallback_height = args.height
    if args.fps is not None:
        cfg.fallback_fps = args.fps
    if args.model_width is not None:
        cfg.model_width = args.model_width
    if args.model_height is not None:
        cfg.model_height = args.model_height
    if args.score is not None:
        cfg.score_threshold = args.score
    if args.nms is not None:
        cfg.nms_iou = args.nms
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.classes is not None:
        cfg.num_classes = args.classes
    if args.frames is not None:
        cfg.frames = args.frames
    if args.udp_host is not None:
        cfg.udp_host = args.udp_host
    if args.udp_port is not None:
        cfg.udp_port = args.udp_port
    if args.udp_port_base is not None:
        cfg.udp_port_base = args.udp_port_base
    if args.bitrate is not None:
        cfg.bitrate_kbps = args.bitrate
    if args.rtsp_udp:
        cfg.tcp = False
    if args.print_backend:
        cfg.print_backend = True
    return cfg


def resolve_model_path(cfg: Config) -> str:
    _name, filename = model_spec()
    if cfg.model_path:
        return str(resolve_app_path(cfg.model_path))
    models_dir = resolve_app_path(cfg.models_dir) if cfg.models_dir else Path(__file__).resolve().parent / "assets" / "models"
    return str(models_dir / filename)


def validate_config(cfg: Config) -> None:
    if not cfg.rtsp_url:
        raise ValueError("RTSP URL must not be empty")
    if not cfg.udp_host:
        raise ValueError("UDP host must not be empty")
    port = cfg.udp_port or cfg.udp_port_base
    if port <= 0 or port > 65535:
        raise ValueError("UDP port must be in 1..65535")
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
    opt.output_caps.memory = pyneat.CapsMemory.SystemMemory
    return opt


def make_model(cfg: Config):
    name, _filename = model_spec()
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.input_max_width = cfg.fallback_width
    opt.preprocess.input_max_height = cfg.fallback_height
    opt.preprocess.input_max_depth = 1
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = cfg.model_width
    opt.preprocess.resize.height = cfg.model_height
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    if name == "yolo26n":
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
    elif name == "yolov8n-seg":
        opt.decode_type = pyneat.BoxDecodeType.YoloV8Seg
    else:
        opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.top_k
    opt.num_classes = cfg.num_classes
    return pyneat.Model(resolve_model_path(cfg), opt)


def build_inference_graph(cfg: Config, width: int, height: int, fps: int):
    source = pyneat.groups.rtsp_decoded_input(make_source_options(cfg, width, height, fps))
    branch = pyneat.graphs.branch("source", ["frame", "model"])

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), make_model(cfg))

    result_graph = pyneat.Graph("result")
    result_graph.add(pyneat.nodes.output("result", pyneat.OutputOptions.every_frame(4)))

    joined = pyneat.graphs.combine(["frame", "result"], "output", pyneat.CombinePolicy.ByFrame)
    graph = pyneat.Graph(project_name() + "_python")
    graph.connect(source, branch)
    graph.connect(branch, frame_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, result_graph)
    graph.connect(frame_graph, joined)
    graph.connect(result_graph, joined)
    return graph


def make_nv12_input_options(width: int, height: int, fps: int):
    input_opt = pyneat.InputOptions()
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = pyneat.Format.NV12
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 1
    input_opt.max_width = width
    input_opt.max_height = height
    input_opt.max_depth = 1
    input_opt.fps_n = max(1, fps)
    input_opt.fps_d = 1
    input_opt.caps_override = f"video/x-raw,format=NV12,width={width},height={height},framerate={max(1, fps)}/1"
    input_opt.use_simaai_pool = False
    return input_opt


def build_source_graph(cfg: Config, width: int, height: int, fps: int):
    graph = pyneat.Graph("source")
    graph.add(pyneat.groups.rtsp_decoded_input(make_source_options(cfg, width, height, fps)))
    graph.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))
    return graph


def build_model_graph(cfg: Config, width: int, height: int, fps: int):
    graph = pyneat.Graph("model")
    graph.add(pyneat.nodes.input(make_nv12_input_options(width, height, fps)))
    graph.add(make_model(cfg))
    graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    return graph


def build_video_graph(cfg: Config, width: int, height: int, fps: int):
    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.udp_host
    sender_opt.channel = 0
    sender_opt.video_port_base = cfg.udp_port or cfg.udp_port_base
    sender_opt.encoder.bitrate_kbps = cfg.bitrate_kbps

    graph = pyneat.Graph("video")
    graph.add(pyneat.nodes.input(make_nv12_input_options(width, height, fps)))
    graph.add(pyneat.groups.video_sender(sender_opt))
    seed_nv12 = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
    seed_nv12[:height, :] = 16
    seed = make_nv12_tensor(seed_nv12, width, height)
    return graph, graph.build([seed]), sender_opt.video_port


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def tensor_bgr_from_decoded(tensor):
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))
    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def tensor_nv12_from_decoded(tensor):
    if not tensor.is_nv12():
        raise RuntimeError("expected decoded NV12 frame")
    width = tensor_dim(tensor, "width")
    height = tensor_dim(tensor, "height")
    payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
    expected = width * height * 3 // 2
    if payload.size < expected:
        raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
    return np.ascontiguousarray(payload[:expected].reshape((height * 3 // 2, width))).copy(), width, height


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


def find_field(sample, label: str):
    if sample is None:
        return None
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = find_field(field, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, index: int):
    field = find_field(sample, label)
    fields = list(getattr(sample, "fields", []))
    if field is not None:
        return field
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > index:
        return fields[index]
    raise RuntimeError(f"joined output missing {label} field")


def frame_tensor_from_sample(sample):
    tensors = extract_tensors(joined_field(sample, "frame", 0))
    if not tensors:
        raise RuntimeError("joined frame field has no tensor")
    return tensors[0]


def result_tensors_from_sample(sample) -> list:
    tensors = extract_tensors(joined_field(sample, "result", 1))
    if not tensors:
        raise RuntimeError("joined result field has no tensors")
    return tensors


def class_label(class_id: int) -> str:
    if 0 <= class_id < len(COCO_LABELS):
        return COCO_LABELS[class_id]
    return f"CLASS {class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
    return PALETTE[max(0, class_id) % len(PALETTE)]


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


def draw_boxes(frame, boxes: list[dict], min_score: float) -> int:
    visible = 0
    for box in boxes:
        if box["score"] < min_score:
            continue
        x1 = max(0, min(frame.shape[1] - 1, int(round(box["x1"]))))
        y1 = max(0, min(frame.shape[0] - 1, int(round(box["y1"]))))
        x2 = max(0, min(frame.shape[1] - 1, int(round(box["x2"]))))
        y2 = max(0, min(frame.shape[0] - 1, int(round(box["y2"]))))
        if x2 <= x1 or y2 <= y1:
            continue
        label = class_label(box["class_id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        visible += 1
    return visible


def draw_y_line(y_plane, x1: int, x2: int, y: int, value: int, thickness: int) -> None:
    height, width = y_plane.shape
    if y < 0 or y >= height:
        return
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    if x2 < x1:
        x1, x2 = x2, x1
    half = thickness // 2
    for yy in range(y - half, y + half + 1):
        if 0 <= yy < height:
            y_plane[yy, x1:x2 + 1] = value


def draw_y_col(y_plane, x: int, y1: int, y2: int, value: int, thickness: int) -> None:
    height, width = y_plane.shape
    if x < 0 or x >= width:
        return
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if y2 < y1:
        y1, y2 = y2, y1
    half = thickness // 2
    for xx in range(x - half, x + half + 1):
        if 0 <= xx < width:
            y_plane[y1:y2 + 1, xx] = value


def draw_uv_rect(uv_plane, x1: int, y1: int, x2: int, y2: int, u_value: int, v_value: int) -> None:
    uv_height, uv_stride = uv_plane.shape
    uv_width = uv_stride // 2
    x1 = max(0, min(uv_width - 1, x1 // 2))
    x2 = max(0, min(uv_width - 1, x2 // 2))
    y1 = max(0, min(uv_height - 1, y1 // 2))
    y2 = max(0, min(uv_height - 1, y2 // 2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    uv_plane[y1, x1 * 2:(x2 + 1) * 2:2] = u_value
    uv_plane[y1, x1 * 2 + 1:(x2 + 1) * 2:2] = v_value
    uv_plane[y2, x1 * 2:(x2 + 1) * 2:2] = u_value
    uv_plane[y2, x1 * 2 + 1:(x2 + 1) * 2:2] = v_value
    uv_plane[y1:y2 + 1, x1 * 2] = u_value
    uv_plane[y1:y2 + 1, x1 * 2 + 1] = v_value
    uv_plane[y1:y2 + 1, x2 * 2] = u_value
    uv_plane[y1:y2 + 1, x2 * 2 + 1] = v_value


def fill_nv12_rect(y_plane, uv_plane, x1: int, y1: int, x2: int, y2: int,
                   y_value: int, u_value: int, v_value: int) -> None:
    height, width = y_plane.shape
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return

    y_plane[y1:y2, x1:x2] = y_value

    uv_y1 = y1 // 2
    uv_y2 = (y2 + 1) // 2
    uv_x1 = x1 & ~1
    uv_x2 = (x2 + 1) & ~1
    uv_y1 = max(0, min(uv_plane.shape[0], uv_y1))
    uv_y2 = max(0, min(uv_plane.shape[0], uv_y2))
    uv_x1 = max(0, min(width, uv_x1))
    uv_x2 = max(0, min(width, uv_x2))
    if uv_x2 <= uv_x1 or uv_y2 <= uv_y1:
        return

    uv_plane[uv_y1:uv_y2, uv_x1:uv_x2:2] = u_value
    uv_plane[uv_y1:uv_y2, uv_x1 + 1:uv_x2:2] = v_value


def draw_boxes_on_nv12(nv12, width: int, height: int, boxes: list[dict], min_score: float) -> int:
    y_plane = nv12[:height, :]
    uv_plane = nv12[height:height + height // 2, :]
    y_value = 76
    u_value = 84
    v_value = 255
    thickness = 3
    visible = 0
    for box in boxes:
        if box["score"] < min_score:
            continue
        x1 = max(0, int(box["x1"]))
        y1 = max(0, int(box["y1"]))
        x2 = min(width - 1, int(box["x2"]))
        y2 = min(height - 1, int(box["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue
        fill_nv12_rect(y_plane, uv_plane, x1, y1, x2 + 1, y1 + thickness,
                       y_value, u_value, v_value)
        fill_nv12_rect(y_plane, uv_plane, x1, y2 - thickness + 1, x2 + 1, y2 + 1,
                       y_value, u_value, v_value)
        fill_nv12_rect(y_plane, uv_plane, x1, y1, x1 + thickness, y2 + 1,
                       y_value, u_value, v_value)
        fill_nv12_rect(y_plane, uv_plane, x2 - thickness + 1, y1, x2 + 1, y2 + 1,
                       y_value, u_value, v_value)
        label = class_label(box["class_id"])
        label_y = y1 - 6 if y1 >= 20 else min(height - 8, y1 + 18)
        cv2.putText(
            y_plane,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            235,
            1,
            cv2.LINE_AA,
        )
        visible += 1
    return visible


def decode_segmentation(tensors: list, width: int, height: int, top_k: int) -> list[dict]:
    decoded = pyneat.decode_segmentation(tensors, clamp_to=(width, height), top_k=top_k, strict=False)
    detections = []
    for item in decoded:
        boxes = np.asarray(item.boxes.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
        masks = np.asarray(item.masks.to_numpy(copy=True), dtype=np.uint8).reshape((-1, 160, 160))
        for row, mask in zip(boxes, masks):
            x1, y1, x2, y2, score, class_id = row.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "score": float(score), "class_id": int(class_id), "mask": mask,
            })
            if len(detections) >= top_k:
                return detections
    return detections


def frame_rect(det: dict, frame) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(round(det["x1"]))))
    y1 = max(0, min(h - 1, int(round(det["y1"]))))
    x2 = max(x1 + 1, min(w, int(round(det["x2"]))))
    y2 = max(y1 + 1, min(h, int(round(det["y2"]))))
    return x1, y1, x2, y2


def overlay_segmentation(frame, detections: list[dict], min_score: float) -> int:
    visible = 0
    for det in detections:
        if det["score"] < min_score:
            continue
        x1, y1, x2, y2 = frame_rect(det, frame)
        mask = cv2.resize(det["mask"], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        color = np.array(class_color(det["class_id"]), dtype=np.uint8)
        roi = frame[y1:y2, x1:x2]
        blend = cv2.addWeighted(roi, 0.45, np.full_like(roi, color), 0.55, 0.0)
        cv2.copyTo(blend, binary, roi)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours, -1, tuple(int(c) for c in color), 2, cv2.LINE_8)
        draw_boxes(frame, [det], min_score)
        visible += 1
    return visible


def bgr_to_nv12(frame_bgr):
    height, width = frame_bgr.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        raise RuntimeError(f"NV12 output requires even dimensions, got {width}x{height}")
    i420 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y_size = width * height
    uv_size = y_size // 4
    y = i420[:y_size].reshape(height, width)
    u = i420[y_size:y_size + uv_size].reshape(height // 2, width // 2)
    v = i420[y_size + uv_size:y_size + uv_size * 2].reshape(height // 2, width // 2)
    uv = np.empty((height // 2, width), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.ascontiguousarray(np.vstack((y, uv)))


def make_nv12_tensor(nv12, width: int, height: int):
    tensor = pyneat.Tensor.from_numpy(
        np.ascontiguousarray(nv12),
        copy=True,
        layout=pyneat.TensorLayout.HW,
        memory=pyneat.TensorMemory.CPU,
    )
    tensor.shape = [height, width]
    tensor.strides_bytes = [width, 1]
    tensor.byte_offset = 0
    image = pyneat.ImageSpec()
    image.format = pyneat.PixelFormat.NV12
    semantic = tensor.semantic
    semantic.image = image
    tensor.semantic = semantic

    y = pyneat.Plane()
    y.role = pyneat.PlaneRole.Y
    y.shape = [height, width]
    y.strides_bytes = [width, 1]
    y.byte_offset = 0

    uv = pyneat.Plane()
    uv.role = pyneat.PlaneRole.UV
    uv.shape = [height // 2, width]
    uv.strides_bytes = [width, 1]
    uv.byte_offset = width * height
    tensor.planes = [y, uv]
    return tensor


def push_video(video_run, frame_bgr) -> None:
    nv12 = bgr_to_nv12(frame_bgr)
    tensor = make_nv12_tensor(nv12, frame_bgr.shape[1], frame_bgr.shape[0])
    if not video_run.push([tensor]):
        raise RuntimeError("video push failed")


def push_nv12_video(video_run, nv12, width: int, height: int) -> None:
    tensor = make_nv12_tensor(nv12, width, height)
    if not video_run.push([tensor]):
        raise RuntimeError("video push failed")


def run(cfg: Config) -> int:
    load_runtime_dependencies()
    os.environ.setdefault("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1")
    validate_config(cfg)
    width, height, fps = probe_rtsp(cfg)
    source_graph = build_source_graph(cfg, width, height, fps)
    model_graph = build_model_graph(cfg, width, height, fps)
    video_graph, video_run, video_port = build_video_graph(cfg, width, height, fps)

    if cfg.print_backend:
        print("Source backend:")
        print(source_graph.describe_backend())
        print("Model backend:")
        print(model_graph.describe_backend())
        print("Video backend:")
        print(video_graph.describe_backend())

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    source_run_options = pyneat.RunOptions()
    source_run_options.preset = pyneat.RunPreset.Realtime
    source_run_options.queue_depth = 3
    source_run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    source_run_options.output_memory = pyneat.OutputMemory.Owned
    source_run = source_graph.build(source_run_options)
    model_run = model_graph.build(run_options)

    print(f"RTSP input: {cfg.rtsp_url}")
    print(f"Model:      {resolve_model_path(cfg)}")
    print(f"UDP output: udp://{cfg.udp_host}:{video_port}")
    print(f"Viewer:     gst-launch-1.0 -v udpsrc port={video_port} caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false")

    processed = 0
    profile_sum = StageProfile()
    run_start = time.perf_counter()
    steady_start = run_start
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            decoder_start = time.perf_counter()
            frame_tensors = source_run.pull_tensors(timeout_ms=20000)
            decoder_end = time.perf_counter()
            if not frame_tensors:
                print("[warn] timed out waiting for RTSP frame", file=sys.stderr)
                continue
            frame_tensor = frame_tensors[0]
            nv12, frame_width, frame_height = tensor_nv12_from_decoded(frame_tensor)
            model_tensor = make_nv12_tensor(nv12, frame_width, frame_height)
            decoder_end = time.perf_counter()

            inference_start = time.perf_counter()
            if not model_run.push([model_tensor]):
                print("[warn] failed to push frame to model", file=sys.stderr)
                continue
            model_sample = model_run.pull("detections", 20000)
            inference_end = time.perf_counter()
            if model_sample is None:
                print("[warn] timed out waiting for model output", file=sys.stderr)
                continue
            tensors = extract_tensors(model_sample)
            overlay_start = time.perf_counter()
            if model_spec()[0] == "yolov8n-seg":
                frame = np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))
                detections = decode_segmentation(tensors, width, height, cfg.top_k)
                visible = overlay_segmentation(frame, detections, cfg.score_threshold)
                overlay_end = time.perf_counter()
                encoder_start = time.perf_counter()
                push_video(video_run, frame)
            else:
                detections = decode_boxes(tensors, width, height, cfg.top_k)
                visible = draw_boxes_on_nv12(nv12, frame_width, frame_height, detections, cfg.score_threshold)
                overlay_end = time.perf_counter()
                encoder_start = time.perf_counter()
                push_nv12_video(video_run, nv12, frame_width, frame_height)
            encoder_end = time.perf_counter()
            processed += 1
            frame_profile = StageProfile(
                decoder_ms=(decoder_end - decoder_start) * 1000.0,
                inference_ms=(inference_end - inference_start) * 1000.0,
                overlay_ms=(overlay_end - overlay_start) * 1000.0,
                encoder_ms=(encoder_end - encoder_start) * 1000.0,
            )
            profile_sum.add(frame_profile)
            if processed == 1:
                steady_start = time.perf_counter()
            if should_log_frame(processed, cfg.frames):
                now = time.perf_counter()
                elapsed = now - run_start
                fps_now = processed / elapsed if elapsed > 0 else 0.0
                steady_elapsed = now - steady_start
                steady_fps = (processed - 1) / steady_elapsed if processed > 1 and steady_elapsed > 0 else fps_now
                avg = profile_sum.average(processed)
                print(
                    f"frame={processed} detections={len(detections)} visible={visible} "
                    f"fps={fps_now:.2f} steady_fps={steady_fps:.2f} "
                    f"ms(decoder={frame_profile.decoder_ms:.2f}, inference={frame_profile.inference_ms:.2f}, "
                    f"overlay={frame_profile.overlay_ms:.2f}, encoder={frame_profile.encoder_ms:.2f}, "
                    f"total={frame_profile.total_ms:.2f}) "
                    f"avg_ms(decoder={avg.decoder_ms:.2f}, inference={avg.inference_ms:.2f}, "
                    f"overlay={avg.overlay_ms:.2f}, encoder={avg.encoder_ms:.2f}, total={avg.total_ms:.2f})",
                    flush=True,
                )
            time.sleep(0)
    finally:
        model_run.close()
        source_run.close()
        video_run.close()
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
