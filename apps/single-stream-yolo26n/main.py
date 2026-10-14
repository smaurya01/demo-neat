#!/usr/bin/env python3
"""RTSP single-stream Neat demo with Python overlay and H.264/RTP UDP output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import os
from pathlib import Path
import subprocess
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
    rtsp_url: str = "<rtsp-url>"
    model_path: str = ""
    models_dir: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 60
    probe_source: bool = True  # read width/height/fps from the stream (ffprobe)
    model_width: int = 640
    model_height: int = 640
    latency_ms: int = 200
    score_threshold: float = 0.25
    nms_iou: float = 0.50
    top_k: int = 100
    num_classes: int = 80
    frames: int = 0
    udp_host: str = "<host-ip>"
    udp_port: int = 9000
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


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "default.conf"


def resolve_app_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def model_spec() -> tuple[str, str]:
    return "yolo26n", "yolo26n-det-bf16-mla_tess-b1.tar.gz"


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
    elif key == "width":
        cfg.width = int(value)
    elif key == "height":
        cfg.height = int(value)
    elif key == "fps":
        cfg.fps = int(value)
    elif key == "probe_source":
        cfg.probe_source = parse_bool(value)
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
    elif key in {"udp_port", "udp_port_base"}:  # udp_port_base: older name, same meaning
        cfg.udp_port = int(value)
    elif key == "bitrate_kbps":
        cfg.bitrate_kbps = int(value)
    elif key == "rtsp_transport":
        transport = value.strip().lower()
        if transport not in {"tcp", "udp"}:
            raise ValueError(f"rtsp_transport must be tcp or udp, got: {value}")
        cfg.tcp = transport == "tcp"
    elif key == "print_backend":
        cfg.print_backend = parse_bool(value)
    else:
        raise ValueError(f"unknown config key: {key}")


def strip_comment(line: str) -> str:
    """Cut a comment only where '#' starts the line or follows whitespace (as the C++ app does),
    so a value that contains '#' -- an RTSP password, say -- survives."""
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def load_config_file(cfg: Config, path: Path, required: bool) -> None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"config file not found: {path}")
        return
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = strip_comment(raw).strip()
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
    # required=True: deriving it from .exists() made a mistyped --config path silently fall back
    # to the dataclass defaults, and the user got a misleading "UDP port must be in 1..65535".
    load_config_file(cfg, args.config, required=True)
    if args.rtsp is not None:
        cfg.rtsp_url = args.rtsp
    if args.model is not None:
        cfg.model_path = args.model
    if args.models_dir is not None:
        cfg.models_dir = args.models_dir
    if args.width is not None:
        cfg.width = args.width
    if args.height is not None:
        cfg.height = args.height
    if args.fps is not None:
        cfg.fps = args.fps
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
    # Command line beats the config; --udp-port-base is the older name for --udp-port.
    if args.udp_port is not None:
        cfg.udp_port = args.udp_port
    elif args.udp_port_base is not None:
        cfg.udp_port = args.udp_port_base
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
    # The shipped config has placeholders (`<rtsp-url>`, `<host-ip>`). Catch them here with a
    # clear message instead of a slow ffprobe timeout and an obscure GStreamer error later.
    for key, flag, value in (("rtsp_url", "--rtsp", cfg.rtsp_url),
                             ("udp_host", "--udp-host", cfg.udp_host)):
        if not value or value.startswith("<"):
            raise ValueError(f"set {key} in the config (or pass {flag}); currently '{value}'")
    port = cfg.udp_port
    if port <= 0 or port > 65535:
        raise ValueError("UDP port must be in 1..65535")
    model_path = Path(resolve_model_path(cfg))
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")


def probe_stream(url: str, tcp: bool = True):
    """((width, height, fps), "") as the RTSP stream reports them, or (None, reason) on failure.

    Neat 0.5.0 sets the decoder up from width/height/fps even when the stream reports its own, and
    a value that differs from the camera stops build() with misconfig.media_caps. So the app reads
    them from the stream at start-up, as the reference Neat apps do. ffprobe runs as a separate
    process, so it cannot clash with Neat's GStreamer instance the way cv2.VideoCapture did.
    """
    cmd = ["ffprobe", "-v", "error"] + (["-rtsp_transport", "tcp"] if tcp else []) + [
        "-timeout", "5000000", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
        "-of", "default=nw=1", url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return None, "ffprobe not installed (sudo apt install ffmpeg)"
    except subprocess.TimeoutExpired:
        return None, "ffprobe timed out"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"ffprobe failed: {exc}"
    vals = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)

    def rate(text):  # "30/1" -> 30, "30000/1001" -> 30 (Neat accepts 29.97 against 30), "0/0" -> 0
        num, _, den = (text or "0").partition("/")
        try:
            value = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            return 0
        fps = int(round(value)) if value > 0 else 0
        # Outside 1..240 it is not a frame rate (e.g. an RTP clock rate such as 90000/1).
        return fps if 1 <= fps <= 240 else 0

    try:
        width, height = int(vals.get("width") or 0), int(vals.get("height") or 0)
    except ValueError:
        width = height = 0
    fps = rate(vals.get("avg_frame_rate")) or rate(vals.get("r_frame_rate"))
    if width > 0 and height > 0 and fps > 0:
        return (width, height, fps), ""
    errors = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
    return None, f"ffprobe: {errors[-1][:200]}" if errors else "no usable video stream reported"


def probe_rtsp(cfg: Config) -> tuple[int, int, int]:
    """Width, height and fps from the stream itself; the config's values only if probing fails."""
    probed, reason = probe_stream(cfg.rtsp_url, cfg.tcp) if cfg.probe_source else (None, "")
    if probed is None:
        if cfg.probe_source:
            print(f"[warn] could not probe {cfg.rtsp_url} ({reason}); using the config's "
                  f"{cfg.width}x{cfg.height}@{cfg.fps}", file=sys.stderr, flush=True)
        return cfg.width, cfg.height, cfg.fps
    if probed != (cfg.width, cfg.height, cfg.fps):
        print(f"[source] stream is {probed[0]}x{probed[1]}@{probed[2]} (config says "
              f"{cfg.width}x{cfg.height}@{cfg.fps}); using the stream's values", flush=True)
    cfg.width, cfg.height, cfg.fps = probed
    return probed


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


def make_model(cfg: Config, source_width: int = 0, source_height: int = 0):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    # Capacity must cover the frame the pipeline actually carries. Binding it to fallback_*
    # while every other node was built from the PROBED stream size made one graph declare two
    # geometries -- a 1080p source against the shipped 720p fallbacks either failed to build or
    # preprocessed a truncated frame.
    opt.preprocess.input_max_width = max(cfg.width, source_width)
    opt.preprocess.input_max_height = max(cfg.height, source_height)
    # 3, not 1: preproc publishes RGB (3 channels), and Neat 0.3.0 enforces this capacity
    # bound. With 1 it aborts: "color_input_requires_input_shape_channels_3".
    opt.preprocess.input_max_depth = 3
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = cfg.model_width
    opt.preprocess.resize.height = cfg.model_height
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.score_threshold
    # YOLO26 heads are NMS-free: YoloV26 skips NMS, so nms_iou has no effect for this model.
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.top_k
    opt.num_classes = cfg.num_classes
    return pyneat.Model(resolve_model_path(cfg), opt)


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
    # Was `use_simaai_pool = False`, deprecated in NEAT 0.4.0. Input.h documents the
    # exact mapping: False -> InputMemoryPolicy.SystemMemory. Behaviour-identical.
    input_opt.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
    return input_opt


def build_source_graph(cfg: Config, width: int, height: int, fps: int):
    graph = pyneat.Graph("source")
    graph.add(pyneat.groups.rtsp_decoded_input(make_source_options(cfg, width, height, fps)))
    # 4 slots, not 1: one slot means the decoder waits on the source thread every gap.
    # drop=True: on 0.4.0 a non-dropping appsink permanently stalls the decoder once behind.
    src_out = pyneat.OutputOptions.every_frame(4)
    src_out.drop = True
    graph.add(pyneat.nodes.output(src_out))
    return graph


def build_model_graph(cfg: Config, width: int, height: int, fps: int):
    graph = pyneat.Graph("model")
    graph.add(pyneat.nodes.input(make_nv12_input_options(width, height, fps)))
    graph.add(make_model(cfg, width, height))
    graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    return graph


def build_video_graph(cfg: Config, width: int, height: int, fps: int):
    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.udp_host
    sender_opt.channel = 0
    sender_opt.video_port_base = cfg.udp_port
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


def class_label(class_id: int) -> str:
    if 0 <= class_id < len(COCO_LABELS):
        return COCO_LABELS[class_id]
    return f"CLASS {class_id}"


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
    # Opened right after the first build, so a later build that fails still closes the runs
    # that were already built.
    video_run = source_run = model_run = None
    try:
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
        # Bound the FAILURE path too. Every `continue` below skips `processed += 1`, so on a dead
        # source the loop never advanced toward cfg.frames and `--frames 30` (the README's smoke
        # test) warned every 20 s forever instead of exiting.
        consecutive_source_timeouts = 0
        max_source_timeouts = 3
        while cfg.frames <= 0 or processed < cfg.frames:
            decoder_start = time.perf_counter()
            frame_tensors = source_run.pull_tensors(timeout_ms=20000)
            decoder_end = time.perf_counter()
            if not frame_tensors:
                consecutive_source_timeouts += 1
                print(f"[warn] timed out waiting for RTSP frame "
                      f"({consecutive_source_timeouts}/{max_source_timeouts})", file=sys.stderr)
                if consecutive_source_timeouts >= max_source_timeouts:
                    print("[err] no RTSP frame after "
                          f"{max_source_timeouts} consecutive timeouts; giving up",
                          file=sys.stderr)
                    break
                continue
            consecutive_source_timeouts = 0
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
                # A timeout does not cancel the in-flight inference: this frame's result is still
                # queued, and the next pull would pair it with the WRONG frame from then on.
                # Drain the sink (0 ms = non-blocking) to resynchronise before continuing.
                drained = 0
                while model_run.pull("detections", 0) is not None:
                    drained += 1
                print(f"[warn] timed out waiting for model output; drained {drained} stale "
                      "result(s)", file=sys.stderr)
                continue
            tensors = extract_tensors(model_sample)
            overlay_start = time.perf_counter()
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
        for run_ in (model_run, source_run, video_run):
            if run_ is not None:
                run_.close()
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
