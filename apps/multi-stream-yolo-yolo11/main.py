#!/usr/bin/env python3
"""Two-stream RTSP -> shared YOLO11 model stage -> annotated H.264/RTP UDP output.

Two RTSP inputs are decoded independently, run through ONE shared Neat YOLO11
model stage (a single compiled archive / model graph), decoded with Neat box
decode, annotated, and published as one H.264/RTP UDP stream per input. Stream
identity is preserved end to end: each decoded frame is inferenced and annotated
in place before the next stream is serviced, and every stream owns its own UDP
output port.

Design notes (all APIs traceable to /workspace/core):
- Per-stream RTSP source graph + video-sender graph, following the three-graph
  shuttle pattern from apps/single-stream-yolo-yolo11/main.py.
- ONE shared model graph (the "shared YOLO11 model stage"): both streams push
  into the same model Run handle round-robin and pull their own result before
  the next stream is serviced, so the bbox result always belongs to the frame
  just pushed. This is the pragmatic single-process form of the multi-stream
  pattern in core/tutorials/015_run_multiple_streams (combine/ByFrame joins
  streams inside one graph; here we keep per-stream sinks instead of joining).
- pyneat best practices per /workspace/overall-learning.md: ModelOptions
  preprocess presets (COCO_YOLO), BoxDecodeType, NO deprecated
  boxdecode_original_width/height fields (Model.h marks them deprecated;
  box decode reads geometry from preprocess metadata).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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


@dataclass
class Config:
    # Two RTSP inputs. Both default to the same source by design for now.
    rtsp_url_0: str = "rtsp://192.168.132.129:8555/stream"
    rtsp_url_1: str = "rtsp://192.168.132.129:8555/stream"
    model_path: str = ""
    models_dir: str = ""
    model_name: str = "yolo11"  # yolo11 | yolo26n (selects BoxDecodeType)
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
    udp_host: str = "192.168.132.129"
    # Stream i publishes on udp_port_base + i * udp_port_stride.
    udp_port_base: int = 5206
    udp_port_stride: int = 2
    bitrate_kbps: int = 4000
    tcp: bool = True
    print_backend: bool = False

    def rtsp_urls(self) -> list[str]:
        return [self.rtsp_url_0, self.rtsp_url_1]


@dataclass
class StreamContext:
    """Everything needed to service one logical stream. Identity = stream_id."""
    stream_id: int
    rtsp_url: str
    source_run: object
    video_run: object
    video_port: int
    width: int
    height: int
    fps: int
    processed: int = 0
    last_detections: int = 0
    last_visible: int = 0


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


def model_filename(cfg: Config) -> str:
    # Default archive name mirrors the single-stream YOLO11 app.
    return "yolo_11n_mpk.tar.gz"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def apply_config_value(cfg: Config, key: str, value: str) -> None:
    if key in {"rtsp_url", "rtsp_url_0"}:
        cfg.rtsp_url_0 = value
    elif key == "rtsp_url_1":
        cfg.rtsp_url_1 = value
    elif key == "model_path":
        cfg.model_path = value
    elif key == "models_dir":
        cfg.models_dir = value
    elif key == "model_name":
        cfg.model_name = value
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
    elif key in {"udp_port", "udp_port_base"}:
        cfg.udp_port_base = int(value)
    elif key == "udp_port_stride":
        cfg.udp_port_stride = int(value)
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
    parser.add_argument("--rtsp0", help="RTSP URL for stream 0")
    parser.add_argument("--rtsp1", help="RTSP URL for stream 1")
    parser.add_argument("--model")
    parser.add_argument("--models-dir")
    parser.add_argument("--model-name")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--model-width", type=int)
    parser.add_argument("--model-height", type=int)
    parser.add_argument("--score", type=float)
    parser.add_argument("--nms", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--classes", type=int)
    parser.add_argument("--frames", type=int, help="frames PER stream; 0 = run forever")
    parser.add_argument("--udp-host")
    parser.add_argument("--udp-port-base", type=int)
    parser.add_argument("--udp-port-stride", type=int)
    parser.add_argument("--bitrate", type=int)
    parser.add_argument("--rtsp-udp", action="store_true")
    parser.add_argument("--print-backend", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config()
    load_config_file(cfg, args.config, required=args.config.exists())
    if args.rtsp0 is not None:
        cfg.rtsp_url_0 = args.rtsp0
    if args.rtsp1 is not None:
        cfg.rtsp_url_1 = args.rtsp1
    if args.model is not None:
        cfg.model_path = args.model
    if args.models_dir is not None:
        cfg.models_dir = args.models_dir
    if args.model_name is not None:
        cfg.model_name = args.model_name
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
    if args.udp_port_base is not None:
        cfg.udp_port_base = args.udp_port_base
    if args.udp_port_stride is not None:
        cfg.udp_port_stride = args.udp_port_stride
    if args.bitrate is not None:
        cfg.bitrate_kbps = args.bitrate
    if args.rtsp_udp:
        cfg.tcp = False
    if args.print_backend:
        cfg.print_backend = True
    return cfg


def resolve_model_path(cfg: Config) -> str:
    if cfg.model_path:
        return str(resolve_app_path(cfg.model_path))
    models_dir = resolve_app_path(cfg.models_dir) if cfg.models_dir else Path(__file__).resolve().parent / "assets" / "models"
    return str(models_dir / model_filename(cfg))


def validate_config(cfg: Config) -> None:
    for idx, url in enumerate(cfg.rtsp_urls()):
        if not url:
            raise ValueError(f"RTSP URL for stream {idx} must not be empty")
    if not cfg.udp_host:
        raise ValueError("UDP host must not be empty")
    if cfg.udp_port_base <= 0 or cfg.udp_port_base > 65535:
        raise ValueError("udp_port_base must be in 1..65535")
    if cfg.udp_port_stride <= 0:
        raise ValueError("udp_port_stride must be positive so streams get distinct ports")
    model_path = Path(resolve_model_path(cfg))
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")


def probe_rtsp(cfg: Config, url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
        cap.release()
        if width > 0 and height > 0:
            return width, height, fps if fps > 0 else cfg.fallback_fps
    return cfg.fallback_width, cfg.fallback_height, cfg.fallback_fps


def make_source_options(cfg: Config, url: str, width: int, height: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
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
    # ModelOptions preprocess presets + BoxDecodeType (overall-learning.md).
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
    # The compile_ready YOLO11 surgery exposes the 6 YoloV26 grouped tensors, so
    # a compiled yolo11n/yolo26n archive decodes with BoxDecodeType.YoloV26.
    if cfg.model_name in {"yolo26n", "yolo11"}:
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
    else:
        opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.top_k
    opt.num_classes = cfg.num_classes
    # NOTE: intentionally NOT setting boxdecode_original_width/height — deprecated
    # in /workspace/core/include/model/Model.h; geometry comes from preprocess meta.
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
    input_opt.use_simaai_pool = False
    return input_opt


def build_source_graph(cfg: Config, url: str, width: int, height: int, fps: int):
    graph = pyneat.Graph(f"source_{width}x{height}")
    graph.add(pyneat.groups.rtsp_decoded_input(make_source_options(cfg, url, width, height, fps)))
    graph.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))
    return graph


def build_model_graph(cfg: Config, width: int, height: int, fps: int):
    graph = pyneat.Graph("model")
    graph.add(pyneat.nodes.input(make_nv12_input_options(width, height, fps)))
    graph.add(make_model(cfg))
    graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(1)))
    return graph


def build_video_graph(cfg: Config, stream_id: int, width: int, height: int, fps: int):
    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.udp_host
    sender_opt.channel = 0
    sender_opt.video_port_base = cfg.udp_port_base + stream_id * cfg.udp_port_stride
    sender_opt.encoder.bitrate_kbps = cfg.bitrate_kbps

    graph = pyneat.Graph(f"video_{stream_id}")
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
    for f in getattr(sample, "fields", []):
        tensors.extend(extract_tensors(f))
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
    uv_y1 = max(0, min(uv_plane.shape[0], y1 // 2))
    uv_y2 = max(0, min(uv_plane.shape[0], (y2 + 1) // 2))
    uv_x1 = max(0, min(width, x1 & ~1))
    uv_x2 = max(0, min(width, (x2 + 1) & ~1))
    if uv_x2 <= uv_x1 or uv_y2 <= uv_y1:
        return
    uv_plane[uv_y1:uv_y2, uv_x1:uv_x2:2] = u_value
    uv_plane[uv_y1:uv_y2, uv_x1 + 1:uv_x2:2] = v_value


def draw_boxes_on_nv12(nv12, width: int, height: int, boxes: list[dict], min_score: float,
                       banner: str) -> int:
    y_plane = nv12[:height, :]
    uv_plane = nv12[height:height + height // 2, :]
    y_value, u_value, v_value, thickness = 76, 84, 255, 3
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
        fill_nv12_rect(y_plane, uv_plane, x1, y1, x2 + 1, y1 + thickness, y_value, u_value, v_value)
        fill_nv12_rect(y_plane, uv_plane, x1, y2 - thickness + 1, x2 + 1, y2 + 1, y_value, u_value, v_value)
        fill_nv12_rect(y_plane, uv_plane, x1, y1, x1 + thickness, y2 + 1, y_value, u_value, v_value)
        fill_nv12_rect(y_plane, uv_plane, x2 - thickness + 1, y1, x2 + 1, y2 + 1, y_value, u_value, v_value)
        label = class_label(box["class_id"])
        label_y = y1 - 6 if y1 >= 20 else min(height - 8, y1 + 18)
        cv2.putText(y_plane, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 235, 1, cv2.LINE_AA)
        visible += 1
    # Per-stream identity banner burned into the Y plane (top-left).
    cv2.putText(y_plane, banner, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 235, 2, cv2.LINE_AA)
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


def service_stream(cfg: Config, ctx: StreamContext, model_run) -> bool:
    """Pull one frame for this stream, run the SHARED model, annotate, publish.

    Returns True if a frame was produced. Stream identity is preserved because
    the frame just pulled from ctx.source_run is the exact frame pushed into the
    shared model_run and then annotated/published on ctx.video_run.
    """
    frame_tensors = ctx.source_run.pull_tensors(timeout_ms=20000)
    if not frame_tensors:
        print(f"[warn] stream {ctx.stream_id}: timed out waiting for RTSP frame", file=sys.stderr)
        return False
    nv12, frame_width, frame_height = tensor_nv12_from_decoded(frame_tensors[0])
    model_tensor = make_nv12_tensor(nv12, frame_width, frame_height)

    if not model_run.push([model_tensor]):
        print(f"[warn] stream {ctx.stream_id}: failed to push frame to shared model", file=sys.stderr)
        return False
    model_sample = model_run.pull("detections", 20000)
    if model_sample is None:
        print(f"[warn] stream {ctx.stream_id}: timed out waiting for model output", file=sys.stderr)
        return False

    tensors = extract_tensors(model_sample)
    detections = decode_boxes(tensors, frame_width, frame_height, cfg.top_k)
    banner = f"STREAM {ctx.stream_id} :{ctx.video_port}"
    visible = draw_boxes_on_nv12(nv12, frame_width, frame_height, detections, cfg.score_threshold, banner)
    push_nv12_video(ctx.video_run, nv12, frame_width, frame_height)

    ctx.processed += 1
    ctx.last_detections = len(detections)
    ctx.last_visible = visible
    return True


def run(cfg: Config) -> int:
    load_runtime_dependencies()
    os.environ.setdefault("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1")
    validate_config(cfg)

    urls = cfg.rtsp_urls()
    # Probe each input; both default to the same source, so caps normally match.
    caps = [probe_rtsp(cfg, url) for url in urls]
    # The shared model graph needs one input caps; all streams must agree.
    model_w, model_h, model_fps = caps[0]
    for idx, (w, h, f) in enumerate(caps):
        if (w, h) != (model_w, model_h):
            raise RuntimeError(
                f"stream {idx} caps {w}x{h} differ from stream 0 {model_w}x{model_h}; "
                "a shared model stage needs matching input geometry"
            )

    # ONE shared model graph/Run for all streams.
    model_graph = build_model_graph(cfg, model_w, model_h, model_fps)
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    model_run = model_graph.build(run_options)

    contexts: list[StreamContext] = []
    for stream_id, url in enumerate(urls):
        w, h, f = caps[stream_id]
        source_graph = build_source_graph(cfg, url, w, h, f)
        src_opts = pyneat.RunOptions()
        src_opts.preset = pyneat.RunPreset.Realtime
        src_opts.queue_depth = 3
        src_opts.overflow_policy = pyneat.OverflowPolicy.KeepLatest
        src_opts.output_memory = pyneat.OutputMemory.Owned
        source_run = source_graph.build(src_opts)
        video_graph, video_run, video_port = build_video_graph(cfg, stream_id, w, h, f)
        contexts.append(StreamContext(
            stream_id=stream_id, rtsp_url=url, source_run=source_run,
            video_run=video_run, video_port=video_port, width=w, height=h, fps=f,
        ))
        if cfg.print_backend:
            print(f"Stream {stream_id} source backend:\n{source_graph.describe_backend()}")
            print(f"Stream {stream_id} video backend:\n{video_graph.describe_backend()}")
    if cfg.print_backend:
        print(f"Shared model backend:\n{model_graph.describe_backend()}")

    print(f"Shared model: {resolve_model_path(cfg)} (decode={cfg.model_name})")
    for ctx in contexts:
        print(f"Stream {ctx.stream_id}: RTSP {ctx.rtsp_url} -> udp://{cfg.udp_host}:{ctx.video_port}")
        print(
            f"  Viewer: gst-launch-1.0 -v udpsrc port={ctx.video_port} "
            f"caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" "
            f"! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false"
        )

    run_start = time.perf_counter()
    total_processed = 0
    try:
        # Round-robin: each stream is serviced once per outer iteration so both
        # UDP outputs advance together and every result stays with its stream.
        while cfg.frames <= 0 or min(c.processed for c in contexts) < cfg.frames:
            for ctx in contexts:
                if cfg.frames > 0 and ctx.processed >= cfg.frames:
                    continue
                if service_stream(cfg, ctx, model_run):
                    total_processed += 1
                    if ctx.processed == 1 or ctx.processed % 30 == 0 or ctx.processed == cfg.frames:
                        elapsed = time.perf_counter() - run_start
                        fps_now = total_processed / elapsed if elapsed > 0 else 0.0
                        print(
                            f"stream={ctx.stream_id} port={ctx.video_port} "
                            f"frame={ctx.processed} detections={ctx.last_detections} "
                            f"visible={ctx.last_visible} agg_fps={fps_now:.2f}",
                            flush=True,
                        )
            time.sleep(0)
    finally:
        model_run.close()
        for ctx in contexts:
            ctx.source_run.close()
            ctx.video_run.close()
    return total_processed


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
