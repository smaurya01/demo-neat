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
    insight_host: str = "<host-ip>"
    channel: int = 0
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    width: int = 1280
    height: int = 720
    fps: int = 60
    probe_source: bool = True  # read width/height/fps from the stream (ffprobe)
    model_width: int = 640
    model_height: int = 640
    latency_ms: int = 200
    score_threshold: float = 0.6
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
    elif key == "bitrate_kbps":
        cfg.bitrate_kbps = int(value)
    elif key == "rtsp_transport":
        cfg.tcp = value.strip().lower() == "tcp"
    elif key == "print_backend":
        cfg.print_backend = parse_bool(value)
    else:
        raise ValueError(f"unknown config key: {key}")


def strip_comment(line: str) -> str:
    """A '#' starts a comment only at the start of a line or after whitespace, so a value
    that contains '#' (for example an RTSP password) is kept intact."""
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def load_config_file(cfg: Config, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
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


def is_placeholder(value: str) -> bool:
    """True for an unedited placeholder such as "<rtsp-url>" or "rtsp://<rtsp-server-ip>/src1"."""
    return not value or value.startswith("<") or "://<" in value


def validate_config(cfg: Config) -> None:
    if is_placeholder(cfg.rtsp_url):
        raise ValueError(f"set rtsp_url in the config (or pass --rtsp) to your camera's RTSP URL; "
                         f"it is still \"{cfg.rtsp_url}\"")
    if is_placeholder(cfg.insight_host):
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


def probe_stream(url: str, tcp: bool = True):
    """((width, height, fps), "") as the RTSP stream reports them, or (None, reason) if it
    cannot be probed.

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
        return None, "ffprobe is not installed on the DevKit (sudo apt install ffmpeg)"
    except subprocess.TimeoutExpired:
        return None, "no reply within 15 s"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    vals = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)

    def rate(text):  # "30/1" -> 30, "30000/1001" -> 30 (Neat accepts 29.97 against 30), "0/0" -> 0
        num, _, den = (text or "0").partition("/")
        try:
            value = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            return 0
        value = int(round(value)) if value > 0 else 0
        # A real camera runs at 1..240 fps; anything else (a 90000/1 timebase) is not a frame rate.
        return value if 1 <= value <= 240 else 0

    try:
        width, height = int(vals.get("width") or 0), int(vals.get("height") or 0)
    except ValueError:
        width = height = 0
    fps = rate(vals.get("avg_frame_rate")) or rate(vals.get("r_frame_rate"))
    if width > 0 and height > 0 and fps > 0:
        return (width, height, fps), ""
    errors = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
    return None, (errors[-1][:200] if errors else "no usable width/height/frame rate in the reply")


def probe_rtsp(cfg: Config) -> tuple[int, int, int]:
    """Width, height and fps from the stream itself; the config's values only if probing fails."""
    probed, reason = probe_stream(cfg.rtsp_url, cfg.tcp) if cfg.probe_source else (None, "")
    if probed is None:
        if cfg.probe_source:
            print(f"[warn] could not probe {cfg.rtsp_url} with ffprobe ({reason}); using the "
                  f"config's {cfg.width}x{cfg.height}@{cfg.fps}", file=sys.stderr, flush=True)
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
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return opt


def make_model(cfg: Config, width: int, height: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.input_max_width = width
    opt.preprocess.input_max_height = height
    # 3, not 1: preproc publishes RGB (3 channels), and Neat 0.3.0 enforces this capacity
    # bound. With 1 it aborts: "color_input_requires_input_shape_channels_3".
    opt.preprocess.input_max_depth = 3
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = cfg.model_width
    opt.preprocess.resize.height = cfg.model_height
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # Zoo yolo_11n exposes raw 64-channel DFL heads -> YoloV8 decode family.
    # A YOLO11 compiled with model-compilation/ keeps the same heads but emits class logits:
    # keep YoloV8 and set decode_type_option = BoxDecodeTypeOption.GroupedByRoleLogit.
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
    for box in boxes:
        if box["score"] < min_score:
            continue
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        w = max(0, min(width - x, int(box["x2"] - box["x1"])))
        h = max(0, min(height - y, int(box["y2"] - box["y1"])))
        if w <= 0 or h <= 0:
            continue
        objects.append({
            "id": f"obj_{len(objects) + 1}",
            "label": class_label(box["class_id"]),
            "confidence": float(box["score"]),
            "bbox": [x, y, w, h],
        })
    return objects


def should_log_frame(processed: int, target_frames: int) -> bool:
    """Frame 1 is reported separately as a startup latency, so windowed stats
    start at the first multiple of 30."""
    return processed % 30 == 0 or (target_frames > 0 and processed == target_frames)


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

    try:
        metadata_options = pyneat.MetadataSenderOptions()
        metadata_options.host = cfg.insight_host
        metadata_options.channel = cfg.channel
        metadata_options.metadata_port_base = cfg.metadata_port_base
        metadata_sender = pyneat.MetadataSender(metadata_options)
        # Stop here, as main.cpp does, rather than stream video with no metadata at all.
        if hasattr(metadata_sender, "ok") and not metadata_sender.ok():
            raise RuntimeError(f"metadata sender failed to start on port {cfg.metadata_port_base}")
    except BaseException:
        run_handle.close()  # the graph is already running; do not leak it
        raise
    metadata_failures = 0

    print(f"RTSP input:  {cfg.rtsp_url} ({width}x{height}@{fps})")
    print(f"Model:       {resolve_model_path(cfg)}")
    print(f"Video out:   udp://{cfg.insight_host}:{video_port} (clean frames, no overlay)")
    print(f"Metadata out: udp://{cfg.insight_host}:{metadata_sender.metadata_port()} "
          f"(object-detection JSON, channel={cfg.channel})")
    print(f"Viewer:      Neat Insight Video Viewer channel {cfg.channel} draws boxes from metadata")

    # Windowed stats: a cumulative average would fold startup cost into every figure.
    processed = 0
    timeouts = 0
    window_frames = 0
    window_pull_ms = 0.0
    window_send_ms = 0.0
    startup_ms = 0.0
    run_start = time.perf_counter()
    steady_start = run_start
    window_start = run_start
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            pull_start = time.perf_counter()
            sample = run_handle.pull("detections", 20000)
            pull_end = time.perf_counter()
            if sample is None:
                timeouts += 1
                print("[warn] timed out waiting for detections", file=sys.stderr)
                # Drop the stalled interval instead of folding it into the
                # window, so one stall shows up as a timeout count rather than
                # silently biasing this line and every line after it.
                window_start = time.perf_counter()
                window_frames = 0
                window_pull_ms = 0.0
                window_send_ms = 0.0
                continue

            tensors = extract_tensors(sample)
            boxes = decode_boxes(tensors, width, height, cfg.top_k)
            objects = build_metadata_objects(boxes, width, height, cfg.score_threshold)

            frame_id = getattr(sample, "frame_id", -1)
            if frame_id is None or frame_id < 0:
                frame_id = processed
            send_start = time.perf_counter()
            # Send every frame, even empty, so stale boxes never linger in the viewer.
            # timestamp_ms = -1 OMITS the field on purpose. The video graph re-stamps each frame
            # with its push time, so no source timestamp can match a video frame; without one the
            # viewer pairs metadata with the newest frame instead.
            # A failed send (False, or an exception such as a network error) is counted and
            # reported, never fatal: the video keeps streaming, as in main.cpp.
            try:
                sent = metadata_sender.send_metadata(
                    "object-detection",
                    json.dumps({"objects": objects}, separators=(",", ":")),
                    -1,
                    str(frame_id),
                )
                send_error = ""
            except Exception as exc:  # e.g. OSError: network unreachable
                sent = False
                send_error = f": {exc}"
            if not sent:
                metadata_failures += 1
                if metadata_failures in (1, 10, 100) or metadata_failures % 1000 == 0:
                    print(f"[warn] metadata send failed ({metadata_failures} so far){send_error}",
                          file=sys.stderr, flush=True)
            send_end = time.perf_counter()

            processed += 1
            if processed == 1:
                # RTSP connect, jitter-buffer fill, decoder init and model load
                # all land inside the first pull. Report that once as a latency
                # and start the steady-state clock after it.
                startup_ms = (pull_end - run_start) * 1000.0
                steady_start = send_end
                window_start = send_end
                print(
                    f"frame=1 detections={len(boxes)} published={len(objects)} "
                    f"startup_ms={startup_ms:.1f}",
                    flush=True,
                )
                continue

            window_frames += 1
            window_pull_ms += (pull_end - pull_start) * 1000.0
            window_send_ms += (send_end - send_start) * 1000.0
            if should_log_frame(processed, cfg.frames) and window_frames > 0:
                now = time.perf_counter()
                window_s = now - window_start
                fps_now = window_frames / window_s if window_s > 0 else 0.0
                print(
                    f"frame={processed} detections={len(boxes)} published={len(objects)} "
                    f"fps={fps_now:.2f} "
                    f"avg_ms(pull={window_pull_ms / window_frames:.2f}, "
                    f"metadata_send={window_send_ms / window_frames:.2f})",
                    flush=True,
                )
                window_start = now
                window_frames = 0
                window_pull_ms = 0.0
                window_send_ms = 0.0
    finally:
        run_handle.close()
    # Frames 2..N over the post-startup interval. Any timeout stall is still in
    # that interval, which is why timeouts are reported alongside it.
    steady_s = time.perf_counter() - steady_start
    avg_fps = (processed - 1) / steady_s if processed > 1 and steady_s > 0 else 0.0
    print(f"processed={processed} timeouts={timeouts} startup_ms={startup_ms:.1f} "
          f"avg_fps={avg_fps:.2f} video={cfg.insight_host}:{video_port} "
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
