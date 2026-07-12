#!/usr/bin/env python3
"""Quad-stream / quad-model pipeline: 4 RTSP inputs -> 4 DIFFERENT models -> 4 UDP sinks.

Each logical stream owns its own RTSP source graph, its own model graph (a
distinct compiled archive / task), and its own H.264/RTP UDP sink graph. Stream
identity is preserved end to end: the frame pulled from stream i's source is the
exact frame pushed into stream i's model, decoded for stream i's task, annotated
in place, and published on stream i's UDP port.

Task routing (config/default.conf, per stream slot 0..3):
  * detection    -> yolo11s      (built-in Neat BoxDecodeType.YoloV26 on-device decode)
  * segmentation -> yolo11s-seg   (raw heads -> host decode, src/decoders.py)
  * pose         -> yolo26s-pose  (raw heads -> host decode)
  * yolox        -> yolox_s        (raw heads -> host decode)

Design provenance (every API traceable to https://github.com/sima-neat/core):
  * three-graph shuttle (source / model / video) — apps/multi-stream-yolo-yolo11/main.py
  * NV12 RTSP source + video_sender groups — pyneat.groups (core/include/neat/node_groups.h)
  * ModelOptions preprocess presets + BoxDecodeType — core/include/model/Model.h,
    core/include/pipeline/BoxDecodeType.h
  * push/pull named endpoints + RunOptions(queue_depth/overflow/preset) — core/include/pipeline/Run.h
  * host task decode is required because the surgery archives expose RAW per-scale
    heads (Neat's built-in fused decode covers only the detection family) — see
    src/decoders.py and ../../model-compilation/compile/_surgery_ultralytics.py.
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
dec = None  # src.decoders, imported after runtime deps

APP_DIR = Path(__file__).resolve().parent
# Archives live in this app's own assets/models/ (git-ignored), so the app is
# self-contained and works from any clone location. Build them with the
# graph-surgery flow in ../../model-compilation and copy them here — see README.
MODELS_DIR = APP_DIR / "assets" / "models"

# Default per-task compiled archives. These are the SAME paths config/default.conf
# sets, so the app behaves identically with or without a config file.
#
# NOTE on `pose`: that archive must be compiled with the keypoint head zero-padded
# 51 -> 64 channels (`pad_channels_to: 64` in
# ../../model-compilation/compile/_surgery_ultralytics.py). The padding is a
# load-bearing PERFORMANCE fix, not cosmetics: with the natural 51 channels the same
# model runs at 1782 ms/frame (0.6 fps), and at 8.5 ms/frame (117 fps) with 64 — a
# 209x speedup for identical weights. Unpadded, pose holds the shared MLA so long
# that the other three streams back up and fail their model push, and the quad
# cannot run at all. src/decoders.py slices channels 51..63 (the padding) back off.
DEFAULT_ARCHIVES = {
    "detection":    str(MODELS_DIR / "yolo11s.compile_ready_mpk.tar.gz"),
    "segmentation": str(MODELS_DIR / "yolo11s-seg.compile_ready_mpk.tar.gz"),
    "pose":         str(MODELS_DIR / "yolo26s-pose.compile_ready_mpk.tar.gz"),
    "yolox":        str(MODELS_DIR / "yolox_s.compile_ready_mpk.tar.gz"),
}
DEFAULT_TASKS = ["detection", "segmentation", "pose", "yolox"]


@dataclass
class StreamSpec:
    stream_id: int
    task: str
    rtsp_url: str
    model_path: str
    port: int


@dataclass
class Config:
    rtsp_default: str = "rtsp://<rtsp-server-ip>:8555/stream"
    udp_host: str = "<host-ip-that-receives-video>"
    udp_port_base: int = 5206
    udp_port_stride: int = 2
    model_width: int = 640
    model_height: int = 640
    fallback_width: int = 1280
    fallback_height: int = 720
    fallback_fps: int = 30
    latency_ms: int = 200
    score_threshold: float = 0.25
    nms_iou: float = 0.50
    top_k: int = 100
    bitrate_kbps: int = 4000
    tcp: bool = True
    queue_depth: int = 3
    frames: int = 0
    num_streams: int = 4
    print_backend: bool = False
    # Frames per stream excluded from the reported FPS/stage means (graph build,
    # model load and RTSP jitter-buffer fill all land on the first few frames).
    warmup_frames: int = 20
    # Run every stage of every stream on one thread (the original round-robin).
    # Slower by design; kept so the pipelined speedup stays reproducible.
    serial: bool = False
    # Frames kept in flight inside each model graph before the first pull.
    # 1 = lock-step push/pull; >1 lets a graph's CVU-preprocess / MLA /
    # box-decode stages overlap across consecutive frames.
    pipeline_depth: int = 2
    # Skip host decode + NV12 annotation, still encode and publish the frame.
    # This isolates the MODEL rate (RTSP -> preprocess -> MLA -> encode) from the
    # A65 host-decode and overlay cost, which for segmentation and pose is large.
    no_overlay: bool = False
    # Four model graphs share one MLA, so a pull can block far longer than a solo
    # run suggests. Keep this generous: a too-short timeout reports a scheduling
    # delay as a model failure.
    pull_timeout_ms: int = 20000
    # Execution target for the model's pre (tessellate/quantize) and post
    # (detessellate/dequantize) CVU stages: AUTO | EV74 | A65.
    # AUTO lets Neat's planner choose. The planner does not always pick the
    # accelerator: see README — with AUTO, yolo26s-pose's post stage lands on the
    # A65 and costs ~1.8 s/frame, while forcing EV74 makes it ~10 ms.
    cvu_pre_target: str = "AUTO"
    cvu_post_target: str = "AUTO"
    # Measure for a fixed wall-clock window instead of a per-stream frame count.
    # This is the correct design for a SHARED-resource throughput test: with a
    # frame cap, a fast stream keeps running (and keeps consuming the one MLA)
    # until the slowest stream also reaches the cap, which starves the slow
    # streams and reports rates that no steady state ever produced.
    # 0 = use --frames instead.
    duration_s: float = 0.0
    # per-slot overrides parsed from config; None => default
    _tasks: dict = field(default_factory=dict)
    _rtsp: dict = field(default_factory=dict)
    _models: dict = field(default_factory=dict)
    _ports: dict = field(default_factory=dict)

    def stream_specs(self) -> list[StreamSpec]:
        specs = []
        for i in range(self.num_streams):
            task = self._tasks.get(i, DEFAULT_TASKS[i % len(DEFAULT_TASKS)])
            rtsp = self._rtsp.get(i, self.rtsp_default)
            model = self._models.get(i, DEFAULT_ARCHIVES[task])
            port = self._ports.get(i, self.udp_port_base + i * self.udp_port_stride)
            specs.append(StreamSpec(i, task, rtsp, model, port))
        return specs


# ── runtime dep loading (board dist-packages, like the reference app) ─────────
def load_runtime_dependencies() -> None:
    global cv2, np, pyneat, dec
    if pyneat is not None:
        return
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module
    cv2, np, pyneat = cv2_module, np_module, pyneat_module
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from src import decoders as decoders_module
    dec = decoders_module


# ── config parsing ────────────────────────────────────────────────────────────
def parse_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "on"}


def apply_config_value(cfg: Config, key: str, value: str) -> None:
    key = key.strip()
    for i in range(8):
        if key == f"stream{i}_task":
            cfg._tasks[i] = value.strip(); return
        if key == f"stream{i}_rtsp":
            cfg._rtsp[i] = value.strip(); return
        if key == f"stream{i}_model":
            cfg._models[i] = str(resolve_app_path(value.strip())); return
        if key == f"stream{i}_port":
            cfg._ports[i] = int(value); return
    simple = {
        "rtsp_default": ("rtsp_default", str), "udp_host": ("udp_host", str),
        "udp_port_base": ("udp_port_base", int), "udp_port_stride": ("udp_port_stride", int),
        "model_width": ("model_width", int), "model_height": ("model_height", int),
        "fallback_width": ("fallback_width", int), "fallback_height": ("fallback_height", int),
        "fallback_fps": ("fallback_fps", int), "latency_ms": ("latency_ms", int),
        "score_threshold": ("score_threshold", float), "nms_iou": ("nms_iou", float),
        "top_k": ("top_k", int), "bitrate_kbps": ("bitrate_kbps", int),
        "queue_depth": ("queue_depth", int), "frames": ("frames", int),
        "num_streams": ("num_streams", int), "warmup_frames": ("warmup_frames", int),
        "pipeline_depth": ("pipeline_depth", int), "pull_timeout_ms": ("pull_timeout_ms", int),
        "cvu_pre_target": ("cvu_pre_target", str), "cvu_post_target": ("cvu_post_target", str),
        "duration_s": ("duration_s", float),
    }
    if key in simple:
        attr, cast = simple[key]
        setattr(cfg, attr, cast(value))
    elif key == "rtsp_transport":
        cfg.tcp = value.strip().lower() == "tcp"
    elif key == "print_backend":
        cfg.print_backend = parse_bool(value)
    elif key == "serial":
        cfg.serial = parse_bool(value)
    elif key == "no_overlay":
        cfg.no_overlay = parse_bool(value)
    else:
        raise ValueError(f"unknown config key: {key}")


def resolve_app_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else APP_DIR / p


def load_config_file(cfg: Config, path: Path) -> None:
    if not path.exists():
        return
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{n}: expected key=value")
        k, v = line.split("=", 1)
        apply_config_value(cfg, k, v)


def parse_args(argv) -> Config:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=APP_DIR / "config" / "default.conf")
    ap.add_argument("--rtsp", help="override RTSP URL for ALL streams")
    ap.add_argument("--num-streams", type=int)
    ap.add_argument("--task", choices=DEFAULT_TASKS,
                    help="run ONE stream with this task only, to measure that model's "
                         "solo rate with the MLA uncontended")
    ap.add_argument("--tasks",
                    help="comma-separated task list, one per stream slot, e.g. "
                         "'detection,segmentation,yolox'. Sets --num-streams to match.")
    ap.add_argument("--udp-host")
    ap.add_argument("--udp-port-base", type=int)
    ap.add_argument("--score", type=float)
    ap.add_argument("--nms", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--queue-depth", type=int)
    ap.add_argument("--frames", type=int, help="frames PER stream; 0 = forever")
    ap.add_argument("--warmup-frames", type=int,
                    help="frames per stream excluded from the reported FPS/stage means")
    ap.add_argument("--pipeline-depth", type=int,
                    help="frames kept in flight inside each model graph (1 = lock-step)")
    ap.add_argument("--serial", action="store_true",
                    help="single-threaded round-robin (the pre-pipelining behaviour)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip host decode + NV12 annotation; isolates the model rate")
    ap.add_argument("--duration", type=float,
                    help="measure for this many seconds after warmup (shared-resource "
                         "throughput test); overrides --frames as the stop condition")
    ap.add_argument("--pre-target", choices=["AUTO", "EV74", "A65"],
                    help="execution target for the model's pre (tessellate/quantize) CVU stage")
    ap.add_argument("--post-target", choices=["AUTO", "EV74", "A65"],
                    help="execution target for the model's post (detess/dequant) CVU stage")
    ap.add_argument("--rtsp-udp", action="store_true")
    ap.add_argument("--print-backend", action="store_true")
    a = ap.parse_args(argv)

    cfg = Config()
    load_config_file(cfg, a.config)
    if a.rtsp is not None:
        cfg.rtsp_default = a.rtsp
        cfg._rtsp = {}
    if a.task is not None:
        # Solo mode: one stream, one model, MLA uncontended.
        cfg.num_streams = 1
        cfg._tasks = {0: a.task}
        cfg._models = {}
    if a.tasks is not None:
        tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
        unknown = [t for t in tasks if t not in DEFAULT_ARCHIVES]
        if unknown:
            raise ValueError(f"unknown task(s): {unknown}; known: {DEFAULT_TASKS}")
        cfg._tasks = {i: t for i, t in enumerate(tasks)}
        cfg._models = {}
        cfg.num_streams = len(tasks)
    if a.num_streams is not None:
        cfg.num_streams = a.num_streams
    if a.udp_host is not None:
        cfg.udp_host = a.udp_host
    if a.udp_port_base is not None:
        cfg.udp_port_base = a.udp_port_base
    if a.score is not None:
        cfg.score_threshold = a.score
    if a.nms is not None:
        cfg.nms_iou = a.nms
    if a.top_k is not None:
        cfg.top_k = a.top_k
    if a.queue_depth is not None:
        cfg.queue_depth = a.queue_depth
    if a.frames is not None:
        cfg.frames = a.frames
    if a.warmup_frames is not None:
        cfg.warmup_frames = a.warmup_frames
    if a.pipeline_depth is not None:
        cfg.pipeline_depth = a.pipeline_depth
    if a.serial:
        cfg.serial = True
    if a.no_overlay:
        cfg.no_overlay = True
    if a.duration is not None:
        cfg.duration_s = a.duration
    if a.pre_target is not None:
        cfg.cvu_pre_target = a.pre_target
    if a.post_target is not None:
        cfg.cvu_post_target = a.post_target
    if a.rtsp_udp:
        cfg.tcp = False
    if a.print_backend:
        cfg.print_backend = True
    return cfg


# ── time profiling ────────────────────────────────────────────────────────────
# Each frame is timed stage by stage, so a slow stream can be attributed to a
# specific stage rather than guessed at. Stage meanings:
#   rtsp    wait + copy of one decoded NV12 frame out of the RTSP source graph
#   prep    NV12 -> pyneat.Tensor for the model input
#   infer   THE MODEL: push + pull (EV74 preprocess, MLA, on-device box decode)
#   decode  task decode. detection = Neat's fused on-device decode (cheap);
#           segmentation / pose / yolox = host A65 NumPy decode of the RAW heads
#   overlay NV12 Y-plane annotation (boxes, labels, masks, skeletons)
#   send    NV12 -> Tensor + push into the H.264/RTP UDP sender graph
#
# `infer` is the number to read for "can this model do 60 fps": it is the model
# stage alone, with no host decode and no overlay in it.
STAGES = ("rtsp", "prep", "infer", "decode", "overlay", "send")


class StageProfile:
    """Per-stage wall-clock samples for one stream."""

    def __init__(self) -> None:
        self.samples: dict = {name: [] for name in STAGES}
        self.total: list = []

    def add(self, timings: dict, total_ms: float) -> None:
        for name, value in timings.items():
            self.samples[name].append(value)
        self.total.append(total_ms)

    @staticmethod
    def _percentile(values: list, pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
        return ordered[idx]

    def mean(self, name: str) -> float:
        values = self.samples[name] if name != "total" else self.total
        return sum(values) / len(values) if values else 0.0

    def p95(self, name: str) -> float:
        values = self.samples[name] if name != "total" else self.total
        return self._percentile(values, 95)

    def frames(self) -> int:
        return len(self.total)


# ── stream context ────────────────────────────────────────────────────────────
@dataclass
class StreamContext:
    spec: StreamSpec
    source_run: object
    model_run: object
    video_run: object
    width: int
    height: int
    fps: int
    is_builtin_decode: bool
    processed: int = 0
    last_objs: int = 0
    dropped: int = 0
    pull_timeouts: int = 0
    model_q: object = None   # set by the pipelined engine
    out_q: object = None     # set by the pipelined engine
    # Frames already delivered when the steady-state window opened. Streams do
    # not cross the warmup mark at the same instant, so the window's frame count
    # must be (processed - steady_base), not (processed - warmup).
    steady_base: int = 0
    profile: StageProfile = field(default_factory=StageProfile)


# ── graph builders (NV12 shuttle; mirrors multi-stream-yolo-yolo11) ───────────
def make_source_options(cfg: Config, url: str, w: int, h: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = w
    opt.fallback_h264_height = h
    opt.fallback_h264_fps = fps
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = w
    opt.output_caps.height = h
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.SystemMemory
    return opt


def make_nv12_input_options(w: int, h: int, fps: int):
    o = pyneat.InputOptions()
    o.payload_type = pyneat.PayloadType.Image
    o.format = pyneat.Format.NV12
    o.width = w; o.height = h; o.depth = 1
    o.max_width = w; o.max_height = h; o.max_depth = 1
    o.fps_n = max(1, fps); o.fps_d = 1
    o.caps_override = f"video/x-raw,format=NV12,width={w},height={h},framerate={max(1, fps)}/1"
    o.use_simaai_pool = False
    return o


def make_model(cfg: Config, spec: StreamSpec):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.input_max_width = cfg.fallback_width
    opt.preprocess.input_max_height = cfg.fallback_height
    opt.preprocess.input_max_depth = 1
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = cfg.model_width
    opt.preprocess.resize.height = cfg.model_height
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # Pin the pre/post CVU stages when asked. Leaving these AUTO lets the planner
    # drop a raw-head model's detessellate+dequantize onto the A65, which is
    # ~180x slower than the EV74 for the pose head layout.
    opt.processcvu.pre_run_target = cfg.cvu_pre_target
    opt.processcvu.post_run_target = cfg.cvu_post_target
    if spec.task == "detection":
        # compile_ready yolo11s exposes the 6 YoloV26 grouped tensors -> on-device decode.
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
        opt.score_threshold = cfg.score_threshold
        opt.nms_iou_threshold = cfg.nms_iou
        opt.top_k = cfg.top_k
        opt.num_classes = 80
    # seg / pose / yolox: leave decode_type Unspecified -> model emits RAW heads,
    # decoded on the host in src/decoders.py. NOT setting boxdecode_original_* (deprecated).
    return pyneat.Model(spec.model_path, opt)


def build_source_graph(cfg: Config, url: str, w: int, h: int, fps: int):
    g = pyneat.Graph(f"source_{w}x{h}")
    g.add(pyneat.groups.rtsp_decoded_input(make_source_options(cfg, url, w, h, fps)))
    g.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))
    return g


def build_model_graph(cfg: Config, spec: StreamSpec, w: int, h: int, fps: int):
    g = pyneat.Graph(f"model_{spec.task}_{spec.stream_id}")
    g.add(pyneat.nodes.input(make_nv12_input_options(w, h, fps)))
    g.add(make_model(cfg, spec))
    endpoint = "detections" if spec.task == "detection" else "heads"
    g.add(pyneat.nodes.output(endpoint, pyneat.OutputOptions.every_frame(1)))
    return g, endpoint


def build_video_graph(cfg: Config, spec: StreamSpec, w: int, h: int, fps: int):
    so = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(w, h, max(1, fps))
    so.host = cfg.udp_host
    so.channel = 0
    so.video_port_base = spec.port
    so.encoder.bitrate_kbps = cfg.bitrate_kbps
    g = pyneat.Graph(f"video_{spec.stream_id}")
    g.add(pyneat.nodes.input(make_nv12_input_options(w, h, fps)))
    g.add(pyneat.groups.video_sender(so))
    seed = np.full((h * 3 // 2, w), 128, dtype=np.uint8)
    seed[:h, :] = 16
    return g, g.build([make_nv12_tensor(seed, w, h)]), so.video_port


# ── NV12 tensor plumbing (verbatim from multi-stream-yolo-yolo11) ─────────────
def tensor_dim(t, name):
    v = getattr(t, name)
    return int(v() if callable(v) else v)


def tensor_nv12_from_decoded(t):
    if not t.is_nv12():
        raise RuntimeError("expected decoded NV12 frame")
    w = tensor_dim(t, "width"); h = tensor_dim(t, "height")
    payload = np.frombuffer(t.copy_payload_bytes(), dtype=np.uint8)
    need = w * h * 3 // 2
    if payload.size < need:
        raise RuntimeError(f"NV12 payload too small: {payload.size} < {need}")
    return np.ascontiguousarray(payload[:need].reshape((h * 3 // 2, w))).copy(), w, h


def make_nv12_tensor(nv12, w, h):
    t = pyneat.Tensor.from_numpy(np.ascontiguousarray(nv12), copy=True,
                                 layout=pyneat.TensorLayout.HW, memory=pyneat.TensorMemory.CPU)
    t.shape = [h, w]; t.strides_bytes = [w, 1]; t.byte_offset = 0
    img = pyneat.ImageSpec(); img.format = pyneat.PixelFormat.NV12
    sem = t.semantic; sem.image = img; t.semantic = sem
    y = pyneat.Plane(); y.role = pyneat.PlaneRole.Y; y.shape = [h, w]; y.strides_bytes = [w, 1]; y.byte_offset = 0
    uv = pyneat.Plane(); uv.role = pyneat.PlaneRole.UV; uv.shape = [h // 2, w]; uv.strides_bytes = [w, 1]; uv.byte_offset = w * h
    t.planes = [y, uv]
    return t


def extract_tensors(sample) -> list:
    if sample is None or not hasattr(sample, "kind"):
        return []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    out = []
    for f in getattr(sample, "fields", []):
        out.extend(extract_tensors(f))
    return out


# ── annotation on the NV12 Y plane ────────────────────────────────────────────
def _fill_rect(y, x1, y1, x2, y2, val):
    hh, ww = y.shape
    x1 = max(0, min(ww, x1)); x2 = max(0, min(ww, x2))
    y1 = max(0, min(hh, y1)); y2 = max(0, min(hh, y2))
    if x2 > x1 and y2 > y1:
        y[y1:y2, x1:x2] = val


def annotate(nv12, w, h, result, task, banner) -> int:
    y = nv12[:h, :]
    th = 3
    drawn = 0
    for d in result.detections:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # box (bright edges on Y plane)
        _fill_rect(y, x1, y1, x2, y1 + th, 235)
        _fill_rect(y, x1, y2 - th, x2, y2, 235)
        _fill_rect(y, x1, y1, x1 + th, y2, 235)
        _fill_rect(y, x2 - th, y1, x2, y2, 235)
        if task in ("detection", "segmentation", "yolox"):
            label = dec.class_label(d.class_id)
        else:
            label = "PERSON"
        ly = y1 - 6 if y1 >= 20 else min(h - 8, y1 + 18)
        cv2.putText(y, f"{label} {d.score:.2f}", (x1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 235, 1, cv2.LINE_AA)
        if d.mask is not None:
            mh, mw = d.mask.shape
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                m = cv2.resize(d.mask, (bw, bh), interpolation=cv2.INTER_NEAREST).astype(bool)
                region = y[y1:y2, x1:x2]
                if region.shape[:2] == m.shape:
                    region[m] = np.clip(region[m].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        if d.keypoints is not None:
            for kx, ky, kv in d.keypoints:
                if kv < 0.3:
                    continue
                cx, cy = int(kx), int(ky)
                _fill_rect(y, cx - 2, cy - 2, cx + 3, cy + 3, 255)
            for a, b in dec.COCO_SKELETON:
                if a < len(d.keypoints) and b < len(d.keypoints):
                    if d.keypoints[a, 2] >= 0.3 and d.keypoints[b, 2] >= 0.3:
                        cv2.line(y, (int(d.keypoints[a, 0]), int(d.keypoints[a, 1])),
                                 (int(d.keypoints[b, 0]), int(d.keypoints[b, 1])), 200, 1, cv2.LINE_AA)
        drawn += 1
    cv2.putText(y, banner, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 235, 2, cv2.LINE_AA)
    return drawn


# ── per-stream service (preserves stream identity) ────────────────────────────
def decode_for_task(cfg, task, tensors, w, h):
    mw, mh = cfg.model_width, cfg.model_height
    kw = dict(model_w=mw, model_h=mh, score_thr=cfg.score_threshold,
              iou_thr=cfg.nms_iou, top_k=cfg.top_k)
    if task == "segmentation":
        return dec.decode_segmentation(tensors, w, h, **kw)
    if task == "pose":
        return dec.decode_pose(tensors, w, h, **kw)
    if task == "yolox":
        return dec.decode_yolox(tensors, w, h, **kw)
    return dec.decode_detection(tensors, w, h, **kw)


def decode_sample(cfg, ctx: StreamContext, tensors, fw: int, fh: int):
    """Task decode for one model output. Detection uses Neat's fused on-device
    box decode; segmentation / pose / yolox decode the RAW heads on the host."""
    if ctx.is_builtin_decode:
        result = dec.DecodeResult([])
        decoded = pyneat.decode_bbox(tensors, clamp_to=(fw, fh), top_k=cfg.top_k)
        for t in decoded:
            arr = np.asarray(t.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
            for x1, y1, x2, y2, sc, cid in arr:
                if sc < cfg.score_threshold:
                    continue
                result.detections.append(dec.Detection(float(x1), float(y1), float(x2),
                                                        float(y2), float(sc), int(cid)))
        return result
    if os.environ.get("QSQM_DEBUG") and ctx.processed == 0:
        shapes = []
        for t in tensors:
            a = t.to_numpy(copy=True) if hasattr(t, "to_numpy") else np.asarray(t)
            shapes.append(tuple(a.shape))
        print(f"[dbg] stream {ctx.spec.stream_id} {ctx.spec.task}: "
              f"{len(tensors)} raw tensors shapes={shapes}", file=sys.stderr, flush=True)
    return decode_for_task(cfg, ctx.spec.task, tensors, fw, fh)


def service_stream(cfg, ctx: StreamContext) -> bool:
    """Serial path: one thread runs every stage of this stream, timed per stage."""
    timings: dict = {}
    frame_start = time.perf_counter()
    endpoint = "detections" if ctx.is_builtin_decode else "heads"

    mark = time.perf_counter()
    frames = ctx.source_run.pull_tensors(timeout_ms=20000)
    if not frames:
        print(f"[warn] stream {ctx.spec.stream_id}: RTSP frame timeout", file=sys.stderr)
        return False
    nv12, fw, fh = tensor_nv12_from_decoded(frames[0])
    timings["rtsp"] = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    tensor = make_nv12_tensor(nv12, fw, fh)
    timings["prep"] = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    if not ctx.model_run.push([tensor]):
        print(f"[warn] stream {ctx.spec.stream_id}: model push failed", file=sys.stderr)
        return False
    try:
        sample = ctx.model_run.pull(endpoint, cfg.pull_timeout_ms)
    except Exception as exc:
        # pyneat raises on pull timeout rather than returning None.
        ctx.pull_timeouts += 1
        print(f"[warn] stream {ctx.spec.stream_id} ({ctx.spec.task}): "
              f"model pull failed: {exc}", file=sys.stderr, flush=True)
        return False
    if sample is None:
        ctx.pull_timeouts += 1
        return False
    timings["infer"] = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    result = None if cfg.no_overlay else decode_sample(cfg, ctx, extract_tensors(sample), fw, fh)
    timings["decode"] = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    if result is not None:
        banner = f"S{ctx.spec.stream_id} {ctx.spec.task.upper()} :{ctx.spec.port}"
        ctx.last_objs = annotate(nv12, fw, fh, result, ctx.spec.task, banner)
    timings["overlay"] = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    if not ctx.video_run.push([make_nv12_tensor(nv12, fw, fh)]):
        raise RuntimeError("video push failed")
    timings["send"] = (time.perf_counter() - mark) * 1000.0

    ctx.profile.add(timings, (time.perf_counter() - frame_start) * 1000.0)
    ctx.processed += 1
    return True


def print_profile(contexts: list, wall_s: float, mode: str, no_overlay: bool) -> None:
    """Per-stream/per-model stage breakdown + delivered FPS.

    Two different FPS numbers are reported and they answer different questions:

      model fps    = 1000 / mean(infer). What the MODEL stage alone sustains for
                     this stream — MLA time including its share of contention with
                     the other three models on the one MLA. This is the "60 fps
                     for the model" number.
      delivered fps = frames actually published to UDP per second of wall clock.
                     Includes host decode + overlay + encode, so for segmentation
                     and pose it is much lower than the model rate.
    """
    tag = "no-overlay" if no_overlay else "with-overlay"
    print(f"\n=== time profile ({mode}, {tag}; ms/frame, mean | p95) ===", flush=True)
    header = f"{'stream':>6} {'task':>13} {'frames':>6}"
    for name in STAGES:
        header += f" {name:>15}"
    header += f" {'latency':>15}"
    print(header, flush=True)
    total = 0
    for ctx in contexts:
        prof = ctx.profile
        base = ctx.steady_base
        if base and prof.frames() > base:
            for name in STAGES:
                prof.samples[name] = prof.samples[name][base:]
            prof.total = prof.total[base:]
        window = max(0, ctx.processed - base)
        total += window
        row = f"{ctx.spec.stream_id:>6} {ctx.spec.task:>13} {window:>6}"
        for name in STAGES:
            row += f" {prof.mean(name):>7.2f}|{prof.p95(name):<7.2f}"
        row += f" {prof.mean('total'):>7.2f}|{prof.p95('total'):<7.2f}"
        print(row, flush=True)

    print(f"\n=== per model-stream FPS (steady-state window {wall_s:.1f}s) ===", flush=True)
    print(f"{'stream':>6} {'task':>13} {'model':>13} {'model fps':>10} "
          f"{'delivered fps':>14} {'dropped':>8} {'pull t/o':>9}", flush=True)
    for ctx in contexts:
        infer = ctx.profile.mean("infer")
        window = max(0, ctx.processed - ctx.steady_base)
        model_fps = 1000.0 / infer if infer else 0.0
        delivered = window / wall_s if wall_s else 0.0
        print(f"{ctx.spec.stream_id:>6} {ctx.spec.task:>13} "
              f"{Path(ctx.spec.model_path).name.split('.')[0]:>13} "
              f"{model_fps:>10.1f} {delivered:>14.2f} {ctx.dropped:>8} "
              f"{ctx.pull_timeouts:>9}", flush=True)
    print(f"\naggregate delivered: {total / wall_s if wall_s else 0.0:.2f} fps "
          f"across {len(contexts)} stream-model pairs", flush=True)
    if wall_s <= 0.0:
        print("[warn] steady-state window never opened: at least one stream never "
              "reached --warmup-frames. FPS columns are meaningless; lower "
              "--warmup-frames or fix the starving stream.", flush=True)
    sys.stdout.flush()


# ── probing + main loop ───────────────────────────────────────────────────────
def probe_rtsp(cfg, url):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
        cap.release()
        if w > 0 and h > 0:
            return w, h, fps if fps > 0 else cfg.fallback_fps
    return cfg.fallback_width, cfg.fallback_height, cfg.fallback_fps


def build_run_options(cfg):
    ro = pyneat.RunOptions()
    ro.preset = pyneat.RunPreset.Realtime
    ro.queue_depth = cfg.queue_depth
    ro.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    # ZeroCopy. OutputMemory.Owned was tried (on the theory that queueing a ZeroCopy
    # Sample to the output thread was a use-after-free behind the intermittent
    # teardown abort) and REJECTED on evidence: it did not stop the abort (a run
    # still segfaulted mid-run under Owned), and it deep-copies every output sample.
    # That copy is expensive for the raw-head models, whose outputs are large:
    # segmentation fell 87 -> 22 model fps and yolox 90 -> 62. Keep ZeroCopy.
    # See README "Known limitations" for the still-open abort.
    ro.output_memory = pyneat.OutputMemory.ZeroCopy
    return ro


def run(cfg: Config) -> int:
    load_runtime_dependencies()
    os.environ.setdefault("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1")
    specs = cfg.stream_specs()
    for s in specs:
        if not Path(s.model_path).exists():
            raise FileNotFoundError(f"stream {s.stream_id} ({s.task}) archive not found: {s.model_path}")

    contexts: list[StreamContext] = []
    for s in specs:
        w, h, fps = probe_rtsp(cfg, s.rtsp_url)
        source_run = build_source_graph(cfg, s.rtsp_url, w, h, fps).build(build_run_options(cfg))
        model_graph, _ = build_model_graph(cfg, s, w, h, fps)
        model_run = model_graph.build(build_run_options(cfg))
        _, video_run, port = build_video_graph(cfg, s, w, h, fps)
        contexts.append(StreamContext(
            spec=s, source_run=source_run, model_run=model_run, video_run=video_run,
            width=w, height=h, fps=fps, is_builtin_decode=(s.task == "detection")))
        print(f"Stream {s.stream_id}: {s.task:12s} {Path(s.model_path).name}")
        print(f"  RTSP {s.rtsp_url} -> udp://{cfg.udp_host}:{port}")
        print(f"  Viewer: gst-launch-1.0 -v udpsrc port={port} "
              f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" '
              f"! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false")

    warmup = min(cfg.warmup_frames, cfg.frames - 1) if cfg.frames > 0 else cfg.warmup_frames
    warmup = max(0, warmup)
    try:
        if cfg.serial:
            return run_serial(cfg, contexts, warmup)
        return run_pipelined(cfg, contexts, warmup)
    finally:
        for c in contexts:
            c.model_run.close(); c.source_run.close(); c.video_run.close()


def run_serial(cfg, contexts: list, warmup: int) -> int:
    """Original single-threaded round-robin over all stream-model pairs.

    One thread runs every stage of every stream, so the four MLA models never
    overlap with each other's host decode/overlay work and the per-stream rate is
    1 / (num_streams * per-frame service time).
    """
    start = time.perf_counter()
    steady_start = None
    total = 0
    try:
        while cfg.frames <= 0 or min(c.processed for c in contexts) < cfg.frames:
            for ctx in contexts:
                if cfg.frames > 0 and ctx.processed >= cfg.frames:
                    continue
                if service_stream(cfg, ctx):
                    total += 1
                    if ctx.processed % 50 == 0:
                        print(f"stream={ctx.spec.stream_id} task={ctx.spec.task} "
                              f"frame={ctx.processed} objs={ctx.last_objs}", flush=True)
            if steady_start is None and min(c.processed for c in contexts) >= warmup:
                steady_start = time.perf_counter()
                for ctx in contexts:
                    ctx.steady_base = ctx.processed
            time.sleep(0)
    finally:
        if steady_start is not None:
            print_profile(contexts, time.perf_counter() - steady_start,
                          "serial", cfg.no_overlay)
    return total


# ── pipelined (threaded) engine ───────────────────────────────────────────────
# Unlike multi-stream-yolo-yolo11, every stream here owns its OWN model graph, so
# there is no shared model stage to serialize on and each stream gets its own
# model thread. The four model threads still contend for the single MLA — that
# contention is real and shows up inside `infer` — but their host work (RTSP copy,
# host head decode, NV12 annotation, encoder push) now overlaps instead of running
# lock-step behind one another.
#
#   source thread (one per stream)  RTSP pull -> NV12 -> Tensor  -> ctx.model_q
#   model thread  (one per stream)  push/pull that stream's Run -> ctx.out_q
#   output thread (one per stream)  task decode -> annotate -> UDP encoder push


def _drop_oldest_put(q, item) -> int:
    """Bounded put with drop-oldest. Returns frames dropped (0 or 1).

    A live 60 fps RTSP source does not wait. If a stage falls behind, blocking
    would grow latency without bound, so the oldest queued frame is discarded and
    the newest kept — the same intent as OverflowPolicy.KeepLatest.
    """
    import queue as queue_mod
    try:
        q.put_nowait(item)
        return 0
    except queue_mod.Full:
        try:
            q.get_nowait()
        except queue_mod.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue_mod.Full:
            return 1
        return 1


def run_pipelined(cfg, contexts: list, warmup: int) -> int:
    import queue as queue_mod
    import threading

    stop = threading.Event()
    errors: list = []
    steady = {"start": None, "lock": threading.Lock()}
    for ctx in contexts:
        ctx.model_q = queue_mod.Queue(maxsize=cfg.queue_depth)
        ctx.out_q = queue_mod.Queue(maxsize=cfg.queue_depth)

    def note_steady() -> None:
        with steady["lock"]:
            if steady["start"] is None and all(c.processed >= warmup for c in contexts):
                steady["start"] = time.perf_counter()
                for c in contexts:
                    c.steady_base = c.processed

    def source_thread(ctx: StreamContext) -> None:
        try:
            while not stop.is_set():
                mark = time.perf_counter()
                frames = ctx.source_run.pull_tensors(timeout_ms=5000)
                if not frames:
                    if stop.is_set():
                        return
                    print(f"[warn] stream {ctx.spec.stream_id}: RTSP frame timeout",
                          file=sys.stderr)
                    continue
                nv12, fw, fh = tensor_nv12_from_decoded(frames[0])
                rtsp_ms = (time.perf_counter() - mark) * 1000.0

                mark = time.perf_counter()
                tensor = make_nv12_tensor(nv12, fw, fh)
                prep_ms = (time.perf_counter() - mark) * 1000.0

                ctx.dropped += _drop_oldest_put(
                    ctx.model_q, (nv12, tensor, fw, fh, rtsp_ms, prep_ms, time.perf_counter()))
        except Exception as exc:
            errors.append(f"source {ctx.spec.stream_id}: {exc}")
            stop.set()

    def model_thread(ctx: StreamContext) -> None:
        """Sole pusher AND sole puller of THIS stream's model Run, so the graph's
        FIFO ordering keeps the Nth sample matched to the Nth frame pushed."""
        endpoint = "detections" if ctx.is_builtin_decode else "heads"
        pending: list = []
        try:
            while True:
                if stop.is_set() and not pending:
                    return
                item = None
                if not stop.is_set():
                    try:
                        item = ctx.model_q.get(timeout=0.2)
                    except queue_mod.Empty:
                        item = None
                if item is not None:
                    nv12, tensor, fw, fh, rtsp_ms, prep_ms, t_in = item
                    push_mark = time.perf_counter()
                    if not ctx.model_run.push([tensor]):
                        print(f"[warn] stream {ctx.spec.stream_id}: model push failed",
                              file=sys.stderr)
                        continue
                    pending.append((nv12, fw, fh, rtsp_ms, prep_ms, t_in, push_mark))

                if pending and (len(pending) >= cfg.pipeline_depth or item is None):
                    nv12, fw, fh, rtsp_ms, prep_ms, t_in, push_mark = pending.pop(0)
                    # Four model graphs share one MLA. Under contention a pull can
                    # block far longer than a solo run would suggest, and pyneat
                    # RAISES on pull timeout rather than returning None. Treat that
                    # as a dropped frame for this stream, not as a fatal error for
                    # the whole pipeline.
                    try:
                        sample = ctx.model_run.pull(endpoint, cfg.pull_timeout_ms)
                    except Exception as exc:
                        if stop.is_set():
                            return
                        ctx.pull_timeouts += 1
                        print(f"[warn] stream {ctx.spec.stream_id} ({ctx.spec.task}): "
                              f"model pull failed: {exc}", file=sys.stderr, flush=True)
                        continue
                    if sample is None:
                        if stop.is_set():
                            return
                        ctx.pull_timeouts += 1
                        continue
                    infer_ms = (time.perf_counter() - push_mark) * 1000.0
                    ctx.dropped += _drop_oldest_put(
                        ctx.out_q, (nv12, sample, fw, fh, rtsp_ms, prep_ms, infer_ms, t_in))
        except Exception as exc:
            errors.append(f"model {ctx.spec.stream_id}: {exc}")
            stop.set()

    def output_thread(ctx: StreamContext) -> None:
        try:
            while not stop.is_set():
                try:
                    nv12, sample, fw, fh, rtsp_ms, prep_ms, infer_ms, t_in = \
                        ctx.out_q.get(timeout=0.2)
                except queue_mod.Empty:
                    continue

                mark = time.perf_counter()
                result = (None if cfg.no_overlay
                          else decode_sample(cfg, ctx, extract_tensors(sample), fw, fh))
                decode_ms = (time.perf_counter() - mark) * 1000.0

                mark = time.perf_counter()
                if result is not None:
                    banner = f"S{ctx.spec.stream_id} {ctx.spec.task.upper()} :{ctx.spec.port}"
                    ctx.last_objs = annotate(nv12, fw, fh, result, ctx.spec.task, banner)
                overlay_ms = (time.perf_counter() - mark) * 1000.0

                mark = time.perf_counter()
                if not ctx.video_run.push([make_nv12_tensor(nv12, fw, fh)]):
                    raise RuntimeError("video push failed")
                send_ms = (time.perf_counter() - mark) * 1000.0

                ctx.profile.add(
                    {"rtsp": rtsp_ms, "prep": prep_ms, "infer": infer_ms,
                     "decode": decode_ms, "overlay": overlay_ms, "send": send_ms},
                    (time.perf_counter() - t_in) * 1000.0,
                )
                ctx.processed += 1
                note_steady()
                if ctx.processed % 60 == 0:
                    print(f"stream={ctx.spec.stream_id} task={ctx.spec.task} "
                          f"frame={ctx.processed} objs={ctx.last_objs} "
                          f"dropped={ctx.dropped}", flush=True)
                if (cfg.duration_s <= 0 and cfg.frames > 0
                        and all(c.processed >= cfg.frames for c in contexts)):
                    stop.set()
        except Exception as exc:
            errors.append(f"output {ctx.spec.stream_id}: {exc}")
            stop.set()

    threads = []
    for ctx in contexts:
        threads.append(threading.Thread(target=source_thread, args=(ctx,),
                                        name=f"src{ctx.spec.stream_id}", daemon=True))
        threads.append(threading.Thread(target=model_thread, args=(ctx,),
                                        name=f"mdl{ctx.spec.stream_id}", daemon=True))
        threads.append(threading.Thread(target=output_thread, args=(ctx,),
                                        name=f"out{ctx.spec.stream_id}", daemon=True))
    print(f"\npipelined: {len(contexts)} x (source + model + output) threads, "
          f"pipeline_depth={cfg.pipeline_depth}, overlay="
          f"{'off' if cfg.no_overlay else 'on'}", flush=True)
    for t in threads:
        t.start()
    try:
        while not stop.is_set():
            if cfg.duration_s > 0:
                with steady["lock"]:
                    started = steady["start"]
                if started is not None and (time.perf_counter() - started) >= cfg.duration_s:
                    stop.set()
                    break
            time.sleep(0.05)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        # Snapshot the window BEFORE joining: the drain in model/output threads
        # would otherwise keep incrementing processed after the clock stopped.
        wall = (time.perf_counter() - steady["start"]) if steady["start"] else 0.0
        final = {c.spec.stream_id: c.processed for c in contexts}
        for t in threads:
            t.join(timeout=5.0)
        for c in contexts:
            c.processed = final[c.spec.stream_id]
        print_profile(contexts, wall, "pipelined", cfg.no_overlay)
        for err in errors:
            print(f"[ERR] {err}", file=sys.stderr, flush=True)
        # Model Runs are built with OutputMemory.ZeroCopy, so a pulled Sample
        # points into runtime-owned memory. Drop every queued Sample while the
        # Runs are still alive, before run()'s finally closes them.
        import gc
        for ctx in contexts:
            for q in (ctx.model_q, ctx.out_q):
                while True:
                    try:
                        q.get_nowait()
                    except queue_mod.Empty:
                        break
        gc.collect()
    return sum(max(0, c.processed - c.steady_base) for c in contexts)


def main(argv=None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
