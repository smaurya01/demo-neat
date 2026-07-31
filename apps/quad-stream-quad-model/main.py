#!/usr/bin/env python3
"""Quad-stream / quad-model pipeline: 4 RTSP inputs -> 4 DIFFERENT models -> 4 UDP sinks.

Each logical stream owns its own RTSP source graph, its own model graph (a
distinct compiled archive / task), and its own H.264/RTP UDP sink graph. Stream
identity is preserved end to end: the frame pulled from stream i's source is the
exact frame pushed into stream i's model, decoded for stream i's task, annotated
in place, and published on stream i's UDP port.

Task routing (config/default.conf, per stream slot 0..3). EVERY stream decodes
on-device with Neat's fused BoxDecode — there is no host-side decode anywhere:

  task          model                     source          Neat decode family
  detection     yolo_11s                  MODEL ZOO       BoxDecodeType.YoloV8
  segmentation  yolo_11s_seg              MODEL ZOO       BoxDecodeType.YoloV8Seg
  pose          yolo26s-pose              self-compiled   BoxDecodeType.YoloV26Pose
  yolox         yolox_s                   self-compiled   BoxDecodeType.YoloX

The decode family is chosen by the shape of the archive's HEAD, not by the model's
version number:
  * The zoo archives keep the raw 64-channel DFL bbox heads -> the YoloV8 family
    (this is why zoo YOLO11 decodes as YoloV8, not as some "YoloV11").
  * The self-compiled archives fold the DFL into the graph and emit 4-channel
    l/t/r/b distance heads -> the YoloV26 family.
Both are correct; they are different contracts. Get it wrong and the app still
runs and still draws boxes — just decoded from the wrong channels.

The zoo has no YOLO-pose and no YOLOX, which is why those two stay self-compiled.

Design provenance (every API traceable to https://github.com/sima-neat/core):
  * three-graph shuttle (source / model / video) — apps/multi-stream-yolo-yolo11/main.py
  * NV12 RTSP source + video_sender groups — pyneat.groups (core/include/neat/node_groups.h)
  * ModelOptions preprocess presets + BoxDecodeType — core/include/model/Model.h,
    core/include/pipeline/BoxDecodeType.h
  * push/pull named endpoints + RunOptions(queue_depth/overflow/preset) — core/include/pipeline/Run.h
  * pyneat.decode_bbox / decode_pose / decode_segmentation read the BBOX payload the
    on-device BoxDecode stage produced — core/python/src/module.cpp.
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
ov = None  # src.overlay (labels + result containers), imported after runtime deps

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
# cannot run at all.
#
# The padding is also compatible with Neat's on-device pose decode: YoloV26Pose
# requires the keypoint head's SLICE depth to be 51 but allows its INPUT depth to be
# larger (see infer_*/keypoint_depth checks in core BoxDecodeStageSemantics.cpp), so
# the 51-of-64 padded head is accepted as-is and Neat ignores the 13 pad channels.
DEFAULT_ARCHIVES = {
    # From the SiMa model zoo — see README "Model Download Command".
    "detection":    str(MODELS_DIR / "yolo_11s_mpk.tar.gz"),
    "segmentation": str(MODELS_DIR / "yolo_11s_seg_mpk.tar.gz"),
    # Not published in the zoo (it has no YOLO-pose and no YOLOX), so these two are
    # built by the graph-surgery flow in ../../model-compilation.
    "pose":         str(MODELS_DIR / "yolo26s-pose.compile_ready_mpk.tar.gz"),
    "yolox":        str(MODELS_DIR / "yolox_s.compile_ready_mpk.tar.gz"),
}
DEFAULT_TASKS = ["detection", "segmentation", "pose", "yolox"]

# Neat on-device decode family per task, keyed by the archive's head contract.
# See the module docstring: the family follows the HEAD SHAPE, not the model name.
DECODE_FAMILY = {
    "detection":    "YoloV8",       # zoo yolo_11s      : 64-ch DFL bbox + 80-ch class
    "segmentation": "YoloV8Seg",    # zoo yolo_11s_seg  : + 32-ch mask coeff + 32x160 proto
    "pose":         "YoloV26Pose",  # yolo26s-pose      : 4-ch l/t/r/b + 1-ch score + 51-ch kpt
    "yolox":        "YoloX",        # yolox_s           : (4, 1, 80) triplets per scale
}
# Pose is single-class ("person"); the rest are COCO-80.
NUM_CLASSES = {"detection": 80, "segmentation": 80, "yolox": 80, "pose": 1}

# Input normalization, per MODEL FAMILY — not a detail you can share across all four.
#
# COCO_YOLO feeds the model x/255 (values in [0,1]). Every Ultralytics model wants that.
# Megvii YOLOX does NOT: it is trained on RAW 0-255 pixels, so it needs normalization
# turned off. Feed YOLOX the COCO_YOLO preset and it sees an image 255x too dark; its
# objectness logits pin negative and it detects NOTHING — at full speed, with healthy
# FPS and no error anywhere. Measured on COCO val 000000000139:
#
#            preset          obj logit max      boxes @0.25
#            COCO_YOLO           -3.47                0
#            None                +1.36                8   <- correct
#   (CPU reference, raw 0-255)   +5.90
#
# This pairs with the archive: yolox_s is compiled with std=1/255 (models.yaml) so its
# input quantization expects 0-255 (q_scale ~ 1.0, vs ~255 for the /255 models). BOTH
# halves are required — the compile-time scale and the runtime preset must agree.
NORMALIZE_PRESET = {
    "detection":    "COCO_YOLO",
    "segmentation": "COCO_YOLO",
    "pose":         "COCO_YOLO",
    "yolox":        "None",       # Megvii YOLOX: raw 0-255 input, no normalization.
}

# Every model graph now ends in an on-device BoxDecode stage, so they all publish a
# decoded BBOX payload on the same endpoint. (This used to be "detections" for the
# detection stream and "heads" for the raw-head streams.)
MODEL_ENDPOINT = "detections"


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
    # --- decoder admission (Neat 0.3.0), mirroring main.cpp ---
    # These feed the decoder-admission LEASE, not plain GStreamer element properties.
    # They only take effect because every stream's decoder now lives in ONE graph/Run:
    # Neat requests an admission plan only when a single graph holds more than one
    # H.264 decoder. Four separate source Runs -- the old layout -- never asked, and
    # un-admitted decoders collectively saturate around 43 fps each.
    decoder_buffers: int = 16        # per-stream decoder OUTPUT pool
    decoder_input_buffers: int = 2   # per-stream compressed-INPUT pool
    decoder_tuning: str = "auto"     # auto | default | low-memory | throughput-low-latency
    rtsp_drop_on_latency: bool = True
    # Leave False for a 59.94 fps source: skip_rtsp_probe pins an INTEGER framerate and
    # 60/1 against a 60000/1001 stream fails negotiation outright (0 fps everywhere).
    skip_rtsp_probe: bool = False
    frames: int = 0
    num_streams: int = 4
    print_backend: bool = False
    # Frames per stream excluded from the reported FPS/stage means (graph build,
    # model load and RTSP jitter-buffer fill all land on the first few frames).
    warmup_frames: int = 20
    # How often the live time profile is printed to the terminal while running.
    # 0 disables it and you only get the summary at exit.
    profile_interval_s: float = 5.0
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
    global cv2, np, pyneat, ov
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
    from src import overlay as overlay_module
    ov = overlay_module


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
        "profile_interval": ("profile_interval_s", float),
        "pipeline_depth": ("pipeline_depth", int), "pull_timeout_ms": ("pull_timeout_ms", int),
        "cvu_pre_target": ("cvu_pre_target", str), "cvu_post_target": ("cvu_post_target", str),
        "duration_s": ("duration_s", float),
        "decoder_buffers": ("decoder_buffers", int),
        "decoder_input_buffers": ("decoder_input_buffers", int),
        "decoder_tuning": ("decoder_tuning", str),
    }
    if key in simple:
        attr, cast = simple[key]
        setattr(cfg, attr, cast(value))
    elif key == "rtsp_transport":
        cfg.tcp = value.strip().lower() == "tcp"
    elif key == "drop_on_latency":
        cfg.rtsp_drop_on_latency = parse_bool(value)
    elif key == "skip_rtsp_probe":
        cfg.skip_rtsp_probe = parse_bool(value)
    elif key == "print_backend":
        cfg.print_backend = parse_bool(value)
    elif key == "serial":
        cfg.serial = parse_bool(value)
    elif key == "no_overlay":
        cfg.no_overlay = parse_bool(value)
    elif key in {"model_queue_depth", "output_queue_depth"}:
        # C++-ONLY knobs. config/default.conf is shared with main.cpp, which splits
        # the in-flight bound into model_queue_depth (frames inside the model graph)
        # and output_queue_depth (frames parked before the output thread). This
        # implementation uses one `queue_depth` for both, so these are accepted and
        # ignored rather than raising -- a shared config must not fail one twin.
        pass
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
    ap.add_argument("--profile-interval", type=float,
                    help="seconds between live time-profile prints (0 = off, default 5)")
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
    if a.profile_interval is not None:
        cfg.profile_interval_s = a.profile_interval
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
# specific stage rather than guessed at. Same stage names as main.cpp, so the two
# implementations are directly comparable.
#
# Stage meanings — note that two of them do NOT measure what their name suggests,
# because the stages either side of the model are PIPELINED on the device:
#
#   decode   pull + copy of one decoded NV12 frame out of the RTSP source graph.
#            ARRIVAL-GATED: this thread blocks until the next frame exists, so on a
#            healthy 60 fps stream it reads ~16.7 ms however fast the decoder is.
#            That is the frame INTERVAL, not the decode cost. The H.264 decode runs
#            on the hardware decoder and is not visible from the host. It only
#            climbs above the interval once the decoder has genuinely fallen behind.
#            (In main.cpp a dedicated source thread absorbs this wait; here it sits
#            in the critical path, which is part of why Python delivers less.)
#   prep     NV12 -> pyneat.Tensor for the model input.
#   infer    THE MODEL: push + pull (EV74 preprocess, MLA, on-device box decode).
#   postproc host-side read of the already-decoded payload + instance build. Every
#            model now box-decodes ON DEVICE, so this is just a NumPy reshape. It
#            used to be a full A65 NumPy decode of the raw heads (~340 ms for seg).
#   overlay  NV12 Y-plane annotation (boxes, labels, masks, skeletons).
#   encode   NV12 -> Tensor + push into the H.264/RTP UDP sender graph. This ENQUEUES
#            to the encoder and returns, so it is encoder HEADROOM, not encode
#            latency: near zero until the encoder falls behind, and only then does
#            its backpressure surface here.
#
# `infer` is the number to read for "can this model do 60 fps": it is the model
# stage alone, with no host postproc and no overlay in it.
STAGES = ("decode", "prep", "infer", "postproc", "overlay", "encode")


class StageProfile:
    """Per-stage wall-clock samples for one stream."""

    def __init__(self) -> None:
        self.samples: dict = {name: [] for name in STAGES}
        self.total: list = []

    def add(self, timings: dict, total_ms: float) -> None:
        # There are two call sites (the serial path and the threaded output thread) and
        # they build this dict independently. A key that drifts from STAGES would either
        # KeyError deep inside a worker thread or, worse, silently drop a stage from the
        # report. Fail loudly and say exactly which key is wrong.
        if timings.keys() != self.samples.keys():
            raise KeyError(
                f"stage timings {sorted(timings)} do not match STAGES {sorted(self.samples)}"
            )
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


# ── live time profile ─────────────────────────────────────────────────────────
# ONE reporter prints the whole table every `profile_interval` seconds, so the terminal
# shows stage timings as the run happens rather than only at exit. Every number is the mean
# over THAT WINDOW (since the previous print), not a cumulative average — a cumulative mean
# hides a stream that degrades halfway through the run.
#
# Same columns and same window semantics as main.cpp, so the two are directly comparable.
class LiveCursor:
    """Where the previous window ended, for one stream."""

    def __init__(self) -> None:
        self.n = 0            # index into profile.samples[*] / profile.total
        self.processed = 0
        self.t = 0.0


def print_live_profile(contexts, cursors, start: float) -> None:
    now = time.perf_counter()
    out = [f"\n── t={now - start:5.1f}s ─── ms/frame, mean over this window ───"]
    out.append(f"{'stream':<7}{'task':<14}"
               + "".join(f"{name:>9}" for name in STAGES)
               + f"{'latency':>9}{'mdl fps':>9}{'deliv fps':>11}{'dropped':>9}{'objs':>6}")

    aggregate = 0.0
    for ctx, cur in zip(contexts, cursors):
        # len() first, then slice to it: list.append is atomic under the GIL, so a slice
        # bounded by a previously-read length can never see a partially written element.
        n = len(ctx.profile.total)

        def window_mean(values: list) -> float:
            chunk = values[cur.n:n]
            return sum(chunk) / len(chunk) if chunk else 0.0

        means = {name: window_mean(ctx.profile.samples[name]) for name in STAGES}
        latency = window_mean(ctx.profile.total)

        dt = now - cur.t
        delivered = (ctx.processed - cur.processed) / dt if dt > 0 else 0.0
        model_fps = 1000.0 / means["infer"] if means["infer"] > 0 else 0.0
        cur.n, cur.processed, cur.t = n, ctx.processed, now
        aggregate += delivered

        out.append(f"{ctx.spec.stream_id:<7}{ctx.spec.task:<14}"
                   + "".join(f"{means[name]:>9.2f}" for name in STAGES)
                   + f"{latency:>9.2f}{model_fps:>9.1f}{delivered:>11.1f}"
                   f"{ctx.dropped:>9}{ctx.last_objs:>6}")

    out.append(" " * 52 + f"aggregate delivered {aggregate:.1f} fps")
    print("\n".join(out), flush=True)


# ── stream context ────────────────────────────────────────────────────────────
@dataclass
class StreamContext:
    spec: StreamSpec
    source_run: object
    # Endpoint name on the SHARED source Run (one Run serves every stream now).
    source_endpoint: str
    model_run: object
    video_run: object
    width: int
    height: int
    fps: int
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
# How many decoded frames the source appsink may hold for the source thread.
#
# DO NOT use OutputOptions.latest() here. Its meaning CHANGED between Neat 0.2.2 and
# 0.3.0 and the change is silent -- no app edit, no warning:
#   0.2.2:  latest() returned the struct defaults -> max_buffers=4, drop=False
#   0.3.0:  latest() -> max_buffers=1, drop=True
# every_frame(4) is byte-for-byte what 0.2.2's latest() produced. This app already
# sheds load at its own drop-oldest queues, which is where a live pipeline should drop.
SOURCE_OUTPUT_BUFFERS = 4


# Short, NOT multi-second.
#
# Every stream's source thread now pulls from the SAME Run (one combined source graph
# is what earns the decoder-admission lease). A long blocking named pull on a shared
# Run starves the other endpoints: with a 5000 ms timeout the four streams collapsed to
# ~1.6 fps aggregate and then to 0. A short timeout lets each thread take its turn.
#
# With a short timeout an empty return is NORMAL (that endpoint simply had nothing
# ready), not an error -- so callers must not warn on every miss.
SOURCE_PULL_TIMEOUT_MS = 20
# Consecutive empty pulls that add up to ~5 s before we call it a real stall.
SOURCE_STALL_MISSES = max(1, 5000 // SOURCE_PULL_TIMEOUT_MS)


def _pull_source_tensors(ctx, timeout_ms: int):
    """Pull THIS stream's frame from the shared source Run, by endpoint name.

    One Run now serves every stream, so the endpoint identifies which camera the frame
    came from; the unnamed pull_tensors() would hand back whichever stream arrived first.

    Use the NAMED pull_tensors overload rather than pull()+extract_tensors: the runtime
    does the extraction itself and handles every Sample kind the decoder can emit.
    Hand-rolling it with extract_tensors() silently returned [] for the decoder's sample
    kind, which this app then reported as "RTSP frame timeout" -- a real failure wearing
    the wrong label.
    """
    return ctx.source_run.pull_tensors(ctx.source_endpoint, timeout_ms)


def source_endpoint(stream_id: int) -> str:
    """Per-stream output endpoint on the ONE shared source Run."""
    return f"frame{stream_id}"


def make_rtsp_encoded_graph(cfg: Config, spec: StreamSpec):
    """RTSP receive + depay + parse only. The decode is a separate, explicit node.

    Returns the group's OWN graph, not a wrapper. Wrapping it in an extra
    `pyneat.Graph(...)` and `.add()`-ing the group adds a nesting level that makes two
    structurally identical source chains merge inside one combined Graph -- exactly the
    failure that left one stream with zero frames in multi-stream-yolo-yolo11. main.cpp
    returns `groups::RtspEncodedInput(enc)` directly for the same reason.
    """
    enc = pyneat.RtspEncodedInputOptions()
    enc.url = spec.rtsp_url
    enc.codec = pyneat.RtspCodec.H264
    enc.latency_ms = cfg.latency_ms
    enc.tcp = cfg.tcp
    enc.drop_on_latency = cfg.rtsp_drop_on_latency
    enc.insert_queue = True
    enc.h264_payload_type = 96
    # With skip_rtsp_probe the configured caps ARE the source contract: pin them on the
    # parser instead of probing each stream at startup.
    enc.auto_caps_from_stream = not cfg.skip_rtsp_probe
    if cfg.skip_rtsp_probe:
        enc.h264_width = cfg.fallback_width
        enc.h264_height = cfg.fallback_height
        enc.h264_fps = cfg.fallback_fps
    enc.fallback_h264_width = cfg.fallback_width
    enc.fallback_h264_height = cfg.fallback_height
    enc.fallback_h264_fps = cfg.fallback_fps
    return pyneat.groups.rtsp_encoded_input(enc)


def make_decoder_graph(cfg: Config, spec: StreamSpec):
    g = pyneat.Graph(f"qsqm_decoder_{spec.stream_id}")

    dec = pyneat.SimaDecodeOptions()
    dec.type = pyneat.SimaDecodeType.H264
    dec.sima_allocator_type = 2
    dec.out_format = pyneat.Format.NV12
    dec.raw_output = True
    dec.dec_width = cfg.fallback_width
    dec.dec_height = cfg.fallback_height
    dec.dec_fps = cfg.fallback_fps
    # Admission-lease inputs. Setting the equivalent GStreamer property on a lone
    # un-admitted decoder does nothing useful; the lease is what matters.
    dec.num_buffers = cfg.decoder_buffers
    dec.input_buffers = cfg.decoder_input_buffers
    dec.decoder_tuning = cfg.decoder_tuning
    g.add(pyneat.nodes.sima_decode(dec))

    g.add(pyneat.nodes.caps_raw("NV12", cfg.fallback_width, cfg.fallback_height,
                                cfg.fallback_fps, pyneat.CapsMemory.Any))
    # 4 slots, not 1: with a single slot the decoder has nowhere to put frame N+1 until
    # the source thread has taken frame N, so every scheduling gap costs a frame.
    #
    # BISECT: drop=True. every_frame() leaves drop=False, i.e. a NON-dropping appsink
    # that back-pressures the hardware decoder when the consumer falls behind. main.cpp
    # can afford that because it drains at 61 fps/stream and sheds at its own mailbox
    # (it dropped 238-418 frames/stream in a healthy 60 s run). This Python drains far
    # slower -- three ~1.4 MB copies per frame, much of it under the GIL -- so the sink
    # fills, the admitted decoder stalls, and every stream goes to 0 permanently.
    # Dropping here lets the decoder keep running and the app simply miss frames.
    src_out = pyneat.OutputOptions.every_frame(SOURCE_OUTPUT_BUFFERS)
    src_out.drop = True
    g.add(pyneat.nodes.output(source_endpoint(spec.stream_id), src_out))
    return g


def make_combined_source_graph(cfg: Config, specs):
    """ONE Graph holding every stream's `rtsp -> decode -> Output("frame<i>")` chain.

    This is what unlocks 60 fps per stream and it is not optional at four streams.

    Neat requests a coordinated plan from the decoder daemon (/tmp/dec-admission-v2.sock)
    ONLY when a single graph contains more than one H.264 decoder. One source graph per
    stream -- the old layout -- never asks, so each decoder picks an uncoordinated default
    without zero-copy output. Measured on this board with the C++ twin:

        4 separate un-admitted source Runs   dec 38.5 fps/stream   152 fps aggregate
        1 shared admitted source Run         dec 60.0 fps/stream   235 fps aggregate

    Verify with SIMA_DECODER_ADMISSION_DEBUG=1 -> "admission_accepted streams=4"; the
    admitted decoder also gains `zero-copy-output=true` in the backend dump.

    The branches are independent -- they share the graph, not any data path -- so each
    stream still pulls its own frames by its own endpoint name.
    """
    g = pyneat.Graph("qsqm_sources")
    for spec in specs:
        g.connect(make_rtsp_encoded_graph(cfg, spec), make_decoder_graph(cfg, spec))
    return g


def make_nv12_input_options(w: int, h: int, fps: int):
    o = pyneat.InputOptions()
    o.payload_type = pyneat.PayloadType.Image
    o.format = pyneat.Format.NV12
    o.width = w; o.height = h; o.depth = 1
    o.max_width = w; o.max_height = h; o.max_depth = 1
    o.fps_n = max(1, fps); o.fps_d = 1
    # NO caps_override and NO use_simaai_pool -- main.cpp sets neither, and both break
    # this app on Neat 0.3.0:
    #   caps_override omitted `depth`, and it WINS over the fields above, so the appsrc
    #     published caps with no depth and the CVU had nothing to size the ingress from.
    #   use_simaai_pool=False maps to InputMemoryPolicy.SystemMemory in 0.3.0, which
    #     makes neatprocesscvu reject the staged buffer:
    #     "Staged input 'input_image' raw GstMemory span overflow from 'sink_pad_0'"
    # Leaving both unset lets Neat build caps carrying depth=1 and pick a SiMa-visible
    # target; the CPU->EV74 copy is already permitted by
    # SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY (set in main).
    return o


def make_model(cfg: Config, spec: StreamSpec):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    # The input_max_* envelope is NOT set, matching main.cpp. Neat 0.3.0 enforces the
    # bound only when it is set explicitly, so setting it can only go wrong: 1 aborts
    # ("color_input_requires_input_shape_channels_3", because color_convert publishes
    # RGB) and 3 mis-sizes the ingress. Let the runtime infer it from the caps.
    # opt.preprocess.input_max_width = cfg.fallback_width
    # opt.preprocess.input_max_height = cfg.fallback_height
    # 3, not 1. This is the ingress CAPACITY bound (max channels the preproc stage will
    # accept), not a statement about the NV12 transport layout. Neat 0.3.0 started
    # enforcing it -- Preproc throws "input depth 3 exceeds max_input_depth 1" -- because
    # color_convert publishes RGB (3 channels) toward normalize/quant/tess. 0.2.2 accepted
    # the 1 silently. Enforcement only triggers when the bound is set EXPLICITLY, which is
    # why main.cpp (where these three lines are commented out) runs unmodified on 0.3.0.
    opt.preprocess.input_max_depth = 3
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = cfg.model_width
    opt.preprocess.resize.height = cfg.model_height
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    # Normalization is per-family: Ultralytics wants x/255, Megvii YOLOX wants raw
    # 0-255. See NORMALIZE_PRESET — getting this wrong silently zeroes the detections.
    opt.preprocess.preset = getattr(pyneat.NormalizePreset, NORMALIZE_PRESET[spec.task])
    # Pin the pre/post CVU stages when asked. Leaving these AUTO lets the planner
    # drop a raw-head model's detessellate+dequantize onto the A65, which is
    # ~180x slower than the EV74 for the pose head layout.
    opt.processcvu.pre_run_target = cfg.cvu_pre_target
    opt.processcvu.post_run_target = cfg.cvu_post_target
    # Every task decodes ON-DEVICE. Leaving decode_type Unspecified would make the
    # model emit raw per-scale heads and force a host (A65) NumPy decode — which is
    # exactly what this app used to do, and what made seg/pose/yolox slow.
    family = DECODE_FAMILY[spec.task]
    opt.decode_type = getattr(pyneat.BoxDecodeType, family)
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.top_k
    opt.num_classes = NUM_CLASSES[spec.task]
    # NOT setting boxdecode_original_width/height — deprecated in core/include/model/Model.h;
    # box decode reads the frame geometry from the preprocess metadata.
    return pyneat.Model(spec.model_path, opt)


# (build_source_graph removed. It built ONE rtsp_decoded_input graph -- and therefore
# one Run -- PER STREAM, the layout that never gets a decoder-admission lease.
# Replaced by make_combined_source_graph, matching main.cpp.)


def build_model_graph(cfg: Config, spec: StreamSpec, w: int, h: int, fps: int):
    g = pyneat.Graph(f"model_{spec.task}_{spec.stream_id}")
    g.add(pyneat.nodes.input(make_nv12_input_options(w, h, fps)))
    g.add(make_model(cfg, spec))
    # One endpoint for all four tasks: the model graph ends in an on-device BoxDecode
    # stage, so what comes out is always a decoded BBOX payload, never raw heads.
    # EveryFrame(max_buffers) = how many FINISHED results the sink may park for the
    # puller. MUST be >= queue_depth or the two settings contradict: the Run is told to
    # keep queue_depth frames in flight, and a 1-slot sink backs the graph up until
    # push() times out. main.cpp uses max(1, model_queue_depth) for the same reason.
    out_buffers = max(1, cfg.queue_depth)
    g.add(pyneat.nodes.output(MODEL_ENDPOINT, pyneat.OutputOptions.every_frame(out_buffers)))
    return g, MODEL_ENDPOINT


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
# COCO_SKELETON as two index arrays, built once. Indexing a NumPy array with a Python
# int returns a boxed np.bool_/np.float32 scalar, so the old `vis[a] and vis[b]` test
# inside a 19-iteration loop per detection cost ~250 boxed scalar reads per frame.
# With these, the whole edge selection is three vectorised ops.
_SKEL_A = None
_SKEL_B = None


def _skeleton_index():
    global _SKEL_A, _SKEL_B
    if _SKEL_A is None:
        pairs = np.asarray(ov.COCO_SKELETON, dtype=np.intp)
        _SKEL_A, _SKEL_B = pairs[:, 0], pairs[:, 1]
    return _SKEL_A, _SKEL_B


# Pixel offsets of one (2r+1)^2 dot, built once.
_DOT_OFF = None


def _draw_dots(y, pts, h, w, val=255, r=2):
    """Stamp a square dot at every point in `pts` with ONE indexed write.

    Drawing these as thick cv2.polylines segments was measured at 24.6 ms of the
    34.4 ms pose overlay (13 objects x 17 joints = 221 segments, each rendered as a
    filled polygon). Expanding the joints to their pixel offsets and doing a single
    fancy-index assignment turns that into one vectorised write of ~5.5k bytes. It
    holds the GIL, but for tens of microseconds rather than milliseconds.
    """
    global _DOT_OFF
    if _DOT_OFF is None:
        _DOT_OFF = np.mgrid[-r:r + 1, -r:r + 1].reshape(2, -1)
    yy = (pts[:, 1:2] + _DOT_OFF[0]).ravel()
    xx = (pts[:, 0:1] + _DOT_OFF[1]).ravel()
    ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    y[yy[ok], xx[ok]] = val


def _fill_rect(y, x1, y1, x2, y2, val):
    hh, ww = y.shape
    x1 = max(0, min(ww, x1)); x2 = max(0, min(ww, x2))
    y1 = max(0, min(hh, y1)); y2 = max(0, min(hh, y2))
    if x2 > x1 and y2 > y1:
        y[y1:y2, x1:x2] = val


def annotate(nv12, w, h, result, task, banner) -> int:
    """Draw on the NV12 Y plane with as few Python->OpenCV calls as possible.

    Call count is what costs, not pixels. Every _fill_rect is a NumPy slice write that
    HOLDS the GIL, and every cv2 call pays Python->C marshalling; with four overlay
    threads the GIL-held ones serialise across streams. Measured on the DevKit at 13
    objects/frame, 4 concurrent threads (the real configuration):

        detection      5.04 ms      (4 fill_rect + 1 putText per box)
        segmentation   9.13 ms      (+ resize + masked add per box)
        pose          64.51 ms      (+ 17 fill_rect + 19 cv2.line per box  == ~470 calls)

    So pose is the whole problem. The fix is batching: the skeleton segments and the
    keypoint dots are accumulated across ALL detections and drawn with ONE
    cv2.polylines each, and the box outline is one cv2.rectangle instead of four
    fill_rect. cv2.polylines/rectangle release the GIL; fill_rect does not.
    """
    y = nv12[:h, :]
    th = 3
    drawn = 0
    # Batched across every detection in this frame -> one cv2 call each at the end.
    skel_segs = []
    dot_pts = []
    for d in result.detections:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # ONE cv2.rectangle instead of four _fill_rect. cv2 clips out-of-frame
        # coordinates itself, so the manual clamping _fill_rect did is not needed.
        cv2.rectangle(y, (x1, y1), (x2 - 1, y2 - 1), 235, th, cv2.LINE_8)
        if task in ("detection", "segmentation", "yolox"):
            label = ov.class_label(d.class_id)
        else:
            label = "PERSON"
        ly = y1 - 6 if y1 >= 20 else min(h - 8, y1 + 18)
        # LINE_8, not LINE_AA. Antialiased glyphs cost ~2x for a label nobody reads
        # at subpixel precision (0.72 -> 0.39 ms per frame at 13 objects).
        cv2.putText(y, f"{label} {d.score:.2f}", (x1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 235, 1, cv2.LINE_8)
        if d.mask is not None:
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                m = cv2.resize(d.mask, (bw, bh), interpolation=cv2.INTER_NEAREST)
                region = y[y1:y2, x1:x2]
                if region.shape[:2] == m.shape:
                    # cv2.add(mask=) instead of `region[m] = clip(region[m] + 60)`.
                    # Two wins, and the second is the big one:
                    #   1. 27.8 -> 5.2 ms per frame (13 objects) — no fancy-index copies.
                    #   2. cv2 RELEASES THE GIL; NumPy fancy indexing does not. With four
                    #      overlay threads (one per stream) the NumPy path serialised all
                    #      four, so its cost showed up multiplied in wall-clock.
                    cv2.add(region, 60, dst=region, mask=m)
        if d.keypoints is not None:
            kp = d.keypoints
            # Vectorised visibility test: one NumPy pass instead of a per-keypoint
            # Python loop with float unpacking.
            vis = kp[:, 2] >= 0.3
            if vis.any():
                pts = kp[:, :2].astype(np.int32)
                # Joints are stamped in one indexed write at the end of the frame.
                dot_pts.append(pts[vis])
                # Whole skeleton selected at once: no Python loop over the 19 edges.
                ia, ib = _skeleton_index()
                ok = vis[ia] & vis[ib]
                if ok.any():
                    skel_segs.append(np.stack([pts[ia[ok]], pts[ib[ok]]], axis=1))
        drawn += 1
    # Two calls for every skeleton and every joint in the frame, however many there are.
    # One concatenate per frame, then a single (N,2,2) array straight into cv2 — no
    # per-segment Python list of arrays.
    if skel_segs:
        cv2.polylines(y, np.concatenate(skel_segs), False, 200, 1, cv2.LINE_8)
    if dot_pts:
        _draw_dots(y, np.concatenate(dot_pts), h, w)
    cv2.putText(y, banner, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 235, 2, cv2.LINE_8)
    return drawn


# ── per-stream service (preserves stream identity) ────────────────────────────
def _boxes_to_detections(arr, score_threshold, keep_index=False):
    """Neat hands back boxes as float32 [N, 6] = (x1, y1, x2, y2, score, class_id),
    already NMS'd, already clamped to frame pixels. All we do is drop sub-threshold
    rows and wrap them for the overlay."""
    out = []
    arr = np.asarray(arr, dtype=np.float32).reshape((-1, 6))
    for i, (x1, y1, x2, y2, sc, cid) in enumerate(arr):
        if sc < score_threshold:
            continue
        det = ov.Detection(float(x1), float(y1), float(x2), float(y2), float(sc), int(cid))
        out.append((i, det) if keep_index else det)
    return out


def decode_sample(cfg, ctx: StreamContext, tensors, fw: int, fh: int):
    """Read the decoded BBOX payload produced by the model graph's on-device
    BoxDecode stage. No anchor grids, no sigmoid, no NMS, no mask assembly here —
    the MLA/EV74 did all of it. This function only reshapes and thresholds.

    pyneat gives one reader per payload kind (core/python/src/module.cpp):
      decode_bbox         -> boxes [N, 6]
      decode_pose         -> boxes [N, 6] + keypoints [N, 17, 3]  (x, y, visibility)
      decode_segmentation -> boxes [N, 6] + masks     [N, 160, 160] uint8
    """
    task = ctx.spec.task
    result = ov.DecodeResult([])

    if task == "pose":
        for r in pyneat.decode_pose(tensors, clamp_to=(fw, fh), top_k=cfg.top_k):
            kpts = np.asarray(r.keypoints.to_numpy(copy=True), dtype=np.float32)
            kpts = kpts.reshape((-1, 17, 3))
            for i, det in _boxes_to_detections(r.boxes.to_numpy(copy=True),
                                               cfg.score_threshold, keep_index=True):
                if i < kpts.shape[0]:
                    det.keypoints = kpts[i]
                result.detections.append(det)
        return result

    if task == "segmentation":
        for r in pyneat.decode_segmentation(tensors, clamp_to=(fw, fh), top_k=cfg.top_k):
            masks = np.asarray(r.masks.to_numpy(copy=True), dtype=np.uint8)
            masks = masks.reshape((-1, 160, 160))
            for i, det in _boxes_to_detections(r.boxes.to_numpy(copy=True),
                                               cfg.score_threshold, keep_index=True):
                if i < masks.shape[0]:
                    # Neat returns a full 160x160 model-space mask per instance. The
                    # overlay wants the box crop, so map the box back into mask space.
                    det.mask = crop_mask_to_box(masks[i], det, fw, fh,
                                                cfg.model_width, cfg.model_height)
                result.detections.append(det)
        return result

    # detection + yolox: plain boxes.
    for t in pyneat.decode_bbox(tensors, clamp_to=(fw, fh), top_k=cfg.top_k):
        result.detections.extend(
            _boxes_to_detections(t.to_numpy(copy=True), cfg.score_threshold))
    return result


def crop_mask_to_box(mask, det, frame_w, frame_h, model_w, model_h):
    """Neat's segmentation mask is [160,160] in LETTERBOXED MODEL space (160 = 640/4).
    The box is already in frame pixels, so undo the letterbox to find the box's
    footprint in the mask and return just that crop."""
    scale = min(model_w / frame_w, model_h / frame_h)
    pad_x = (model_w - frame_w * scale) / 2.0
    pad_y = (model_h - frame_h * scale) / 2.0
    q = model_w / mask.shape[1]          # model px per mask px (640/160 = 4)
    x1 = int(np.clip((det.x1 * scale + pad_x) / q, 0, mask.shape[1] - 1))
    y1 = int(np.clip((det.y1 * scale + pad_y) / q, 0, mask.shape[0] - 1))
    x2 = int(np.clip((det.x2 * scale + pad_x) / q, 0, mask.shape[1]))
    y2 = int(np.clip((det.y2 * scale + pad_y) / q, 0, mask.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return None
    return mask[y1:y2, x1:x2]


def service_stream(cfg, ctx: StreamContext) -> bool:
    """Serial path: one thread runs every stage of this stream, timed per stage."""
    timings: dict = {}
    frame_start = time.perf_counter()
    endpoint = MODEL_ENDPOINT

    mark = time.perf_counter()
    # Same shared-Run fairness rule as the pipelined path: poll briefly, many times,
    # instead of one long blocking pull that would starve the other streams.
    frames = []
    for _ in range(SOURCE_STALL_MISSES):
        frames = _pull_source_tensors(ctx, SOURCE_PULL_TIMEOUT_MS)
        if frames:
            break
    if not frames:
        print(f"[warn] stream {ctx.spec.stream_id}: RTSP frame timeout", file=sys.stderr)
        return False
    nv12, fw, fh = tensor_nv12_from_decoded(frames[0])
    timings["decode"] = (time.perf_counter() - mark) * 1000.0

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
    timings["postproc"] = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    if result is not None:
        banner = f"S{ctx.spec.stream_id} {ctx.spec.task.upper()} :{ctx.spec.port}"
        ctx.last_objs = annotate(nv12, fw, fh, result, ctx.spec.task, banner)
    timings["overlay"] = (time.perf_counter() - mark) * 1000.0

    mark = time.perf_counter()
    if not ctx.video_run.push([make_nv12_tensor(nv12, fw, fh)]):
        raise RuntimeError("video push failed")
    timings["encode"] = (time.perf_counter() - mark) * 1000.0

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
        model_graph, _ = build_model_graph(cfg, s, w, h, fps)
        model_run = model_graph.build(build_run_options(cfg))
        _, video_run, port = build_video_graph(cfg, s, w, h, fps)
        contexts.append(StreamContext(
            spec=s, source_run=None, source_endpoint=source_endpoint(s.stream_id),
            model_run=model_run, video_run=video_run,
            width=w, height=h, fps=fps))
        print(f"Stream {s.stream_id}: {s.task:12s} {Path(s.model_path).name}")
        print(f"  RTSP {s.rtsp_url} -> udp://{cfg.udp_host}:{port}")
        print(f"  Viewer: gst-launch-1.0 -v udpsrc port={port} "
              f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" '
              f"! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false")

    # Start the RTSP sources LAST, after every model is loaded and immediately before
    # the threads that drain them. Graph.build() STARTS the pipeline: building sources
    # inside the loop above meant stream 0's camera streamed while streams 1-3 were
    # still loading their MLA archives (seconds each), nobody pulling, and its edge
    # queue filled -- dead on arrival. main.cpp carries the same rule.
    source_graph = make_combined_source_graph(cfg, specs)
    if cfg.print_backend:
        print(f"--- combined source graph ({len(specs)} decoders)\n"
              f"{source_graph.describe_backend()}")
    source_run = source_graph.build(build_run_options(cfg))
    for c in contexts:
        c.source_run = source_run

    warmup = min(cfg.warmup_frames, cfg.frames - 1) if cfg.frames > 0 else cfg.warmup_frames
    warmup = max(0, warmup)
    try:
        if cfg.serial:
            return run_serial(cfg, contexts, warmup)
        return run_pipelined(cfg, contexts, warmup)
    finally:
        for c in contexts:
            c.model_run.close(); c.video_run.close()
        source_run.close()   # shared by every stream, so closed once, last.


def run_serial(cfg, contexts: list, warmup: int) -> int:
    """Original single-threaded round-robin over all stream-model pairs.

    One thread runs every stage of every stream, so the four MLA models never
    overlap with each other's host decode/overlay work and the per-stream rate is
    1 / (num_streams * per-frame service time).
    """
    start = time.perf_counter()
    steady_start = None
    total = 0
    # Live profile: no per-frame heartbeat, just the table on an interval.
    cursors = [LiveCursor() for _ in contexts]
    for cur in cursors:
        cur.t = start
    next_profile = (start + cfg.profile_interval_s) if cfg.profile_interval_s > 0 else None
    try:
        while cfg.frames <= 0 or min(c.processed for c in contexts) < cfg.frames:
            for ctx in contexts:
                if cfg.frames > 0 and ctx.processed >= cfg.frames:
                    continue
                if service_stream(cfg, ctx):
                    total += 1
            if steady_start is None and min(c.processed for c in contexts) >= warmup:
                steady_start = time.perf_counter()
                for ctx in contexts:
                    ctx.steady_base = ctx.processed
            if next_profile is not None and time.perf_counter() >= next_profile:
                print_live_profile(contexts, cursors, start)
                next_profile = time.perf_counter() + cfg.profile_interval_s
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
        misses = 0
        try:
            while not stop.is_set():
                mark = time.perf_counter()
                frames = _pull_source_tensors(ctx, SOURCE_PULL_TIMEOUT_MS)
                if not frames:
                    if stop.is_set():
                        return
                    # Empty is normal at this timeout; only a long unbroken run of
                    # misses means the source has actually stalled.
                    misses += 1
                    if misses % SOURCE_STALL_MISSES == 0:
                        print(f"[warn] stream {ctx.spec.stream_id}: no RTSP frame for "
                              f"{misses * SOURCE_PULL_TIMEOUT_MS / 1000:.0f}s",
                              file=sys.stderr)
                    continue
                misses = 0
                nv12, fw, fh = tensor_nv12_from_decoded(frames[0])
                decode_ms = (time.perf_counter() - mark) * 1000.0

                mark = time.perf_counter()
                tensor = make_nv12_tensor(nv12, fw, fh)
                prep_ms = (time.perf_counter() - mark) * 1000.0

                ctx.dropped += _drop_oldest_put(
                    ctx.model_q, (nv12, tensor, fw, fh, decode_ms, prep_ms, time.perf_counter()))
        except Exception as exc:
            errors.append(f"source {ctx.spec.stream_id}: {exc}")
            stop.set()

    def model_thread(ctx: StreamContext) -> None:
        """Sole pusher AND sole puller of THIS stream's model Run, so the graph's
        FIFO ordering keeps the Nth sample matched to the Nth frame pushed."""
        endpoint = MODEL_ENDPOINT
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
                    nv12, tensor, fw, fh, decode_ms, prep_ms, t_in = item
                    push_mark = time.perf_counter()
                    if not ctx.model_run.push([tensor]):
                        print(f"[warn] stream {ctx.spec.stream_id}: model push failed",
                              file=sys.stderr)
                        continue
                    pending.append((nv12, fw, fh, decode_ms, prep_ms, t_in, push_mark))

                if pending and (len(pending) >= cfg.pipeline_depth or item is None):
                    nv12, fw, fh, decode_ms, prep_ms, t_in, push_mark = pending.pop(0)
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
                        ctx.out_q, (nv12, sample, fw, fh, decode_ms, prep_ms, infer_ms, t_in))
        except Exception as exc:
            errors.append(f"model {ctx.spec.stream_id}: {exc}")
            stop.set()

    def output_thread(ctx: StreamContext) -> None:
        try:
            while not stop.is_set():
                try:
                    nv12, sample, fw, fh, decode_ms, prep_ms, infer_ms, t_in = \
                        ctx.out_q.get(timeout=0.2)
                except queue_mod.Empty:
                    continue

                mark = time.perf_counter()
                result = (None if cfg.no_overlay
                          else decode_sample(cfg, ctx, extract_tensors(sample), fw, fh))
                postproc_ms = (time.perf_counter() - mark) * 1000.0

                mark = time.perf_counter()
                if result is not None:
                    banner = f"S{ctx.spec.stream_id} {ctx.spec.task.upper()} :{ctx.spec.port}"
                    ctx.last_objs = annotate(nv12, fw, fh, result, ctx.spec.task, banner)
                overlay_ms = (time.perf_counter() - mark) * 1000.0

                mark = time.perf_counter()
                if not ctx.video_run.push([make_nv12_tensor(nv12, fw, fh)]):
                    raise RuntimeError("video push failed")
                encode_ms = (time.perf_counter() - mark) * 1000.0

                # Keys MUST match STAGES — StageProfile.samples is keyed on it.
                ctx.profile.add(
                    {"decode": decode_ms, "prep": prep_ms, "infer": infer_ms,
                     "postproc": postproc_ms, "overlay": overlay_ms, "encode": encode_ms},
                    (time.perf_counter() - t_in) * 1000.0,
                )
                ctx.processed += 1
                note_steady()
                # No per-frame heartbeat: the reporter thread prints the live profile.
                if (cfg.duration_s <= 0 and cfg.frames > 0
                        and all(c.processed >= cfg.frames for c in contexts)):
                    stop.set()
        except Exception as exc:
            errors.append(f"output {ctx.spec.stream_id}: {exc}")
            stop.set()

    run_start = time.perf_counter()
    cursors = [LiveCursor() for _ in contexts]
    for cur in cursors:
        cur.t = run_start

    def reporter_thread() -> None:
        """Print the live time profile every profile_interval_s until the run stops."""
        if cfg.profile_interval_s <= 0:
            return
        # Event.wait() returns True the moment stop is set, so Ctrl-C and --duration are
        # honoured immediately instead of after a full interval of sleeping.
        while not stop.wait(cfg.profile_interval_s):
            print_live_profile(contexts, cursors, run_start)

    threads = []
    for ctx in contexts:
        threads.append(threading.Thread(target=source_thread, args=(ctx,),
                                        name=f"src{ctx.spec.stream_id}", daemon=True))
        threads.append(threading.Thread(target=model_thread, args=(ctx,),
                                        name=f"mdl{ctx.spec.stream_id}", daemon=True))
        threads.append(threading.Thread(target=output_thread, args=(ctx,),
                                        name=f"out{ctx.spec.stream_id}", daemon=True))
    threads.append(threading.Thread(target=reporter_thread, name="profile", daemon=True))
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
