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

Design provenance (every API traceable to /workspace/core):
  * three-graph shuttle (source / model / video) — apps/multi-stream-yolo-yolo11/main.py
  * NV12 RTSP source + video_sender groups — pyneat.groups (core/include/neat/node_groups.h)
  * ModelOptions preprocess presets + BoxDecodeType — core/include/model/Model.h,
    core/include/pipeline/BoxDecodeType.h
  * push/pull named endpoints + RunOptions(queue_depth/overflow/preset) — core/include/pipeline/Run.h
  * host task decode is required because the surgery archives expose RAW per-scale
    heads (Neat's built-in fused decode covers only the detection family) — see
    src/decoders.py and model-compilation/work/<model>/reports/SURGERY.md.
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
WORK = "/workspace/demo-neat/model-compilation/work"

# Default per-task compiled archives (referenced in place; not copied/committed).
DEFAULT_ARCHIVES = {
    "detection": f"{WORK}/yolo11s/compile_int8/yolo11s.compile_ready/yolo11s.compile_ready_mpk.tar.gz",
    "segmentation": f"{WORK}/yolo11s-seg/compile_int8/yolo11s-seg.compile_ready/yolo11s-seg.compile_ready_mpk.tar.gz",
    "pose": f"{WORK}/yolo26s-pose/compile_int8/yolo26s-pose.compile_ready/yolo26s-pose.compile_ready_mpk.tar.gz",
    "yolox": f"{WORK}/yolox_s/compile_int8/yolox_s.compile_ready/yolox_s.compile_ready_mpk.tar.gz",
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
    rtsp_default: str = "rtsp://192.168.132.129:8555/stream"
    udp_host: str = "192.168.132.129"
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
        "num_streams": ("num_streams", int),
    }
    if key in simple:
        attr, cast = simple[key]
        setattr(cfg, attr, cast(value))
    elif key == "rtsp_transport":
        cfg.tcp = value.strip().lower() == "tcp"
    elif key == "print_backend":
        cfg.print_backend = parse_bool(value)
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
    ap.add_argument("--udp-host")
    ap.add_argument("--udp-port-base", type=int)
    ap.add_argument("--score", type=float)
    ap.add_argument("--nms", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--queue-depth", type=int)
    ap.add_argument("--frames", type=int, help="frames PER stream; 0 = forever")
    ap.add_argument("--rtsp-udp", action="store_true")
    ap.add_argument("--print-backend", action="store_true")
    a = ap.parse_args(argv)

    cfg = Config()
    load_config_file(cfg, a.config)
    if a.rtsp is not None:
        cfg.rtsp_default = a.rtsp
        cfg._rtsp = {}
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
    if a.rtsp_udp:
        cfg.tcp = False
    if a.print_backend:
        cfg.print_backend = True
    return cfg


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


def service_stream(cfg, ctx: StreamContext) -> bool:
    frames = ctx.source_run.pull_tensors(timeout_ms=20000)
    if not frames:
        print(f"[warn] stream {ctx.spec.stream_id}: RTSP frame timeout", file=sys.stderr)
        return False
    nv12, fw, fh = tensor_nv12_from_decoded(frames[0])
    if not ctx.model_run.push([make_nv12_tensor(nv12, fw, fh)]):
        print(f"[warn] stream {ctx.spec.stream_id}: model push failed", file=sys.stderr)
        return False
    endpoint = "detections" if ctx.is_builtin_decode else "heads"
    sample = ctx.model_run.pull(endpoint, 20000)
    if sample is None:
        print(f"[warn] stream {ctx.spec.stream_id}: model output timeout", file=sys.stderr)
        return False
    tensors = extract_tensors(sample)

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
    else:
        if os.environ.get("QSQM_DEBUG") and ctx.processed == 0:
            shapes = []
            for t in tensors:
                a = t.to_numpy(copy=True) if hasattr(t, "to_numpy") else np.asarray(t)
                shapes.append(tuple(a.shape))
            print(f"[dbg] stream {ctx.spec.stream_id} {ctx.spec.task}: "
                  f"{len(tensors)} raw tensors shapes={shapes}", file=sys.stderr, flush=True)
        result = decode_for_task(cfg, ctx.spec.task, tensors, fw, fh)

    banner = f"S{ctx.spec.stream_id} {ctx.spec.task.upper()} :{ctx.spec.port}"
    ctx.last_objs = annotate(nv12, fw, fh, result, ctx.spec.task, banner)
    if not ctx.video_run.push([make_nv12_tensor(nv12, fw, fh)]):
        raise RuntimeError("video push failed")
    ctx.processed += 1
    return True


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

    start = time.perf_counter()
    per = {c.spec.stream_id: {"t": 0.0, "n": 0} for c in contexts}
    total = 0
    try:
        while cfg.frames <= 0 or min(c.processed for c in contexts) < cfg.frames:
            for ctx in contexts:
                if cfg.frames > 0 and ctx.processed >= cfg.frames:
                    continue
                t0 = time.perf_counter()
                if service_stream(cfg, ctx):
                    dt = time.perf_counter() - t0
                    per[ctx.spec.stream_id]["t"] += dt
                    per[ctx.spec.stream_id]["n"] += 1
                    total += 1
                    if ctx.processed == 1 or ctx.processed % 20 == 0 or ctx.processed == cfg.frames:
                        el = time.perf_counter() - start
                        s_fps = per[ctx.spec.stream_id]["n"] / per[ctx.spec.stream_id]["t"] if per[ctx.spec.stream_id]["t"] else 0
                        print(f"stream={ctx.spec.stream_id} task={ctx.spec.task} "
                              f"frame={ctx.processed} objs={ctx.last_objs} "
                              f"stream_fps={s_fps:.2f} agg_fps={total/el:.2f}", flush=True)
            time.sleep(0)
    finally:
        el = max(1e-6, time.perf_counter() - start)
        print("\n=== per-stream summary ===")
        for c in contexts:
            p = per[c.spec.stream_id]
            fps = p["n"] / p["t"] if p["t"] else 0.0
            print(f"stream {c.spec.stream_id} {c.spec.task:12s} frames={c.processed} "
                  f"stream_fps={fps:.2f} (service-time based)")
        print(f"aggregate: {total} frames in {el:.1f}s = {total/el:.2f} agg_fps "
              f"across {len(contexts)} streams")
        for c in contexts:
            c.model_run.close(); c.source_run.close(); c.video_run.close()
    return total


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
