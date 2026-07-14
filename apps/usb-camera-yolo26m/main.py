#!/usr/bin/env python3
"""USB camera (UVC/MJPEG) -> YOLO26m object detection -> H.264 RTP/UDP.

Python port of main.cpp. Two pipeline modes, selected by `pipeline_mode` in the config.

── pipeline_mode=push (default, works on runtime 0.2.2) ──────────────────────

    source graph:  custom(v4l2src ! jpegparse ! jpegdec ! videoconvert ! NV12) -> output("frame")
    model  graph:  input("image") -> Model(yolo26m) -> output("detections")
    udp    graph:  input("video") -> video_sender (H264EncodeSima -> RTP -> UDP)

    The app pulls each NV12 frame, pushes it into the model graph (appsrc), pulls the
    boxes, draws them onto the frame, and pushes the annotated frame to the encoder.
    Pushing through appsrc is what lands the frame in SiMa DMA memory, which the CVU
    requires -- see SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY below.

── pipeline_mode=graph (zero-copy, needs a newer runtime) ────────────────────

    custom(camera) -> branch -+-> video_sender
                              +-> Model -> output("detections")

    Strictly better on paper: the frame never touches the CPU. But on runtime 0.2.2 the
    CVU silently reads system-memory buffers as black frames and yields zero detections,
    because the private `neatcamerabridge` element that lands OS buffers into SiMa DMA
    memory does not exist in libsima_neat.so.2.1.2. Set camera_bridge=true once you are
    on a runtime that ships it. See LEARNING.md.

MJPEG is decoded on the CPU (jpegdec), NOT on the SiMa hardware decoder. That is
deliberate and measured: neatdecoder in mjpeg mode runs at ~4 fps on this camera's
JPEGs, while jpegdec sustains the camera's full 30 fps. See LEARNING.md.

Run:
    source ~/pyneat/bin/activate
    python3 main.py [config/default.conf]
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Lets Neat copy a CPU-resident appsrc buffer into EV74/SiMa memory for the CVU.
# Must be set before pyneat initializes its GStreamer runtime.
os.environ.setdefault("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1")

import pyneat  # noqa: E402

DEFAULT_CONFIG = "./config/default.conf"

_stop = False


def _handle_signal(_signum, _frame) -> None:
    global _stop
    _stop = True


COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class StageStat:
    """Per-stage timing.

    Two different means, on purpose:

      * `window_mean()` covers only the current reporting window. A cumulative mean
        silently hides a pipeline that degrades halfway through a run -- it just drifts
        slowly, and by frame 6000 one bad minute is invisible.
      * `mean()` / `p95()` cover the whole run, for the exit summary. p95 is what tells
        you a stage is *occasionally* slow, which a mean never will.

    Memory is O(1): the mean comes from a running sum, p95 from a bounded ring of recent
    samples. A `frames=0` run streams for hours; an unbounded list would grow without end.
    """

    RING = 20000

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0
        self._win_sum = 0.0
        self._win_count = 0
        self._ring: list[float] = []
        self._pos = 0

    def add(self, ms: float) -> None:
        self._sum += ms
        self._count += 1
        self._win_sum += ms
        self._win_count += 1
        if len(self._ring) < self.RING:
            self._ring.append(ms)
        else:
            self._ring[self._pos] = ms
            self._pos = (self._pos + 1) % self.RING

    def window_mean(self) -> float:
        return self._win_sum / self._win_count if self._win_count else 0.0

    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    def p95(self) -> float:
        if not self._ring:
            return 0.0
        ordered = sorted(self._ring)
        return ordered[int(0.95 * (len(ordered) - 1))]

    def reset_window(self) -> None:
        self._win_sum = 0.0
        self._win_count = 0


class Profile:
    """capture = wait for camera frame · infer = CVU+MLA+boxdecode · overlay = CPU draw ·
    encode = push into the H.264 graph · total = the whole per-frame loop."""

    def __init__(self) -> None:
        self.capture = StageStat()
        self.infer = StageStat()
        self.overlay = StageStat()
        self.encode = StageStat()
        self.total = StageStat()

    def reset_windows(self) -> None:
        for s in (self.capture, self.infer, self.overlay, self.encode, self.total):
            s.reset_window()


def print_profile_summary(prof: Profile, frames: int, elapsed_s: float) -> None:
    if frames < 2 or elapsed_s <= 0:
        return

    fps = (frames - 1) / elapsed_s

    print("\n── time profile ──────────────────────────────")
    print(f"{'stage':<10}{'mean ms':>10}{'p95 ms':>10}")
    for name, stat in (
        ("capture", prof.capture),
        ("infer", prof.infer),
        ("overlay", prof.overlay),
        ("encode", prof.encode),
        ("total", prof.total),
    ):
        print(f"{name:<10}{stat.mean():>10.2f}{stat.p95():>10.2f}")

    print(f"\nframes {frames}   elapsed {elapsed_s:.1f}s   steady-state {fps:.2f} fps")

    # Say plainly what is holding the pipeline back. `infer` is the only stage on the
    # accelerator, so its mean is the MLA's ceiling. If we deliver well under that,
    # something upstream (the camera) is the constraint -- not the SoC.
    infer_ms = prof.infer.mean()
    if infer_ms > 0:
        ceiling = 1000.0 / infer_ms
        if ceiling > fps * 1.05:
            print(
                f"bottleneck: THE CAMERA. Inference takes {infer_ms:.1f} ms, so the MLA "
                f"could sustain ~{ceiling:.1f} fps; you are getting {fps:.1f}. "
                f"A smaller/faster model will not help."
            )
        else:
            print(
                f"bottleneck: INFERENCE. The MLA tops out near {ceiling:.1f} fps and you "
                f"are delivering {fps:.1f}. A smaller model would raise this."
            )


@dataclass
class Config:
    camera_device: str = "/dev/video16"
    width: int = 1920
    height: int = 1080
    fps: int = 30

    model_path: str = "./assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz"
    model_width: int = 640
    model_height: int = 640
    score_threshold: float = 0.30
    nms_iou: float = 0.50
    top_k: int = 100
    num_classes: int = 80

    udp_host: str = ""
    udp_port: int = 5205
    bitrate_kbps: int = 4000
    metadata_host: str = ""
    metadata_port: int = 9100

    pipeline_mode: str = "push"   # push | graph
    flip: str = "none"            # none | rotate-180 | horizontal-flip | vertical-flip
    overlay: bool = True
    frames: int = 0
    profile_interval: float = 1.0  # seconds between profile lines; 0 = off
    queue_depth: int = 3
    print_backend: bool = False
    verbose_planner: bool = False  # dump the MPK contract + route/fusion decisions
    camera_bridge: bool = False
    source_override: str = ""


_INT_KEYS = {
    "width", "height", "fps", "model_width", "model_height", "top_k", "num_classes",
    "udp_port", "bitrate_kbps", "metadata_port", "frames", "queue_depth",
}
_FLOAT_KEYS = {"score_threshold", "nms_iou", "profile_interval"}
_BOOL_KEYS = {"print_backend", "verbose_planner", "camera_bridge", "overlay"}


def read_config(path: str) -> Config:
    cfg = Config()
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if not hasattr(cfg, key):
            raise ValueError(f"unknown config key: {key}")
        if key in _INT_KEYS:
            setattr(cfg, key, int(value))
        elif key in _FLOAT_KEYS:
            setattr(cfg, key, float(value))
        elif key in _BOOL_KEYS:
            setattr(cfg, key, value in ("true", "1", "yes"))
        else:
            setattr(cfg, key, value)
    if not cfg.udp_host:
        raise ValueError("config missing: udp_host")
    if cfg.pipeline_mode not in ("push", "graph"):
        raise ValueError("pipeline_mode must be push or graph")
    return cfg


def camera_fragment(cfg: Config) -> str:
    """GStreamer fragment for the USB camera source.

    Neat has no V4L2 source node, so this goes through the custom() escape hatch.

    io-mode=mmap      zero-copy DMA from the UVC driver (io-mode=rw memcpys every frame)
    image/jpeg caps   pins MJPEG. Without it v4l2src negotiates YUYV, which the Brio 100
                      only offers at 5 fps for 1080p (USB 2.0 bandwidth limit).
    queue leaky       drop stale frames rather than stall the camera when the MLA is busy
    jpegdec           CPU MJPEG decode -- faster than the SiMa HW decoder here
    videoconvert      I420 (jpegdec's native output) -> NV12 for the CVU and the encoder

    The fragment must not end on a bare caps string: gst_parse_launch parses a trailing
    "video/x-raw,..." as an element name and fails with `no element "video"`. Terminating
    on a real element keeps the caps a capsfilter.
    """
    if cfg.source_override:
        return cfg.source_override

    frag = (
        f"v4l2src device={cfg.camera_device} io-mode=mmap"
        f" ! image/jpeg,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1"
        f" ! queue leaky=downstream max-size-buffers=2"
        f" ! jpegparse"
        f" ! jpegdec"
    )
    # COCO models lose confidence on inverted scenes; correct an upside-down mount here.
    if cfg.flip != "none":
        frag += f" ! videoflip method={cfg.flip}"
    frag += (
        f" ! videoconvert n-threads=4"
        f" ! video/x-raw,format=NV12,width={cfg.width},height={cfg.height}"
        f",framerate={cfg.fps}/1"
    )
    if cfg.camera_bridge:
        frag += " ! neatcamerabridge buffer-name=camera num-buffers=4 copy-allowed=true"
    return frag + " ! queue leaky=downstream max-size-buffers=2"


def make_model(cfg: Config) -> pyneat.Model:
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.input_max_width = cfg.width
    opt.preprocess.input_max_height = cfg.height
    opt.preprocess.input_max_depth = 1
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = cfg.model_width
    opt.preprocess.resize.height = cfg.model_height
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO

    # YOLO26 uses NMS-free raw l/t/r/b distance heads. Neat decodes these on the EV74.
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.top_k
    opt.num_classes = cfg.num_classes
    # Dumps the MPK contract and the planner's route decisions -- how it maps the packaged
    # stages onto CVU/MLA nodes, and what it fuses into boxdecode. This is what shows, e.g.,
    # `post_fusion=user_boxdecode(cast+detess+dequant)->boxdecode` for the int8 package.
    if cfg.verbose_planner:
        opt.verbose.level = pyneat.VerbosityLevel.Verbose
        opt.verbose.planner = True
    return pyneat.Model(cfg.model_path, opt)


def make_nv12_input_options(cfg: Config) -> pyneat.InputOptions:
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Image
    opt.format = pyneat.Format.NV12
    opt.width = cfg.width
    opt.height = cfg.height
    opt.depth = 1
    opt.max_width = cfg.width
    opt.max_height = cfg.height
    opt.max_depth = 1
    opt.fps_n = cfg.fps
    opt.fps_d = 1
    opt.caps_override = (
        f"video/x-raw,format=NV12,width={cfg.width},height={cfg.height}"
        f",framerate={cfg.fps}/1"
    )
    opt.use_simaai_pool = False
    return opt


def make_video_options(cfg: Config) -> pyneat.VideoSenderOptions:
    opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(cfg.width, cfg.height, cfg.fps)
    opt.host = cfg.udp_host
    opt.channel = 0
    opt.video_port_base = cfg.udp_port
    opt.encoder.bitrate_kbps = cfg.bitrate_kbps
    return opt


def make_run_options(cfg: Config) -> pyneat.RunOptions:
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Realtime
    opt.queue_depth = cfg.queue_depth
    opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    opt.output_memory = pyneat.OutputMemory.ZeroCopy
    return opt


def class_label(class_id: int) -> str:
    if 0 <= class_id < len(COCO_LABELS):
        return COCO_LABELS[class_id]
    return f"class_{class_id}"


def extract_tensors(sample) -> list:
    if sample is None:
        return []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    if sample.kind == pyneat.SampleKind.Bundle:
        out = []
        for field in sample.fields:
            out.extend(extract_tensors(field))
        return out
    return []


def decode_boxes(sample, cfg: Config) -> list[dict]:
    tensors = extract_tensors(sample)
    if not tensors:
        return []
    decoded = pyneat.decode_bbox(tensors, clamp_to=(cfg.width, cfg.height), top_k=cfg.top_k)
    boxes: list[dict] = []
    for tensor in decoded:
        arr = np.asarray(tensor.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
        for x1, y1, x2, y2, score, class_id in arr:
            if float(score) < cfg.score_threshold:
                continue
            cid = int(class_id)
            boxes.append(
                {
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "score": round(float(score), 3),
                    "class_id": cid,
                    "label": class_label(cid),
                }
            )
    return boxes[: cfg.top_k]


# A 5x7 bitmap font, drawn straight onto the NV12 planes. There is no OpenCV in this app
# (and no font on the board), so the glyphs are hand-coded -- the same approach the other
# single-stream apps use. Mirrors glyph_for() in main.cpp exactly.
_GLYPHS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10011", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("10010", "10010", "10010", "11111", "00010", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01111", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "11110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}
_GLYPH_UNKNOWN = ("11111", "00001", "00010", "00100", "00100", "00000", "00100")  # '?'


def _fill_nv12_rect(y, uv, width, height, x1, y1, x2, y2, y_val, u_val, v_val) -> None:
    """Fill a rectangle on both NV12 planes. Coordinates are clamped."""
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return

    y[y1:y2, x1:x2] = y_val

    # UV is half-height, full-width, with U and V interleaved per 2x2 block, so the
    # column span must be snapped to even boundaries or U and V get swapped.
    uy1, uy2 = y1 // 2, (y2 + 1) // 2
    ux1, ux2 = x1 & ~1, (x2 + 1) & ~1
    uy2 = min(uy2, height // 2)
    ux2 = min(ux2, width)
    if uy2 <= uy1 or ux2 <= ux1:
        return
    uv[uy1:uy2, ux1:ux2:2] = u_val
    uv[uy1:uy2, ux1 + 1 : ux2 : 2] = v_val


def _draw_nv12_text(y, uv, width, height, x, y0, text, y_val, u_val, v_val, scale) -> None:
    cx = x
    for raw in text:
        glyph = _GLYPHS.get(raw.upper(), _GLYPH_UNKNOWN)
        for row in range(7):
            for col in range(5):
                if glyph[row][col] == "1":
                    _fill_nv12_rect(
                        y, uv, width, height,
                        cx + col * scale, y0 + row * scale,
                        cx + (col + 1) * scale, y0 + (row + 1) * scale,
                        y_val, u_val, v_val,
                    )
        cx += 6 * scale  # 5px glyph + 1px gap
        if cx >= width - 6 * scale:
            break


def draw_boxes_on_nv12(buf: np.ndarray, width: int, height: int, boxes: list[dict]) -> None:
    """Draw red boxes with class-name labels onto the NV12 buffer (Y + interleaved UV)."""
    y_size = width * height
    y = buf[:y_size].reshape((height, width))
    uv = buf[y_size:].reshape((height // 2, width))

    box_y, box_u, box_v = 76, 84, 255      # red   — box + label bar
    txt_y, txt_u, txt_v = 235, 128, 128    # white — label text
    t = 3

    # 5x7 glyphs are unreadable at 1080p unscaled. Scale with the frame: 3 at 1080p.
    scale = max(2, height // 360)
    glyph_h, glyph_w = 7 * scale, 6 * scale
    pad = scale

    for b in boxes:
        x1 = max(0, min(width - 1, b["x1"]))
        y1 = max(0, min(height - 1, b["y1"]))
        x2 = max(0, min(width - 1, b["x2"]))
        y2 = max(0, min(height - 1, b["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue

        _fill_nv12_rect(y, uv, width, height, x1, y1, x2 + 1, y1 + t, box_y, box_u, box_v)
        _fill_nv12_rect(y, uv, width, height, x1, y2 - t + 1, x2 + 1, y2 + 1, box_y, box_u, box_v)
        _fill_nv12_rect(y, uv, width, height, x1, y1, x1 + t, y2 + 1, box_y, box_u, box_v)
        _fill_nv12_rect(y, uv, width, height, x2 - t + 1, y1, x2 + 1, y2 + 1, box_y, box_u, box_v)

        # "PERSON 0.93"
        label = f"{b['label']} {b['score']:.2f}"
        bar_w = min(len(label) * glyph_w + 2 * pad, width - x1)
        bar_h = glyph_h + 2 * pad

        # Above the box by default; if there is no room up there, drop it just inside the
        # top edge so a detection touching the top of the frame still gets a label.
        bar_y = y1 - bar_h if y1 - bar_h >= 0 else y1 + t

        # Filled bar first: white glyphs on a bright background would be invisible.
        _fill_nv12_rect(y, uv, width, height, x1, bar_y, x1 + bar_w, bar_y + bar_h,
                        box_y, box_u, box_v)
        _draw_nv12_text(y, uv, width, height, x1 + pad, bar_y + pad, label,
                        txt_y, txt_u, txt_v, scale)


def nv12_tensor_from_numpy(buf: np.ndarray, width: int, height: int) -> pyneat.Tensor:
    """Wrap a flat NV12 byte buffer as a Tensor.

    NV12 is semi-planar, so the Y and UV planes must be described explicitly -- a bare
    from_numpy() gives the encoder a shapeless buffer it cannot interpret.
    """
    tensor = pyneat.Tensor.from_numpy(
        np.ascontiguousarray(buf.reshape((height * 3 // 2, width))),
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


def summarize(boxes: list[dict]) -> str:
    return " ".join(f"{b['label']}({b['score']:.2f})" for b in boxes[:5])


def print_banner(cfg: Config, has_metadata: bool) -> None:
    overlay = " (with overlay)" if cfg.pipeline_mode == "push" and cfg.overlay else ""
    print(f"Mode:    {cfg.pipeline_mode}{overlay}")
    print(f"Camera:  {cfg.camera_device} MJPEG {cfg.width}x{cfg.height}@{cfg.fps}")
    print(f"Model:   {cfg.model_path} (YOLO26, {cfg.model_width}x{cfg.model_height})")
    print(f"Video:   udp://{cfg.udp_host}:{cfg.udp_port} H264/RTP payload=96")
    if has_metadata:
        print(f"Metadata: udp://{cfg.metadata_host}:{cfg.metadata_port}")
    print(
        "\nView the stream with:\n"
        f"  gst-launch-1.0 -v udpsrc port={cfg.udp_port} "
        'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" '
        "! rtpjitterbuffer ! rtph264depay ! h264parse ! avdec_h264 "
        "! videoconvert ! autovideosink sync=false\n"
    )
    print("Running. Press Ctrl-C to stop.")


def make_metadata_sender(cfg: Config):
    if not cfg.metadata_host:
        return None
    try:
        opt = pyneat.MetadataSenderOptions()
        opt.host = cfg.metadata_host
        opt.channel = 0
        opt.metadata_port_base = cfg.metadata_port
        return pyneat.MetadataSender(opt)
    except RuntimeError as exc:
        print(f"[warn] metadata sender failed to init ({exc}); continuing without it")
        return None


def run_push(cfg: Config) -> int:
    source_graph = pyneat.Graph("usb_camera_source")
    source_graph.add(pyneat.nodes.custom(camera_fragment(cfg), pyneat.InputRole.Source))
    source_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.latest()))

    model_graph = pyneat.Graph("usb_camera_model")
    model_graph.add(pyneat.nodes.input("image", make_nv12_input_options(cfg)))
    model_graph.add(make_model(cfg))
    model_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    udp_graph = pyneat.Graph("usb_camera_udp")
    udp_graph.add(pyneat.nodes.input("video", make_nv12_input_options(cfg)))
    udp_graph.add(pyneat.groups.video_sender(make_video_options(cfg)))

    if cfg.print_backend:
        print("Source backend:\n" + source_graph.describe_backend())
        print("Model backend:\n" + model_graph.describe_backend())
        print("UDP backend:\n" + udp_graph.describe_backend())

    run_opt = make_run_options(cfg)
    source_run = source_graph.build(run_opt)
    model_run = model_graph.build(run_opt)

    # The encoder graph owns its buffers; seed it so appsrc caps are fixed at build time.
    udp_opt = make_run_options(cfg)
    udp_opt.output_memory = pyneat.OutputMemory.Owned
    seed = np.zeros(cfg.width * cfg.height * 3 // 2, dtype=np.uint8)
    udp_run = udp_graph.build(
        [nv12_tensor_from_numpy(seed, cfg.width, cfg.height)], udp_opt
    )

    metadata = make_metadata_sender(cfg)
    print_banner(cfg, metadata is not None)

    processed = 0
    prof = Profile()
    steady_start = time.monotonic()
    last_log = steady_start
    last_log_frames = 0

    while not _stop and (cfg.frames == 0 or processed < cfg.frames):
        # pyneat's pull() returns Optional[Sample]: None on timeout or closed stream.
        t0 = time.perf_counter()
        frame_sample = source_run.pull("frame", 20000)
        t1 = time.perf_counter()
        if frame_sample is None:
            print("[warn] no camera frame (timeout or source closed)", file=sys.stderr)
            continue

        frame_tensors = extract_tensors(frame_sample)
        if not frame_tensors:
            print("[warn] camera sample has no tensors", file=sys.stderr)
            continue
        frame = frame_tensors[0]

        # Pushing through appsrc is what lands the frame in SiMa DMA memory for the CVU.
        t2 = time.perf_counter()
        if not model_run.push("image", [frame]):
            print("[warn] failed to push frame to model", file=sys.stderr)
            continue

        det_sample = model_run.pull("detections", 20000)
        if det_sample is None:
            print("[warn] no detections (timeout or pipeline closed)", file=sys.stderr)
            continue

        boxes = decode_boxes(det_sample, cfg)
        t3 = time.perf_counter()

        # to_numpy() rejects the camera's NV12 tensor ("__dlpack__ only supports dense
        # tensors") because it is semi-planar. copy_payload_bytes() hands back the raw
        # Y+UV bytes, which is what we want anyway.
        t4 = time.perf_counter()
        buf = np.frombuffer(frame.copy_payload_bytes(), dtype=np.uint8).copy()
        expected = cfg.width * cfg.height * 3 // 2
        if buf.size != expected:
            print(
                f"[warn] unexpected NV12 payload {buf.size} != {expected}", file=sys.stderr
            )
            continue

        if cfg.overlay:
            draw_boxes_on_nv12(buf, cfg.width, cfg.height, boxes)
        t5 = time.perf_counter()

        udp_run.push("video", [nv12_tensor_from_numpy(buf, cfg.width, cfg.height)])
        t6 = time.perf_counter()

        if metadata:
            metadata.send_metadata(
                "detection",
                json.dumps({"boxes": boxes}),
                int(time.time() * 1000),
                str(det_sample.frame_id),
            )

        processed += 1
        if processed == 1:
            steady_start = time.monotonic()
            last_log = steady_start
            last_log_frames = 1

        prof.capture.add((t1 - t0) * 1000.0)
        prof.infer.add((t3 - t2) * 1000.0)
        prof.overlay.add((t5 - t4) * 1000.0)
        prof.encode.add((t6 - t5) * 1000.0)
        prof.total.add((t6 - t0) * 1000.0)

        now = time.monotonic()
        since_log = now - last_log
        if cfg.profile_interval > 0 and since_log >= cfg.profile_interval:
            # fps over THIS window, not since start -- so a stall shows up immediately
            # instead of being averaged away across thousands of earlier good frames.
            win_fps = (processed - last_log_frames) / since_log if since_log > 0 else 0.0
            print(
                f"frame={processed} fps={win_fps:.1f} boxes={len(boxes)} "
                f"ms(capture={prof.capture.window_mean():.1f} "
                f"infer={prof.infer.window_mean():.1f} "
                f"overlay={prof.overlay.window_mean():.1f} "
                f"encode={prof.encode.window_mean():.1f} "
                f"total={prof.total.window_mean():.1f}) {summarize(boxes)}"
            )
            prof.reset_windows()
            last_log = now
            last_log_frames = processed

    print_profile_summary(prof, processed, time.monotonic() - steady_start)

    udp_run.close()
    model_run.close()
    source_run.close()
    return 130 if _stop else 0


def run_graph(cfg: Config) -> int:
    source = pyneat.nodes.custom(camera_fragment(cfg), pyneat.InputRole.Source)
    branch = pyneat.graphs.branch("camera", ["video", "model"])

    video_graph = pyneat.Graph("video")
    video_graph.connect(
        pyneat.nodes.input("video"), pyneat.groups.video_sender(make_video_options(cfg))
    )

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), make_model(cfg))

    detections_graph = pyneat.Graph("detections")
    detections_graph.add(
        pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4))
    )

    # RealtimeLatestByStream: if one branch falls behind, drop its stale frames rather
    # than back-pressuring the camera. The video branch must never stall the MLA.
    live = pyneat.GraphLinkOptions()
    live.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

    # connect() registers the source; calling add(source) too would emit the camera
    # fragment twice and start two v4l2src elements.
    graph = pyneat.Graph("usb_camera_yolo26m_python")
    graph.connect(source, branch)
    graph.connect(branch, video_graph, live)
    graph.connect(branch, model_graph, live)
    graph.connect(model_graph, detections_graph)

    if cfg.print_backend:
        print("Backend:\n" + graph.describe_backend())

    run = graph.build(make_run_options(cfg))
    metadata = make_metadata_sender(cfg)
    print_banner(cfg, metadata is not None)

    processed = 0
    prof = Profile()
    steady_start = time.monotonic()
    last_log = steady_start
    last_log_frames = 0

    while not _stop and (cfg.frames == 0 or processed < cfg.frames):
        # In graph mode the whole camera -> CVU -> MLA -> boxdecode chain runs inside the
        # pipeline, so this single pull IS the pipeline. There is no separate capture or
        # encode stage to time here: `infer` is the wait for the next result, and
        # `overlay` is the CPU box decode.
        t0 = time.perf_counter()
        sample = run.pull("detections", 20000)
        t1 = time.perf_counter()
        if sample is None:
            print("[warn] no detections (timeout or pipeline closed)", file=sys.stderr)
            continue

        boxes = decode_boxes(sample, cfg)
        t2 = time.perf_counter()

        if metadata:
            metadata.send_metadata(
                "detection",
                json.dumps({"boxes": boxes}),
                int(time.time() * 1000),
                str(sample.frame_id),
            )

        processed += 1
        if processed == 1:
            steady_start = time.monotonic()
            last_log = steady_start
            last_log_frames = 1

        prof.infer.add((t1 - t0) * 1000.0)
        prof.overlay.add((t2 - t1) * 1000.0)
        prof.total.add((t2 - t0) * 1000.0)

        now = time.monotonic()
        since_log = now - last_log
        if cfg.profile_interval > 0 and since_log >= cfg.profile_interval:
            win_fps = (processed - last_log_frames) / since_log if since_log > 0 else 0.0
            print(
                f"frame={processed} fps={win_fps:.1f} boxes={len(boxes)} "
                f"ms(pipeline={prof.infer.window_mean():.1f} "
                f"decode={prof.overlay.window_mean():.1f} "
                f"total={prof.total.window_mean():.1f}) {summarize(boxes)}"
            )
            prof.reset_windows()
            last_log = now
            last_log_frames = processed

    print_profile_summary(prof, processed, time.monotonic() - steady_start)

    run.close()
    return 130 if _stop else 0


def main() -> int:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg = read_config(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
    return run_graph(cfg) if cfg.pipeline_mode == "graph" else run_push(cfg)


if __name__ == "__main__":
    sys.exit(main())
