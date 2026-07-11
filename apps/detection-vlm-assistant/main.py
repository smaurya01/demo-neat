#!/usr/bin/env python3
"""Detection-to-VLM assistant: always-on YOLO11 detector -> trigger-gated VLM captions.

Shape of the pipeline
----------------------
    frame (RTSP or still image) --> YOLO11 detector (cheap, every frame)
                                        |
                                        v
                              trigger gate + dedup (which crops deserve a VLM call)
                                        |
                                        v
                              bounded background VLM worker --> natural-language caption

The detector is cheap and always-on (~37 fps measured on the 2-stream YOLO11 app);
the VLM is expensive (seconds per call). So the VLM is NOT run per frame. Selected
detections are cropped, de-duplicated, rate-limited, and handed to a bounded
background worker so the detection loop never blocks on VLM latency.

Split validation (see README):
  * The DETECTION leg is validated live on the DevKit (RTSP or --image in -> boxes out).
  * The VLM leg is code-complete + API-checked but NOT executed here. Run with
    ``--no-vlm`` (the default when no VLM model dir is configured) to log the crop and
    the exact prompt that WOULD be sent, without touching the VLM. The owner runs the
    real VLM manually.

APIs traceable to /workspace/core and the sibling apps:
  * Detector: adapted from apps/examples/genai/detection-to-vlm-assistant/src/python/main.py
    and apps/multi-stream-yolo-yolo11/main.py (Agent A, live-validated on this exact
    compiled yolo11n archive).
  * Box decode: ``pyneat.decode_bbox`` (core/python/src/module.cpp), the same call Agent A
    validated on this archive; a compile_ready yolo11n decodes with BoxDecodeType.YoloV26.
  * VLM: pyneat.genai.VisionLanguageModel (src/vlm_commenter.py).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import glob
import os
from pathlib import Path
import sys
import time

# Bound late so `--help` and py_compile work off-board where pyneat is absent.
cv2 = None
np = None
pyneat = None

# COCO 80-class labels (index == class_id from the detector). Kept upper-case for
# on-screen use; matching against config allow-lists is case-insensitive.
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
    # --- input ---
    source: str = "rtsp"                # rtsp | image
    rtsp_url: str = "rtsp://192.168.132.129:8555/stream"
    image_path: str = ""               # file, directory, or glob (source=image)
    rtsp_transport_tcp: bool = True
    fallback_width: int = 1280
    fallback_height: int = 720
    fallback_fps: int = 30
    latency_ms: int = 200

    # --- detector ---
    model_path: str = ""
    models_dir: str = ""
    model_name: str = "yolo11"         # yolo11 | yolo26n => BoxDecodeType.YoloV26
    model_width: int = 640             # compiled model input size (Neat resizes to this)
    model_height: int = 640
    score_threshold: float = 0.25
    nms_iou: float = 0.50
    top_k: int = 100
    frames: int = 0                    # 0 = run until interrupted (rtsp) / all images
    timeout_ms: int = 20000

    # --- VLM trigger / commenter (src/vlm_commenter.py) ---
    vlm_enabled: bool = True
    vlm_model_dir: str = "/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4"
    vlm_trigger_classes: tuple[str, ...] = ("person",)
    vlm_trigger_min_score: float = 0.55
    vlm_trigger_min_area_frac: float = 0.02   # crop must cover >= 2% of the frame
    vlm_interval_seconds: float = 5.0         # min seconds between VLM calls
    vlm_max_pending: int = 1                  # bounded queue of in-flight/queued crops
    vlm_dedup_iou: float = 0.6                # same object if IoU >= this
    vlm_dedup_cooldown_s: float = 15.0        # re-trigger the same object after this
    vlm_max_new_tokens: int = 96
    vlm_prompt: str = (
        "You are watching a security camera. This image is a crop of one detected "
        "{label}. Describe what this {label} is doing and anything notable in one "
        "short sentence."
    )

    def resolved_model_path(self) -> str:
        if self.model_path:
            return str(_resolve_app_path(self.model_path))
        models_dir = (
            _resolve_app_path(self.models_dir)
            if self.models_dir
            else Path(__file__).resolve().parent / "assets" / "models"
        )
        return str(models_dir / "yolo_11n_mpk.tar.gz")


def _resolve_app_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "default.conf"


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


# --------------------------------------------------------------------------- #
# Config file (key=value .conf, same convention as the other apps/ entries)
# --------------------------------------------------------------------------- #
def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def apply_config_value(cfg: Config, key: str, value: str) -> None:
    if key == "source":
        cfg.source = value.strip().lower()
    elif key == "rtsp_url":
        cfg.rtsp_url = value
    elif key == "image_path":
        cfg.image_path = value
    elif key == "rtsp_transport":
        cfg.rtsp_transport_tcp = value.strip().lower() == "tcp"
    elif key == "fallback_width":
        cfg.fallback_width = int(value)
    elif key == "fallback_height":
        cfg.fallback_height = int(value)
    elif key == "fallback_fps":
        cfg.fallback_fps = int(value)
    elif key == "latency_ms":
        cfg.latency_ms = int(value)
    elif key == "model_path":
        cfg.model_path = value
    elif key == "models_dir":
        cfg.models_dir = value
    elif key == "model_name":
        cfg.model_name = value
    elif key == "model_width":
        cfg.model_width = int(value)
    elif key == "model_height":
        cfg.model_height = int(value)
    elif key == "score_threshold":
        cfg.score_threshold = float(value)
    elif key == "nms_iou":
        cfg.nms_iou = float(value)
    elif key == "top_k":
        cfg.top_k = int(value)
    elif key == "frames":
        cfg.frames = int(value)
    elif key == "timeout_ms":
        cfg.timeout_ms = int(value)
    elif key == "vlm_enabled":
        cfg.vlm_enabled = _parse_bool(value)
    elif key == "vlm_model_dir":
        cfg.vlm_model_dir = value
    elif key == "vlm_trigger_classes":
        cfg.vlm_trigger_classes = _csv(value)
    elif key == "vlm_trigger_min_score":
        cfg.vlm_trigger_min_score = float(value)
    elif key == "vlm_trigger_min_area_frac":
        cfg.vlm_trigger_min_area_frac = float(value)
    elif key == "vlm_interval_seconds":
        cfg.vlm_interval_seconds = float(value)
    elif key == "vlm_max_pending":
        cfg.vlm_max_pending = max(1, int(value))
    elif key == "vlm_dedup_iou":
        cfg.vlm_dedup_iou = float(value)
    elif key == "vlm_dedup_cooldown_s":
        cfg.vlm_dedup_cooldown_s = float(value)
    elif key == "vlm_max_new_tokens":
        cfg.vlm_max_new_tokens = int(value)
    elif key == "vlm_prompt":
        cfg.vlm_prompt = value
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


def parse_args(argv: list[str] | None) -> tuple[Config, bool]:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--source", choices=["rtsp", "image"])
    parser.add_argument("--rtsp", dest="rtsp_url")
    parser.add_argument("--image", dest="image_path",
                        help="still image file / directory / glob; implies --source image")
    parser.add_argument("--model")
    parser.add_argument("--model-name")
    parser.add_argument("--score", type=float)
    parser.add_argument("--nms", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--vlm-model-dir")
    parser.add_argument("--no-vlm", action="store_true",
                        help="dry-run: log the crop + the prompt that WOULD be sent, "
                             "never load or call the VLM (this is how the detection leg "
                             "is validated without touching the VLM)")
    args = parser.parse_args(argv)

    cfg = Config()
    load_config_file(cfg, args.config, required=args.config.exists())
    if args.source is not None:
        cfg.source = args.source
    if args.rtsp_url is not None:
        cfg.rtsp_url = args.rtsp_url
    if args.image_path is not None:
        cfg.image_path = args.image_path
        cfg.source = "image"
    if args.model is not None:
        cfg.model_path = args.model
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.score is not None:
        cfg.score_threshold = args.score
    if args.nms is not None:
        cfg.nms_iou = args.nms
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.frames is not None:
        cfg.frames = args.frames
    if args.vlm_model_dir is not None:
        cfg.vlm_model_dir = args.vlm_model_dir
    return cfg, args.no_vlm


# --------------------------------------------------------------------------- #
# Detector (adapted from the reference detection-to-vlm app; BGR image input)
# --------------------------------------------------------------------------- #
def build_model(cfg: Config):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    # Explicit letterbox resize to the compiled model input (mirrors the proven
    # preprocess block in apps/multi-stream-yolo-yolo11/main.py). Without this the
    # planner does not resize a full-size frame to 640x640 and the model yields no
    # detections (observed on this archive with the auto-only path).
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = cfg.model_width
    opt.preprocess.resize.height = cfg.model_height
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    # OpenCV-decoded frames (and our NV12->BGR RTSP conversion) are BGR; the model
    # wants normalized RGB. If instead you feed a raw NV12 frame straight from the
    # decoder, set input_format to PreprocessColorFormat.NV12 as the multi-stream app does.
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # A compile_ready yolo11n surgery exposes the YoloV26 grouped-tensor head, so this
    # archive decodes with YoloV26 (NOT YoloV8). Verified by Agent A on this exact file.
    if cfg.model_name in {"yolo11", "yolo26n"}:
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
    else:
        opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.top_k
    opt.num_classes = 80
    # NOTE: intentionally NOT setting boxdecode_original_width/height — deprecated in
    # core/include/model/Model.h; box decode reads geometry from preprocess metadata.
    return pyneat.Model(cfg.resolved_model_path(), opt)


def build_detector_run(cfg: Config, width: int, height: int):
    model = build_model(cfg)
    input_opt = model.input_appsrc_options(False)
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = pyneat.Format.BGR
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 3

    graph = pyneat.Graph("detector")
    graph.add(pyneat.nodes.input(input_opt))
    graph.add(model.graph())
    # Named output + push/pull (Agent A's live-validated pattern). The synchronous
    # run([...]) helper does NOT surface this archive's model-managed box-decode
    # output; push + pull("detections") does. Verified on the DevKit.
    graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(1)))

    seed = pyneat.Tensor.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return model, graph, graph.build([seed], run_opt)


def _extract_tensors(sample) -> list:
    if sample is None or not hasattr(sample, "kind"):
        return []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    tensors = []
    for f in getattr(sample, "fields", []):
        tensors.extend(_extract_tensors(f))
    return tensors


def detect(cfg: Config, detector_run, frame_bgr) -> list[dict]:
    """Run the detector on a BGR frame and return boxes as x1,y1,x2,y2 dicts."""
    height, width = frame_bgr.shape[:2]
    model_input = pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame_bgr),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )
    if not detector_run.push([model_input]):
        raise RuntimeError("detector push failed")
    sample = detector_run.pull("detections", cfg.timeout_ms)
    tensors = _extract_tensors(sample)
    if not tensors:
        return []
    # decode_bbox turns the model's BBOX-format output into [N,6] float rows
    # (x1, y1, x2, y2, score, class_id), clamped to the frame. Same call Agent A
    # validated on this archive (core/python/src/module.cpp).
    decoded = pyneat.decode_bbox(tensors, clamp_to=(width, height), top_k=cfg.top_k)
    boxes: list[dict] = []
    for tensor in decoded:
        arr = np.asarray(tensor.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
        for x1, y1, x2, y2, score, class_id in arr:
            if x2 > x1 and y2 > y1:
                boxes.append({
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "score": float(score), "class_id": int(class_id),
                })
    return boxes[: cfg.top_k]


def label_for(class_id: int) -> str:
    if 0 <= class_id < len(COCO_LABELS):
        return COCO_LABELS[class_id]
    return f"CLASS_{class_id}"


# --------------------------------------------------------------------------- #
# Input sources
# --------------------------------------------------------------------------- #
def make_rtsp_source(cfg: Config, width: int, height: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp_url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.rtsp_transport_tcp
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

    graph = pyneat.Graph("rtsp_source")
    graph.add(pyneat.groups.rtsp_decoded_input(opt))
    graph.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))
    run_opt = pyneat.RunOptions()
    run_opt.preset = pyneat.RunPreset.Realtime
    run_opt.queue_depth = 3
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return graph, graph.build(run_opt)


def _tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def decoded_tensor_to_bgr(tensor):
    """Convert a decoded RTSP frame (NV12/I420/other) to a contiguous uint8 BGR array."""
    if tensor.is_nv12():
        width = _tensor_dim(tensor, "width")
        height = _tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))
    if tensor.is_i420():
        width = _tensor_dim(tensor, "width")
        height = _tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420))
    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


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


def list_images(cfg: Config) -> list[str]:
    p = Path(cfg.image_path)
    if p.is_dir():
        files = sorted(
            str(f) for f in p.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    elif any(ch in cfg.image_path for ch in "*?[]"):
        files = sorted(glob.glob(cfg.image_path))
    elif p.is_file():
        files = [str(p)]
    else:
        files = []
    return files


# --------------------------------------------------------------------------- #
# Main loops
# --------------------------------------------------------------------------- #
def run_image_mode(cfg: Config, commenter) -> int:
    files = list_images(cfg)
    if not files:
        print(f"no images found for image_path={cfg.image_path!r}", file=sys.stderr)
        return 3
    # Size the detector graph from the first image; re-probe if a later image differs.
    processed = 0
    detector_run = None
    built_wh: tuple[int, int] | None = None
    try:
        for path in files:
            if cfg.frames > 0 and processed >= cfg.frames:
                break
            frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                print(f"[warn] failed to read {path}", file=sys.stderr)
                continue
            height, width = frame_bgr.shape[:2]
            if built_wh != (width, height):
                if detector_run is not None:
                    detector_run.close()
                _model, _graph, detector_run = build_detector_run(cfg, width, height)
                built_wh = (width, height)
            boxes = detect(cfg, detector_run, frame_bgr)
            commenter.on_frame(frame_bgr, boxes)
            processed += 1
            summary = ", ".join(
                f"{label_for(b['class_id'])}:{b['score']:.2f}" for b in boxes[:8]
            )
            print(f"image={Path(path).name} detections={len(boxes)} [{summary}]",
                  flush=True)
        return 0 if processed > 0 else 3
    finally:
        if detector_run is not None:
            detector_run.close()


def run_rtsp_mode(cfg: Config, commenter) -> int:
    width, height, fps = probe_rtsp(cfg)
    print(f"rtsp={cfg.rtsp_url} stream={width}x{height}@{fps}", flush=True)
    _src_graph, source_run = make_rtsp_source(cfg, width, height, fps)
    _model, _graph, detector_run = build_detector_run(cfg, width, height)
    frame_id = 0
    run_start = time.perf_counter()
    try:
        while cfg.frames <= 0 or frame_id < cfg.frames:
            tensors = source_run.pull_tensors(timeout_ms=cfg.timeout_ms)
            if not tensors:
                print("RTSP stream ended or pull timed out", file=sys.stderr)
                break
            frame_bgr = decoded_tensor_to_bgr(tensors[0])
            boxes = detect(cfg, detector_run, frame_bgr)
            commenter.on_frame(frame_bgr, boxes)
            frame_id += 1
            if frame_id == 1 or frame_id % 30 == 0:
                elapsed = time.perf_counter() - run_start
                fps_now = frame_id / elapsed if elapsed > 0 else 0.0
                print(f"frame={frame_id} detections={len(boxes)} fps={fps_now:.2f}",
                      flush=True)
        return 0 if frame_id > 0 else 3
    finally:
        detector_run.close()
        source_run.close()


def run(cfg: Config, force_no_vlm: bool) -> int:
    load_runtime_dependencies()
    os.environ.setdefault("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1")

    model_path = Path(cfg.resolved_model_path())
    if not model_path.exists():
        print(f"model archive not found: {model_path}", file=sys.stderr)
        return 2

    # Import here so the module imports even where pyneat/numpy are unavailable.
    from src.vlm_commenter import VlmCommenter

    dry_run = force_no_vlm or not cfg.vlm_enabled or not Path(cfg.vlm_model_dir).is_dir()
    commenter = VlmCommenter(cfg, dry_run=dry_run, label_fn=label_for)
    commenter.start()
    print(
        f"detector={model_path.name} decode={cfg.model_name} "
        f"vlm={'DRY-RUN (crop+prompt logged, VLM not called)' if dry_run else cfg.vlm_model_dir}",
        flush=True,
    )
    try:
        if cfg.source == "image":
            return run_image_mode(cfg, commenter)
        return run_rtsp_mode(cfg, commenter)
    finally:
        commenter.close()


def main(argv: list[str] | None = None) -> int:
    try:
        cfg, force_no_vlm = parse_args(argv)
        return run(cfg, force_no_vlm)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level guard prints a clean message
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
