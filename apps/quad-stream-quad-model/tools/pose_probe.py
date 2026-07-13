#!/usr/bin/env python3
"""Standalone probe for ONE compiled model — built to debug `yolo26s-pose`.

Why this exists
---------------
In the full quad pipeline, `yolo26s-pose` costs ~1.8 s per frame inside the model
graph (push -> pull), ~200x its peers, even with host decode and overlay switched
off. That is far too slow to explain by its MLA cost (1.21x yolo11s by compiler
cycle count). This script strips everything else away so the model can be studied
on its own:

  * ONE model graph: input(NV12) -> model -> output. No source graph fan-out, no
    UDP encoder, no threads, no overlay.
  * Lock-step push/pull, so the reported `infer` is the TRUE per-frame model cost
    (with frames in flight, push->pull would measure graph latency instead).
  * Prints every raw output tensor's shape/dtype, so the head layout is visible.
  * Runs the host decoder and sanity-checks the decoded boxes/keypoints, so we can
    tell "slow but correct" from "slow and wrong".
  * `--sweep` tries the runtime knobs that could plausibly move the number, so a
    fix (or the absence of one) is demonstrated rather than assumed.

Usage
-----
    # reproduce the pose cost against the live stream, 20 frames
    python tools/pose_probe.py --iters 20

    # control: the same probe on a model that is known-fast
    python tools/pose_probe.py --task segmentation --iters 20

    # is any runtime option able to move it?
    python tools/pose_probe.py --sweep

    # verify correctness on a still image and eyeball the result
    python tools/pose_probe.py --image /path/frame.jpg --save-out /tmp/pose.jpg
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import statistics
import sys
import time

# The app sets this too. Without it, pushing a CPU-backed NV12 Tensor into the
# model's EV74 route is refused outright. With it, Neat performs a "slow
# compatibility copy" in runner.push(). That copy is the same 1280x720 NV12 for
# every model, so it is NOT what makes pose special — but it is a real per-push
# cost for ALL models, and building the Tensor with TensorMemory.EV74 instead
# would avoid it. Tracked as a separate optimisation lead.
os.environ.setdefault("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1")

APP_DIR = Path(__file__).resolve().parent.parent

# Read the archive map straight from the app, so the probe can never measure a
# DIFFERENT model than the pipeline actually deploys. (It did, once: the probe
# kept its own copy of the map, so it silently benchmarked the OLD pose archive
# after the app had been pointed at the fixed one.)
sys.path.insert(0, str(APP_DIR))
from main import DEFAULT_ARCHIVES as ARCHIVES  # noqa: E402

cv2 = None
np = None
pyneat = None
ov = None


def load_deps() -> None:
    global cv2, np, pyneat, dec
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cv2 as _cv2
    import numpy as _np
    import pyneat as _pyneat
    cv2, np, pyneat = _cv2, _np, _pyneat
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from src import overlay as _ov
    ov = _ov


# Decode families, kept in lockstep with main.py (the whole point of this probe is
# that its numbers transfer to the app, so the model must be built identically).
DECODE_FAMILY = {
    "detection": "YoloV8",       # zoo yolo_11s
    "segmentation": "YoloV8Seg",  # zoo yolo_11s_seg
    "pose": "YoloV26Pose",        # yolo26s-pose
    "yolox": "YoloX",             # yolox_s
}
NUM_CLASSES = {"detection": 80, "segmentation": 80, "yolox": 80, "pose": 1}


def decode_on_device(task: str, tensors, w: int, h: int):
    """Read the BBOX payload the model graph's on-device BoxDecode stage produced.
    This is a payload READ, not a decode — the MLA/EV74 already did the work."""
    result = ov.DecodeResult([])
    if task == "pose":
        for r in pyneat.decode_pose(tensors, clamp_to=(w, h), top_k=100):
            boxes = np.asarray(r.boxes.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
            kpts = np.asarray(r.keypoints.to_numpy(copy=True), dtype=np.float32).reshape((-1, 17, 3))
            for i, (x1, y1, x2, y2, sc, cid) in enumerate(boxes):
                if sc < 0.25:
                    continue
                d = ov.Detection(float(x1), float(y1), float(x2), float(y2), float(sc), int(cid))
                if i < kpts.shape[0]:
                    d.keypoints = kpts[i]
                result.detections.append(d)
        return result
    if task == "segmentation":
        for r in pyneat.decode_segmentation(tensors, clamp_to=(w, h), top_k=100):
            boxes = np.asarray(r.boxes.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
            for x1, y1, x2, y2, sc, cid in boxes:
                if sc >= 0.25:
                    result.detections.append(
                        ov.Detection(float(x1), float(y1), float(x2), float(y2), float(sc), int(cid)))
        return result
    for t in pyneat.decode_bbox(tensors, clamp_to=(w, h), top_k=100):
        arr = np.asarray(t.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
        for x1, y1, x2, y2, sc, cid in arr:
            if sc >= 0.25:
                result.detections.append(
                    ov.Detection(float(x1), float(y1), float(x2), float(y2), float(sc), int(cid)))
    return result


# ── NV12 plumbing (identical contract to the app, so results transfer) ────────
def make_nv12_tensor(nv12, w: int, h: int):
    t = pyneat.Tensor.from_numpy(np.ascontiguousarray(nv12), copy=True,
                                 layout=pyneat.TensorLayout.HW,
                                 memory=pyneat.TensorMemory.CPU)
    t.shape = [h, w]
    t.strides_bytes = [w, 1]
    t.byte_offset = 0
    img = pyneat.ImageSpec()
    img.format = pyneat.PixelFormat.NV12
    sem = t.semantic
    sem.image = img
    t.semantic = sem
    y = pyneat.Plane()
    y.role = pyneat.PlaneRole.Y
    y.shape = [h, w]
    y.strides_bytes = [w, 1]
    y.byte_offset = 0
    uv = pyneat.Plane()
    uv.role = pyneat.PlaneRole.UV
    uv.shape = [h // 2, w]
    uv.strides_bytes = [w, 1]
    uv.byte_offset = w * h
    t.planes = [y, uv]
    return t


def bgr_to_nv12(bgr):
    """OpenCV BGR -> NV12 (Y plane, then interleaved UV), the layout Neat expects."""
    h, w = bgr.shape[:2]
    h -= h % 2
    w -= w % 2
    bgr = bgr[:h, :w]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)   # (h*3/2, w): Y, then U, then V
    y = i420[:h]
    u = i420[h:h + h // 4].reshape(-1)
    v = i420[h + h // 4:].reshape(-1)
    uv = np.empty(w * h // 2, dtype=np.uint8)
    uv[0::2] = u                                        # NV12 = U,V interleaved
    uv[1::2] = v
    return np.ascontiguousarray(np.vstack([y, uv.reshape(h // 2, w)])), w, h


def tensor_nv12_from_decoded(t):
    w = int(t.width() if callable(t.width) else t.width)
    h = int(t.height() if callable(t.height) else t.height)
    payload = np.frombuffer(t.copy_payload_bytes(), dtype=np.uint8)
    need = w * h * 3 // 2
    return np.ascontiguousarray(payload[:need].reshape((h * 3 // 2, w))).copy(), w, h


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


# ── graphs ────────────────────────────────────────────────────────────────────
def make_model(archive: str, task: str, args, w: int, h: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.input_max_width = w
    opt.preprocess.input_max_height = h
    opt.preprocess.input_max_depth = 1
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = 640
    opt.preprocess.resize.height = 640
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.processcvu.pre_run_target = args.pre_target
    opt.processcvu.post_run_target = args.post_target
    if args.mla_pool_buffers:
        opt.processmla.output_pool_buffers = args.mla_pool_buffers
    if args.async_queue_depth:
        opt.async_queue_depth = args.async_queue_depth
    # Every task decodes on-device, same contract as the app (see main.DECODE_FAMILY).
    opt.decode_type = getattr(pyneat.BoxDecodeType, DECODE_FAMILY[task])
    opt.score_threshold = 0.25
    opt.nms_iou_threshold = 0.5
    opt.top_k = 100
    opt.num_classes = NUM_CLASSES[task]
    if args.planner:
        opt.verbose.planner = True
        opt.verbose.level = pyneat.VerbosityLevel.Verbose
    return pyneat.Model(archive, opt)


def nv12_input_options(w: int, h: int, fps: int):
    o = pyneat.InputOptions()
    o.payload_type = pyneat.PayloadType.Image
    o.format = pyneat.Format.NV12
    o.width = w; o.height = h; o.depth = 1
    o.max_width = w; o.max_height = h; o.max_depth = 1
    o.fps_n = max(1, fps); o.fps_d = 1
    o.caps_override = (f"video/x-raw,format=NV12,width={w},height={h},"
                       f"framerate={max(1, fps)}/1")
    o.use_simaai_pool = False
    return o


def run_options():
    ro = pyneat.RunOptions()
    ro.preset = pyneat.RunPreset.Realtime
    ro.queue_depth = 3
    ro.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    ro.output_memory = pyneat.OutputMemory.ZeroCopy
    return ro


def build_model_run(archive: str, task: str, args, w: int, h: int, fps: int):
    endpoint = "detections" if task == "detection" else "heads"
    g = pyneat.Graph(f"probe_{task}")
    g.add(pyneat.nodes.input(nv12_input_options(w, h, fps)))
    g.add(make_model(archive, task, args, w, h))
    g.add(pyneat.nodes.output(endpoint, pyneat.OutputOptions.every_frame(1)))
    t0 = time.perf_counter()
    run = g.build(run_options())
    return run, endpoint, (time.perf_counter() - t0) * 1000.0


def build_rtsp_run(url: str, w: int, h: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = 200
    opt.tcp = True
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
    g = pyneat.Graph("probe_source")
    g.add(pyneat.groups.rtsp_decoded_input(opt))
    g.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))
    ro = run_options()
    ro.output_memory = pyneat.OutputMemory.Owned
    return g.build(ro)


# ── frame sources ─────────────────────────────────────────────────────────────
def synthetic_nv12(w: int, h: int):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, w, 64):
        img[:, i:i + 32] = (60, 140, 220)
    cv2.rectangle(img, (w // 3, h // 4), (2 * w // 3, h), (200, 200, 200), -1)
    return bgr_to_nv12(img)


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))]


# ── report helpers ────────────────────────────────────────────────────────────
def describe_outputs(tensors) -> None:
    print(f"\n--- raw model outputs: {len(tensors)} tensors ---")
    total = 0
    for i, t in enumerate(tensors):
        a = t.to_numpy(copy=True) if hasattr(t, "to_numpy") else np.asarray(t)
        total += a.size
        # channel-last (1,H,W,C) is what the MLA emits
        c = a.shape[-1] if a.ndim >= 2 else -1
        print(f"  [{i}] shape={tuple(a.shape)} dtype={a.dtype} C={c} "
              f"min={float(a.min()):+.3f} max={float(a.max()):+.3f}")
    print(f"  total elements: {total:,}")


def check_pose(result, w: int, h: int) -> None:
    """Sanity-check decoded pose output: are boxes and keypoints in-frame and real?"""
    n = len(result.detections)
    print(f"\n--- host decode: {n} person(s) ---")
    if n == 0:
        print("  [!] zero detections — either the scene has no people or the decode is wrong")
        return
    for d in result.detections[:5]:
        kp = d.keypoints
        if kp is None:
            print(f"  box=({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f}) "
                  f"score={d.score:.2f}  [!] no keypoints attached")
            continue
        vis = int((kp[:, 2] >= 0.3).sum())
        inside = int(((kp[:, 0] >= 0) & (kp[:, 0] < w) &
                      (kp[:, 1] >= 0) & (kp[:, 1] < h)).sum())
        print(f"  box=({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f}) score={d.score:.2f} "
              f"kpts={kp.shape} visible={vis}/17 in-frame={inside}/17")
    boxes_ok = all(0 <= d.x1 < d.x2 <= w and 0 <= d.y1 < d.y2 <= h
                   for d in result.detections)
    print(f"  all boxes within frame bounds: {boxes_ok}")


def annotate(bgr, result, task: str):
    for d in result.detections:
        cv2.rectangle(bgr, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
        cv2.putText(bgr, f"{d.score:.2f}", (int(d.x1), max(12, int(d.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if d.keypoints is not None:
            for kx, ky, kv in d.keypoints:
                if kv >= 0.3:
                    cv2.circle(bgr, (int(kx), int(ky)), 3, (0, 0, 255), -1)
            for a, b in ov.COCO_SKELETON:
                if a < len(d.keypoints) and b < len(d.keypoints) and \
                        d.keypoints[a, 2] >= 0.3 and d.keypoints[b, 2] >= 0.3:
                    cv2.line(bgr,
                             (int(d.keypoints[a, 0]), int(d.keypoints[a, 1])),
                             (int(d.keypoints[b, 0]), int(d.keypoints[b, 1])),
                             (255, 128, 0), 2, cv2.LINE_AA)
    return bgr


# ── main probe ────────────────────────────────────────────────────────────────
def probe(args, task: str, quiet: bool = False):
    archive = args.model or ARCHIVES[task]
    if not Path(archive).exists():
        raise FileNotFoundError(f"archive not found: {archive}")

    # ---- get frames -----------------------------------------------------------
    src_run = None
    bgr_ref = None
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise FileNotFoundError(f"cannot read image: {args.image}")
        nv12, w, h = bgr_to_nv12(bgr)
        bgr_ref = bgr.copy()
        frames = [(nv12, w, h)]
    elif args.synthetic:
        nv12, w, h = synthetic_nv12(1280, 720)
        frames = [(nv12, w, h)]
    else:
        w, h, fps = 1280, 720, 60
        src_run = build_rtsp_run(args.rtsp, w, h, fps)
        frames = []

    # ---- build the model graph ------------------------------------------------
    run, endpoint, build_ms = build_model_run(archive, task, args, w, h, 60)
    if not quiet:
        print(f"\n=== {task}  {Path(archive).name}")
        print(f"    graph build: {build_ms:.0f} ms   endpoint='{endpoint}'   "
              f"pre={args.pre_target} post={args.post_target}")

    infer_ms, decode_ms = [], []
    last_tensors, last_result, last_nv12 = None, None, None
    try:
        for i in range(args.iters):
            if src_run is not None:
                ts = src_run.pull_tensors(timeout_ms=20000)
                if not ts:
                    print("[warn] RTSP frame timeout", file=sys.stderr)
                    continue
                nv12, w, h = tensor_nv12_from_decoded(ts[0])
            else:
                nv12, w, h = frames[0]

            tensor = make_nv12_tensor(nv12, w, h)
            t0 = time.perf_counter()
            if not run.push([tensor]):
                print("[warn] push failed", file=sys.stderr)
                continue
            sample = run.pull(endpoint, 30000)          # lock-step: true model cost
            dt = (time.perf_counter() - t0) * 1000.0
            if sample is None:
                print("[warn] pull returned None", file=sys.stderr)
                continue
            infer_ms.append(dt)

            tensors = extract_tensors(sample)
            t0 = time.perf_counter()
            result = decode_on_device(task, tensors, w, h)
            decode_ms.append((time.perf_counter() - t0) * 1000.0)
            last_tensors, last_result, last_nv12 = tensors, result, nv12
            if not quiet and i == 0:
                describe_outputs(tensors)
    finally:
        run.close()
        if src_run is not None:
            src_run.close()

    if not infer_ms:
        raise RuntimeError("no successful iterations")

    # The first frames pay graph/model warmup; they are not the steady-state cost.
    warm = getattr(args, "warmup", 0)
    if warm and len(infer_ms) > warm:
        infer_ms = infer_ms[warm:]
        decode_ms = decode_ms[warm:]

    mean_infer = statistics.mean(infer_ms)
    if not quiet:
        print(f"\n--- timing over {len(infer_ms)} frames (lock-step push/pull) ---")
        print(f"  infer  mean {mean_infer:8.2f} ms   p95 {pct(infer_ms, 95):8.2f} ms"
              f"   -> model rate {1000.0 / mean_infer:6.1f} fps")
        print(f"  decode mean {statistics.mean(decode_ms):8.2f} ms   "
              f"p95 {pct(decode_ms, 95):8.2f} ms   (BBOX payload read, host)")
        if task == "pose":
            check_pose(last_result, w, h)
        else:
            print(f"\n--- host decode: {len(last_result.detections)} object(s) ---")

        if args.save_out and last_result is not None:
            if bgr_ref is not None:
                canvas = bgr_ref.copy()
            else:
                canvas = cv2.cvtColor(last_nv12, cv2.COLOR_YUV2BGR_NV12)
            cv2.imwrite(args.save_out, annotate(canvas, last_result, task))
            print(f"\n  wrote {args.save_out}")

    return mean_infer, len(last_result.detections) if last_result else 0


def sweep(args):
    """Try every runtime knob that could plausibly move the pose number.

    If none of them moves it, the cost is baked into the compiled MPK graph and
    the fix is a recompile — which is exactly what we want to establish.
    """
    print("\n############ runtime option sweep (pose) ############")
    combos = [
        ("baseline           ", dict(pre_target="AUTO", post_target="AUTO")),
        ("post=EV74          ", dict(pre_target="AUTO", post_target="EV74")),
        ("post=A65           ", dict(pre_target="AUTO", post_target="A65")),
        ("pre=EV74,post=EV74 ", dict(pre_target="EV74", post_target="EV74")),
        ("mla_pool_buffers=8 ", dict(mla_pool_buffers=8)),
        ("async_queue_depth=4", dict(async_queue_depth=4)),
    ]
    base = vars(args).copy()
    results = []
    for label, over in combos:
        a = argparse.Namespace(**{**base, **{"pre_target": "AUTO", "post_target": "AUTO",
                                             "mla_pool_buffers": 0, "async_queue_depth": 0},
                                  **over})
        try:
            ms, _ = probe(a, "pose", quiet=True)
            results.append((label, ms))
            print(f"  {label}  infer {ms:9.2f} ms   ({1000.0 / ms:6.2f} fps)", flush=True)
        except Exception as exc:
            print(f"  {label}  FAILED: {exc}", flush=True)
    if results:
        best = min(results, key=lambda r: r[1])
        worst = max(results, key=lambda r: r[1])
        print(f"\n  best {best[0].strip()}: {best[1]:.1f} ms | "
              f"worst {worst[0].strip()}: {worst[1]:.1f} ms | "
              f"spread {worst[1] - best[1]:.1f} ms")
        if worst[1] - best[1] < 0.15 * worst[1]:
            print("  => No runtime option moves it materially. The cost is baked into the\n"
                  "     compiled MPK post-MLA graph; the fix is a RECOMPILE, not a flag.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(ARCHIVES), default="pose")
    ap.add_argument("--model", help="archive path (defaults to the task's compiled archive)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3,
                    help="iterations excluded from the timing means (graph/model warmup)")
    ap.add_argument("--rtsp", default="rtsp://<rtsp-server-ip>:8555/stream")
    ap.add_argument("--image", help="decode a still image instead of RTSP")
    ap.add_argument("--synthetic", action="store_true", help="use a synthetic frame (no source)")
    ap.add_argument("--save-out", help="write an annotated JPEG of the last frame")
    ap.add_argument("--pre-target", default="AUTO", choices=["AUTO", "EV74", "A65"])
    ap.add_argument("--post-target", default="AUTO", choices=["AUTO", "EV74", "A65"])
    ap.add_argument("--mla-pool-buffers", type=int, default=0)
    ap.add_argument("--async-queue-depth", type=int, default=0)
    ap.add_argument("--planner", action="store_true",
                    help="dump the MPK contract / planner advisories")
    ap.add_argument("--sweep", action="store_true",
                    help="try all runtime knobs on pose and report whether any helps")
    args = ap.parse_args()

    load_deps()
    try:
        if args.sweep:
            sweep(args)
        else:
            probe(args, args.task)
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
