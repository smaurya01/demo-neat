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
WORK = "/workspace/demo-neat/model-compilation/work"

# Read the archive map straight from the app, so the probe can never measure a
# DIFFERENT model than the pipeline actually deploys. (It did, once: the probe
# kept its own copy of the map, so it silently benchmarked the OLD pose archive
# after the app had been pointed at the fixed one.)
sys.path.insert(0, str(APP_DIR))
from main import DEFAULT_ARCHIVES as ARCHIVES  # noqa: E402

cv2 = None
np = None
pyneat = None
dec = None


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
    from src import decoders as _dec
    dec = _dec


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
    if task == "detection":
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
        opt.score_threshold = 0.25
        opt.nms_iou_threshold = 0.5
        opt.top_k = 100
        opt.num_classes = 80
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
            for a, b in dec.COCO_SKELETON:
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
            if task == "detection":
                result = dec.DecodeResult([])
                for t in pyneat.decode_bbox(tensors, clamp_to=(w, h), top_k=100):
                    arr = np.asarray(t.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
                    for x1, y1, x2, y2, sc, cid in arr:
                        if sc >= 0.25:
                            result.detections.append(
                                dec.Detection(float(x1), float(y1), float(x2),
                                              float(y2), float(sc), int(cid)))
            else:
                result = dec.HOST_DECODERS[
                    task if task != "detection" else "detection_host"](
                    tensors, w, h, model_w=640, model_h=640,
                    score_thr=0.25, iou_thr=0.5, top_k=100)
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
              f"p95 {pct(decode_ms, 95):8.2f} ms   (host, A65)")
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


def kpt_formula_scan(args):
    """Find the correct YOLO26 keypoint decode empirically.

    The person BOXES decode correctly (verified visually), so they are a reliable
    reference: a correct keypoint decode must place most of a person's visible
    keypoints inside (or very near) that person's box. Each candidate formula is
    scored by exactly that. This beats guessing which Ultralytics variant applies
    to a YOLO26 `one2one_cv4_kpts` head.

    Candidates (k = raw conv value, i = integer grid index, s = stride):
      A  (2k + i) * s          Ultralytics v8/v11 pose (what src/decoders.py uses)
      B  (k + i + 0.5) * s     no 2x, anchor-centred
      C  (k + i) * s           no 2x, integer grid
      D  (2k + i + 0.5) * s    2x, anchor-centred
      E  (2*sigmoid(k) - 0.5 + i) * s   sigmoid-gated offset
    """
    archive = args.model or ARCHIVES["pose"]
    w, h, fps = 1280, 720, 60
    src_run = build_rtsp_run(args.rtsp, w, h, fps)
    run, endpoint, _ = build_model_run(archive, "pose", args, w, h, fps)

    FORMULAS = {
        "A (2k+i)*s        [current]": lambda k, g, s: (k * 2.0 + g) * s,
        "B (k+i+0.5)*s             ": lambda k, g, s: (k + g + 0.5) * s,
        "C (k+i)*s                 ": lambda k, g, s: (k + g) * s,
        "D (2k+i+0.5)*s            ": lambda k, g, s: (k * 2.0 + g + 0.5) * s,
        "E (2*sig(k)-0.5+i)*s      ": lambda k, g, s: (2.0 * dec._sigmoid(k) - 0.5 + g) * s,
    }
    # per formula: inside, total_visible, span_sum, nose_ok, nose_n, det_n
    score = {name: [0, 0, 0.0, 0.0, 0, 0] for name in FORMULAS}

    try:
        for _ in range(args.iters):
            ts = src_run.pull_tensors(timeout_ms=20000)
            if not ts:
                continue
            nv12, fw, fh = tensor_nv12_from_decoded(ts[0])
            if not run.push([make_nv12_tensor(nv12, fw, fh)]):
                continue
            sample = run.pull(endpoint, 30000)
            if sample is None:
                continue
            tensors = extract_tensors(sample)

            b = dec._classify(tensors, 640)
            geom = dec.LetterboxGeom.compute(fw, fh, 640, 640)
            all_boxes, all_scores, per_scale = [], [], []
            for hh in sorted(b["bbox"], reverse=True):
                if hh not in b["cls"] or hh not in b["kpt"]:
                    continue
                ww = b["bbox"][hh].shape[2]
                stride = 640.0 / hh
                bbox = b["bbox"][hh].reshape(4, -1)
                cls = dec._sigmoid(b["cls"][hh].reshape(-1))
                kpt = b["kpt"][hh].reshape(51, -1)
                gx05, gy05 = dec._anchor_grid(hh, ww, 0.5)
                gxi, gyi = dec._anchor_grid(hh, ww, 0.0)
                boxes = dec._dist2bbox(bbox, gx05, gy05, stride)
                all_boxes.append(boxes)
                all_scores.append(cls)
                per_scale.append((kpt, gxi, gyi, stride, boxes.shape[0]))
            if not all_boxes:
                continue
            boxes = np.concatenate(all_boxes, 0)
            scores = np.concatenate(all_scores, 0)
            offsets = np.cumsum([0] + [p[4] for p in per_scale])

            m = scores >= 0.25
            idx_map = np.nonzero(m)[0]
            if not idx_map.size:
                continue
            keep = dec._nms(boxes[m], scores[m], 0.5, 100)

            for kk in keep:
                gidx = idx_map[kk]
                sidx = int(np.searchsorted(offsets, gidx, side="right") - 1)
                kpt, gxi, gyi, stride, _n = per_scale[sidx]
                li = gidx - offsets[sidx]
                bx = boxes[gidx]
                fx1, fy1 = geom.to_frame_xy(bx[0], bx[1])
                fx2, fy2 = geom.to_frame_xy(bx[2], bx[3])
                vis = dec._sigmoid(kpt[2::3, li])
                for name, fn in FORMULAS.items():
                    kx = fn(kpt[0::3, li], gxi[li], stride)
                    ky = fn(kpt[1::3, li], gyi[li], stride)
                    px, py = geom.to_frame_xy(kx, ky)
                    sel = vis >= 0.3
                    if not sel.any():
                        continue
                    # generous margin: a limb may extend slightly past the box
                    mx, my = 0.25 * (fx2 - fx1), 0.10 * (fy2 - fy1)
                    inside = ((px[sel] >= fx1 - mx) & (px[sel] <= fx2 + mx) &
                              (py[sel] >= fy1 - my) & (py[sel] <= fy2 + my))
                    st = score[name]
                    st[0] += int(inside.sum())
                    st[1] += int(sel.sum())
                    st[5] += 1

                    # "inside the box" alone is gameable: a formula that collapses
                    # every keypoint onto the anchor scores 100% while being useless
                    # (formula E's sigmoid bounds the offset to ~1 cell, which does
                    # exactly that). These two checks cannot be gamed that way.
                    box_h = max(1e-6, fy2 - fy1)
                    # (a) the skeleton must SPAN the person, not clump on the anchor
                    st[2] += float((py[sel].max() - py[sel].min()) / box_h)
                    # (b) anatomy: the nose must sit above the ankles
                    if vis[0] >= 0.3 and (vis[15] >= 0.3 or vis[16] >= 0.3):
                        ank = [py[i] for i in (15, 16) if vis[i] >= 0.3]
                        st[3] += 1.0 if py[0] < (sum(ank) / len(ank)) else 0.0
                        st[4] += 1
    finally:
        run.close()
        src_run.close()

    print("\n############ keypoint decode formula scan ############")
    print("A correct decode must satisfy ALL THREE, not just the first:")
    print("  inside%     visible keypoints inside their (known-correct) person box")
    print("  span        vertical spread of skeleton / box height. Want ~0.8-1.0.")
    print("              A formula that collapses keypoints onto the anchor scores ~0")
    print("              here while still scoring 100% on inside% — that is the trap.")
    print("  nose<ankle  anatomy: head above feet. Want ~100%.\n")
    print(f"  {'formula':<28} {'inside%':>8} {'span':>7} {'nose<ankle':>11}")
    rows = []
    for name, (ins, tot, span_sum, nose_ok, nose_n, det_n) in score.items():
        if not tot or not det_n:
            continue
        rows.append((name, ins / tot, span_sum / det_n, nose_ok / max(1, nose_n)))
    # a formula must win on all three; span is capped at 1.0 so overshoot is not rewarded
    ranked = sorted(rows, key=lambda r: -(r[1] * min(r[2], 1.0) * r[3]))
    for name, insf, span, nosef in ranked:
        print(f"  {name:<28} {insf * 100:7.1f}% {span:7.2f} {nosef * 100:10.1f}%")
    print(f"\n  WINNER: {ranked[0][0].strip()}")


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
    ap.add_argument("--rtsp", default="rtsp://192.168.2.105:8555/stream")
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
    ap.add_argument("--kpt-scan", action="store_true",
                    help="empirically find the correct YOLO26 keypoint decode formula")
    args = ap.parse_args()

    load_deps()
    try:
        if args.sweep:
            sweep(args)
        elif args.kpt_scan:
            kpt_formula_scan(args)
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
