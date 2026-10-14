#!/usr/bin/env python3
"""Standalone probe for ONE compiled model of the quad app (built to debug `yolo26s-pose`).

Why this exists
---------------
The quad pipeline runs four models on one shared MLA across a dozen threads, which is the worst
place to work out why one model is slow or wrong. This script strips everything else away so the
model can be studied on its own:

  * ONE model graph: input(NV12) -> model -> output, built with main.py's own helpers and
    config/default.conf, so the model, input and run options are the app's. No UDP encoder,
    no threads, no overlay.
  * Lock-step push/pull, so the reported `infer` is the TRUE per-frame model cost
    (with frames in flight, push->pull would measure graph latency instead).
  * Prints every output tensor's shape/dtype.
  * Reads the on-device decode result and sanity-checks the boxes/keypoints, so we can
    tell "slow but correct" from "slow and wrong".
  * `--sweep` tries the runtime knobs that could plausibly move the number, so a
    fix (or the absence of one) is demonstrated rather than assumed.

Usage
-----
    # the pose model against a live stream, 20 frames
    dk ./tools/pose_probe.py --rtsp <rtsp-url> --iters 20

    # control: the same probe on another model, no camera needed
    dk ./tools/pose_probe.py --task segmentation --synthetic --iters 20

    # is any runtime option able to move it?
    dk ./tools/pose_probe.py --rtsp <rtsp-url> --sweep

    # verify correctness on a still image and eyeball the result
    dk ./tools/pose_probe.py --image /path/frame.jpg --save-out /tmp/pose.jpg

Without --rtsp, --image or --synthetic the probe reads the RTSP URL from the config.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

APP_DIR = Path(__file__).resolve().parent.parent

# Build everything with the app's own helpers, so the probe can never measure a DIFFERENT model,
# or the same model built differently, than the pipeline deploys.
sys.path.insert(0, str(APP_DIR))
import main as app  # noqa: E402

ARCHIVES = app.DEFAULT_ARCHIVES

cv2 = None
np = None
pyneat = None
ov = None


def load_deps() -> None:
    global cv2, np, pyneat, ov
    # As in main.py: the pushed NV12 tensor is CPU memory and the preprocess route is on the
    # EV74, so allow the runtime's compatibility copy in push().
    os.environ.setdefault("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1")
    app.load_runtime_dependencies()
    cv2, np, pyneat, ov = app.cv2, app.np, app.pyneat, app.ov


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


# ── graphs (the app's own builders) ──────────────────────────────────────────
def load_app_config(args):
    """config/default.conf, as main.py reads it, with this probe's overrides."""
    cfg = app.Config()
    app.load_config_file(cfg, Path(args.config))
    if args.pre_target:
        cfg.cvu_pre_target = args.pre_target
    if args.post_target:
        cfg.cvu_post_target = args.post_target
    return cfg


def slot_for(cfg, task: str):
    """The config slot that runs `task`, so the probe picks up that slot's archive and URL."""
    for i in range(cfg.num_streams):
        if cfg._tasks.get(i, app.DEFAULT_TASKS[i % len(app.DEFAULT_TASKS)]) == task:
            return i
    return None


def make_model(cfg, spec, args):
    opt = app.model_options(cfg, spec)
    if args.mla_pool_buffers:
        opt.processmla.output_pool_buffers = args.mla_pool_buffers
    if args.async_queue_depth:
        opt.async_queue_depth = args.async_queue_depth
    if args.planner:
        opt.verbose.planner = True
        opt.verbose.level = pyneat.VerbosityLevel.Verbose
    return pyneat.Model(spec.model_path, opt)


def build_model_run(cfg, spec, args, w: int, h: int, fps: int):
    """Same graph as main.build_model_graph(), plus the probe's runtime knobs."""
    g = pyneat.Graph(f"probe_{spec.task}")
    g.add(pyneat.nodes.input(app.make_nv12_input_options(w, h, fps)))
    g.add(make_model(cfg, spec, args))
    g.add(pyneat.nodes.output(app.MODEL_ENDPOINT,
                              pyneat.OutputOptions.every_frame(max(1, cfg.queue_depth))))
    t0 = time.perf_counter()
    run = g.build(app.build_run_options(cfg, model=True))
    return run, app.MODEL_ENDPOINT, (time.perf_counter() - t0) * 1000.0


def build_rtsp_run(cfg, spec):
    """main.py's source graph for one stream: RTSP -> explicit decoder -> dropping sink."""
    return app.make_combined_source_graph(cfg, [spec]).build(app.build_run_options(cfg))


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
    cfg = load_app_config(args)
    slot = slot_for(cfg, task)
    archive = args.model or (cfg._models.get(slot) if slot is not None else None) or ARCHIVES[task]
    if not Path(archive).exists():
        raise FileNotFoundError(f"archive not found: {archive}")

    # ---- get frames -----------------------------------------------------------
    rtsp_url = ""
    if not args.image and not args.synthetic:
        rtsp_url = args.rtsp or (cfg._rtsp.get(slot, cfg.rtsp_default) if slot is not None
                                 else cfg.rtsp_default)
        if rtsp_url.startswith("<"):
            raise ValueError("no frame source: pass --rtsp <url>, --image <file> or --synthetic "
                             "(or set the RTSP URL in the config)")
    spec = app.StreamSpec(0, task, rtsp_url, str(archive), 0)

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
        nv12, w, h = synthetic_nv12(cfg.width - cfg.width % 2, cfg.height - cfg.height % 2)
        frames = [(nv12, w, h)]
    else:
        # Size and rate come from the stream itself, as in main.py; the config is the fallback.
        geometry, reason = (app.probe_stream(rtsp_url, cfg.tcp) if cfg.probe_source
                            else (None, ""))
        if geometry is not None:
            cfg.width, cfg.height, cfg.fps = geometry
        elif cfg.probe_source:
            print(f"[warn] could not probe {rtsp_url} with ffprobe: {reason}; using the "
                  f"config's {cfg.width}x{cfg.height}@{cfg.fps}", file=sys.stderr)
        w, h = cfg.width, cfg.height
        frames = []

    # ---- build the model graph, then the source (as main.py does) ---------------
    run, endpoint, build_ms = build_model_run(cfg, spec, args, w, h, cfg.fps)
    if rtsp_url:
        src_run = build_rtsp_run(cfg, spec)
    if not quiet:
        print(f"\n=== {task}  {Path(archive).name}")
        print(f"    graph build: {build_ms:.0f} ms   endpoint='{endpoint}'   "
              f"pre={cfg.cvu_pre_target} post={cfg.cvu_post_target}")

    infer_ms, decode_ms = [], []
    last_result, last_nv12 = None, None
    decode_ctx = SimpleNamespace(spec=spec)
    try:
        for i in range(args.iters):
            if src_run is not None:
                ts = src_run.pull_tensors(app.source_endpoint(spec.stream_id), 20000)
                if not ts:
                    print("[warn] RTSP frame timeout", file=sys.stderr)
                    continue
                nv12, w, h = app.tensor_nv12_from_decoded(ts[0])
            else:
                nv12, w, h = frames[0]

            tensor = app.make_nv12_tensor(nv12, w, h)
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

            tensors = app.extract_tensors(sample)
            t0 = time.perf_counter()
            result = app.decode_sample(cfg, decode_ctx, tensors, w, h)
            decode_ms.append((time.perf_counter() - t0) * 1000.0)
            last_result, last_nv12 = result, nv12
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
    ap.add_argument("--config", default=str(APP_DIR / "config" / "default.conf"),
                    help="the app's config file (model options, thresholds, fallback geometry)")
    ap.add_argument("--model", help="archive path (defaults to the config's archive for the task)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3,
                    help="iterations excluded from the timing means (graph/model warmup)")
    ap.add_argument("--rtsp", help="RTSP URL (defaults to the config's URL for the task)")
    ap.add_argument("--image", help="decode a still image instead of RTSP")
    ap.add_argument("--synthetic", action="store_true", help="use a synthetic frame (no source)")
    ap.add_argument("--save-out", help="write an annotated JPEG of the last frame")
    ap.add_argument("--pre-target", choices=["AUTO", "EV74", "A65"],
                    help="override the config's cvu_pre_target")
    ap.add_argument("--post-target", choices=["AUTO", "EV74", "A65"],
                    help="override the config's cvu_post_target")
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
