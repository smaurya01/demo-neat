#!/usr/bin/env python3
"""STEP 4b - can Neat's ON-DEVICE box decoder consume this archive?

    # DevKit only (needs pyneat + the MLA)
    python compile/test_box_decode.py --model-id yolov8s
    python compile/test_box_decode.py --all

WHY this exists, separately from test_model.py:

  test_model.py proves the raw head tensors come back with the right SHAPES. It says nothing
  about whether Neat itself can turn them into boxes. That matters, because a `compile_ready`
  archive has had its decode tail surgically removed and emits raw heads -- so an app either
  decodes them on the host (slow: 143-337 ms/frame measured for seg/yolox) or lets Neat's
  model-route box-decode stage do it on-device (0.6 ms measured). Which BoxDecodeType can read
  a given archive is NOT predictable from the model's family name:

    * a compile_ready YOLO11 / YOLOv8 archive decodes as BoxDecodeType.YoloV26, NOT YoloV8 --
      the decode type follows the 6-tensor SURGERY CONTRACT, not the model name;
    * BoxDecodeType.YoloV8 is REJECTED AT BUILD TIME on those same archives
      ("could not validate grouped-by-role raw DFL output order");
    * YOLOX's 9-tensor decoupled head needs YoloX + Split3Interleaved.

  So this script PROBES: for every detection model it tries each candidate
  (BoxDecodeType, BoxDecodeTypeOption) pair, reports which ones build, and runs the ones that
  do against real images to confirm they produce sensible boxes. A type that builds but returns
  zero boxes on every image is a FAILURE, not a pass -- that is exactly how a wrong decode
  layout presents.

Output: a per-model matrix plus the winning decode type, which is what an app should set as
`ModelOptions.decode_type`.
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from common import ROOT, archive_path, load_registry, model_cfg

# Candidate decode types per registry `decode:` kind. Ordered best-guess first; the script
# tries them all anyway so a wrong guess here costs nothing but a few seconds.
CANDIDATES = {
    "yolo_raw_heads": [
        ("YoloV26", "Auto"),
        ("YoloV26", "GroupedByRole"),
        ("YoloV8", "Auto"),
        ("YoloV6", "Auto"),
    ],
    "yolox_split_heads": [
        ("YoloX", "Split3Interleaved"),
        ("YoloX", "Auto"),
    ],
}


def load_labels() -> list[str]:
    p = ROOT / "assets/labels/coco80.txt"
    if p.exists():
        return [x.strip() for x in p.read_text().splitlines() if x.strip()]
    return []


def images(project: dict, limit: int) -> list:
    d = ROOT / project.get("infer_dir", "assets/inference")
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:limit]


def letterbox(bgr, mw: int, mh: int, pad: int = 114):
    """Aspect-preserving resize onto an mw x mh canvas. Returns (canvas, scale, dx, dy)."""
    import cv2

    h0, w0 = bgr.shape[:2]
    scale = min(mw / w0, mh / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    dx, dy = (mw - nw) // 2, (mh - nh) // 2
    canvas = np.full((mh, mw, 3), pad, np.uint8)
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, dx, dy


def normalize_preset(pyneat, cfg):
    """Pick the Neat preprocess preset that reproduces the mean/std the archive was COMPILED with.

    The compiler applies (x/255 - mean) / std. So:
      mean=0, std=1        -> plain x/255            -> COCO_YOLO   (Ultralytics)
      std=1/255            -> cancels the /255, raw  -> None        (Megvii YOLOX)
    Getting this wrong is silent and total: YOLOX fed x/255 sees an image 255x too dark and
    detects NOTHING while still running at full speed.
    """
    std = [float(x) for x in cfg.get("std", [1, 1, 1])]
    if all(abs(s - 1.0 / 255.0) < 1e-6 for s in std):
        return getattr(pyneat.NormalizePreset, "None")
    return pyneat.NormalizePreset.COCO_YOLO


def try_decode(pyneat, cfg, dtype: str, dopt: str, imgs, conf, iou, top_k, num_classes,
               timeout_ms: int):
    """Build the archive with one decode type and run it. Returns (status, detail, per_image).

    Uses the IMAGE route (raw BGR in, Neat preprocesses), not the tensor route: the tensor
    route mis-negotiates caps against a box-decode stage and fails with
    `misconfig.caps ... InputStream::pull_and_discard`. On the image route Neat also inverts
    the box coordinates from its own preprocess metadata, so detections come back in ORIGINAL
    image space with no manual rescaling.
    """
    import cv2

    arc = archive_path(cfg["id"])
    _, _, mh, mw = cfg["input_shape"]

    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = normalize_preset(pyneat, cfg)
    opt.preprocess.input_max_width = mw
    opt.preprocess.input_max_height = mh
    opt.preprocess.input_max_depth = 3          # 0.3.0 enforces this; an explicit 1 aborts
    opt.decode_type = getattr(pyneat.BoxDecodeType, dtype)
    opt.decode_type_option = getattr(pyneat.BoxDecodeTypeOption, dopt)
    opt.score_threshold = conf
    opt.nms_iou_threshold = iou
    opt.top_k = top_k
    opt.num_classes = num_classes

    def tens(bgr):
        return pyneat.Tensor.from_numpy(np.ascontiguousarray(bgr, dtype=np.uint8), copy=True,
                                        image_format=pyneat.PixelFormat.BGR,
                                        memory=pyneat.TensorMemory.EV74)

    # appsrc caps are FIXED at build time from the seed tensor, so every pushed frame must be
    # exactly the seed's size -- a differently-sized frame does not error, it silently stalls the
    # pipeline until the pull times out. The sample images are 640x426/427/480 and 427x640, so we
    # letterbox each onto one 640x640 canvas and push that, then invert the letterbox ourselves.
    seed = np.zeros((mh, mw, 3), np.uint8)
    try:
        model = pyneat.Model(str(arc), opt)
        runner = model.build([tens(seed)])
        runner.run([tens(seed)], timeout_ms=timeout_ms)   # warmup; also surfaces
                                                          # a bad route as a timeout
    except Exception as e:                       # build-time rejection is the expected failure
        first = str(e).strip().splitlines()[0] if str(e).strip() else e.__class__.__name__
        return "BUILD-REJECTED", first[:96], []

    per_image = []
    try:
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            canvas, scale, dx, dy = letterbox(bgr, mw, mh)
            out = runner.run([tens(canvas)], timeout_ms=timeout_ms)
            boxes = pyneat.decode_bbox(out, clamp_to=(mw, mh), top_k=top_k)
            b = (np.asarray(boxes[0].to_numpy(copy=True)).reshape(-1, 6).copy()
                 if boxes else np.zeros((0, 6), np.float32))
            if b.size:                                  # canvas space -> original pixels
                b[:, [0, 2]] = (b[:, [0, 2]] - dx) / scale
                b[:, [1, 3]] = (b[:, [1, 3]] - dy) / scale
            per_image.append((p.name, b))
    except Exception as e:
        return "RUN-FAILED", str(e).strip().splitlines()[0][:96], per_image
    finally:
        try:
            runner.close()
        except Exception:
            pass

    total = sum(len(b) for _, b in per_image)
    if total == 0:
        return "ZERO-BOXES", "builds, but decodes nothing on any image", per_image
    return "OK", f"{total} boxes over {len(per_image)} images", per_image


def run_model(pyneat, model_id, project, args, labels) -> tuple[str | None, list]:
    cfg, _ = model_cfg(model_id)
    arc = archive_path(model_id)
    print(f"\n=== {model_id}  ({cfg['task']}, decode={cfg['decode']}) ===")
    if arc is None:
        print("    no _mpk.tar.gz -- compile it first")
        return None, []

    cands = CANDIDATES.get(cfg["decode"])
    if not cands:
        print(f"    no box-decode candidates for decode kind '{cfg['decode']}' -- skipping")
        return None, []

    imgs = images(project, args.limit)
    rows, winner, winner_dets = [], None, []
    for dtype, dopt in cands:
        status, detail, per_image = try_decode(pyneat, cfg, dtype, dopt, imgs,
                                               args.conf, args.iou, args.top_k, args.num_classes,
                                               args.timeout_ms)
        rows.append((dtype, dopt, status, detail))
        mark = {"OK": "PASS", "ZERO-BOXES": "FAIL", "BUILD-REJECTED": "----",
                "RUN-FAILED": "FAIL"}[status]
        print(f"    [{mark}] {dtype:<8} {dopt:<18} {status:<15} {detail}")
        if status == "OK" and winner is None:
            winner, winner_dets = f"{dtype}/{dopt}", per_image

    if winner:
        print(f"    -> USE BoxDecodeType.{winner.split('/')[0]}"
              + (f" (option {winner.split('/')[1]})" if winner.split('/')[1] != "Auto" else ""))
        for name, b in winner_dets[:args.limit]:
            top = sorted(b.tolist(), key=lambda r: -r[4])[:4]
            txt = ", ".join(
                f"{labels[int(r[5])] if int(r[5]) < len(labels) else int(r[5])} {r[4]:.2f}"
                for r in top)
            print(f"       {name:<22} {len(b):>3} boxes  {txt}")
    else:
        print("    -> NO candidate decode type works for this archive")
    return winner, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=3, help="images per model")
    ap.add_argument("--num-classes", type=int, default=80, help="COCO-80 for every model here")
    ap.add_argument("--timeout-ms", type=int, default=20000)
    args = ap.parse_args()

    import pyneat

    project, models = load_registry()
    if args.all:
        # DETECTION only. Segmentation and pose archives also carry the 6 detection heads, so
        # they would "pass" this probe on their box heads alone while their real contract
        # (YoloV26Seg + decode_segmentation, YoloV26Pose + decode_pose) goes untested. Ask for
        # them explicitly with --model-id if that is what you want.
        ids = [m["id"] for m in models
               if m.get("enabled", True) and m["task"] == "detection"
               and m["decode"] in CANDIDATES]
    elif args.model_id:
        ids = [args.model_id]
    else:
        raise SystemExit("give --model-id <id> or --all")

    results = {}
    for mid in ids:
        try:
            cfg, _ = model_cfg(mid)
            winner, _ = run_model(pyneat, mid, project, args, load_labels())
            results[mid] = (winner, cfg["task"])
        except SystemExit:
            raise
        except Exception:
            traceback.print_exc()
            results[mid] = (None, "unknown")

    print("\n" + "=" * 68)
    print(f"{'model':<20} {'on-device box decode':<28} verdict")
    print("-" * 68)
    for mid, (w, task) in results.items():
        verdict = "PASS" if w else ("n/a" if task != "detection" else "FAIL")
        print(f"{mid:<20} {(w or '-'):<28} {verdict}")

    # only a DETECTION model with no working decode type is a real failure
    bad = [m for m, (w, t) in results.items() if not w and t == "detection"]
    skipped = [m for m, (w, t) in results.items() if not w and t != "detection"]
    if skipped:
        print(f"\nnot a detection contract ({len(skipped)}): {', '.join(skipped)}"
              "\n  pose -> BoxDecodeType.YoloV26Pose + pyneat.decode_pose"
              "\n  seg  -> BoxDecodeType.YoloV26Seg  + pyneat.decode_segmentation")
    if bad:
        print(f"\nno working decode type ({len(bad)}): {', '.join(bad)}")
        return 1
    print("\nevery detection archive decodes on-device")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
