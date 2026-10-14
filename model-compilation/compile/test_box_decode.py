#!/usr/bin/env python3
"""STEP 4b - can Neat's ON-DEVICE box decoder consume this archive?

    # DevKit only (needs pyneat + the MLA)
    dk compile/test_box_decode.py --model-id yolov8s
    dk compile/test_box_decode.py --all

WHY this exists, separately from test_model.py:

  test_model.py proves the raw head tensors come back with the right SHAPES. It says nothing
  about whether Neat itself can turn them into boxes. A `compile_ready` archive has had its
  decode tail surgically removed and emits raw heads, so an app either decodes them on the host
  (slow: 143-337 ms/frame measured for seg/yolox) or lets Neat's model-route box-decode stage do
  it on-device (0.6 ms measured).

  Every model family decodes with ITS OWN BoxDecodeType, recorded in models.yaml as
  `box_decode` / `box_decode_option`:

    * YOLOv8 / YOLO11 (+ YOLO-World): raw 64-ch DFL bbox heads + class logits
        -> YoloV8 (YoloV8Seg for -seg), option GroupedByRoleLogit
    * YOLO26: one2one heads (4 distance channels), NMS-free by design
        -> YoloV26 (YoloV26Pose for -pose), option Auto
    * YOLOX: 9-tensor decoupled head -> YoloX, option Split3Interleaved

  This script builds each archive with exactly that pair and runs it on real images. It FAILS when
    * the build is rejected, or no image produces a box (wrong layout),
    * a score falls outside [0, 1] (logits read as probabilities: wrong option), or
    * same-class boxes overlapping above --iou are systematic (more than one pair per five boxes:
      NMS did not run). This last check is the one that matters on Neat >= 0.5.0, whose YoloV26
      decoder skips NMS: a one-to-many head decoded as YoloV26 returns a stack of near-identical
      boxes per object, and a bare "did it return boxes" check reports that as a pass. An isolated
      pair is printed as a warning.
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from common import ROOT, archive_path, load_registry, model_cfg

# int8 / bf16: which build of the model to test (None = the model's precision in models.yaml).
PRECISION: str | None = None

# Registry `decode:` kinds whose archives go through the on-device box decoder.
BOX_DECODE_KINDS = {"yolo_raw_heads", "yolox_split_heads"}

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
    `misconfig.caps ... InputStream::pull_and_discard`. Detections come back in the space of the
    pushed frame; this script pushes a host-letterboxed canvas (see below), so it inverts that
    letterbox itself.
    """
    import cv2

    arc = archive_path(cfg["id"], PRECISION)
    _, _, mh, mw = cfg["input_shape"]

    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = normalize_preset(pyneat, cfg)
    opt.preprocess.input_max_width = mw
    opt.preprocess.input_max_height = mh
    opt.preprocess.input_max_depth = 3          # Neat enforces this; an explicit 1 aborts
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
    runner = None
    try:
        model = pyneat.Model(str(arc), opt)
        runner = model.build([tens(seed)])
        runner.run([tens(seed)], timeout_ms=timeout_ms)   # warmup; also surfaces
                                                          # a bad route as a timeout
    except Exception as e:                       # build-time rejection is the expected failure
        first = str(e).strip().splitlines()[0] if str(e).strip() else e.__class__.__name__
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        return "BUILD-REJECTED", first[:96], []

    per_image = []
    try:
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            canvas, scale, dx, dy = letterbox(bgr, mw, mh)
            out = runner.run([tens(canvas)], timeout_ms=timeout_ms)
            # each decode type has its own payload reader; seg and pose results carry .boxes
            if dtype == "YoloV8Seg":
                boxes = [r.boxes for r in pyneat.decode_segmentation(out, clamp_to=(mw, mh), top_k=top_k)]
            elif dtype == "YoloV26Pose":
                boxes = [r.boxes for r in pyneat.decode_pose(out, clamp_to=(mw, mh), top_k=top_k)]
            else:
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


def iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def duplicate_pairs(b, iou: float) -> int:
    """Same-class box pairs overlapping above `iou`: what NMS should have removed."""
    rows = b.tolist()
    return sum(1 for i in range(len(rows)) for j in range(i + 1, len(rows))
               if int(rows[i][5]) == int(rows[j][5]) and iou_xyxy(rows[i], rows[j]) > iou)


SKIPPED = "skipped"     # run_model verdict for a model with no box_decode: not a failure


def run_model(pyneat, model_id, project, args, labels) -> tuple[str | None, list]:
    cfg, _ = model_cfg(model_id)
    arc = archive_path(model_id, PRECISION)
    dtype, dopt = cfg.get("box_decode"), cfg.get("box_decode_option", "Auto")
    print(f"\n=== {model_id}  ({cfg['task']}, BoxDecodeType.{dtype} / {dopt}) ===")
    if not dtype:
        print("    no box_decode in models.yaml -- skipping")
        return SKIPPED, []
    if arc is None:
        print("    no _mpk.tar.gz -- compile it first")
        return None, []

    imgs = images(project, args.limit)
    num_classes = int(cfg.get("num_classes", args.num_classes))   # pose decoders require 1
    status, detail, per_image = try_decode(pyneat, cfg, dtype, dopt, imgs, args.conf, args.iou,
                                           args.top_k, num_classes, args.timeout_ms)
    problems = [] if status == "OK" else [f"{status}: {detail}"]
    for name, b in per_image:
        dups = duplicate_pairs(b, args.iou) if len(b) else 0
        bad_scores = int(((b[:, 4] < 0) | (b[:, 4] > 1)).sum()) if len(b) else 0
        top = sorted(b.tolist(), key=lambda r: -r[4])[:4]
        txt = ", ".join(
            f"{labels[int(r[5])] if int(r[5]) < len(labels) else int(r[5])} {r[4]:.2f}"
            for r in top)
        print(f"       {name:<22} {len(b):>3} boxes  {dups:>3} duplicate pairs  {txt}")
        # Missing NMS shows as stacks (yolo11n through YoloV26: 254 pairs over 67 boxes). An
        # isolated pair is possible on a correct decode (a quantized one-to-one YOLO26 head, INT8 or
        # bf16, emits the odd near-duplicate that FP32 does not), so it is reported, not failed.
        if dups > max(1, len(b) // 5):
            problems.append(f"{name}: {dups} same-class pairs with IoU > {args.iou} (NMS did not run)")
        elif dups:
            print(f"    [WARN] {name}: {dups} overlapping same-class pair(s); isolated, not a missing-NMS stack")
        if bad_scores:
            problems.append(f"{name}: {bad_scores} scores outside [0, 1] (logits read as "
                            "probabilities -- check box_decode_option)")
    for p in problems:
        print(f"    [FAIL] {p}")
    verdict = None if problems else f"{dtype}/{dopt}"
    return verdict, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--precision", choices=["int8", "bf16"], default=None,
                    help="test the int8 or bf16 build (default: the model's precision in models.yaml)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=3, help="images per model")
    ap.add_argument("--num-classes", type=int, default=80,
                    help="class count when models.yaml gives none (COCO-80); pose models set 1")
    ap.add_argument("--timeout-ms", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="only 1 is supported: Neat 0.5.0's box decode handles one image per call")
    args = ap.parse_args()
    global PRECISION
    PRECISION = args.precision
    if args.batch_size != 1:
        # Measured on Neat 0.5.0 (yolo11n batch 6): the image route times out in preprocess, and
        # on the tensor route BoxDecode returns ONE result for the whole batch. Decoding a
        # batched archive needs host-side decode of the raw heads until Neat supports it.
        print(f"SKIP: batch {args.batch_size}: Neat 0.5.0's on-device box decode returns one "
              "result per call, not one per batch item, so it cannot check a batched archive.\n"
              f"      Check the batch-{args.batch_size} build on the DevKit with: dk compile/test_model.py "
              f"--model-id {getattr(args, 'model_id', None) or '<id>'}"
              f"{' --precision ' + args.precision if getattr(args, 'precision', None) else ''}"
              f" --batch-size {args.batch_size}   (compares every item's raw heads "
              "with the batch-1 build)")
        return 0

    import pyneat

    project, models = load_registry()
    if args.all:
        # Every archive with an on-device box decoder: detection, segmentation and pose. Seg and
        # pose results carry boxes too, so the duplicate / score checks apply to all three.
        ids = [m["id"] for m in models
               if m.get("enabled", True) and m["decode"] in BOX_DECODE_KINDS]
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
        verdict = "SKIP" if w == SKIPPED else "PASS" if w else "FAIL"
        print(f"{mid:<20} {(w if w and w != SKIPPED else '-'):<28} {verdict}")

    bad = [m for m, (w, _) in results.items() if not w]
    if bad:
        print(f"\nfailed ({len(bad)}): {', '.join(bad)}")
        return 1
    print("\nevery archive decodes on-device with its own BoxDecodeType")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
