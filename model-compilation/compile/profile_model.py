#!/usr/bin/env python3
"""Profile a compiled archive on the DevKit: image in -> image out, step by step, on real images.

    # batch 1 (Neat image route: preprocess -> MLA -> box decode on the device)
    dk compile/profile_model.py --model-id yolo26m --precision int8 --images <dir of 50+ jpgs>
    # batch 6 (tensor route; also profiles batch 1 on the same route, for a like-for-like table)
    dk compile/profile_model.py --model-id yolo26m --precision int8 --batch-size 6 --images <dir>

What is timed, per frame (ms; for a batch, the batch-level steps are also shown per call):

  read + JPEG decode      open the file, decode it to BGR (resize to the frame size if it differs)
  host preprocess         tensor route only: letterbox to the model size, RGB, (x/255-mean)/std
  to Neat tensor          wrap the frame (or the stacked batch) as a pyneat.Tensor in EV74 memory
  Neat run()              push -> the archive's stages -> pull, one call at a time
     per stage            Neat's own counters (SIMA_RUNTIME_PROFILE): preprocess / quantize on the
                          EV74, the MLA, dequantize / box decode
  boxes                   image route: decode_bbox() of Neat's result; tensor route: decode the raw
                          heads on the host (YOLO26: direct box distances, no NMS; YOLOv8/11: DFL + NMS)
  draw                    boxes and labels on the frame
  JPEG encode             the "image out" (the first --save frames are also written to --out)

Then a pipelined run (--depth calls in flight, inputs prepared in advance) for the throughput the
same route reaches when the host keeps the device busy.

Why two routes: on Neat 0.5.0 a batched archive cannot use the image route (its preprocess times
out) and its on-device box decode returns one result per call, so a batch-N archive runs on the
tensor route with host preprocess and host decode (README > Multi-batch compilation).

Why 120 calls: Neat prints its per-stage counters every 120 calls of an element, so each route
runs exactly --calls calls (a multiple of 120); the first --warmup calls are not host-timed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

from common import ROOT, archive_path, model_batch_size, model_cfg, model_precision

STEP_ORDER = ["read + JPEG decode", "host preprocess", "to Neat tensor", "Neat run()", "boxes",
              "draw", "JPEG encode"]


# ----------------------------------------------------------------------------- Neat stage counters
class StageCapture:
    """Redirect fd 2 to a file while a runner works, then parse Neat's [*-profile] lines."""

    LINE = re.compile(r"\[(?:runtime|boxdecode)-profile\] (?:component|plugin)=(\w+) stage=(\S+) "
                      r".*?(?:samples|frames)=(\d+) avg_ms\{([^}]*)\}")

    def __enter__(self):
        sys.stderr.flush()
        self.tmp = tempfile.TemporaryFile(mode="w+b")
        self.saved = os.dup(2)
        os.dup2(self.tmp.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.tmp.seek(0)
        self.text = self.tmp.read().decode("utf-8", "replace")
        self.tmp.close()
        return False

    def stages(self) -> list[dict]:
        latest: dict[str, dict] = {}
        for comp, stage, n, body in self.LINE.findall(getattr(self, "text", "")):
            fields = dict(kv.split("=", 1) for kv in body.split(",") if "=" in kv)
            row = {"component": comp, "stage": stage, "calls": int(n),
                   "total_ms": float(fields.get("total", "nan")),
                   "exec_ms": float(fields["dispatcher_exec"]) if "dispatcher_exec" in fields else None}
            if stage not in latest or row["calls"] >= latest[stage]["calls"]:
                latest[stage] = row
        return list(latest.values())


def stage_label(row: dict) -> str:
    comp, stage = row["component"], row["stage"]
    if comp == "processmla":
        return "MLA inference"
    if comp in ("neatobjectdecode", "boxdecode") or stage.startswith("boxdecode"):
        return "box decode"
    if stage.startswith(("preproc", "quanttess", "quant")):
        return "preprocess / quantize (EV74)"
    if stage.startswith(("postproc", "detess", "dequant")):
        return "dequantize (EV74)"
    return f"{comp} {stage}"


# ----------------------------------------------------------------------------- host-side helpers
def letterbox(bgr, mw, mh, pad=114):
    import cv2
    import numpy as np

    h0, w0 = bgr.shape[:2]
    s = min(mw / w0, mh / h0)
    nw, nh = int(round(w0 * s)), int(round(h0 * s))
    dx, dy = (mw - nw) // 2, (mh - nh) // 2
    canvas = np.full((mh, mw, 3), pad, np.uint8)
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, s, dx, dy


def nms(boxes, scores, iou):
    import numpy as np

    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        o = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][o <= iou]
    return keep


def decode_heads(heads, item, conf, iou, top_k, nms_needed):
    """Raw YOLO heads [bbox_0..2, class_logit_0..2] of one batch item -> [N,6] x1,y1,x2,y2,score,cls
    in model-input pixels. 4-channel bbox heads are box distances (YOLO26); 64 are DFL bins."""
    import numpy as np

    rows = []
    for k in range(3):
        box = heads[k][item] if heads[k].ndim == 4 else heads[k]
        cls = heads[k + 3][item] if heads[k + 3].ndim == 4 else heads[k + 3]
        h = box.shape[0]
        stride = heads_model_size[0] // h          # 640 / 80 = 8, / 40 = 16, / 20 = 32
        score = 1.0 / (1.0 + np.exp(-cls.max(-1)))
        keep = score > conf
        if not keep.any():
            continue
        gy, gx = np.nonzero(keep)
        b = box[gy, gx]
        if b.shape[-1] == 64:                      # DFL: softmax over 16 bins per side
            b = b.reshape(-1, 4, 16)
            e = np.exp(b - b.max(-1, keepdims=True))
            b = (e / e.sum(-1, keepdims=True)) @ np.arange(16, dtype=np.float32)
        cx, cy = (gx + 0.5) * stride, (gy + 0.5) * stride
        rows.append(np.stack([cx - b[:, 0] * stride, cy - b[:, 1] * stride,
                              cx + b[:, 2] * stride, cy + b[:, 3] * stride,
                              score[gy, gx], cls[gy, gx].argmax(-1)], 1))
    if not rows:
        return np.zeros((0, 6), np.float32)
    out = np.concatenate(rows)
    if nms_needed:
        kept = []
        for c in np.unique(out[:, 5]):
            idx = np.nonzero(out[:, 5] == c)[0]
            kept += idx[nms(out[idx, :4], out[idx, 4], iou)].tolist()
        out = out[kept]
    return out[np.argsort(-out[:, 4])][:top_k]


heads_model_size: tuple[int, int] | None = None


def draw(bgr, boxes, names):
    import cv2

    for x1, y1, x2, y2, score, cls in boxes:
        c = int(cls)
        cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(bgr, f"{names[c] if 0 <= c < len(names) else c} {score:.2f}",
                    (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)


def summary(xs):
    xs = sorted(xs)
    return {"mean": statistics.fmean(xs), "p50": xs[len(xs) // 2],
            "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))], "max": xs[-1], "n": len(xs)}


# ----------------------------------------------------------------------------- one route
def profile_route(route, arc, batch, cfg, frames, names, args, out_dir):
    """Run `args.calls` calls of `batch` frames through `route`; return the timing record."""
    import cv2
    import numpy as np
    import pyneat

    global heads_model_size
    _, _, mh, mw = cfg["input_shape"]
    heads_model_size = (mh, mw)
    fh, fw = frames[0][1].shape[:2]
    mean = np.array(cfg.get("mean", [0, 0, 0]), np.float32)
    std = np.array(cfg.get("std", [1, 1, 1]), np.float32)
    dtype = cfg.get("box_decode", "YoloV26")
    nms_needed = not dtype.startswith("YoloV26")

    opt = pyneat.ModelOptions()
    if route == "image":
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        opt.preprocess.input_max_width, opt.preprocess.input_max_height = fw, fh
        opt.preprocess.input_max_depth = 3
        opt.decode_type = getattr(pyneat.BoxDecodeType, dtype)
        opt.decode_type_option = getattr(pyneat.BoxDecodeTypeOption,
                                         cfg.get("box_decode_option", "Auto"))
        opt.score_threshold, opt.nms_iou_threshold = args.conf, args.iou
        opt.top_k, opt.num_classes = args.top_k, cfg.get("num_classes", 80)
    else:
        opt.preprocess.kind = pyneat.InputKind.Tensor
        opt.preprocess.input_max_width, opt.preprocess.input_max_height = mw, mh
        opt.preprocess.input_max_depth = 3

    def to_tensor(arr):
        if route == "image":
            return pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.BGR,
                                            memory=pyneat.TensorMemory.EV74)
        t = pyneat.Tensor.from_numpy(arr, copy=True, layout=pyneat.TensorLayout.HWC,
                                     memory=pyneat.TensorMemory.EV74)
        if arr.ndim == 4:
            A = pyneat.TensorAxisSemantic
            t.axis_semantics = [A.N, A.H, A.W, A.C]
        return t

    def host_pre(bgr):
        canvas, s, dx, dy = letterbox(bgr, mw, mh)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray((rgb - mean) / std, np.float32), (s, dx, dy)

    def seed():
        if route == "image":
            return frames[0][1].copy()
        x = host_pre(frames[0][1])[0]
        return np.stack([x] * batch) if batch > 1 else x

    steps = {k: [] for k in STEP_ORDER}
    per_call = {"to Neat tensor": [], "Neat run()": []}
    totals, nboxes, saved = [], [], 0
    model = pyneat.Model(str(arc), opt)
    with StageCapture() as cap:
        runner = model.build([to_tensor(seed())])
        fi = 0
        for call in range(args.calls):
            timed = call >= args.warmup
            t_frame = [0.0] * batch
            bgrs, metas, xs = [], [], []
            for b in range(batch):                     # host, per frame
                path, _ = frames[fi % len(frames)]
                fi += 1
                t0 = time.perf_counter()
                bgr = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if bgr.shape[:2] != (fh, fw):
                    bgr = cv2.resize(bgr, (fw, fh))
                t1 = time.perf_counter()
                if route == "tensor":
                    x, meta = host_pre(bgr)
                    xs.append(x)
                    metas.append(meta)
                t2 = time.perf_counter()
                bgrs.append((path, bgr))
                if timed:
                    steps["read + JPEG decode"].append((t1 - t0) * 1e3)
                    if route == "tensor":
                        steps["host preprocess"].append((t2 - t1) * 1e3)
                t_frame[b] += (t2 - t0) * 1e3
            t0 = time.perf_counter()                   # device, per call
            if route == "image":
                inp = to_tensor(bgrs[0][1])
            else:
                inp = to_tensor(np.stack(xs) if batch > 1 else xs[0])
            t1 = time.perf_counter()
            out = runner.run([inp], timeout_ms=30000)
            t2 = time.perf_counter()
            if timed:
                per_call["to Neat tensor"].append((t1 - t0) * 1e3)
                per_call["Neat run()"].append((t2 - t1) * 1e3)
            shared = (t2 - t0) * 1e3 / batch
            if route == "tensor":
                heads = [np.asarray(t.to_numpy(copy=True)) for t in out]
            for b, (path, bgr) in enumerate(bgrs):     # host, per frame
                t0 = time.perf_counter()
                if route == "image":
                    res = pyneat.decode_bbox(out, clamp_to=(fw, fh), top_k=args.top_k)
                    boxes = (np.asarray(res[0].to_numpy(copy=True)).reshape(-1, 6) if res
                             else np.zeros((0, 6)))
                else:
                    boxes = decode_heads(heads, b, args.conf, args.iou, args.top_k, nms_needed)
                    s, dx, dy = metas[b]
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / s
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / s
                t1 = time.perf_counter()
                draw(bgr, boxes, names)
                t2 = time.perf_counter()
                ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                t3 = time.perf_counter()
                t_frame[b] += (t3 - t0) * 1e3 + shared
                if timed:
                    steps["boxes"].append((t1 - t0) * 1e3)
                    steps["draw"].append((t2 - t1) * 1e3)
                    steps["JPEG encode"].append((t3 - t2) * 1e3)
                    totals.append(t_frame[b])
                    nboxes.append(len(boxes))
                if out_dir and saved < args.save:
                    (out_dir / f"{route}_b{batch}_{path.name}").write_bytes(jpg.tobytes())
                    saved += 1
        runner.close()

    for k in per_call:                                 # per-frame share of the per-call steps
        steps[k] = [v / batch for v in per_call[k]]

    # Pipelined: a fresh runner (its own counters), prepared inputs, `depth` calls in flight.
    prepared = []
    for i in range(min(8, len(frames))):
        bgr = frames[i][1]
        if route == "image":
            prepared.append(to_tensor(bgr.copy()))
        else:
            x = host_pre(bgr)[0]
            prepared.append(to_tensor(np.stack([x] * batch) if batch > 1 else x))
    with StageCapture():
        r2 = pyneat.Model(str(arc), opt).build([prepared[0]])
        for _ in range(5):
            r2.run([prepared[0]], timeout_ms=30000)
        n, pushed, pulled = args.calls, 0, 0
        t0 = time.perf_counter()
        while pulled < n:
            while pushed < n and pushed - pulled < args.depth:
                r2.push([prepared[pushed % len(prepared)]])
                pushed += 1
            r2.pull(30000)
            pulled += 1
        dt = time.perf_counter() - t0
        r2.close()

    return {
        "route": route, "batch": batch, "archive": str(arc), "calls": args.calls,
        "timed_frames": len(totals), "frame_size": [fw, fh],
        "steps_per_frame_ms": {k: summary(v) for k, v in steps.items() if v},
        "per_call_ms": {k: summary(v) for k, v in per_call.items()},
        "total_per_frame_ms": summary(totals),
        "stages_per_call_ms": cap.stages(),
        "boxes_per_frame": [min(nboxes), max(nboxes)],
        "pipelined": {"depth": args.depth, "calls_per_s": n / dt, "frames_per_s": n * batch / dt},
    }


def print_record(r):
    b = r["batch"]
    print(f"\n### {r['route']} route, batch {b}  ({Path(r['archive']).parent.parent.name})")
    print(f"{r['timed_frames']} frames timed ({r['calls']} calls, frame {r['frame_size'][0]}x"
          f"{r['frame_size'][1]}); boxes per frame {r['boxes_per_frame'][0]}-{r['boxes_per_frame'][1]}\n")
    print("| step (per frame) | mean ms | p50 ms | p95 ms |")
    print("|---|---|---|---|")
    for k in STEP_ORDER:
        s = r["steps_per_frame_ms"].get(k)
        if s:
            note = f" (per call {r['per_call_ms'][k]['mean']:.2f})" if b > 1 and k in r["per_call_ms"] else ""
            print(f"| {k}{note} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} |")
    t = r["total_per_frame_ms"]
    print(f"| **total** | **{t['mean']:.2f}** | {t['p50']:.2f} | {t['p95']:.2f} |")
    print(f"\nend to end, one call at a time: {1000 / t['mean']:.1f} frames/s")
    if r["stages_per_call_ms"]:
        print("\n| inside Neat run(), per call | avg ms | calls |")
        print("|---|---|---|")
        for row in r["stages_per_call_ms"]:
            print(f"| {stage_label(row)} (`{row['stage']}`) | {row['total_ms']:.2f} | {row['calls']} |")
    p = r["pipelined"]
    print(f"\npipelined ({p['depth']} calls in flight, inputs prepared): {p['frames_per_s']:.1f} frames/s "
          f"({p['calls_per_s']:.1f} calls/s)")


def main() -> int:
    os.environ.setdefault("SIMA_RUNTIME_PROFILE", "1")   # Neat's per-stage counters
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--precision", choices=["int8", "bf16"], default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="profile the batch-N build (default: models.yaml batch_size, else 1)")
    ap.add_argument("--images", type=Path, required=True,
                    help="real images, all the camera's size (e.g. 1280x720 frames)")
    ap.add_argument("--min-images", type=int, default=50)
    ap.add_argument("--calls", type=int, default=120, help="calls per route (multiple of 120)")
    ap.add_argument("--warmup", type=int, default=10, help="first calls not host-timed")
    ap.add_argument("--depth", type=int, default=4, help="calls in flight for the pipelined run")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None, help="write the first --save annotated frames here")
    ap.add_argument("--save", type=int, default=6)
    ap.add_argument("--json", type=Path, default=None, help="write all numbers here")
    args = ap.parse_args()
    if args.calls < 120 or args.calls % 120:
        raise SystemExit("--calls must be a positive multiple of 120 (Neat prints its stage counters "
                         "every 120 calls)")
    if not 0 <= args.warmup < args.calls:
        raise SystemExit("--warmup must be >= 0 and smaller than --calls")

    import cv2
    import numpy as np

    cfg, _ = model_cfg(args.model_id)
    if cfg.get("box_decode") not in ("YoloV8", "YoloV26"):
        # decode_heads() reads the 6 raw heads [bbox_0..2, class_logit_0..2] of a YOLOv8/YOLO11 or
        # YOLO26 detector; seg/pose/YOLOX/classification outputs are laid out differently.
        raise SystemExit(f"{args.model_id}: profile_model.py supports YOLO detection models "
                         f"(box_decode YoloV8 or YoloV26), not box_decode={cfg.get('box_decode')}")
    prec = model_precision(args.model_id, args.precision)
    batch = model_batch_size(args.model_id, args.batch_size)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) < args.min_images:
        raise SystemExit(f"{len(paths)} images in {args.images}; profile on at least {args.min_images} "
                         "real images (--min-images)")
    frames = [(p, cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)) for p in paths]
    labels = ROOT / "assets/labels/coco80.txt"
    names = [x.strip() for x in labels.read_text().splitlines() if x.strip()] if labels.exists() else []
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    plan = []
    if batch == 1:
        plan.append(("image", 1))
    else:
        plan += [("tensor", 1), ("tensor", batch)]
    records = []
    for route, b in plan:
        arc = archive_path(args.model_id, prec, b)
        if arc is None:
            raise SystemExit(f"no {prec} batch-{b} archive for {args.model_id}; compile it first "
                             f"(compiler.py --model-id {args.model_id} --precision {prec}"
                             f"{'' if b == 1 else f' --batch-size {b}'})")
        print(f"[profile] {args.model_id} {prec} batch {b}, {route} route, {len(paths)} images, "
              f"{args.calls} calls ...", flush=True)
        records.append(profile_route(route, arc, b, cfg, frames, names, args, args.out))
    print(f"\n## {args.model_id} {prec}: {len(paths)} images from {args.images}")
    for r in records:
        print_record(r)
    if args.json:
        args.json.write_text(json.dumps({"model": args.model_id, "precision": prec, "images": len(paths),
                                         "records": records}, indent=2))
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
