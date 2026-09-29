#!/usr/bin/env python3
"""Measure the accuracy cost of INT8 / bf16 against FP32 on a COCO val2017 subset.

Every precision goes through the SAME preprocessing (letterbox to 640), the SAME raw-head decoder
and the SAME COCO scorer, so the only thing that differs between the numbers is the precision.

    # 1. FP32 reference - SDK host, onnxruntime on the compile-ready ONNX
    python accuracy/eval_coco.py detect --model-id yolo26s --backend onnx

    # 2. INT8 and bf16 - DevKit, the compiled archives through pyneat
    python accuracy/eval_coco.py detect --model-id yolo26s --backend neat --precision int8
    python accuracy/eval_coco.py detect --model-id yolo26s --backend neat --precision bf16

    # 3. Score everything - SDK host (needs pycocotools)
    python accuracy/eval_coco.py score

`detect` writes accuracy/results/<model>_<precision>.json (COCO detection format). `score` runs
the standard COCO evaluation on each file and prints one table.

Works for the YOLO26 detection models in models.yaml (raw one2one heads, 4 box + 80 class
channels per scale). Run accuracy/prepare_coco_subset.py first.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent                   # model-compilation/
sys.path.insert(0, str(ROOT_DIR / "compile"))
from common import archive_path, compile_input_onnx, model_precision  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
ANN = DATA / "instances_val2017_subset.json"
IMG_DIR = DATA / "val2017_subset"
IMGSZ = 640
STRIDES = {80: 8, 40: 16, 20: 32}     # head grid size -> stride, for a 640x640 input


# ----------------------------------------------------------------------------- preprocessing
def letterbox(bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Ultralytics-style letterbox: keep aspect ratio, pad to 640x640 with gray 114.

    Returns the model input (HWC, RGB, float32 in [0, 1]), the scale and the (x, y) padding, so
    boxes can be mapped back to the original image.
    """
    import cv2

    h, w = bgr.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (IMGSZ - nw) / 2, (IMGSZ - nh) / 2
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb), r, (left, top)


# ----------------------------------------------------------------------------- decoding
def decode(heads: list[np.ndarray], channels_last: bool, conf: float, max_det: int):
    """YOLO26 one2one raw heads -> (boxes xyxy in 640 space, scores, class ids).

    Each scale has a 4-channel box head (left/top/right/bottom distances from the cell centre, in
    stride units) and an 80-channel class head (logits). YOLO26 is NMS-free: take the top
    `max_det` (anchor, class) scores, as Ultralytics does.
    """
    box_heads, cls_heads = {}, {}
    for a in heads:
        a = np.asarray(a, dtype=np.float32)
        a = a.reshape(a.shape[-3:]) if a.ndim == 4 else a
        if not channels_last:                      # CHW -> HWC
            a = np.transpose(a, (1, 2, 0))
        g, c = a.shape[0], a.shape[2]
        (box_heads if c == 4 else cls_heads)[g] = a
    boxes, scores = [], []
    for g in sorted(box_heads, reverse=True):      # 80, 40, 20
        s = STRIDES[g]
        ys, xs = np.mgrid[0:g, 0:g].astype(np.float32) + 0.5
        d = box_heads[g].reshape(-1, 4)
        cx, cy = xs.reshape(-1), ys.reshape(-1)
        boxes.append(np.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], 1) * s)
        scores.append(1.0 / (1.0 + np.exp(-cls_heads[g].reshape(-1, cls_heads[g].shape[2]))))
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    flat = scores.reshape(-1)
    k = min(max_det, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[flat[idx] >= conf]
    anchor, cls = np.divmod(idx, scores.shape[1])
    return boxes[anchor], flat[idx], cls


# ----------------------------------------------------------------------------- backends
class OnnxBackend:
    """FP32 reference: the compile-ready ONNX (same graph the compiler sees) on onnxruntime."""
    channels_last = False

    def __init__(self, model_id: str):
        import onnxruntime as ort
        self.path = compile_input_onnx(model_id)
        self.sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray) -> list[np.ndarray]:
        return self.sess.run(None, {self.inp: rgb.transpose(2, 0, 1)[None]})


class NeatBackend:
    """INT8 / bf16: the compiled archive on the DevKit MLA, fed the same float tensor."""
    channels_last = True

    def __init__(self, model_id: str, precision: str):
        import pyneat
        self.pyneat = pyneat
        self.path = archive_path(model_id, precision)
        if self.path is None:
            raise SystemExit(f"no {precision} archive for {model_id}; compile it first")
        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Tensor
        opt.preprocess.input_max_width = IMGSZ
        opt.preprocess.input_max_height = IMGSZ
        opt.preprocess.input_max_depth = 3
        self.model = pyneat.Model(str(self.path), opt)
        self.runner = self.model.build([self._tensor(np.zeros((IMGSZ, IMGSZ, 3), np.float32))])

    def _tensor(self, a: np.ndarray):
        p = self.pyneat
        return p.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32), copy=True,
                                   layout=p.TensorLayout.HWC, memory=p.TensorMemory.EV74)

    def __call__(self, rgb: np.ndarray) -> list[np.ndarray]:
        return [np.asarray(t.to_numpy(copy=True)) for t in self.runner.run([self._tensor(rgb)])]


# ----------------------------------------------------------------------------- commands
def result_path(model_id: str, precision: str, limit: int) -> Path:
    """results/<model>_<precision>.json; a --limit run gets its own file so it never replaces a full run."""
    return RESULTS / (f"{model_id}_{precision}_n{limit}.json" if limit else f"{model_id}_{precision}.json")


def artifact_name(path: Path) -> str:
    """The archive or ONNX path relative to model-compilation/, so the precision folder is recorded."""
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def cmd_detect(args) -> int:
    import cv2

    coco = json.loads(ANN.read_text())
    cat_ids = [c["id"] for c in sorted(coco["categories"], key=lambda c: c["id"])]  # 80 -> COCO id
    images = coco["images"][: args.limit] if args.limit else coco["images"]

    if args.backend == "onnx":
        if args.precision:
            raise SystemExit("--precision applies to --backend neat only (onnx is always FP32)")
        backend, precision = OnnxBackend(args.model_id), "fp32"
    else:
        precision = model_precision(args.model_id, args.precision)
        backend = NeatBackend(args.model_id, precision)
    print(f"[detect] {args.model_id} {precision}: {backend.path.name}, {len(images)} images")

    dets, infer_ms = [], []
    for i, im in enumerate(images, 1):
        bgr = cv2.imread(str(args.image_dir / im["file_name"]))
        if bgr is None:
            raise SystemExit(f"cannot read {args.image_dir / im['file_name']}")
        rgb, r, (px, py) = letterbox(bgr)
        t0 = time.perf_counter()
        heads = backend(rgb)
        infer_ms.append((time.perf_counter() - t0) * 1000)
        boxes, scores, cls = decode(heads, backend.channels_last, args.conf, args.max_det)
        boxes = (boxes - [px, py, px, py]) / r                       # back to original pixels
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, im["width"])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, im["height"])
        for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), cls.tolist()):
            dets.append({"image_id": im["id"], "category_id": cat_ids[c],
                         "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                         "score": round(s, 5)})
        if i % 100 == 0 or i == len(images):
            print(f"   {i}/{len(images)}  median inference {np.median(infer_ms):.1f} ms", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = result_path(args.model_id, precision, args.limit)
    out.write_text(json.dumps({"model": args.model_id, "precision": precision,
                               "artifact": artifact_name(backend.path), "images": len(images),
                               "image_ids": [im["id"] for im in images],
                               "median_infer_ms": round(float(np.median(infer_ms)), 2),
                               "detections": dets}))
    print(f"[detect] wrote {len(dets)} detections -> {out}")
    return 0


def cmd_score(args) -> int:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    files = [Path(f) for f in args.files] or sorted(RESULTS.glob("*.json"))
    if not files:
        raise SystemExit(f"no result files in {RESULTS}; run `detect` first")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN))
    order = {"pytorch": 0, "fp32": 1, "bf16": 2, "int8": 3}
    runs = sorted((json.loads(f.read_text()) for f in files),
                  key=lambda r: (r["model"], order.get(r["precision"], 9)))

    rows, base = [], {}
    for r in runs:
        with contextlib.redirect_stdout(io.StringIO()):
            if r["detections"]:
                ev = COCOeval(gt, gt.loadRes(r["detections"]), "bbox")
                # Score exactly the images the run covered (older result files predate image_ids;
                # they always covered the first N images of the sorted subset).
                ev.params.imgIds = r.get("image_ids") or sorted(
                    im["id"] for im in gt.dataset["images"])[: r["images"]]
                ev.evaluate(); ev.accumulate(); ev.summarize()
                m, m50 = ev.stats[0] * 100, ev.stats[1] * 100
            else:
                m = m50 = 0.0                  # no detections at all: nothing to evaluate
        if r["precision"] == "fp32":
            base[r["model"]] = m
        rows.append((r["model"], r["precision"], m, m50, r["images"], r["median_infer_ms"]))

    print("\nvs FP32 = difference from the fp32 row (compile-ready ONNX, same decoder as bf16/int8)")
    print(f"\n{'model':<10}{'precision':<11}{'mAP50-95':>9}{'mAP50':>8}{'vs FP32':>9}"
          f"{'images':>8}   median inference")
    for model, prec, m, m50, n, ms in rows:
        delta = f"{m - base[model]:+.2f}" if model in base and prec != "fp32" else "-"
        where = {"pytorch": "host CPU, Ultralytics", "fp32": "host CPU"}.get(prec, "DevKit MLA")
        print(f"{model:<10}{prec:<11}{m:9.2f}{m50:8.2f}{delta:>9}{n:8d}   {ms:.1f} ms ({where})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="run one model/precision over the subset")
    d.add_argument("--model-id", required=True)
    d.add_argument("--backend", choices=["onnx", "neat"], required=True,
                   help="onnx = FP32 on the host; neat = compiled archive on the DevKit")
    d.add_argument("--precision", choices=["int8", "bf16"], default=None,
                   help="which compiled archive (neat backend only)")
    d.add_argument("--conf", type=float, default=0.001, help="score threshold (low, for mAP)")
    d.add_argument("--max-det", type=int, default=300)
    d.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    d.add_argument("--image-dir", type=Path, default=IMG_DIR,
                   help="where the subset images are (default accuracy/data/val2017_subset). On the "
                        "DevKit, point this at a local copy to avoid reading them over NFS")
    s = sub.add_parser("score", help="COCO mAP for result files (default: all in results/)")
    s.add_argument("files", nargs="*")
    args = ap.parse_args()
    return cmd_detect(args) if args.cmd == "detect" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
