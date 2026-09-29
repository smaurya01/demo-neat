# Accuracy: FP32 vs bf16 vs INT8 on the MLA

A model is trained in FP32 (32-bit floating point). The MLA does not run FP32, so the compiler turns
the model into one of two lower-precision forms:

- **bf16** — 16-bit floating point. Same number range as FP32, fewer digits of precision.
- **INT8** — 8-bit integers. Each tensor's values are mapped onto 256 steps, using ranges the
  compiler learns from **calibration images**.

Both lose some information, so every deployment has to ask: **how much accuracy does compiling
cost?** This page answers that for Ultralytics `yolo26s` on COCO val2017 (4,980 of its 5,000
images), and explains how the answer relates to the accuracy Ultralytics publishes.

**To replicate these numbers**, follow [`accuracy/README.md`](accuracy/README.md): it has every
step, from downloading the model and COCO val2017 to compiling both precisions, running each row on
the host or the DevKit, and scoring — with the expected output of each step.

## Contents

- [The answer](#the-answer)
- [What was compared](#what-was-compared)
- [Results](#results)
- [How the comparison is kept fair](#how-the-comparison-is-kept-fair)
- [Why this differs from the accuracy Ultralytics publishes](#why-this-differs-from-the-accuracy-ultralytics-publishes)
- [Choosing a precision](#choosing-a-precision)
- [Limits](#limits)
- [Reproduce it](#reproduce-it)

---

## The answer

| Precision | Accuracy cost vs FP32 (mAP50-95) | Speed on the MLA | Archive size |
| --- | --- | --- | --- |
| **bf16** | **−0.07** — no measurable loss | 18.4 ms per inference | 40 MB |
| **INT8** | **−2.67** — about 5.6% of the model's accuracy | 11.1 ms per inference (1.7× faster) | 23 MB |

For `yolo26s` on COCO: **bf16 keeps FP32 accuracy; INT8 trades 2.67 mAP for 1.7× the speed.**

---

## What was compared

One model, four ways of running it:

| Row | What runs | Where |
| --- | --- | --- |
| **Ultralytics** | The original `yolo26s.pt`, through Ultralytics' own Python pipeline (`predict()`): its preprocessing, PyTorch, its box decoding | SDK host CPU |
| **FP32** | The same model after export to ONNX and this repository's graph surgery (`yolo26s.compile_ready.onnx`), in `onnxruntime` | SDK host CPU |
| **bf16** | That compile-ready ONNX, compiled with `compiler.py --precision bf16` | DevKit MLA |
| **INT8** | That compile-ready ONNX, compiled with `compiler.py --precision int8` (20 real COCO calibration images, `mse`) | DevKit MLA |

The model is Ultralytics' published `yolo26s.pt` (release dated 2026-01-05, 10.0 M parameters,
AGPL-3.0), downloaded by `compile/convert_to_onnx.py`. The bf16 and INT8 archives were compiled from
it in the Neat SDK 2.1.3 (Model Compiler 2.1.0) and run on a Modalix DevKit with Neat 0.4.0.

**Dataset:** COCO val2017 — **4,980 images, 36,178 objects**. That is all 5,000 val2017 images
except the 20 in `assets/calibration/`, which the INT8 build was calibrated on; a model should not be
scored on the images it was calibrated with.

---

## Results

| Row | mAP50-95 | mAP50 | vs FP32 (mAP50-95) | Runs on | Inference |
| --- | --- | --- | --- | --- | --- |
| Ultralytics | 47.41 | 63.85 | −0.30 | SDK host CPU | 171.9 ms |
| **FP32** (the reference) | **47.71** | 64.31 | — | SDK host CPU | 231.6 ms |
| **bf16** | **47.64** | 64.26 | **−0.07** | DevKit MLA | **18.4 ms** |
| **INT8** | **45.05** | 61.82 | **−2.67** | DevKit MLA | **11.1 ms** |

How to reproduce each row, in [`accuracy/README.md`](accuracy/README.md): step 3 produces the FP32
and Ultralytics rows, step 5 the bf16 row, step 6 the INT8 row, and step 7 prints this table.

| Term | Meaning |
| --- | --- |
| **mAP50-95** | The standard COCO score: average precision over the 80 classes, averaged over box-overlap (IoU) thresholds 0.50, 0.55, … 0.95. Rewards boxes that are *tight* |
| **mAP50** | The same at one IoU threshold, 0.50. Rewards finding the object at all |
| **vs FP32** | The row's mAP50-95 minus the FP32 row's. Negative means accuracy was lost |
| **Inference** | Median time per image. DevKit rows: one `runner.run()` call — tensor copy to the device, the EV74 cast or quantize steps, the MLA, and the steps back; letterbox and box decoding excluded. Host rows are CPU times, shown only to say where the reference ran — not comparable with the DevKit |

**What the results say:**

1. **bf16 is lossless in practice.** −0.07 mAP50-95 and −0.05 mAP50 over nearly 5,000 images is
   noise, not a trend.
2. **INT8 costs 2.67 mAP50-95 and 2.49 mAP50.** The loss is a little larger at strict IoU
   thresholds: INT8 finds almost the same objects but places some boxes less precisely.
3. **INT8 is 1.7× faster than bf16** on the MLA (11.1 ms against 18.4 ms), and its archive is
   smaller (23 MB against 40 MB).
4. **Exporting and preparing the model did not cost accuracy.** The FP32 row (after ONNX export and
   graph surgery) scores 0.3 *above* the Ultralytics row, not below it. That gap most likely comes
   from preprocessing, [explained below](#why-this-differs-from-the-accuracy-ultralytics-publishes).

An earlier run on the first 500 of these images showed the same picture — `yolo26s` bf16 −0.12,
INT8 −2.60 — and also measured **`yolo26m` in bf16: 55.55 against 55.42 for FP32 (+0.13)**, at
30.9 ms per inference.

---

## How the comparison is kept fair

Between the FP32, bf16 and INT8 rows, **only the precision changes**:

| Stage | Identical for FP32, bf16 and INT8 |
| --- | --- |
| Model graph | The same `work/yolo26s/surgery/yolo26s.compile_ready.onnx`. FP32 runs it; bf16 and INT8 are compiled from it |
| Images | The same 4,980 images |
| Preprocessing | The same letterbox: keep the aspect ratio, resize the long side to 640, pad with gray (114) to 640×640, RGB, divide by 255 |
| Decoding | The same Python decoder turns the model's six raw output heads into boxes (YOLO26 is NMS-free: the 300 highest scores are kept). Before any run, it was checked against Ultralytics' own decoder on the original export: identical boxes, scores and classes (a one-off check; there is no script for it here) |
| Scoring | `pycocotools`, the official COCO evaluator. Score threshold 0.001, at most 300 boxes per image — the usual settings for mAP |

FP32 runs on the host because the MLA cannot run FP32. bf16 and INT8 run on the DevKit through
`pyneat`, which returns the raw heads (Neat's own on-device box decode is not used), so all three go
through the same decoder.

Two checks back the setup:

- **Reproducibility.** The INT8 run was repeated — once reading the images over the shared
  workspace, once from the DevKit's NVMe. The detections were byte-for-byte identical.
- **An independent reference.** The Ultralytics row uses none of this repository's code: not its
  export, surgery, letterbox or decoder. It lands within 0.3 mAP of the FP32 row.

---

## Why this differs from the accuracy Ultralytics publishes

Ultralytics publishes **47.8 mAP50-95** for `yolo26s`. The Ultralytics row here is **47.41**, and
the FP32 row **47.71**. The model is the same; the measurement is not. Each of these differences
could plausibly move mAP50-95 by a few tenths of a point. None was measured on its own here:

| Difference | Published number | This comparison |
| --- | --- | --- |
| **How the model is run** | `yolo val`: images grouped by aspect ratio into batches of 32, each batch resized to one shared shape with some extra padding | Ultralytics row: `predict()`, one image at a time, padded only up to a multiple of 32. FP32/bf16/INT8 rows: every image letterboxed to a full 640×640 |
| **Images** | All 5,000 val2017 images | 4,980 — the 20 calibration images are left out |
| **Evaluator** | Ultralytics' built-in mAP code | `pycocotools`. The two treat crowd regions and the precision/recall curve slightly differently, and are known to disagree by a few tenths |
| **Which published column** | The YOLO26 docs table lists a standard mAP and an end-to-end (NMS-free) mAP | The end-to-end (NMS-free) head, so the fair comparison is with the end-to-end column |
| **Hardware and version** | Usually measured on a GPU, often in FP16 | CPU in FP32, Ultralytics 8.4.136 (the checkpoint was saved with 8.3.222) |

**Preprocessing is the likely main cause.** The same `.pt` scores 47.41 through `predict()`, and the
same weights score 47.71 through a fixed 640×640 letterbox. Since the decoder was checked to match
Ultralytics' own, that 0.3 gap is most likely input size and padding. The FP32 row sits 0.09 below
the published 47.8.

**None of this affects the conclusions.** bf16 and INT8 are compared with the FP32 row, which shares
their images, preprocessing, decoder and evaluator exactly. −0.07 and −2.67 are what compilation
costs, whatever pipeline produced the absolute scale.

**To reproduce the published number itself** (not done here), run Ultralytics' own validator on all
5,000 images:

```bash
yolo val model=yolo26s.pt data=coco-val.yaml imgsz=640 batch=32 conf=0.001 save_json=True
```

`coco-val.yaml` is a val-only dataset file you write, with the labels converted to Ultralytics'
YOLO text format from `accuracy/data/annotations/instances_val2017.json`. Ultralytics' standard
`coco.yaml` would also download the 18 GB training set. With `save_json=True` the validator reports
both its own mAP and the `pycocotools` mAP, which would show how much each difference above
contributes.

---

## Choosing a precision

| If | Use |
| --- | --- |
| Accuracy matters most, and 18 ms per `yolo26s` frame is fast enough | **bf16** — no 8-bit ranges to get wrong, no measurable accuracy loss |
| You need the speed, and 2–3 mAP is acceptable for your task | **INT8** — then check it on your own data, and try more calibration images |
| INT8 fails to compile or loses far more than this | **bf16** — e.g. `yolov8s-worldv2`, which [fails INT8 outright](MODEL-COMPILATION.md#-int8-for-yolo-world--passes-placement-fails-the-sim-check) |

Build either with the same four steps; only `compiler.py --precision` changes
([COMPILE-COMMANDS.md, section 14](COMPILE-COMMANDS.md#14-yolo26s--detection-int8-and-bf16-accuracy-study)).

---

## Limits

- **One calibration setup.** INT8 used this repository's default: 20 real COCO images, `mse`
  calibration. More or different calibration images may narrow the 2.67 gap
  (`compiler.py --calib-dir <dir> --num-calib-samples N`, with images outside the evaluation set).
  Not measured here.
- **One model.** Other architectures react to INT8 very differently — measure the model you deploy.
- **COCO is not your data.** If your scenes, objects or lighting differ, score a sample of your own
  labelled images.
- **Absolute mAP depends on the pipeline** (see the section above). Compare precisions within one
  pipeline, as this page does, not across pipelines.

---

## Reproduce it

Step-by-step commands — download the model, compile both precisions, download COCO val2017, run
FP32 and the Ultralytics reference on the host, run bf16 and INT8 on the DevKit, score — are in
**[`accuracy/README.md`](accuracy/README.md)**. The whole study takes about an hour and a half,
most of it the two host CPU runs.
