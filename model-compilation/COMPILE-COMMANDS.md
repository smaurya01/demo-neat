# Compile commands

One section per model: the commands, then the output to expect. Do the
[Setup](README.md#setup) first, and compile **one model at a time**.

Every section has the same shape:

- **Commands**: the four steps, run with `python` on the SDK host.
- **Host**: what the compile and the archive check print. Look for `A65 : 0` and `[PASS] ... elf=1 so=0`.
- **DevKit**: the real-image test, run with `dk`.

The notes under a model only say what is different about it; each links to the full explanation.

## Table of Contents

- [Compile a single model](#compile-a-single-model)
  - [1. `resnet50` — classification, no surgery](#1-resnet50--classification-no-surgery)
  - [2. `convnext_tiny` — classification, no surgery, **bf16**](#2-convnext_tiny--classification-no-surgery-bf16)
  - [3. `densenet169` — classification, no surgery](#3-densenet169--classification-no-surgery)
  - [4. `efficientnet_v2_s` — classification, no surgery, 384×384 input](#4-efficientnet_v2_s--classification-no-surgery-384384-input)
  - [5. `yolov8s` — detection, surgery (**head at `model.22`, no attention**)](#5-yolov8s--detection-surgery-head-at-model22-no-attention)
  - [6. `yolo11n` — detection, surgery](#6-yolo11n--detection-surgery)
  - [7. `yolo11s` — detection, surgery](#7-yolo11s--detection-surgery)
  - [8. `yolo26n` — detection, surgery (**one-to-one head, `YoloV26`**)](#8-yolo26n--detection-surgery-one-to-one-head-yolov26)
  - [9. `yolo11s-seg` — segmentation, surgery](#9-yolo11s-seg--segmentation-surgery)
  - [10. `yolo26s-pose` — pose, surgery (**carries the 209× fix**)](#10-yolo26s-pose--pose-surgery-carries-the-209-fix)
  - [11. `yolox_s` — detection, **different surgery**](#11-yolox_s--detection-different-surgery)
  - [12. `yolov8s-worldv2` — open-vocabulary, **bf16 not INT8**](#12-yolov8s-worldv2--open-vocabulary-bf16-not-int8)
    - [Change the vocabulary](#change-the-vocabulary)
  - [13. `yolo26m` — detection, **bf16**](#13-yolo26m--detection-bf16)
  - [14. `yolo26s` — detection, **INT8 and bf16** (accuracy study)](#14-yolo26s--detection-int8-and-bf16-accuracy-study)
- [Compile every model](#compile-every-model)
- [Other builds](#other-builds)

---

## Compile a single model

One model per section. Nothing is downloaded by hand — **step 1 fetches the weights for you**.

### 1. `resnet50` — classification, no surgery

```bash
python compile/convert_to_onnx.py --model-id resnet50      # downloads torchvision weights -> 102 MB ONNX
python compile/graph_surgery.py   --model-id resnet50      # prints "kind=none ... skipping"
python compile/compiler.py        --model-id resnet50      # the long step; ≈3 min
python compile/test_model.py      --model-id resnet50 --validate-only
```

**Host:**

```text
[compile] resnet50: precision=int8 batch=1 onnx=resnet50.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] resnet50 (int8): rc=0  -> .../model-compilation/work/resnet50/compile_int8
[PASS] resnet50         elf=1 so=0 batch=1  (compile_int8/resnet50_mpk.tar.gz)
```

**DevKit:**

```bash
dk compile/test_model.py --model-id resnet50
```

```text
[PASS] resnet50         elf=1 so=0 batch=1  (compile_int8/resnet50_mpk.tar.gz)
[test] resnet50: imagenet_topk on 3 real image(s) from inference/
   000000000139.jpg       home theater 0.07, television 0.06, restaurant 0.04
   000000000885.jpg       racket 0.67, tennis ball 0.02, book jacket 0.01
   000000001000.jpg       racket 0.12, miniskirt 0.08, ballplayer 0.03
```

---

### 2. `convnext_tiny` — classification, no surgery, **bf16**

```bash
python compile/convert_to_onnx.py --model-id convnext_tiny   # -> 114 MB ONNX
python compile/graph_surgery.py   --model-id convnext_tiny   # skipped (surgery: none)
python compile/compiler.py        --model-id convnext_tiny   # precision: bf16; the long step, ≈8 min
python compile/test_model.py      --model-id convnext_tiny --validate-only
```

> Built as **bf16**: its INT8 build loses most of its accuracy ([why](PRECISION.md#why-these-three-are-bf16)).

**Host:**

```text
[compile] convnext_tiny: precision=bf16 batch=1 onnx=convnext_tiny.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] convnext_tiny (bf16): rc=0  -> .../model-compilation/work/convnext_tiny/compile_bf16
[PASS] convnext_tiny    elf=1 so=0 batch=1  (compile_bf16/convnext_tiny_mpk.tar.gz)
```

**DevKit:**

```bash
dk compile/test_model.py --model-id convnext_tiny
```

```text
[PASS] convnext_tiny    elf=1 so=0 batch=1  (compile_bf16/convnext_tiny_mpk.tar.gz)
[test] convnext_tiny: imagenet_topk on 3 real image(s) from inference/
   000000000139.jpg       home theater 0.18, television 0.12, entertainment center 0.08
   000000000885.jpg       racket 0.75, tennis ball 0.03, book jacket 0.00
   000000001000.jpg       racket 0.40, volleyball 0.02, reel 0.02
```

---

### 3. `densenet169` — classification, no surgery

```bash
python compile/convert_to_onnx.py --model-id densenet169   # -> 57 MB ONNX
python compile/graph_surgery.py   --model-id densenet169   # skipped
python compile/compiler.py        --model-id densenet169   # the long step; ≈15 min
python compile/test_model.py      --model-id densenet169 --validate-only
```

**Host:**

```text
[compile] densenet169: precision=int8 batch=1 onnx=densenet169.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] densenet169 (int8): rc=0  -> .../model-compilation/work/densenet169/compile_int8
[PASS] densenet169      elf=1 so=0 batch=1  (compile_int8/densenet169_mpk.tar.gz)
```

**DevKit:**

```bash
dk compile/test_model.py --model-id densenet169
```

```text
[PASS] densenet169      elf=1 so=0 batch=1  (compile_int8/densenet169_mpk.tar.gz)
[test] densenet169: imagenet_topk on 3 real image(s) from inference/
   000000000139.jpg       barbershop 0.72, restaurant 0.17, barber chair 0.06
   000000000885.jpg       racket 0.98, tennis ball 0.02, baseball 0.00
   000000001000.jpg       volleyball 0.73, swimming trunks 0.09, racket 0.04
```

---

### 4. `efficientnet_v2_s` — classification, no surgery, 384×384 input

```bash
python compile/convert_to_onnx.py --model-id efficientnet_v2_s   # -> 86 MB ONNX
python compile/graph_surgery.py   --model-id efficientnet_v2_s   # skipped
python compile/compiler.py        --model-id efficientnet_v2_s   # the long step; ≈11 min
python compile/test_model.py      --model-id efficientnet_v2_s --validate-only
```

**Host:**

```text
[compile] efficientnet_v2_s: precision=int8 batch=1 onnx=efficientnet_v2_s.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] efficientnet_v2_s (int8): rc=0  -> .../model-compilation/work/efficientnet_v2_s/compile_int8
[PASS] efficientnet_v2_s elf=1 so=0 batch=1  (compile_int8/efficientnet_v2_s_mpk.tar.gz)
```

**DevKit:**

```bash
dk compile/test_model.py --model-id efficientnet_v2_s
```

```text
[PASS] efficientnet_v2_s elf=1 so=0 batch=1  (compile_int8/efficientnet_v2_s_mpk.tar.gz)
[test] efficientnet_v2_s: imagenet_topk on 3 real image(s) from inference/
   000000000139.jpg       home theater 0.22, vacuum 0.13, swab 0.10
   000000000885.jpg       racket 0.72, tennis ball 0.03, ping-pong ball 0.00
   000000001000.jpg       racket 0.18, ballplayer 0.16, soccer ball 0.14
```

---

### 5. `yolov8s` — detection, surgery (**head at `model.22`, no attention**)

```bash
python compile/convert_to_onnx.py --model-id yolov8s   # downloads yolov8s.pt -> 45 MB ONNX
python compile/graph_surgery.py   --model-id yolov8s   # cuts the decode tail, exposes 6 raw heads
python compile/compiler.py        --model-id yolov8s   # the long step; ≈5 min
python compile/test_model.py      --model-id yolov8s --validate-only
```

**Host:**

```text
[surgery] yolov8s: OK  outputs=['bbox_0', 'bbox_1', 'bbox_2', 'class_logit_0', 'class_logit_1', 'class_logit_2']
[compile] yolov8s: precision=int8 batch=1 onnx=yolov8s.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolov8s (int8): rc=0  -> .../model-compilation/work/yolov8s/compile_int8
[PASS] yolov8s          elf=1 so=0 batch=1  (compile_int8/yolov8s.compile_ready_mpk.tar.gz)
```

**DevKit** — 6 tensors, the same as `yolo11n`/`yolo11s`:

```bash
dk compile/test_model.py --model-id yolov8s
dk compile/test_box_decode.py --model-id yolov8s
```

```text
[PASS] yolov8s          elf=1 so=0 batch=1  (compile_int8/yolov8s.compile_ready_mpk.tar.gz)
[test] yolov8s: yolo_raw_heads on 3 real image(s) from inference/
   000000000139.jpg       6 head tensor(s): (1, 80, 80, 64) (1, 40, 40, 64) (1, 20, 20, 64) (1, 80, 80, 80) (1, 40, 40, 80) (1, 20, 20, 80)
   ...
```

**`yolov8l`:** run the same commands with `--model-id yolov8l`.

> The surgery differs from YOLO11 in two ways that `graph_surgery.py` already handles: the head is
> at `model.22`, and there is no attention block
> ([details](MODEL-COMPILATION.md#-yolov8-head-at-model22-no-attention)).

---

### 6. `yolo11n` — detection, surgery

```bash
python compile/convert_to_onnx.py --model-id yolo11n   # downloads yolo11n.pt -> 11 MB ONNX
python compile/graph_surgery.py   --model-id yolo11n   # cuts the decode tail, exposes 6 raw heads
python compile/compiler.py        --model-id yolo11n   # the long step; ≈5 min
python compile/test_model.py      --model-id yolo11n --validate-only
```

**Host:**

```text
[surgery] yolo11n: OK  outputs=['bbox_0', 'bbox_1', 'bbox_2', 'class_logit_0', 'class_logit_1', 'class_logit_2']
[compile] yolo11n: precision=int8 batch=1 onnx=yolo11n.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolo11n (int8): rc=0  -> .../model-compilation/work/yolo11n/compile_int8
[PASS] yolo11n          elf=1 so=0 batch=1  (compile_int8/yolo11n.compile_ready_mpk.tar.gz)
```

**DevKit** — the surgery contract, **6 tensors, NHWC**:

```bash
dk compile/test_model.py --model-id yolo11n
dk compile/test_box_decode.py --model-id yolo11n
```

```text
[PASS] yolo11n          elf=1 so=0 batch=1  (compile_int8/yolo11n.compile_ready_mpk.tar.gz)
[test] yolo11n: yolo_raw_heads on 3 real image(s) from inference/
   000000000139.jpg       6 head tensor(s): (1, 80, 80, 64) (1, 40, 40, 64) (1, 20, 20, 64) (1, 80, 80, 80) (1, 40, 40, 80) (1, 20, 20, 80)
   ...
```

bbox = 64 ch (4 × 16 DFL bins) × 3 scales · class = 80 ch × 3 scales.

---

### 7. `yolo11s` — detection, surgery

```bash
python compile/convert_to_onnx.py --model-id yolo11s   # -> 38 MB ONNX
python compile/graph_surgery.py   --model-id yolo11s
python compile/compiler.py        --model-id yolo11s   # the long step; ≈5 min
python compile/test_model.py      --model-id yolo11s --validate-only
```

**Host:**

```text
[compile] yolo11s: precision=int8 batch=1 onnx=yolo11s.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolo11s (int8): rc=0  -> .../model-compilation/work/yolo11s/compile_int8
[PASS] yolo11s          elf=1 so=0 batch=1  (compile_int8/yolo11s.compile_ready_mpk.tar.gz)
```

**DevKit** — the same 6 tensors as `yolo11n`:

```bash
dk compile/test_model.py --model-id yolo11s
dk compile/test_box_decode.py --model-id yolo11s
```

```text
[PASS] yolo11s          elf=1 so=0 batch=1  (compile_int8/yolo11s.compile_ready_mpk.tar.gz)
[test] yolo11s: yolo_raw_heads on 3 real image(s) from inference/
   000000000139.jpg       6 head tensor(s): (1, 80, 80, 64) (1, 40, 40, 64) (1, 20, 20, 64) (1, 80, 80, 80) (1, 40, 40, 80) (1, 20, 20, 80)
   ...
```

---

### 8. `yolo26n` — detection, surgery (**one-to-one head, `YoloV26`**)

```bash
python compile/convert_to_onnx.py --model-id yolo26n   # -> 9.9 MB ONNX
python compile/graph_surgery.py   --model-id yolo26n   # exposes the one2one_cv* heads (4 ch)
python compile/compiler.py        --model-id yolo26n   # the long step; ≈5 min
python compile/test_model.py      --model-id yolo26n --validate-only
```

**Host:**

```text
[compile] yolo26n: precision=int8 batch=1 onnx=yolo26n.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolo26n (int8): rc=0  -> .../model-compilation/work/yolo26n/compile_int8
[PASS] yolo26n          elf=1 so=0 batch=1  (compile_int8/yolo26n.compile_ready_mpk.tar.gz)
```

**DevKit:**

```bash
dk compile/test_model.py --model-id yolo26n
dk compile/test_box_decode.py --model-id yolo26n
```

```text
[PASS] yolo26n          elf=1 so=0 batch=1  (compile_int8/yolo26n.compile_ready_mpk.tar.gz)
[test] yolo26n: yolo_raw_heads on 3 real image(s) from inference/
   000000000139.jpg       6 head tensor(s): (1, 80, 80, 4) (1, 40, 40, 4) (1, 20, 20, 4) (1, 80, 80, 80) (1, 40, 40, 80) (1, 20, 20, 80)
   ...
```

> The box heads are 4 channels, not 64, and one-to-one: this archive decodes with `YoloV26`, which
> needs no NMS ([decoders](BOX-DECODE.md)).

---

### 9. `yolo11s-seg` — segmentation, surgery

```bash
python compile/convert_to_onnx.py --model-id yolo11s-seg   # -> 41 MB ONNX
python compile/graph_surgery.py   --model-id yolo11s-seg   # + mask-coeff heads and the proto head
python compile/compiler.py        --model-id yolo11s-seg   # the long step; ≈7 min
python compile/test_model.py      --model-id yolo11s-seg --validate-only
```

**Host:**

```text
[compile] yolo11s-seg: precision=int8 batch=1 onnx=yolo11s-seg.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolo11s-seg (int8): rc=0  -> .../model-compilation/work/yolo11s-seg/compile_int8
[PASS] yolo11s-seg      elf=1 so=0 batch=1  (compile_int8/yolo11s-seg.compile_ready_mpk.tar.gz)
```

**DevKit** — **10 tensors**: the 6 detection heads + 3 mask-coefficient heads + the proto:

```bash
dk compile/test_model.py --model-id yolo11s-seg
dk compile/test_box_decode.py --model-id yolo11s-seg
```

```text
[PASS] yolo11s-seg      elf=1 so=0 batch=1  (compile_int8/yolo11s-seg.compile_ready_mpk.tar.gz)
[test] yolo11s-seg: yolo_raw_heads on 3 real image(s) from inference/
   000000000139.jpg       10 head tensor(s): (1, 80, 80, 64) (1, 40, 40, 64) (1, 20, 20, 64) (1, 80, 80, 80) (1, 40, 40, 80) (1, 20, 20, 80) (1, 80, 80, 32) (1, 40, 40, 32) (1, 20, 20, 32) (1, 160, 160, 32)
   ...
```

---

### 10. `yolo26s-pose` — pose, surgery (**carries the 209× fix**)

```bash
python compile/convert_to_onnx.py --model-id yolo26s-pose   # -> 42 MB ONNX
python compile/graph_surgery.py   --model-id yolo26s-pose   # + keypoint heads, PADDED 51 -> 64 ch
python compile/compiler.py        --model-id yolo26s-pose   # the long step; ≈8 min
python compile/test_model.py      --model-id yolo26s-pose --validate-only
```

**Host:**

```text
[compile] yolo26s-pose: precision=int8 batch=1 onnx=yolo26s-pose.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolo26s-pose (int8): rc=0  -> .../model-compilation/work/yolo26s-pose/compile_int8
[PASS] yolo26s-pose     elf=1 so=0 batch=1  (compile_int8/yolo26s-pose.compile_ready_mpk.tar.gz)
```

**DevKit** — **9 tensors**, note the keypoint heads are **64**, not 51:

```bash
dk compile/test_model.py --model-id yolo26s-pose
dk compile/test_box_decode.py --model-id yolo26s-pose
```

```text
[PASS] yolo26s-pose     elf=1 so=0 batch=1  (compile_int8/yolo26s-pose.compile_ready_mpk.tar.gz)
[test] yolo26s-pose: yolo_raw_heads on 3 real image(s) from inference/
   000000000139.jpg       9 head tensor(s): (1, 80, 80, 4) (1, 40, 40, 4) (1, 20, 20, 4) (1, 80, 80, 1) (1, 40, 40, 1) (1, 20, 20, 1) (1, 80, 80, 64) (1, 40, 40, 64) (1, 20, 20, 64)
   ...
```

> ⚠️ **Do not remove the padding** (`pad_channels_to: 64` in `compile/_surgery_ultralytics.py`).
> Without it this model runs 209× slower ([why](MODEL-COMPILATION.md#-the-209-pose-fix-padding-51--64-channels)).

---

### 11. `yolox_s` — detection, **different surgery**

```bash
python compile/convert_to_onnx.py --model-id yolox_s   # downloads Megvii's official ONNX (no torch)
python compile/graph_surgery.py   --model-id yolox_s   # decoupled head -> 9 raw heads (bbox/obj/cls per scale, pre-sigmoid)
python compile/compiler.py        --model-id yolox_s   # the long step; ≈9 min
python compile/test_model.py      --model-id yolox_s --validate-only
```

**Host:**

```text
[surgery] yolox_s: OK  outputs=['bbox_0', 'obj_logit_0', 'class_logit_0', 'bbox_1', 'obj_logit_1', 'class_logit_1', 'bbox_2', 'obj_logit_2', 'class_logit_2']
[compile] yolox_s: precision=int8 batch=1 onnx=yolox_s.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolox_s (int8): rc=0  -> .../model-compilation/work/yolox_s/compile_int8
[PASS] yolox_s          elf=1 so=0 batch=1  (compile_int8/yolox_s.compile_ready_mpk.tar.gz)
```

**DevKit** — **9 tensors**: per scale a 4-ch bbox, a 1-ch objectness logit and an 80-ch class logit (`YoloX` / `Split3Interleaved`):

```bash
dk compile/test_model.py --model-id yolox_s
dk compile/test_box_decode.py --model-id yolox_s
```

```text
[PASS] yolox_s          elf=1 so=0 batch=1  (compile_int8/yolox_s.compile_ready_mpk.tar.gz)
[test] yolox_s: yolox_split_heads on 3 real image(s) from inference/
   000000000139.jpg       9 head tensor(s): (1, 80, 80, 4) (1, 80, 80, 1) (1, 80, 80, 80) (1, 40, 40, 4) (1, 40, 40, 1) (1, 40, 40, 80) (1, 20, 20, 4) (1, 20, 20, 1) (1, 20, 20, 80)
   ...
```

---

### 12. `yolov8s-worldv2` — open-vocabulary, **bf16 not INT8**

> - **bf16 only**: INT8 fails the compile ([why](PRECISION.md#why-these-three-are-bf16)).
>   `models.yaml` sets bf16, so the plain commands build it.
> - **The 80 COCO classes are baked in at export**, so no text encoder runs on the board
>   ([how](MODEL-COMPILATION.md#-yolo-world-bake-the-vocabulary-then-flatten-the-5-d-attention)).
>   To use your own classes, see [Change the vocabulary](#change-the-vocabulary).

```bash
python compile/convert_to_onnx.py --model-id yolov8s-worldv2   # bake COCO-80 + 4D attn patch
python compile/graph_surgery.py   --model-id yolov8s-worldv2   # fold contrastive head; raw DFL bbox
python compile/compiler.py        --model-id yolov8s-worldv2   # precision: bf16 from models.yaml
python compile/test_model.py      --model-id yolov8s-worldv2 --validate-only
```

**Host** — same 6-head contract as the other detectors:

```text
[surgery] yolov8s-worldv2: OK  outputs=['bbox_0', 'bbox_1', 'bbox_2', 'class_logit_0', 'class_logit_1', 'class_logit_2']
[compile] yolov8s-worldv2: precision=bf16 batch=1 onnx=yolov8s-worldv2.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolov8s-worldv2 (bf16): rc=0  -> .../model-compilation/work/yolov8s-worldv2/compile_bf16
[PASS] yolov8s-worldv2  elf=1 so=0 batch=1  (compile_bf16/yolov8s-worldv2.compile_ready_mpk.tar.gz)
```

**DevKit:**

```bash
dk compile/test_model.py --model-id yolov8s-worldv2
dk compile/test_box_decode.py --model-id yolov8s-worldv2
```

```text
[PASS] yolov8s-worldv2  elf=1 so=0 batch=1  (compile_bf16/yolov8s-worldv2.compile_ready_mpk.tar.gz)
[test] yolov8s-worldv2: yolo_raw_heads on 3 real image(s) from inference/
   000000000139.jpg       6 head tensor(s): (80, 80, 64) (40, 40, 64) (20, 20, 64) (80, 80, 80) (40, 40, 80) (20, 20, 80)
   ...
```

`class_logit_*` has one channel per baked class (80 here).

#### Change the vocabulary

No retraining. Bake a different class list and recompile — the `class_logit_*` head width follows it:

```bash
python compile/_export_world.py  --model-id yolov8s-worldv2 --labels /path/to/my_classes.txt --force
python compile/graph_surgery.py  --model-id yolov8s-worldv2 --force
python compile/compiler.py       --model-id yolov8s-worldv2
```

One class name per line. Pass `--num-classes <N>` to `test_box_decode.py` and `profile_model.py` if
N is not 80, and note that `assets/labels/coco80.txt` will no longer match the baked vocabulary.

---

### 13. `yolo26m` — detection, **bf16**

> - Built as **bf16** ([why](PRECISION.md#why-these-three-are-bf16)); `models.yaml` sets it, so the
>   plain commands build it.
> - The compile takes ≈16 min and ≈13 GB of RAM on a 16 GB SDK host. Compile nothing else alongside it.

```bash
python compile/convert_to_onnx.py --model-id yolo26m   # -> 82 MB ONNX
python compile/graph_surgery.py   --model-id yolo26m   # same one2one heads as yolo26n
python compile/compiler.py        --model-id yolo26m   # precision: bf16; the long step, ≈16 min
python compile/test_model.py      --model-id yolo26m --validate-only
```

**Host:**

```text
[surgery] yolo26m: OK  outputs=['bbox_0', 'bbox_1', 'bbox_2', 'class_logit_0', 'class_logit_1', 'class_logit_2']
[compile] yolo26m: precision=bf16 batch=1 onnx=yolo26m.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolo26m (bf16): rc=0  -> .../model-compilation/work/yolo26m/compile_bf16
[PASS] yolo26m          elf=1 so=0 batch=1  (compile_bf16/yolo26m.compile_ready_mpk.tar.gz)
```

**DevKit:**

```bash
dk compile/test_model.py      --model-id yolo26m
dk compile/test_box_decode.py --model-id yolo26m
```

```text
   000000000139.jpg       6 head tensor(s): (80, 80, 4) (40, 40, 4) (20, 20, 4) (80, 80, 80) (40, 40, 80) (20, 20, 80)
yolo26m              YoloV26/Auto                 PASS
```

(Neat 0.5.0: 17 / 8 / 17 boxes on the three images, with one or two isolated `[WARN]` pairs.)

To confirm the archive really is bf16, see [Check which precision you built](PRECISION.md#check-which-precision-you-built).
For an INT8 build at batch 6, see [`MULTI-BATCH.md`](MULTI-BATCH.md).

---

### 14. `yolo26s` — detection, **INT8 and bf16** (accuracy study)

```bash
python compile/convert_to_onnx.py --model-id yolo26s   # -> 38 MB ONNX
python compile/graph_surgery.py   --model-id yolo26s
python compile/compiler.py        --model-id yolo26s   # INT8 -> work/yolo26s/compile_int8/
python compile/test_model.py      --model-id yolo26s --validate-only
```

**DevKit:**

```bash
dk compile/test_model.py      --model-id yolo26s
dk compile/test_box_decode.py --model-id yolo26s
```

**Also in bf16, for the accuracy comparison.** Each precision has its own folder, so the two builds
sit side by side:

```bash
python compile/compiler.py        --model-id yolo26s --precision bf16   # -> work/yolo26s/compile_bf16/
python compile/test_model.py      --model-id yolo26s --precision bf16 --validate-only
```

> On COCO, bf16 loses 0.07 mAP50-95 against FP32 and INT8 loses 2.67. Results:
> [`ACCURACY.md`](ACCURACY.md); to rerun the comparison: [`accuracy/README.md`](accuracy/README.md).

---

## Compile every model

The same four steps as in the per-model sections above, with `--all` in place of `--model-id`:
every enabled model in `models.yaml`, one at a time.

```bash
python compile/convert_to_onnx.py --all
python compile/graph_surgery.py   --all
python compile/compiler.py        --all       # serial; the long step
python compile/test_model.py      --all --validate-only
```

`compiler.py` reads each model's `precision:` from `models.yaml`, so the bf16 models are built as
bf16 (into `work/<id>/compile_bf16/`) and the rest as INT8 (into `work/<id>/compile_int8/`). With
`--all`, a model that fails is reported and the next one is built.

Expected final line:

```text
all archives: one .elf, zero .so
```

Then, on the DevKit, confirm Neat decodes every detection, segmentation and pose archive **on-device**, each with its own family's decoder (see
[`BOX-DECODE.md`](BOX-DECODE.md)):

```bash
dk compile/test_box_decode.py --all
```

```text
model                on-device box decode         verdict
--------------------------------------------------------------------
yolov8s              YoloV8/GroupedByRoleLogit    PASS
yolo11n              YoloV8/GroupedByRoleLogit    PASS
yolo11s              YoloV8/GroupedByRoleLogit    PASS
yolo26n              YoloV26/Auto                 PASS
yolo26s              YoloV26/Auto                 PASS
yolo26m              YoloV26/Auto                 PASS
yolo11s-seg          YoloV8Seg/GroupedByRoleLogit PASS
yolo26s-pose         YoloV26Pose/Auto             PASS
yolov8s-worldv2      YoloV8/GroupedByRoleLogit    PASS
yolox_s              YoloX/Split3Interleaved      PASS

every archive decodes on-device with its own BoxDecodeType
```

(Neat 0.5.0 / SDK 2.1.3.1, all models built in their `models.yaml` precision. `--all` also checks
`yolov8l`, which is not shown here. Each block above it prints per-image box counts and duplicate
pairs; a `[WARN]` line for an isolated overlapping pair on a YOLO26 model is expected, see
[`BOX-DECODE.md`](BOX-DECODE.md).)

---

## Other builds

| Build | Where |
| --- | --- |
| Any model at the other precision (`--precision int8\|bf16`) | [`PRECISION.md`](PRECISION.md#build-a-model-at-the-other-precision) |
| N images per MLA call (`--batch-size N`), e.g. `yolo26m` INT8 at batch 6 | [`MULTI-BATCH.md`](MULTI-BATCH.md) |
| `yolo26x` INT8, compile and runtime profile | [`YOLO26X-STUDY.md`](YOLO26X-STUDY.md) |
