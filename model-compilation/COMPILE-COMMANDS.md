# Compilation commands

Copy-paste blocks for compiling the ten models. Run the [setup](README.md#setup) first — activate
the model-compiler env, `pip install ultralytics`, and `cd model-compilation`.

← Back to [`README.md`](README.md) · Why any of this is needed: [`MODEL-COMPILATION.md`](MODEL-COMPILATION.md)

> ⚠️ **Compile strictly ONE model at a time.** The compiler is memory-hungry; concurrent compiles OOM.

## Table of Contents

- [Compile everything](#compile-everything)
  - [How long each model takes](#how-long-each-model-takes)
- [Compile a single model](#compile-a-single-model)
  - [1. `resnet50` — classification, no surgery](#1-resnet50--classification-no-surgery)
  - [2. `convnext_tiny` — classification, no surgery](#2-convnext_tiny--classification-no-surgery)
  - [3. `densenet169` — classification, no surgery](#3-densenet169--classification-no-surgery)
  - [4. `efficientnet_v2_s` — classification, no surgery, 384×384 input](#4-efficientnet_v2_s--classification-no-surgery-384384-input)
  - [5. `yolo11n` — detection, surgery](#5-yolo11n--detection-surgery)
  - [6. `yolo11s` — detection, surgery](#6-yolo11s--detection-surgery)
  - [7. `yolo26n` — detection, surgery (**no DFL rebuild**)](#7-yolo26n--detection-surgery-no-dfl-rebuild)
  - [8. `yolo11s-seg` — segmentation, surgery](#8-yolo11s-seg--segmentation-surgery)
  - [9. `yolo26s-pose` — pose, surgery (**carries the 209× fix**)](#9-yolo26s-pose--pose-surgery-carries-the-209-fix)
  - [10. `yolox_s` — detection, **different surgery**](#10-yolox_s--detection-different-surgery)

---

## Compile everything

```bash
./compile_all.sh                 # all ten, serial, ~2 h; progress in compile_all.log
```

Same four steps per model as below, just scripted and safe to leave running. It collects each
model's artifacts into `assets/models/<id>/`.

Or step by step across all models:

```bash
python compile/convert_to_onnx.py --all
python compile/graph_surgery.py   --all
python compile/compiler.py        --all       # serial; the long step
python compile/test_model.py      --all --validate-only
```

Expected final line:

```text
all archives: one .elf, zero .so
```

### How long each model takes

Measured **end-to-end** — download + export + surgery + INT8 compile + validate — from a full
`compile_all.sh` run on the SDK container. Not compile-only estimates.

| # | Model | End-to-end | | # | Model | End-to-end |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `resnet50` | 4m32s | | 6 | `yolo11s` | 9m48s |
| 2 | `convnext_tiny` | 9m59s | | 7 | `yolo26n` | 10m05s |
| 3 | `densenet169` | 19m27s | | 8 | `yolo11s-seg` | 12m02s |
| 4 | `efficientnet_v2_s` | 19m43s | | 9 | `yolo26s-pose` | 12m16s |
| 5 | `yolo11n` | 8m32s | | 10 | `yolox_s` | 16m02s |

**Total ≈ 2 h**, serial. The compile step dominates; download and export are a minute or two each.
Times scale with host CPU — treat them as ratios, not promises.

---

## Compile a single model

One model per section. Nothing is downloaded by hand — **step 1 fetches the weights for you**.

### 1. `resnet50` — classification, no surgery

```bash
python compile/convert_to_onnx.py --model-id resnet50      # downloads torchvision weights -> 98 MB ONNX
python compile/graph_surgery.py   --model-id resnet50      # prints "kind=none ... skipping"
python compile/compiler.py        --model-id resnet50      # the long step; whole block ≈ 4m32s
python compile/test_model.py      --model-id resnet50 --validate-only
```

**Host:**

```text
[compile] resnet50: rc=0            ...  A65 : 0
[PASS] resnet50   elf=1 so=0  (resnet50_mpk.tar.gz)
```

**DevKit** — `python compile/test_model.py --model-id resnet50`:

```text
[test] resnet50: imagenet_topk on real image(s)
   000000000885.jpg   racket 0.59, tennis ball 0.02
```

---

### 2. `convnext_tiny` — classification, no surgery

```bash
python compile/convert_to_onnx.py --model-id convnext_tiny   # -> 110 MB ONNX
python compile/graph_surgery.py   --model-id convnext_tiny   # skipped (surgery: none)
python compile/compiler.py        --model-id convnext_tiny   # the long step; whole block ≈ 9m59s
python compile/test_model.py      --model-id convnext_tiny --validate-only
```

**Host:**

```text
[compile] convnext_tiny: rc=0       ...  A65 : 0
[PASS] convnext_tiny   elf=1 so=0
```

**DevKit:**

```text
   000000000139.jpg   home theater 0.29, television 0.16
   000000000885.jpg   racket 0.57, tennis ball 0.05
```

---

### 3. `densenet169` — classification, no surgery

```bash
python compile/convert_to_onnx.py --model-id densenet169   # -> 55 MB ONNX
python compile/graph_surgery.py   --model-id densenet169   # skipped
python compile/compiler.py        --model-id densenet169   # the long step; whole block ≈ 19m27s
python compile/test_model.py      --model-id densenet169 --validate-only
```

**Host:**

```text
[compile] densenet169: rc=0         ...  A65 : 0
[PASS] densenet169   elf=1 so=0
```

**DevKit:**

```text
   000000000885.jpg   racket 0.99, tennis ball 0.01
```

---

### 4. `efficientnet_v2_s` — classification, no surgery, 384×384 input

```bash
python compile/convert_to_onnx.py --model-id efficientnet_v2_s   # -> 82 MB ONNX
python compile/graph_surgery.py   --model-id efficientnet_v2_s   # skipped
python compile/compiler.py        --model-id efficientnet_v2_s   # the long step; whole block ≈ 19m43s
python compile/test_model.py      --model-id efficientnet_v2_s --validate-only
```

**Host:**

```text
[compile] efficientnet_v2_s: rc=0   ...  A65 : 0
[PASS] efficientnet_v2_s   elf=1 so=0
```

**DevKit:**

```text
   000000000885.jpg   racket 0.75, tennis ball 0.04
```

---

### 5. `yolo11n` — detection, surgery

```bash
python compile/convert_to_onnx.py --model-id yolo11n   # downloads yolo11n.pt -> 11 MB ONNX
python compile/graph_surgery.py   --model-id yolo11n   # cuts the decode tail, exposes 6 raw heads
python compile/compiler.py        --model-id yolo11n   # the long step; whole block ≈ 8m32s
python compile/test_model.py      --model-id yolo11n --validate-only
```

**Host:**

```text
[surgery] yolo11n: OK  outputs=['bbox_0','bbox_1','bbox_2','class_logit_0','class_logit_1','class_logit_2']
[compile] yolo11n: rc=0             ...  A65 : 0
[PASS] yolo11n   elf=1 so=0
```

**DevKit** — the surgery contract, **6 tensors, NHWC**:

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

bbox = 4 ch × 3 scales · class = 80 ch × 3 scales.

---

### 6. `yolo11s` — detection, surgery

```bash
python compile/convert_to_onnx.py --model-id yolo11s   # -> 37 MB ONNX
python compile/graph_surgery.py   --model-id yolo11s
python compile/compiler.py        --model-id yolo11s   # the long step; whole block ≈ 9m48s
python compile/test_model.py      --model-id yolo11s --validate-only
```

**Host:**

```text
[compile] yolo11s: rc=0             ...  A65 : 0
[PASS] yolo11s   elf=1 so=0
```

**DevKit** — identical contract to `yolo11n`; head node names are scale-invariant, so `n`→`s` is a
free retarget:

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

---

### 7. `yolo26n` — detection, surgery (**no DFL rebuild**)

```bash
python compile/convert_to_onnx.py --model-id yolo26n   # -> 9.5 MB ONNX
python compile/graph_surgery.py   --model-id yolo26n   # one2one_cv* heads; DFL step skipped
python compile/compiler.py        --model-id yolo26n   # the long step; whole block ≈ 10m05s
python compile/test_model.py      --model-id yolo26n --validate-only
```

**Host:**

```text
[compile] yolo26n: rc=0             ...  A65 : 0
[PASS] yolo26n   elf=1 so=0
```

**DevKit:**

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

YOLO26's heads are already 4-channel, so the DFL reconstruction that YOLO11 needs is skipped.

---

### 8. `yolo11s-seg` — segmentation, surgery

```bash
python compile/convert_to_onnx.py --model-id yolo11s-seg   # -> 39 MB ONNX
python compile/graph_surgery.py   --model-id yolo11s-seg   # + mask-coeff heads and the proto head
python compile/compiler.py        --model-id yolo11s-seg   # the long step; whole block ≈ 12m02s
python compile/test_model.py      --model-id yolo11s-seg --validate-only
```

**Host:**

```text
[compile] yolo11s-seg: rc=0         ...  A65 : 0
[PASS] yolo11s-seg   elf=1 so=0
```

**DevKit** — **10 tensors**: the 6 detection heads + 3 mask-coefficient heads + the proto:

```text
   10 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                      (1,80,80,80) (1,40,40,80) (1,20,20,80)
                      (1,80,80,32) (1,40,40,32) (1,20,20,32)     <- mask coeffs (32 ch)
                      (1,160,160,32)                             <- proto
```

---

### 9. `yolo26s-pose` — pose, surgery (**carries the 209× fix**)

```bash
python compile/convert_to_onnx.py --model-id yolo26s-pose   # -> 40 MB ONNX
python compile/graph_surgery.py   --model-id yolo26s-pose   # + keypoint heads, PADDED 51 -> 64 ch
python compile/compiler.py        --model-id yolo26s-pose   # the long step; whole block ≈ 12m16s
python compile/test_model.py      --model-id yolo26s-pose --validate-only
```

**Host:**

```text
[compile] yolo26s-pose: rc=0        ...  A65 : 0
[PASS] yolo26s-pose   elf=1 so=0
```

**DevKit** — **9 tensors**, note the keypoint heads are **64**, not 51:

```text
   9 head tensor(s): (1,80,80,4)  (1,40,40,4)  (1,20,20,4)      <- bbox
                     (1,80,80,1)  (1,40,40,1)  (1,20,20,1)      <- class (1 = person)
                     (1,80,80,64) (1,40,40,64) (1,20,20,64)     <- keypoints, padded 51 -> 64
```

> ⚠️ **Do not remove the padding.** Keep `pad_channels_to: 64` in
> `compile/_surgery_ultralytics.py`. Unpadded, this model runs at **1782 ms/frame**; padded, at
> **8.5 ms/frame** — a **209× speedup** for identical weights. Full story:
> [the 209× pose fix](MODEL-COMPILATION.md#-the-209-pose-fix-padding-51--64-channels).

---

### 10. `yolox_s` — detection, **different surgery**

```bash
python compile/convert_to_onnx.py --model-id yolox_s   # downloads Megvii's official ONNX (no torch)
python compile/graph_surgery.py   --model-id yolox_s   # decoupled anchor-free head -> 3 raw heads
python compile/compiler.py        --model-id yolox_s   # the long step; whole block ≈ 16m02s
python compile/test_model.py      --model-id yolox_s --validate-only
```

**Host:**

```text
[surgery] yolox_s: OK  outputs=['yolox_head_0','yolox_head_1','yolox_head_2']
[compile] yolox_s: rc=0             ...  A65 : 0
[PASS] yolox_s   elf=1 so=0
```

**DevKit** — **3 tensors**, 85 ch each (4 box + 1 obj + 80 class):

```text
   3 head tensor(s): (1,80,80,85) (1,40,40,85) (1,20,20,85)
```

---
