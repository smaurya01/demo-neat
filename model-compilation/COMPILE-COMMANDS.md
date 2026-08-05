# Setup

Everything below runs in the model-compiler environment:

```bash
source /sdk-extensions/model-compiler/bin/activate     # afe + onnx + torch
cd model-compilation
```

**`ultralytics` is not included** in that environment — install it separately, or the YOLO models
will fail at export:

```bash
pip install ultralytics
```

Steps 1–3 and `test_model.py --validate-only` use `python` here in the SDK container. Step 4's real
inference and step 4b run on the DevKit with `dk` — `/workspace` is NFS-mounted there at the same
path, so nothing is copied.

> ⚠️ **Compile strictly ONE model at a time.** The compiler is memory-hungry; concurrent compiles OOM.

## Table of Contents

- [Compile everything](#compile-everything)
  - [How long each model takes](#how-long-each-model-takes)
- [Compile a single model](#compile-a-single-model)
  - [1. `resnet50` — classification, no surgery](#1-resnet50--classification-no-surgery)
  - [2. `convnext_tiny` — classification, no surgery](#2-convnext_tiny--classification-no-surgery)
  - [3. `densenet169` — classification, no surgery](#3-densenet169--classification-no-surgery)
  - [4. `efficientnet_v2_s` — classification, no surgery, 384×384 input](#4-efficientnet_v2_s--classification-no-surgery-384384-input)
  - [5. `yolov8s` — detection, surgery (**head at `model.22`, no attention**)](#5-yolov8s--detection-surgery-head-at-model22-no-attention)
  - [6. `yolo11n` — detection, surgery](#6-yolo11n--detection-surgery)
  - [7. `yolo11s` — detection, surgery](#7-yolo11s--detection-surgery)
  - [8. `yolo26n` — detection, surgery (**no DFL rebuild**)](#8-yolo26n--detection-surgery-no-dfl-rebuild)
  - [9. `yolo11s-seg` — segmentation, surgery](#9-yolo11s-seg--segmentation-surgery)
  - [10. `yolo26s-pose` — pose, surgery (**carries the 209× fix**)](#10-yolo26s-pose--pose-surgery-carries-the-209-fix)
  - [11. `yolox_s` — detection, **different surgery**](#11-yolox_s--detection-different-surgery)
  - [12. `yolov8s-worldv2` — open-vocabulary, **bf16 not INT8**](#12-yolov8s-worldv2--open-vocabulary-bf16-not-int8)
    - [Change the vocabulary](#change-the-vocabulary)

---

## Compile everything

```bash
./compile_all.sh                 # eleven models, serial, ~2 h; progress in compile_all.log
```

Same four steps per model as below, just scripted and safe to leave running. It collects each
model's artifacts into `assets/models/<id>/`.

**`yolov8s-worldv2` is deliberately left out**, so this covers eleven of the twelve models. It needs
`--bf16-weights --bf16-activations`, which `compile_all.sh` has no way to pass per model — compiling
it with the default INT8 flags fails the compiler's sim check. Build it separately afterwards:
[section 12](#12-yolov8s-worldv2--open-vocabulary-bf16-not-int8).

Or step by step across all models:

```bash
python compile/convert_to_onnx.py --all
python compile/graph_surgery.py   --all
python compile/compiler.py        --all       # serial; the long step
python compile/test_model.py      --all --validate-only
```

> ⚠️ **`--all` is not the same as `compile_all.sh` here.** `--all` means *every enabled model in
> `models.yaml`*, which **includes `yolov8s-worldv2`** — so `compiler.py --all` compiles it as INT8
> and it fails. `compile_all.sh` skips it by name and is safe. If you use the `--all` form, either
> accept that one failure and rebuild worldv2 with the bf16 flags, or set `enabled: false` on it
> first.

Expected final line:

```text
all archives: one .elf, zero .so
```

Then, on the DevKit, confirm Neat can decode the detection heads **on-device** (see
[step 4b](README.md#step-4b--on-device-box-decode-detection-models)):

```bash
dk compile/test_box_decode.py --all
```

```text
model                on-device box decode         verdict
--------------------------------------------------------------------
yolov8s              YoloV26/Auto                 PASS
yolo11n              YoloV26/Auto                 PASS
yolo11s              YoloV26/Auto                 PASS
yolo26n              YoloV26/Auto                 PASS
yolov8s-worldv2      YoloV26/Auto                 PASS
yolox_s              YoloX/Split3Interleaved      PASS

every detection archive decodes on-device
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

`yolov8s` was added after that run, so it is not in the table. Its **compile step** measures ≈10 min
cold — the same range as `yolo11s`, as expected for the same size class and the same 6-head contract.
(A re-run over a populated `work/<id>/compile_int8` finishes in ~3 min; that is a warm rebuild, not a
comparable number.)

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

**DevKit:**

```bash
dk compile/test_model.py --model-id resnet50
```

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

```bash
dk compile/test_model.py --model-id convnext_tiny
```

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

```bash
dk compile/test_model.py --model-id densenet169
```

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

```bash
dk compile/test_model.py --model-id efficientnet_v2_s
```

```text
   000000000885.jpg   racket 0.75, tennis ball 0.04
```

---

### 5. `yolov8s` — detection, surgery (**head at `model.22`, no attention**)

```bash
python compile/convert_to_onnx.py --model-id yolov8s   # downloads yolov8s.pt -> 45 MB ONNX
python compile/graph_surgery.py   --model-id yolov8s   # cuts the decode tail, exposes 6 raw heads
python compile/compiler.py        --model-id yolov8s   # the long step; ≈10 min
python compile/test_model.py      --model-id yolov8s --validate-only
```

**Host** — note `attention_rewrites` is **empty**, unlike every YOLO11/YOLO26 model here:

```text
[surgery] yolov8s: OK  outputs=['bbox_0','bbox_1','bbox_2','class_logit_0','class_logit_1','class_logit_2']
[compile] yolov8s: rc=0             ...  MLA : 1   EV74: 16   A65 : 0
[PASS] yolov8s   elf=1 so=0  (yolov8s.compile_ready_mpk.tar.gz)
```

**DevKit** — identical 6-tensor contract to `yolo11n`/`yolo11s`:

```bash
dk compile/test_model.py --model-id yolov8s
dk compile/test_box_decode.py --model-id yolov8s
```

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

Decoding those heads on-device gives detections within 0.01–0.06 of the float ONNX on the same
images (`000000000139`: tv 0.907 vs 0.921 float, chair 0.836 vs 0.849) — the INT8 calibration is
sound, and scores are **not** capped the way a badly-quantized package's would be.

Two things differ from YOLO11 and both are handled in `compile/_surgery_ultralytics.py`:

- **The Detect head is at `model.22`, not `model.23`.** YOLO11 inserts a C2PSA block, which shifts
  every head node name by one. Copying the YOLO11 sources verbatim fails with
  `ValueError: missing head tensors`.
- **YOLOv8 has no attention block at all** — the export contains **0 `MatMul` nodes** — so
  `attention_blocks` is empty and the `MatMul → Einsum` rewrite is a no-op. That is expected, not a
  misconfiguration.

The DFL rebuild is still needed: the `cv2.*` bbox heads emit 64 channels (4 × 16 bins), same as
YOLO11.

---

### 6. `yolo11n` — detection, surgery

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

```bash
dk compile/test_model.py --model-id yolo11n
dk compile/test_box_decode.py --model-id yolo11n
```

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

bbox = 4 ch × 3 scales · class = 80 ch × 3 scales.

---

### 7. `yolo11s` — detection, surgery

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

```bash
dk compile/test_model.py --model-id yolo11s
dk compile/test_box_decode.py --model-id yolo11s
```

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

---

### 8. `yolo26n` — detection, surgery (**no DFL rebuild**)

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

```bash
dk compile/test_model.py --model-id yolo26n
dk compile/test_box_decode.py --model-id yolo26n
```

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

YOLO26's heads are already 4-channel, so the DFL reconstruction that YOLO11 needs is skipped.

---

### 9. `yolo11s-seg` — segmentation, surgery

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

```bash
dk compile/test_model.py --model-id yolo11s-seg
```

```text
   10 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                      (1,80,80,80) (1,40,40,80) (1,20,20,80)
                      (1,80,80,32) (1,40,40,32) (1,20,20,32)     <- mask coeffs (32 ch)
                      (1,160,160,32)                             <- proto
```

---

### 10. `yolo26s-pose` — pose, surgery (**carries the 209× fix**)

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

```bash
dk compile/test_model.py --model-id yolo26s-pose
```

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

### 11. `yolox_s` — detection, **different surgery**

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

```bash
dk compile/test_model.py --model-id yolox_s
dk compile/test_box_decode.py --model-id yolox_s
```

```text
   3 head tensor(s): (1,80,80,85) (1,40,40,85) (1,20,20,85)
```

---

### 12. `yolov8s-worldv2` — open-vocabulary, **bf16 not INT8**

YOLO-World is **open-vocabulary**: normally it takes an image *plus* text prompts, and a CLIP text
encoder turns the prompts into class embeddings at runtime. A SiMa archive is a fixed graph, so the
vocabulary is **baked at export time** (`set_classes` → COCO-80), which drops the CLIP text encoder
and leaves an image-only graph. That is why this model has its own export and surgery steps.

> ⚠️ **This model does not compile with the default INT8 flags, and it is NOT in `compile_all.sh`.**
> `models.yaml` carries `precision: bf16`, but `compiler.py` **does not read that field** — you must
> pass the flags yourself. `python compile/compiler.py --all` would compile it as INT8 and fail.

```bash
python compile/convert_to_onnx.py --model-id yolov8s-worldv2   # bake COCO-80 + 4D attn patch
python compile/graph_surgery.py   --model-id yolov8s-worldv2   # fold contrastive head + DFL
python compile/compiler.py        --model-id yolov8s-worldv2 --bf16-weights --bf16-activations
python compile/test_model.py      --model-id yolov8s-worldv2 --validate-only
```

**Host** — same 6-head contract as the other detectors:

```text
[surgery] yolov8s-worldv2: OK  outputs=['bbox_0','bbox_1','bbox_2','class_logit_0','class_logit_1','class_logit_2']
[compile] yolov8s-worldv2: rc=0     ...  MLA : 1   EV74: 16   A65 : 0
[PASS] yolov8s-worldv2   elf=1 so=0  (yolov8s-worldv2.compile_ready_mpk.tar.gz)
```

**DevKit:**

```bash
dk compile/test_model.py --model-id yolov8s-worldv2
dk compile/test_box_decode.py --model-id yolov8s-worldv2
```

```text
   6 head tensor(s): (80,80,4) (40,40,4) (20,20,4)
                     (80,80,80) (40,40,80) (20,20,80)
```

`class_logit_*` is 80 channels because 80 classes were baked — that width follows your vocabulary,
not the model. On-device box decode works with `BoxDecodeType.YoloV26`, same as the other detectors.

**Why bf16 and not INT8.** With the 4-D attention rewrite the graph places as a single MLA segment in
*both* precisions, but INT8 fails the compiler's sim/bit-accuracy check: the per-head decomposition
exposes the wide-range text-similarity score maps as intermediate INT8 tensors (saturation warnings
on the attn `proj_conv` bias), which the original fused einsum never did. bf16 has no quantization
step, passes at `rc=0`, and is the same format the official yolo26 detection archives ship in.
Calibration images are still used — they just do not drive a quantization scale.

#### Change the vocabulary

No retraining. Bake a different class list and recompile — the `class_logit_*` head width follows it:

```bash
python compile/_export_world.py  --model-id yolov8s-worldv2 --labels /path/to/my_classes.txt --force
python compile/graph_surgery.py  --model-id yolov8s-worldv2 --force
python compile/compiler.py       --model-id yolov8s-worldv2 --bf16-weights --bf16-activations
```

One class name per line. Pass `--num-classes <N>` to `test_box_decode.py` if N is not 80, and note
that `assets/labels/coco80.txt` will no longer match the baked vocabulary.

---
