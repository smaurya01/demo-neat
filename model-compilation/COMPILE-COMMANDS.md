# Compile commands

## Setup

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

`convert_to_onnx.py`, `graph_surgery.py`, `compiler.py` and `test_model.py --validate-only` run with
`python` here in the SDK container. `test_model.py` without `--validate-only`, `test_box_decode.py`
and `profile_model.py` run on the DevKit with `dk` — `/workspace` is NFS-mounted there at the same
path, so nothing is copied. `compiler.py` needs the SDK's `quantize_compile.py` driver; see
[README > Setup](README.md#setup).

> ⚠️ **Compile strictly ONE model at a time.** The compiler is memory-hungry; concurrent compiles OOM.

## Table of Contents

- [Setup](#setup)
- [Choosing the precision: INT8 or bf16](#choosing-the-precision-int8-or-bf16)
- [Compile everything](#compile-everything)
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
  - [15. `yolo26m` at batch 6 — multi-batch, **INT8**](#15-yolo26m-at-batch-6--multi-batch-int8)

---

## Choosing the precision: INT8 or bf16

Every model can be compiled at either precision:

- **INT8** — 8-bit integers. Smaller and faster, but the compiler must learn each tensor's value
  range from the calibration images, and some accuracy is lost.
- **bf16** — 16-bit floats. Keeps FP32 accuracy and does not depend on 8-bit ranges, but the program
  is bigger and slower.

For `yolo26s` on COCO (Neat 0.4.0), bf16 lost 0.07 mAP50-95 against FP32 and ran at 18.4 ms; INT8 lost 2.67 and
ran at 11.1 ms. See [`ACCURACY.md`](ACCURACY.md) for the full comparison.

### How `compiler.py` decides

**One rule: every model's entry in [`models.yaml`](models.yaml) has a `precision:` line — `int8` or
`bf16` — and that is what a plain command builds.** The test scripts check the same precision.

```yaml
  - id: yolo26s
    ...
    precision: int8        # python compile/compiler.py --model-id yolo26s  ->  INT8

  - id: yolo26m
    ...
    precision: bf16        # python compile/compiler.py --model-id yolo26m  ->  bf16
```

`--precision int8|bf16` on the command line overrides it **for that one build only** — for example
to build `yolo26s` in bf16 as well, for the accuracy study. The older flags
`--bf16-weights --bf16-activations` mean the same as `--precision bf16`.

For bf16, `compiler.py` adds `--bf16-weights --bf16-activations` to the SiMa compiler command;
without them the compiler quantizes to INT8. A model with no `precision:` line stops with a message
asking you to add one.

### Each model's default

| `precision:` in `models.yaml` | Models |
| --- | --- |
| **`bf16`** | `convnext_tiny`, `yolo26m`, `yolov8s-worldv2` |
| **`int8`** | `resnet50`, `densenet169`, `efficientnet_v2_s`, `yolov8s`, `yolov8l`, `yolo11n`, `yolo11s`, `yolo26n`, `yolo26s`, `yolo11s-seg`, `yolo26s-pose`, `yolox_s` |

`yolov8s-worldv2` must stay bf16: it fails the compiler's check as INT8
([section 12](#12-yolov8s-worldv2--open-vocabulary-bf16-not-int8)).

### Build a model at the other precision

**For one build**, add `--precision` to `compiler.py`, and the same flag to the test scripts so they
check that build:

```bash
python compile/compiler.py        --model-id yolo26n --precision bf16
python compile/test_model.py      --model-id yolo26n --precision bf16 --validate-only
dk compile/test_model.py          --model-id yolo26n --precision bf16
dk compile/test_box_decode.py     --model-id yolo26n --precision bf16
```

**To change a model's default**, edit the `precision:` line in its `models.yaml` entry. Then the
plain commands, `--all`, and the test scripts all use it:

```yaml
  - id: yolo26n
    ...
    precision: bf16
    enabled: true
```

Each precision is built into its own folder, so one model can have both builds side by side:

```text
work/<id>/compile_int8/<...>_mpk.tar.gz
work/<id>/compile_bf16/<...>_mpk.tar.gz
```

`compile_all.sh` collects archives from `compile_int8/` only, so build bf16 models with the
per-model commands below.

### Check which precision you built

- The first line `compiler.py` prints: `[compile] yolo26m: precision=bf16 ...`
- The folder: `compile_bf16/` or `compile_int8/`, and the command it ran in
  `work/<id>/reports/compile_<precision>.command.txt`
- The archive itself: its first step is `cast_transform` for bf16 and `quantization_transform` for
  INT8 (see [section 13](#13-yolo26m--detection-bf16) for the check, and the
  [Quick Start Guide, Chapter 20](../devkit-quick-start/chapters/20-model-archive.md#how-to-tell-which-one-you-have))

### Which one to pick

| If | Use |
| --- | --- |
| Accuracy matters most, or INT8 fails the compile | **bf16** |
| You need the speed and a smaller archive, and a small accuracy loss is acceptable | **INT8** — then measure it on real images; calibration quality decides how much is lost |
| Not sure | Build both and compare them with [`accuracy/`](accuracy/README.md) |

---

## Compile everything

```bash
./compile_all.sh                 # eleven models, serial, ~1.5–2 h; progress in compile_all.log
```

Same four steps per model as below, just scripted and safe to leave running. It collects each
model's artifacts into `assets/models/<id>/`.

`compile_all.sh` builds eleven models: the ten in the prebuilt zip plus `yolov8s`, each in its
`models.yaml` precision (`convnext_tiny` is bf16, the rest INT8). The other bf16
models — `yolov8s-worldv2` and `yolo26m` — and `yolo26s` are built on their own afterwards:
[sections 12–14](#12-yolov8s-worldv2--open-vocabulary-bf16-not-int8).

Or step by step across all models:

```bash
python compile/convert_to_onnx.py --all
python compile/graph_surgery.py   --all
python compile/compiler.py        --all       # serial; the long step
python compile/test_model.py      --all --validate-only
```

> **`--all` is not the same as `compile_all.sh`.** `--all` means *every enabled model in
> `models.yaml`*: the eleven above plus `yolov8l`, `yolov8s-worldv2`, `yolo26m` and `yolo26s`.
> `compiler.py` reads each model's `precision:` from `models.yaml`, so the bf16 models are built as
> bf16 (into `work/<id>/compile_bf16/`) and the rest as INT8 (into `work/<id>/compile_int8/`).

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
yolov8l              YoloV8/GroupedByRoleLogit    PASS
yolo11n              YoloV8/GroupedByRoleLogit    PASS
yolo11s              YoloV8/GroupedByRoleLogit    PASS
yolo26n              YoloV26/Auto                 PASS
yolo26s              YoloV26/Auto                 PASS
yolo26m              YoloV26/Auto                 PASS
yolo11s-seg          YoloV8Seg/GroupedByRoleLogit PASS
yolo26s-pose         YoloV26Pose/Auto             PASS
yolov8s-worldv2      YoloV8/GroupedByRoleLogit    PASS
yolox_s              YoloX/Split3Interleaved      PASS

every archive decodes on-device with its own BoxDecodeType, NMS applied
```

(Neat 0.5.0 / SDK 2.1.3.1, all models built in their `models.yaml` precision. Each block above it
prints per-image box counts and duplicate pairs; a `[WARN]` line for an isolated overlapping pair on a
YOLO26 model is expected, see [`BOX-DECODE.md`](BOX-DECODE.md).)

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

`models.yaml` builds `convnext_tiny` as **bf16**. With Model Compiler 2.1.3 its INT8 build compiles
cleanly (1 `.elf`, 0 `.so`) but loses most of its accuracy: on `000000000885` it scores racket 0.20,
against 0.73 for the FP32 ONNX. The bf16 build scores 0.75. The other three classifiers are fine in
INT8.

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

**Host** (`reports/` notes an **empty** `attention_rewrites`, unlike every YOLO11/YOLO26 model here):

```text
[surgery] yolov8s: OK  outputs=['bbox_0', 'bbox_1', 'bbox_2', 'class_logit_0', 'class_logit_1', 'class_logit_2']
[compile] yolov8s: precision=int8 batch=1 onnx=yolov8s.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolov8s (int8): rc=0  -> .../model-compilation/work/yolov8s/compile_int8
[PASS] yolov8s          elf=1 so=0 batch=1  (compile_int8/yolov8s.compile_ready_mpk.tar.gz)
```

**DevKit** — identical 6-tensor contract to `yolo11n`/`yolo11s`:

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

Decoding those heads on-device gives detections within 0.01–0.06 of the float ONNX on the same
images (`000000000139`: tv 0.93 vs 0.921 float, chair 0.83 vs 0.849) — the INT8 calibration is
sound, and scores are **not** capped the way a badly-quantized package's would be.

Two things differ from YOLO11 and both are handled in `compile/_surgery_ultralytics.py`:

- **The Detect head is at `model.22`, not `model.23`.** YOLO11 inserts a C2PSA block, which shifts
  every head node name by one. Copying the YOLO11 sources verbatim fails with
  `ValueError: missing head tensors`.
- **YOLOv8 has no attention block at all** — the export contains **0 `MatMul` nodes** — so
  `attention_blocks` is empty and the `MatMul → Einsum` rewrite is a no-op. That is expected, not a
  misconfiguration.

The `cv2.*` bbox heads emit 64 channels (4 × 16 bins), same as YOLO11. The surgery exposes them raw:
Neat's `YoloV8` decoder integrates the DFL and runs NMS on the board.

**`yolov8l`** uses the same surgery spec (Detect at `model.22`, no attention): run the same commands
with `--model-id yolov8l`.

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

**DevKit** — identical contract to `yolo11n`; head node names are scale-invariant, so `n`→`s` is a
free retarget:

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

YOLO26's `one2one_*` heads are already 4-channel distances and one-to-one, so this archive decodes
with `BoxDecodeType.YoloV26`, which runs no NMS (none is needed). YOLOv8/YOLO11 instead keep their
64-channel DFL heads and decode with `YoloV8`.

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

> ⚠️ **Do not remove the padding.** Keep `pad_channels_to: 64` in
> `compile/_surgery_ultralytics.py`. Unpadded, this model runs at **1782 ms/frame**; padded, at
> **8.5 ms/frame** — a **209× speedup** for identical weights. Full story:
> [the 209× pose fix](MODEL-COMPILATION.md#-the-209-pose-fix-padding-51--64-channels).

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

YOLO-World is **open-vocabulary**: normally it takes an image *plus* text prompts, and a CLIP text
encoder turns the prompts into class embeddings at runtime. A SiMa archive is a fixed graph, so the
vocabulary is **baked at export time** (`set_classes` → COCO-80), which drops the CLIP text encoder
and leaves an image-only graph. That is why this model has its own export and surgery steps.

> ⚠️ **This model does not compile as INT8, and it is NOT in `compile_all.sh`.** `models.yaml`
> carries `precision: bf16`, which `compiler.py` reads, so the plain command below builds it as
> bf16. The older form with `--bf16-weights --bf16-activations` still works and means the same.

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

`class_logit_*` is 80 channels because 80 classes were baked — that width follows your vocabulary,
not the model. On-device box decode uses `BoxDecodeType.YoloV8` with `GroupedByRoleLogit`, the same as
the other YOLOv8/YOLO11 detectors.

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
python compile/compiler.py       --model-id yolov8s-worldv2
```

One class name per line. Pass `--num-classes <N>` to `test_box_decode.py` if N is not 80, and note
that `assets/labels/coco80.txt` will no longer match the baked vocabulary.

---

### 13. `yolo26m` — detection, **bf16**

The medium YOLO26, compiled in **bf16** instead of INT8. bf16 keeps weights and activations as
16-bit floats, so there are no 8-bit quantization ranges to learn. The compiler still reads the
calibration images (the `calib=` line below), but the archive has no INT8 quantize steps, so the
choice of images matters far less. The cost is a larger program. `models.yaml` sets `precision: bf16` for this model,
so the usual four commands build it:

```bash
python compile/convert_to_onnx.py --model-id yolo26m   # -> 82 MB ONNX
python compile/graph_surgery.py   --model-id yolo26m   # same one2one heads as yolo26n
python compile/compiler.py        --model-id yolo26m   # precision: bf16; the long step, ≈16 min
python compile/test_model.py      --model-id yolo26m --validate-only
```

To build any other model in bf16, add `--precision bf16` to `compiler.py` (and to the test scripts):
`python compile/compiler.py --model-id yolo26n --precision bf16`.

**Host:**

```text
[surgery] yolo26m: OK  outputs=['bbox_0', 'bbox_1', 'bbox_2', 'class_logit_0', 'class_logit_1', 'class_logit_2']
[compile] yolo26m: precision=bf16 batch=1 onnx=yolo26m.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... afe.backends.mpk.interface - INFO - Plugin distribution per backend:
    ... afe.backends.mpk.interface - INFO - 	A65 : 0
[compile] yolo26m (bf16): rc=0  -> .../model-compilation/work/yolo26m/compile_bf16
[PASS] yolo26m          elf=1 so=0 batch=1  (compile_bf16/yolo26m.compile_ready_mpk.tar.gz)
```

Expect the compile step to take ≈16 min and ≈13 GB of RAM on a 16 GB SDK host — compile nothing
else alongside it. The archive is 69 MB; almost all of it is the `.elf` (137 MB uncompressed).

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

**How to confirm it really is bf16.** Count the kernels in the archive's contract. A bf16 build
converts with `cast_transform` and has no `quantization_transform`:

```bash
tar xzf work/yolo26m/compile_bf16/*/yolo26m.compile_ready_mpk.tar.gz -O --wildcards '*_mpk.json' \
  | grep -o '"kernel": "[a-z_]*"' | sort | uniq -c
```

```text
      7 "kernel": "cast_transform"
      6 "kernel": "detessellation_transform"
      1 "kernel": "pass_through"
      1 "kernel": "tessellation_transform"
      1 "kernel": "unpack_transform"
```

This build does its tessellation on the EV74 (the `tessellation_transform` step). SiMa's Model Zoo
archive `yolo26m-det-bf16-mla_tess-b1` does it inside the MLA instead, which is what `mla_tess` in
its name means; both run with the same Neat app code. The
[Quick Start Guide, Chapter 20](../devkit-quick-start/chapters/20-model-archive.md) explains every
file in the archive.

**Accuracy.** bf16 matched FP32 for `yolo26m` (+0.13 mAP50-95, on 500 COCO images). See
[`ACCURACY.md`](ACCURACY.md).

---

### 14. `yolo26s` — detection, **INT8 and bf16** (accuracy study)

`yolo26s` is compiled **both ways** — the same compile-ready ONNX, once per precision — to measure
what each precision costs in accuracy against FP32 on COCO. Each build gets its own folder, so they
do not overwrite each other:

```bash
python compile/convert_to_onnx.py --model-id yolo26s   # -> 38 MB ONNX
python compile/graph_surgery.py   --model-id yolo26s
python compile/compiler.py        --model-id yolo26s --precision int8   # -> work/yolo26s/compile_int8/
python compile/compiler.py        --model-id yolo26s --precision bf16   # -> work/yolo26s/compile_bf16/
python compile/test_model.py      --model-id yolo26s --precision int8 --validate-only
python compile/test_model.py      --model-id yolo26s --precision bf16 --validate-only
```

On COCO val2017 (4,980 images, all but the 20 calibration images), bf16 loses 0.07 mAP50-95 and
INT8 loses 2.67 against FP32. The
comparison — dataset, method, results and why they differ from Ultralytics' published numbers — is
in [`ACCURACY.md`](ACCURACY.md); the commands to rerun it are in
[`accuracy/README.md`](accuracy/README.md).

---

### 15. `yolo26m` at batch 6 — multi-batch, **INT8**

`--batch-size N` compiles the archive for N images per MLA call. The build goes to its own folder,
`compile_<precision>_b<N>/`. The batch-1 build of the same precision is needed to check it. Read
[`MULTI-BATCH.md`](MULTI-BATCH.md) first: on Neat 0.5.0 a
batched archive is slower, less accurate, and cannot be box-decoded on the device.

```bash
python compile/convert_to_onnx.py --model-id yolo26m
python compile/graph_surgery.py   --model-id yolo26m
python compile/compiler.py        --model-id yolo26m --precision int8 --batch-size 6   # -> compile_int8_b6/, ≈18 min
python compile/compiler.py        --model-id yolo26m --precision int8                  # -> compile_int8/, ≈8 min
python compile/test_model.py      --model-id yolo26m --precision int8 --batch-size 6 --validate-only
```

**Host:**

```text
[compile] yolo26m: precision=int8 batch=6 onnx=yolo26m.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... Plugin distribution per backend:
    ...   A65 : 0
    desired batch 6, achieved 6
[compile] yolo26m (int8_b6): rc=0  -> .../model-compilation/work/yolo26m/compile_int8_b6
[PASS] yolo26m          elf=1 so=0 batch=6  (compile_int8_b6/yolo26m.compile_ready_mpk.tar.gz)
```

The archive is 72 MB; the batch-1 INT8 build is 35 MB. Both times were measured on a 24-core host.

**Not in bf16.** `yolo26m`'s default precision is bf16, but that fails at batch 6.
`compiler.py --model-id yolo26m --batch-size 6` stops after calibration with:

```text
    mlc.compiler.model_graph.large_tensor_helper.CannotSplitLargeTensorError: MLA_0/conv2d_add_0: Cannot be segmented due to batch size. Please try with batch size 1 ...
[compile] yolo26m: batch 6 is not possible for this model at bf16: layer MLA_0/conv2d_add_0 needs large-tensor splitting, which the compiler cannot combine with batching. Use batch 1, or try --precision int8 (half-size tensors) if this was bf16.
[compile] yolo26m (bf16_b6): rc=1  -> .../model-compilation/work/yolo26m/compile_bf16_b6
```

**DevKit:** the five images in `assets/inference/` (the first one repeated to fill the sixth slot)
go in as one `[6, 640, 640, 3]` tensor, and each item is compared with the batch-1 build:

```bash
dk compile/test_model.py --model-id yolo26m --precision int8 --batch-size 6
```

```text
[PASS] yolo26m          elf=1 so=0 batch=6  (compile_int8_b6/yolo26m.compile_ready_mpk.tar.gz)
[test] yolo26m: batch 6 archive, 6 image(s) as one [6, 640, 640, 3] tensor
   6 output tensor(s): (6, 80, 80, 4) (6, 40, 40, 4) (6, 20, 20, 4) (6, 80, 80, 80) (6, 40, 40, 80) (6, 20, 20, 80)
   item 0 000000000139.jpg       vs batch-1: cosine min 0.95118 over 6 heads
   item 1 000000000885.jpg       vs batch-1: cosine min 0.92421 over 6 heads
   ...
   NOTE: batch items differ from the batch-1 build (worst cosine 0.92421; ...). Compare the accuracy and latency of batch 1 and batch 6 before choosing one: MULTI-BATCH.md

all archives: one .elf, zero .so
```

The batch-6 build runs, but it is **less accurate and slower** than batch 1, so compare both before
choosing a batch size:

- The two builds have the same quantization parameters, so the difference comes from the batch-6
  program.
- Against the FP32 ONNX, the batch-1 build stays at a worst-head cosine of 0.97–0.99, and the
  batch-6 build at 0.92–0.97.
- The batch-6 build finds fewer FP32 boxes: 10/15 and 3/6 on two images, where batch 1 finds
  14/15 and 6/6.

Speed (`model-benchmark`, 1000 synthetic frames, package default route; see
[`MULTI-BATCH.md`](MULTI-BATCH.md#7-results-yolo26m-int8-batch-1-vs-batch-6)):

| | throughput | latency |
|---|---|---|
| batch 1 | **189.4 img/s** | 9.10 ms |
| batch 6 | 156.5 img/s | 61.24 ms per 6 images |

`test_box_decode.py` skips a batched build, because Neat 0.5.0's box decode returns one result per call.

Profiling on 60 real images (`compile/profile_model.py`), `model-benchmark` and the full
analysis: [`MULTI-BATCH.md`](MULTI-BATCH.md).
