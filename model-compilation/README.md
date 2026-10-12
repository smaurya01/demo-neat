# Model Compilation for SiMa Modalix

Take a public model → get a **single `.tar.gz` containing a single `.elf`** that runs entirely on the
MLA → prove it works on real images.

Fourteen models are covered: 4 classification CNNs, 7 YOLO detectors, 1 open-vocabulary detector,
1 segmentation, 1 pose. Most are INT8; `yolov8s-worldv2` and `yolo26m` are **bf16**, and `yolo26s`
is built both ways for an [accuracy comparison](ACCURACY.md) of INT8 and bf16 against FP32.

**Two ways to get them:**

| | | |
| --- | --- | --- |
| **Download** the prebuilt archives | ~5 min | [→ jump](#download-the-prebuilt-archives) |
| **Compile** from source | ~2 h for the eleven in `compile_all.sh` | [→ jump](#setup) |

Compile from source when you want to change a model, try a different size, or understand the chain.
Otherwise, download.

**Why any of this is necessary** — graph surgery, the INT8 calibration trap, what worked and what
didn't: **[`MODEL-COMPILATION.md`](MODEL-COMPILATION.md)**.

**What compiling costs in accuracy** — FP32 vs bf16 vs INT8 on COCO:
**[`ACCURACY.md`](ACCURACY.md)**, summarized [below](#accuracy-fp32-vs-bf16-vs-int8).

## Table of Contents

- [Download the prebuilt archives](#download-the-prebuilt-archives)
  - [Manifest](#manifest)
- [Setup](#setup)
- [The scripts](#the-scripts)
- [Compile](#compile)
- [Multi-batch compilation](#multi-batch-compilation)
- [Accuracy: FP32 vs bf16 vs INT8](#accuracy-fp32-vs-bf16-vs-int8)
- [If something goes wrong](#if-something-goes-wrong)

---

## Download the prebuilt archives

Ten of the fourteen models have already been compiled from their **upstream original weights** through
this repo's own export → surgery → INT8 → compile chain. Nothing is pulled pre-compiled from the SiMa
model zoo. `yolov8s`, `yolov8s-worldv2`, `yolo26s` and `yolo26m` were added after that zip was
built — compile them with the [four steps](#the-scripts).

> **⚠️ The zip predates the decode fix.** Its YOLOv8/YOLO11 archives (`yolo11n`, `yolo11s`,
> `yolo11s-seg`) have their DFL folded into 4-channel l/t/r/b heads, which only decode as
> `YoloV26`. Since Neat 0.5.0 that decoder skips NMS, so those archives return a stack of duplicate
> boxes per object. Recompile them with the current surgery, which keeps YOLO11's own head and
> decodes as `YoloV8` (see [Step 4b](#step-4b--on-device-box-decode)). The YOLO26, YOLOX and
> classification archives in the zip are unaffected.

> **📦 [Models-v1.zip](https://drive.google.com/drive/folders/1t-itiF25pUWF8AEPSEPrCDcpFiVs2phY?usp=sharing)**

Unpack it into `model-compilation/assets/models/`. Nothing there is committed to git — `.pt`,
`.onnx` and `.tar.gz` are all ignored.

Each model folder holds the whole chain:

```text
assets/models/<model-id>/
    <model-id>.pt                    original weights   (Ultralytics models only)
    <model-id>.onnx                  the ONNX export that was compiled
    <model-id>..._mpk.tar.gz         the compiled SiMa archive   <-- this is what an app loads
```

torchvision and Megvii models have no `.pt` — torchvision weights come from its pretrained API,
Megvii ships a pre-exported ONNX. In both cases **the `.onnx` is the original**.

### Manifest

Every archive is verified: **1 `.elf`, 0 `.so`, `A65: 0`** — the whole graph runs on the MLA.

| Model | Task | Original | **Compiled archive** |
| --- | --- | --- | --- |
| `resnet50` | classification | *(torchvision)* | `resnet50_mpk.tar.gz` |
| `densenet169` | classification | *(torchvision)* | `densenet169_mpk.tar.gz` |
| `convnext_tiny` | classification | *(torchvision)* | `convnext_tiny_mpk.tar.gz` |
| `efficientnet_v2_s` | classification | *(torchvision)* | `efficientnet_v2_s_mpk.tar.gz` |
| `yolov8s` | detection | `.pt` | `yolov8s.compile_ready_mpk.tar.gz` — **not in the zip**, [compile it](COMPILE-COMMANDS.md#5-yolov8s--detection-surgery-head-at-model22-no-attention) |
| `yolo11n` | detection | `.pt` | `yolo11n.compile_ready_mpk.tar.gz` |
| `yolo11s` | detection | `.pt` | `yolo11s.compile_ready_mpk.tar.gz` |
| `yolo26n` | detection | `.pt` | `yolo26n.compile_ready_mpk.tar.gz` |
| `yolo11s-seg` | segmentation | `.pt` | `yolo11s-seg.compile_ready_mpk.tar.gz` |
| `yolo26s-pose` | pose | `.pt` | `yolo26s-pose.compile_ready_mpk.tar.gz` |
| `yolox_s` | detection | *(Megvii ONNX)* | `yolox_s.compile_ready_mpk.tar.gz` |
| `yolov8s-worldv2` | open-vocabulary detection | `.pt` | `yolov8s-worldv2.compile_ready_mpk.tar.gz` — **bf16, not in the zip**, [compile it](COMPILE-COMMANDS.md#12-yolov8s-worldv2--open-vocabulary-bf16-not-int8) |
| `yolo26m` | detection | `.pt` | `yolo26m.compile_ready_mpk.tar.gz` — **bf16, not in the zip**, [compile it](COMPILE-COMMANDS.md#13-yolo26m--detection-bf16) |
| `yolo26s` | detection | `.pt` | `yolo26s.compile_ready_mpk.tar.gz` — **not in the zip** (INT8; also built in bf16 for the [accuracy comparison](ACCURACY.md)), [compile it](COMPILE-COMMANDS.md#14-yolo26s--detection-int8-and-bf16-accuracy-study) |


---

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

Steps 1–3 and `test_model.py --validate-only` use `python` here in the SDK container. Step 4's real
inference and step 4b run on the DevKit with `dk` — `/workspace` is NFS-mounted there at the same
path, so nothing is copied.

> ⚠️ **Compile strictly ONE model at a time.** The compiler is memory-hungry; concurrent compiles
> OOM. This is the single most common way to waste an hour here.

---

## The scripts

One registry — `models.yaml` — drives all four steps, so a model's input name, shape, normalization
and output contract can never drift between them.

```
compile/
  convert_to_onnx.py   # 1. download weights + export  -> work/<id>/onnx/<id>.onnx
  graph_surgery.py     # 2. make it MLA-ready          -> work/<id>/surgery/<id>.compile_ready.onnx
  compiler.py          # 3. INT8/bf16 + compile        -> work/<id>/compile_<precision>/<...>_mpk.tar.gz
  test_model.py        # 4. validate contract + run on REAL images
  test_box_decode.py   # 4b. does Neat decode the heads ON-DEVICE with the family's own type?
  profile_model.py     # 5. DevKit: image in -> image out timing, step by step, on 50+ real images
```

Every script takes `--model-id <id>` for one model, or `--all` for every enabled model.

**The generic recipe — identical for every model:**

```bash
python compile/convert_to_onnx.py --model-id <ID>    # downloads the weights automatically
python compile/graph_surgery.py   --model-id <ID>    # no-op for CNNs (surgery: none)
python compile/compiler.py        --model-id <ID>    # the long step
python compile/test_model.py      --model-id <ID> --validate-only   # contract check, no board
```

The behavioural check runs on the DevKit, with `dk`:

```bash
source /usr/local/bin/devkit.sh <devkit-ip> sima 22   # dk is a bash function, once per shell
dk compile/test_model.py --model-id <ID>
```

`dk` needs a real terminal — in a non-interactive context (CI, scripts) it hangs; there, ssh in and call
`/home/sima/pyneat/bin/python` directly instead.

Useful flags on `compiler.py`: `--num-calib-samples N`, `--calib-dir <dir>`. Anything else passes
straight through to the compiler (e.g. `--calib_method min_max`).

**Precision: INT8 or bf16.** Every model in `models.yaml` has a `precision:` line (`int8` or
`bf16`), and `compiler.py` builds that. `--precision int8|bf16` overrides it for one build, and
each precision gets its own output folder, `work/<id>/compile_int8/` or `work/<id>/compile_bf16/`. INT8 learns
8-bit ranges from the calibration images; bf16 keeps 16-bit floats, does not depend on 8-bit
ranges, and produces a bigger program. `test_model.py` and `test_box_decode.py` take the same `--precision`
flag. How to pick the precision per model:
[COMPILE-COMMANDS.md](COMPILE-COMMANDS.md#choosing-the-precision-int8-or-bf16). What each precision
costs in accuracy is measured in [`ACCURACY.md`](ACCURACY.md).

**Both checks matter.** `--validate-only` proves the graph is *on the MLA*; it proves nothing about
accuracy. Only the DevKit run on real images catches a model that compiled perfectly and is
numerically wrong — see [what didn't work](MODEL-COMPILATION.md#what-worked-what-didnt).

### Step 4b — on-device box decode

`test_model.py` prints the raw head **shapes**. It does not tell you whether Neat can turn those
heads into boxes itself. That is worth knowing: an app that host-decodes raw heads pays 143–337
ms/frame, while Neat's on-device `neatobjectdecode` stage costs ~0.6 ms.

```bash
dk compile/test_box_decode.py --all               # every detection, segmentation and pose model
dk compile/test_box_decode.py --model-id yolov8s
```

**Every family decodes with its own Neat `BoxDecodeType`**, which `models.yaml` records per model as
`box_decode` / `box_decode_option` (set exactly this pair in your app's `ModelOptions`):

| Archive | `decode_type` | `decode_type_option` | Head the surgery exposes |
| --- | --- | --- | --- |
| `yolov8s`, `yolov8l`, `yolo11n`, `yolo11s`, `yolov8s-worldv2` | `YoloV8` | `GroupedByRoleLogit` | raw 64-channel DFL bbox heads + class logits |
| `yolo11s-seg` | `YoloV8Seg` | `GroupedByRoleLogit` | the same, plus mask coefficients and protos |
| `yolo26n`, `yolo26s`, `yolo26m` | `YoloV26` | `Auto` | one-to-one heads, 4 distance channels |
| `yolo26s-pose` | `YoloV26Pose` | `Auto` | one-to-one heads plus keypoints |
| `yolox_s` | `YoloX` | `Split3Interleaved` | decoupled bbox / objectness / class heads |

Why the family matters: YOLOv8 and YOLO11 heads are **one-to-many** (several anchors fire per
object) and need NMS, which `YoloV8` runs on the board. YOLO26 heads are **one-to-one**, and since
Neat 0.5.0 `YoloV26` skips NMS by design. A YOLO11 decoded as `YoloV26` therefore returns a stack of
near-identical boxes per object. `GroupedByRoleLogit` tells `YoloV8` the class heads are logits (the
surgery cuts before Ultralytics' sigmoid; keeping logits through INT8 preserves recall), so the
decoder applies the sigmoid itself. Without it, scores come out above 1.

The script builds each archive with exactly that pair, runs real images, and **fails** if the build
is rejected, if no image produces a box, if a score falls outside [0, 1], or if same-class boxes
overlapping above the NMS IoU are systematic (more than one pair per five boxes: NMS did not run). An
isolated overlapping pair is printed as a warning. YOLO26's one-to-one head, compiled INT8 or bf16,
occasionally returns one or two near-identical pairs that FP32 Ultralytics does not (a second
anchor's score crosses the threshold after quantization). `YoloV26` runs no NMS, so they reach the
app; measured on `yolo26s`: 0–3 such pairs per COCO image. An app that must never show one can run a
light class-aware NMS on the decoded boxes.

---

## Compile

**→ [`COMPILE-COMMANDS.md`](COMPILE-COMMANDS.md)** — the copy-paste commands: `compile_all.sh` for
eleven INT8 models — the ten in the zip plus `yolov8s` (~2 h, serial) — or a per-model block for
each of the fourteen with its exact expected output.

---

## Multi-batch compilation

**→ [`MULTI-BATCH.md`](MULTI-BATCH.md)** is the full guide: the multi-batch INT8 compile, its checks,
step-by-step runtime profiling on 60 real images with `compile/profile_model.py`, the
`model-benchmark` numbers, and the `yolo26m` batch 1 vs batch 6 results. The summary follows.

A model can be compiled to run **N images per MLA call** instead of one. Example: `yolo26m` at
batch 6, in **INT8**. bf16 does not fit at batch 6; see [Which models can be batched](#which-models-can-be-batched).

```bash
python compile/compiler.py   --model-id yolo26m --precision int8 --batch-size 6
python compile/compiler.py   --model-id yolo26m --precision int8      # batch 1, to compare against
python compile/test_model.py --model-id yolo26m --precision int8 --batch-size 6 --validate-only  # host
python compile/test_model.py --model-id yolo26m --precision int8 --batch-size 6                  # DevKit
```

- **Where it goes:** `work/yolo26m/compile_int8_b6/`, next to the batch-1 build in `compile_int8/`.
  The precision rules are unchanged, and `--precision` still overrides the registry.
- **What changes:**
  - Calibration still sees one image at a time; the ONNX input stays `[1,3,640,640]`.
  - The compiler builds the batch-6 MLA program, and the archive's input becomes `[6,640,640,3]`.
  - `compiler.py` fails if the compiler's log does not report `Achieved batch size: 6`.
- **Making it the default:** for a model that should always be batched, set `batch_size: 6` in
  its `models.yaml` entry instead of passing the flag.
- **Checking it:**
  - `test_model.py --batch-size 6` checks that the archive declares batch 6.
  - On the DevKit it also runs six images as one tensor and compares each item's raw outputs with
    the batch-1 build, which must be compiled too.
  - `test_box_decode.py` skips batched archives (see below).

### Which models can be batched

At batch 1, the compiler splits a layer that does not fit in MLA memory into spatial tiles
("Trigger large tensor support due to: …" in `reports/compile_<precision>.log`). It cannot also
split such a layer by batch. Any batch above 1 then fails with
`CannotSplitLargeTensorError: … Cannot be segmented due to batch size`, and `compiler.py` says so.

| model | batch-1 log needs large-tensor splitting | batch 6 |
|---|---|---|
| `yolo11n` INT8 | no | ✅ compiles |
| `yolo26m` INT8 | no | ✅ compiles (≈18 min) |
| `yolo26m` bf16 | yes, on the input (bf16 doubles its size) | ❌ fails at the first conv |
| `yolo26x` INT8 | yes, first conv | ❌ fails at the first conv |

Grep the batch-1 log before you spend a compile on a batched build.

### Running a batched archive

**Before you use a batched archive, know what Neat 0.5.0 does with it** (measured, `yolo11n` INT8
and `yolo26m` INT8):

| | batch 1 | batch N |
|---|---|---|
| `Model.compiled_batch_size()` | 1 | N |
| Tensor route (raw heads, or `Model.benchmark()`) | works | works. Input is one `[N,H,W,3]` tensor with `tensor.axis_semantics = [N, H, W, C]` (otherwise it is rejected). Each head comes back as `[N, …]`, e.g. `(6, 80, 80, 4)`. |
| Image route (Neat preprocess + box decode) | works | **times out** in the preprocess stage |
| On-device box decode | one result per image | **one result for the whole batch**: the other N−1 images are lost |

So, on Neat 0.5.0, a batched YOLO archive can be run only on the tensor route. You do the
letterbox and normalization yourself, and you decode the raw heads on the host.

**It is not automatically faster either.** Throughput is images per second with frames kept in
flight; latency is one call:

| model | batch 1 | batch 6 |
|---|---|---|
| `yolo11n` INT8 | **614.7 img/s**, 5.7 ms | 507.6 img/s, 31.3 ms per 6 images |
| `yolo26m` INT8 | **187.9 img/s**, 9.1 ms | 156.4 img/s, 61.3 ms per 6 images |

A pipelined batch-1 runner, with several frames in flight through `push`/`pull`, keeps the MLA as
busy as a batched one, without the batch latency.

**And it was less accurate.** `yolo26m` INT8, the same image through both builds, each compared with
the FP32 ONNX:

- **Raw heads:** the batch-1 build agrees with FP32 at a worst-head cosine of 0.97–0.99. The batch-6
  build agrees at only 0.92–0.97, mostly in the box heads.
- **Detections found** (FP32 boxes matched at IoU > 0.5):

  | image | batch 1 | batch 6 |
  |---|---|---|
  | `000000000885` | 14/15 | 10/15 |
  | `000000001296` | 6/6 | 3/6 |

- **Not the calibration:** the two builds have the same quantization parameters, and the batch-6
  program is deterministic, with no data leaking between items.
- `test_model.py --batch-size 6` fails such a build, because its items do not match the batch-1 build.

Measure a batched build for speed **and** accuracy before you use it. For these models on Neat 0.5.0,
batch 1 was the better choice on both.

---

## Accuracy: FP32 vs bf16 vs INT8

The MLA does not run FP32, so every model is compiled to **bf16** (16-bit floats) or **INT8** (8-bit
integers, calibrated on real images). What that costs was measured for Ultralytics `yolo26s` on the
full COCO val2017 set (4,980 images — all 5,000 minus the 20 calibration images):

| Precision | mAP50-95 | vs FP32 | Inference | Archive |
| --- | --- | --- | --- | --- |
| FP32 (reference, SDK host) | 47.71 | — | — | — |
| **bf16** (DevKit MLA) | **47.64** | **−0.07** | 18.4 ms | 40 MB |
| **INT8** (DevKit MLA) | **45.05** | **−2.67** | 11.1 ms | 23 MB |

- **bf16 keeps FP32 accuracy.** `yolo26m` in bf16 also matched FP32 (+0.13, on 500 images).
- **INT8 costs 2.67 mAP50-95** (about 5.6%) and runs **1.7× faster** than bf16.
- The three rows share the same graph, images, preprocessing, decoder and COCO evaluator, so the
  differences are the compiler alone.
- Ultralytics publishes **47.8** for `yolo26s`. The FP32 row here is 47.71, and the original `.pt`
  through Ultralytics' own `predict()` scores 47.41 in this setup. These gaps most likely come from
  how the model is run (`predict()` vs `val()`, padding), the 20 missing images and a different
  evaluator, not from the model. [`ACCURACY.md`](ACCURACY.md#why-this-differs-from-the-accuracy-ultralytics-publishes)
  explains each one.

**→ [`ACCURACY.md`](ACCURACY.md)** — the full comparison: method, results, the published-number
gap, which precision to choose, and limits.

**→ [`accuracy/README.md`](accuracy/README.md)** — how to replicate these numbers: download the model
and COCO val2017, compile INT8 and bf16, run FP32 on the host and bf16 / INT8 on the DevKit, and
score, step by step with expected output.

---

## If something goes wrong

| Symptom | Cause |
| --- | --- |
| `[FAIL] ... no _mpk.tar.gz produced` | the compile failed — read `work/<id>/reports/compile_<precision>.log` (`compile.log` for builds made before the precision option) |
| `so=1` or more | part of the graph fell back to the host CPU; surgery did not remove everything the MLA cannot place |
| `A65 : <non-zero>` in the log | same thing, visible earlier |
| `[compile] REFUSING: calibration set looks SYNTHETIC` | you pointed `--calib-dir` at generated images. Quantization needs **real** images, or the model compiles clean and is quietly wrong |
| compiler killed / OOM | you ran two compiles at once. One at a time |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` — it is not in the model-compiler env |

**A green compile is not a working model.** `rc=0` + one `.elf` + zero `.so` proves the graph is on
the MLA. It proves *nothing* about accuracy — always run the DevKit test on real images. We shipped
a model that passed every contract check and predicted an unrelated class on every image; the story
is in [`MODEL-COMPILATION.md`](MODEL-COMPILATION.md#what-worked-what-didnt).
