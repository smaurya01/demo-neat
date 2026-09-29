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
- [Accuracy: FP32 vs bf16 vs INT8](#accuracy-fp32-vs-bf16-vs-int8)
- [If something goes wrong](#if-something-goes-wrong)

---

## Download the prebuilt archives

Ten of the fourteen models have already been compiled from their **upstream original weights** through
this repo's own export → surgery → INT8 → compile chain. Nothing is pulled pre-compiled from the SiMa
model zoo. `yolov8s`, `yolov8s-worldv2`, `yolo26s` and `yolo26m` were added after that zip was
built — compile them with the [four steps](#the-scripts).

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
  test_box_decode.py   # 4b. can Neat decode the heads ON-DEVICE?  (detection models)
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

### Step 4b — on-device box decode (detection models)

`test_model.py` prints the raw head **shapes**. It does not tell you whether Neat can turn those
heads into boxes itself. That is worth knowing: an app that host-decodes raw heads pays 143–337
ms/frame, while Neat's on-device `neatobjectdecode` stage costs ~0.6 ms.

```bash
dk compile/test_box_decode.py --all               # every detection model
dk compile/test_box_decode.py --model-id yolov8s
```

It tries each candidate `BoxDecodeType` against the archive and reports which one works, because
**the decode type does not follow the model's name**:

| Archive | Works | Rejected |
| --- | --- | --- |
| `yolov8s`, `yolo11n`, `yolo11s`, `yolo26n`, `yolov8s-worldv2` | `BoxDecodeType.YoloV26` | `YoloV8` — *"failed to issue model-managed boxdecode contract"* |
| `yolox_s` | `BoxDecodeType.YoloX` (`Split3Interleaved`) | — |

A decode type that builds but returns **zero boxes on every image** is a failure, not a pass — that
is exactly how a wrong layout presents, so the script runs real images and counts detections.

Pose and segmentation are excluded from `--all`: their archives carry the same 6 detection heads,
so they would pass on box heads alone while their real contract (`YoloV26Pose` + `decode_pose`,
`YoloV26Seg` + `decode_segmentation`) went untested.

---

## Compile

**→ [`COMPILE-COMMANDS.md`](COMPILE-COMMANDS.md)** — the copy-paste commands: `compile_all.sh` for
eleven INT8 models — the ten in the zip plus `yolov8s` (~2 h, serial) — or a per-model block for
each of the fourteen with its exact expected output.

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
