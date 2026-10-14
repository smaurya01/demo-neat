# Model Compilation for SiMa Modalix

Turn a public model into a SiMa archive (a `.tar.gz` holding **one `.elf`** that runs entirely on the
MLA), then check it on real images.

**How to use this page:**

1. [Set up](#setup) once.
2. [Pick your model](#pick-your-model) and open its commands.
3. [Run the four steps](#run-the-steps) and check for `PASS`.

Just want the archives? [Download the prebuilt archives](#download-the-prebuilt-archives).
Want to know how it works? See [Learn more](#learn-more).

## Table of Contents

- [Setup](#setup)
- [Pick your model](#pick-your-model)
- [Run the steps](#run-the-steps)
- [Download the prebuilt archives](#download-the-prebuilt-archives)
- [If something goes wrong](#if-something-goes-wrong)
- [Learn more](#learn-more)

---

## Setup

In the SDK container, activate the model-compiler environment and install `ultralytics`, which it
does not include (the YOLO models need it for export):

```bash
source /sdk-extensions/model-compiler/bin/activate
pip install ultralytics
cd model-compilation
```

- **The compile driver.** `compiler.py` calls the SDK's `quantize_compile.py`, installed with the
  `sima-model-quantize-compile` skill under `~/.codex/skills/` or `~/.claude/skills/`. If yours is
  elsewhere: `export SIMA_QUANTIZE_COMPILE=/path/to/quantize_compile.py`.
- **The DevKit.** The real-image test runs on the DevKit with `dk`
  ([Quick Start, Chapter 16](../devkit-quick-start/chapters/16-devkit-tool-dk.md)). `/workspace` is
  NFS-mounted there at the same path, so nothing is copied.

> ⚠️ **Compile one model at a time.** The compiler is memory-hungry; two compiles at once run out of
> memory.

---

## Pick your model

Click a model for its commands and expected output. All of them except `yolov8l` are also in the
[prebuilt zip](#download-the-prebuilt-archives).

| Model | Task | Precision |
| --- | --- | --- |
| [`resnet50`](COMPILE-COMMANDS.md#1-resnet50--classification-no-surgery) | classification | INT8 |
| [`convnext_tiny`](COMPILE-COMMANDS.md#2-convnext_tiny--classification-no-surgery-bf16) | classification | bf16 |
| [`densenet169`](COMPILE-COMMANDS.md#3-densenet169--classification-no-surgery) | classification | INT8 |
| [`efficientnet_v2_s`](COMPILE-COMMANDS.md#4-efficientnet_v2_s--classification-no-surgery-384384-input) | classification | INT8 |
| [`yolov8s`](COMPILE-COMMANDS.md#5-yolov8s--detection-surgery-head-at-model22-no-attention) / `yolov8l` | detection | INT8 |
| [`yolo11n`](COMPILE-COMMANDS.md#6-yolo11n--detection-surgery) | detection | INT8 |
| [`yolo11s`](COMPILE-COMMANDS.md#7-yolo11s--detection-surgery) | detection | INT8 |
| [`yolo26n`](COMPILE-COMMANDS.md#8-yolo26n--detection-surgery-one-to-one-head-yolov26) | detection | INT8 |
| [`yolo26s`](COMPILE-COMMANDS.md#14-yolo26s--detection-int8-and-bf16-accuracy-study) | detection | INT8 |
| [`yolo26m`](COMPILE-COMMANDS.md#13-yolo26m--detection-bf16) | detection | bf16 |
| [`yolox_s`](COMPILE-COMMANDS.md#11-yolox_s--detection-different-surgery) | detection | INT8 |
| [`yolo11s-seg`](COMPILE-COMMANDS.md#9-yolo11s-seg--segmentation-surgery) | segmentation | INT8 |
| [`yolo26s-pose`](COMPILE-COMMANDS.md#10-yolo26s-pose--pose-surgery-carries-the-209-fix) | pose | INT8 |
| [`yolov8s-worldv2`](COMPILE-COMMANDS.md#12-yolov8s-worldv2--open-vocabulary-bf16-not-int8) | open-vocabulary detection | bf16 |

Precision is set per model in [`models.yaml`](models.yaml). Any model can be built at the other
precision: see [`PRECISION.md`](PRECISION.md).

---

## Run the steps

Every model uses the same four steps on the SDK host. Replace `<ID>` with the model name:

```bash
python compile/convert_to_onnx.py --model-id <ID>                   # 1. download weights, export ONNX
python compile/graph_surgery.py   --model-id <ID>                   # 2. make it MLA-ready (no-op for classifiers)
python compile/compiler.py        --model-id <ID>                   # 3. compile: the long step, 3–16 min
python compile/test_model.py      --model-id <ID> --validate-only   # 4. check the archive
```

Step 4 must print `[PASS] <ID> elf=1 so=0`: the whole graph is on the MLA. The archive is in
`work/<ID>/compile_<precision>/`.

Then run it on real images on the DevKit. For detection, segmentation and pose models, also check
that Neat decodes the boxes:

```bash
dk compile/test_model.py      --model-id <ID>
dk compile/test_box_decode.py --model-id <ID>     # detection, segmentation, pose only
```

**Do not skip the DevKit test.** `elf=1 so=0` proves the model is on the MLA, not that it is
accurate. A model can compile perfectly and still be wrong; only real images show it.

> `dk` needs a real terminal. In a script or CI it hangs; ssh to the DevKit and run
> `/home/sima/pyneat/bin/python` instead.

---

## Download the prebuilt archives

> **📦 [models.zip](https://drive.google.com/drive/folders/1t-itiF25pUWF8AEPSEPrCDcpFiVs2phY?usp=sharing)** (475 MB):
> every model in the [table above](#pick-your-model) except `yolov8l`, built from the upstream
> weights with these scripts.

Unzip it inside `model-compilation/assets/`:

```bash
cd model-compilation/assets
unzip <download-dir>/models.zip
```

Each model gets its own folder, holding its archive at the precision shown in the table:

```text
assets/models/<model-id>/<archive>_mpk.tar.gz      <-- what an app loads
```

Two models also have extra builds, each in its own subfolder:

| Folder | What it holds |
| --- | --- |
| `yolo26s/bf16/` | `yolo26s` in bf16, for the [accuracy comparison](ACCURACY.md) |
| `yolo26m/int8/` | `yolo26m` in INT8, batch 1 |
| `yolo26m/int8_b6/` | `yolo26m` in INT8, batch 6 ([`MULTI-BATCH.md`](MULTI-BATCH.md)) |

`assets/models/` is git-ignored.

---

## If something goes wrong

| Symptom | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics`; it is not in the model-compiler environment |
| `quantize_compile.py driver was not found` | set `SIMA_QUANTIZE_COMPILE`; see [Setup](#setup) |
| compiler killed / out of memory | two compiles ran at once. Run one at a time |
| `[FAIL] ... no _mpk.tar.gz produced` | the compile failed. Read the log `compiler.py` prints: `work/<ID>/reports/compile_<precision>.log` |
| `so=1` or more, or `A65 : <non-zero>` in the log | part of the graph fell back to the host CPU. See [why surgery is needed](MODEL-COMPILATION.md#why-graph-surgery-is-needed) |
| `[compile] REFUSING: calibration set looks SYNTHETIC` | `--calib-dir` points at generated images. Use **real** images |
| `[surgery] ... the existing file is stale -- rerun with --force` | rerun `graph_surgery.py --model-id <ID> --force` |

---

## Learn more

You do not need these to compile a model.

| Read | To learn |
| --- | --- |
| [`MODEL-COMPILATION.md`](MODEL-COMPILATION.md) | How compilation works: the one-`.elf` target, graph surgery, calibration, and what worked and what didn't |
| [`PRECISION.md`](PRECISION.md) | INT8 vs bf16: the difference, which one each model uses and why, how to build and check the other |
| [`ACCURACY.md`](ACCURACY.md) | What each precision costs in accuracy, measured on COCO. To rerun it: [`accuracy/README.md`](accuracy/README.md) |
| [`BOX-DECODE.md`](BOX-DECODE.md) | Which Neat box decoder each model needs in your app, and what `test_box_decode.py` checks |
| [`MULTI-BATCH.md`](MULTI-BATCH.md) | Compiling for N images per MLA call (`--batch-size`), and profiling batch 1 against batch N |
| [`YOLO26X-STUDY.md`](YOLO26X-STUDY.md) | A case study: compiling and profiling the largest YOLO26, `yolo26x`, in INT8 |
| [`COMPILE-COMMANDS.md`](COMPILE-COMMANDS.md#compile-every-model) | Compiling every model in one go with `--all` |
