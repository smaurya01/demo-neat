# Model Compilation for SiMa Modalix

Take a public model → get a **single `.tar.gz` containing a single `.elf`** that runs entirely on the
MLA → prove it works on real images.

Fifteen models are covered: 4 classification CNNs, 8 YOLO detectors, 1 open-vocabulary detector,
1 segmentation, 1 pose. Most are INT8; `convnext_tiny`, `yolov8s-worldv2` and `yolo26m` are **bf16**, and `yolo26s`
is built both ways for an [accuracy comparison](ACCURACY.md) of INT8 and bf16 against FP32.

**Two ways to get them:**

| | | |
| --- | --- | --- |
| **Download** the prebuilt archives | ~5 min | [→ jump](#download-the-prebuilt-archives) |
| **Compile** from source | ~1.5–2 h for the eleven in `compile_all.sh` | [→ jump](#setup) |

Compile from source when you want to change a model, try a different size, or understand the chain.
Otherwise, download.

**Why any of this is necessary** — graph surgery, the INT8 calibration trap, what worked and what
didn't: **[`MODEL-COMPILATION.md`](MODEL-COMPILATION.md)**.

**What compiling costs in accuracy** — FP32 vs bf16 vs INT8 on COCO:
**[`ACCURACY.md`](ACCURACY.md)**.

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

Ten of the fifteen models have already been compiled from their **upstream original weights** through
this repo's own export → surgery → INT8 → compile chain. Nothing is pulled pre-compiled from the SiMa
model zoo. `yolov8s`, `yolov8l`, `yolov8s-worldv2`, `yolo26s` and `yolo26m` were added after that zip was
built — compile them with the [four steps](#the-scripts).

> **⚠️ The zip predates the decode fix. Recompile four of its archives.**
> - `yolo11n`, `yolo11s` and `yolo11s-seg` have their DFL folded into 4-channel l/t/r/b heads, which
>   only decode as `YoloV26`. Since Neat 0.5.0 that decoder skips NMS, so they return a stack of
>   duplicate boxes per object. The current surgery keeps YOLO11's own head, decoded as `YoloV8`.
> - `yolox_s` exposes 3 packed 85-channel heads, which `YoloX` / `Split3Interleaved` cannot decode.
>   The current surgery exposes the 9 split heads.
>
> The YOLO26 and classification archives in the zip are unaffected. See [`BOX-DECODE.md`](BOX-DECODE.md).

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

Every archive compiles to **1 `.elf`, 0 `.so`, `A65: 0`**: the whole graph runs on the MLA.

| Model | Task | Original | **Compiled archive** |
| --- | --- | --- | --- |
| `resnet50` | classification | *(torchvision)* | `resnet50_mpk.tar.gz` |
| `densenet169` | classification | *(torchvision)* | `densenet169_mpk.tar.gz` |
| `convnext_tiny` | classification | *(torchvision)* | `convnext_tiny_mpk.tar.gz` (INT8 in the zip; it now compiles as **bf16**, [why](COMPILE-COMMANDS.md#2-convnext_tiny--classification-no-surgery-bf16)) |
| `efficientnet_v2_s` | classification | *(torchvision)* | `efficientnet_v2_s_mpk.tar.gz` |
| `yolov8s` | detection | `.pt` | `yolov8s.compile_ready_mpk.tar.gz` — **not in the zip**, [compile it](COMPILE-COMMANDS.md#5-yolov8s--detection-surgery-head-at-model22-no-attention) |
| `yolov8l` | detection | `.pt` | `yolov8l.compile_ready_mpk.tar.gz` — **not in the zip**, compile it like [`yolov8s`](COMPILE-COMMANDS.md#5-yolov8s--detection-surgery-head-at-model22-no-attention) with `--model-id yolov8l` |
| `yolo11n` | detection | `.pt` | `yolo11n.compile_ready_mpk.tar.gz` — **recompile** (see the note above) |
| `yolo11s` | detection | `.pt` | `yolo11s.compile_ready_mpk.tar.gz` — **recompile** (see the note above) |
| `yolo26n` | detection | `.pt` | `yolo26n.compile_ready_mpk.tar.gz` |
| `yolo11s-seg` | segmentation | `.pt` | `yolo11s-seg.compile_ready_mpk.tar.gz` — **recompile** (see the note above) |
| `yolo26s-pose` | pose | `.pt` | `yolo26s-pose.compile_ready_mpk.tar.gz` |
| `yolox_s` | detection | *(Megvii ONNX)* | `yolox_s.compile_ready_mpk.tar.gz` — **recompile** (see the note above) |
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

`compiler.py` calls the SDK's `quantize_compile.py` driver, from the `sima-model-quantize-compile`
skill that the Neat SDK installs for its coding agents. It looks in
`~/.codex/skills/sima-model-quantize-compile/scripts/` and then `~/.claude/skills/...`; if yours is
elsewhere, `export SIMA_QUANTIZE_COMPILE=/path/to/quantize_compile.py`.

Steps 1–3 and `test_model.py --validate-only` run with `python` here in the SDK container. Step 4's
real inference, step 4b and step 5 run on the DevKit with `dk` — `/workspace` is NFS-mounted there
at the same path, so nothing is copied.

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
  compiler.py          # 3. INT8/bf16 + compile        -> work/<id>/compile_<precision>[_b<N>]/<...>_mpk.tar.gz
  test_model.py        # 4. validate contract + run on REAL images
  test_box_decode.py   # 4b. does Neat decode the heads on the device? -> BOX-DECODE.md
  profile_model.py     # 5. DevKit: image in -> image out timing, step by step, on 50+ real images
```

Steps 1–4b take `--model-id <id>` for one model, or `--all` for every enabled model.
`profile_model.py` takes `--model-id` and `--images` (see [`MULTI-BATCH.md`](MULTI-BATCH.md)).

**The generic recipe — identical for every model:**

```bash
python compile/convert_to_onnx.py --model-id <ID>    # downloads the weights automatically
python compile/graph_surgery.py   --model-id <ID>    # no-op for CNNs (surgery: none)
python compile/compiler.py        --model-id <ID>    # the long step
python compile/test_model.py      --model-id <ID> --validate-only   # contract check, no board
```

The behavioural check runs on the DevKit, with `dk` (available in the SDK once it is paired; see
[Quick Start, Chapter 16](../devkit-quick-start/chapters/16-devkit-tool-dk.md)):

```bash
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

**Which decoder each model needs** (`decode_type` / `decode_type_option` for your app) and how
`test_box_decode.py` checks it on the DevKit: **[`BOX-DECODE.md`](BOX-DECODE.md)**.

---

## Compile

**→ [`COMPILE-COMMANDS.md`](COMPILE-COMMANDS.md)** — the copy-paste commands: `compile_all.sh` for
eleven models — the ten in the zip plus `yolov8s` (~1.5–2 h, serial) — or a per-model block for
each model with its exact expected output.

---

## Multi-batch compilation

`compiler.py --batch-size N` builds an archive that runs N images per MLA call, in its own folder
`compile_<precision>_b<N>/`. Not every model can be batched, and on Neat 0.5.0 a batched archive
cannot be box-decoded on the device; for `yolo26m`, batch 6 was slower and less accurate than
batch 1, and for `yolo11n` it was slower too. The compile, checks, profiling and results: **[`MULTI-BATCH.md`](MULTI-BATCH.md)**.
The `yolo26x` INT8 compile and runtime profile: [`YOLO26X-STUDY.md`](YOLO26X-STUDY.md).

---

## Accuracy: FP32 vs bf16 vs INT8

The MLA does not run FP32, so every model is compiled to bf16 or INT8. Measured on `yolo26s` over
COCO val2017 (on Neat 0.4.0), bf16 stays within 0.07 mAP50-95 of FP32, while INT8 loses 2.67 but runs 1.7× faster
than bf16.
The method and full results are in **[`ACCURACY.md`](ACCURACY.md)**; to reproduce them, follow
[`accuracy/README.md`](accuracy/README.md).

---

## If something goes wrong

| Symptom | Cause |
| --- | --- |
| `[FAIL] ... no _mpk.tar.gz produced` | the compile failed — read `work/<id>/reports/compile_<precision>[_b<N>].log` (`compiler.py` prints the path; `compile.log` for builds made before the precision option) |
| `so=1` or more | part of the graph fell back to the host CPU; surgery did not remove everything the MLA cannot place |
| `A65 : <non-zero>` in the log | same thing, visible earlier |
| `[compile] REFUSING: calibration set looks SYNTHETIC` | you pointed `--calib-dir` at generated images. Quantization needs **real** images, or the model compiles clean and is quietly wrong |
| compiler killed / OOM | you ran two compiles at once. One at a time |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` — it is not in the model-compiler env |

**A green compile is not a working model.** `rc=0` + one `.elf` + zero `.so` proves the graph is on
the MLA. It proves *nothing* about accuracy — always run the DevKit test on real images. A model can
pass every contract check and still predict an unrelated class on every image; see
[`MODEL-COMPILATION.md`](MODEL-COMPILATION.md#what-worked-what-didnt).
