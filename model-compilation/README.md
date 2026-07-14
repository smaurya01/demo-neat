# Model Compilation for SiMa Modalix

Take a public model → get a **single `.tar.gz` containing a single `.elf`** that runs entirely on the
MLA → prove it works on real images.

Ten models are covered: 4 classification CNNs, 4 YOLO detectors, 1 segmentation, 1 pose.

**Two ways to get them:**

| | | |
| --- | --- | --- |
| **Download** the prebuilt archives | ~5 min | [→ jump](#download-the-prebuilt-archives) |
| **Compile** from source | ~2 h for all ten | [→ jump](#setup) |

Compile from source when you want to change a model, try a different size, or understand the chain.
Otherwise, download.

**Why any of this is necessary** — graph surgery, the INT8 calibration trap, what worked and what
didn't: **[`MODEL-COMPILATION.md`](MODEL-COMPILATION.md)**.

---

## Download the prebuilt archives

All ten models have already been compiled from their **upstream original weights** through this
repo's own export → surgery → INT8 → compile chain. Nothing is pulled pre-compiled from the SiMa
model zoo.

> **📦 [Models-v1.zip](https://drive.google.com/file/d/10zOUhD56VrZaY3urSHgqpjGZjIY3jEKt/view?usp=sharing)**

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
| `yolo11n` | detection | `.pt` | `yolo11n.compile_ready_mpk.tar.gz` |
| `yolo11s` | detection | `.pt` | `yolo11s.compile_ready_mpk.tar.gz` |
| `yolo26n` | detection | `.pt` | `yolo26n.compile_ready_mpk.tar.gz` |
| `yolo11s-seg` | segmentation | `.pt` | `yolo11s-seg.compile_ready_mpk.tar.gz` |
| `yolo26s-pose` | pose | `.pt` | `yolo26s-pose.compile_ready_mpk.tar.gz` |
| `yolox_s` | detection | *(Megvii ONNX)* | `yolox_s.compile_ready_mpk.tar.gz` |


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

Steps 1–3 run on the **host** (SDK container). Step 4's inference runs on the **DevKit** —
`/workspace` is NFS-mounted there at the same path, so nothing is copied.

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
  compiler.py          # 3. INT8 quantize + compile    -> work/<id>/compile_int8/<...>_mpk.tar.gz
  test_model.py        # 4. validate contract + run on REAL images
```

Every script takes `--model-id <id>` for one model, or `--all` for every enabled model.

**The generic recipe — identical for every model:**

```bash
python compile/convert_to_onnx.py --model-id <ID>    # downloads the weights automatically
python compile/graph_surgery.py   --model-id <ID>    # no-op for CNNs (surgery: none)
python compile/compiler.py        --model-id <ID>    # the long step
python compile/test_model.py      --model-id <ID> --validate-only   # host: contract check

# then on the DevKit, for the behavioural check:
ssh sima@<devkit-ip>
source ~/pyneat/bin/activate && cd model-compilation
python compile/test_model.py --model-id <ID>
```

Useful flags on `compiler.py`: `--num-calib-samples N`, `--calib-dir <dir>`. Anything else passes
straight through to the compiler (e.g. `--calib_method min_max`).

**Both checks matter.** `--validate-only` proves the graph is *on the MLA*; it proves nothing about
accuracy. Only the DevKit run on real images catches a model that compiled perfectly and is
numerically wrong — see [what didn't work](MODEL-COMPILATION.md#what-worked-what-didnt).

---

## Compile

**→ [`COMPILE-COMMANDS.md`](COMPILE-COMMANDS.md)** — the copy-paste commands: `compile_all.sh` for
all ten (~2 h, serial), or a per-model block for each of the ten with its exact expected output.

---

## If something goes wrong

| Symptom | Cause |
| --- | --- |
| `[FAIL] ... no _mpk.tar.gz produced` | the compile failed — read `work/<id>/reports/compile.log` |
| `so=1` or more | part of the graph fell back to the host CPU; surgery did not remove everything the MLA cannot place |
| `A65 : <non-zero>` in the log | same thing, visible earlier |
| `[compile] REFUSING: calibration set looks SYNTHETIC` | you pointed `--calib-dir` at generated images. Quantization needs **real** images, or the model compiles clean and is quietly wrong |
| compiler killed / OOM | you ran two compiles at once. One at a time |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` — it is not in the model-compiler env |

**A green compile is not a working model.** `rc=0` + one `.elf` + zero `.so` proves the graph is on
the MLA. It proves *nothing* about accuracy — always run the DevKit test on real images. We shipped
a model that passed every contract check and predicted an unrelated class on every image; the story
is in [`MODEL-COMPILATION.md`](MODEL-COMPILATION.md#what-worked-what-didnt).
