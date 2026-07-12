# Replication — compile every model, one at a time

> ### Don't want to compile? Download the prebuilt archives.
>
> Every model below has already been compiled, and the artifacts — **original weights + ONNX export +
> the compiled SiMa `.tar.gz`** — are published as a download. Grab those, drop them into the app's
> `assets/models/`, and skip this whole document.
>
> **→ See [`assets/models/README.md`](assets/models/README.md)** for the download link, the manifest,
> and exactly which archive each app expects (including the two that need renaming).
>
> Compile from source when you want to change a model, retarget a different size, or understand the
> chain. Otherwise, download.

Copy-paste blocks. One model per section. Nothing is downloaded by hand — **step 1 fetches the
weights for you**.

Run steps 1–3 on the **host** (SDK container). Run step 4's board test on the **DevKit**
(`/workspace` is NFS-mounted there, so there is nothing to copy).

> **Compile ONE model at a time.** The compiler is memory-hungry — concurrent compiles OOM.

## Compile everything unattended

`compile_all.sh` runs all ten, strictly serially, and collects each model's artifacts into
`assets/models/<id>/`:

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation
./compile_all.sh                 # ~2 h; progress in compile_all.log
```

It is the same four steps per model as the sections below — just scripted, and safe to leave running.

Common setup, once per shell:

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation
```

Board test, same pattern for every model (replace `<ID>`):

```bash
ssh sima@<devkit-ip>
source ~/pyneat/bin/activate
cd model-compilation
python compile/test_model.py --model-id <ID>
```

Measured **end-to-end** wall time per model — all four steps, i.e. download + export + surgery +
INT8 compile + validate. These are the real numbers from a full `compile_all.sh` run on the SDK
container, not compile-only estimates:

| # | Model | End-to-end |
| --- | --- | --- |
| 1 | `resnet50` | 4m32s |
| 2 | `convnext_tiny` | 9m59s |
| 3 | `densenet169` | 19m27s |
| 4 | `efficientnet_v2_s` | 19m43s |
| 5 | `yolo11n` | 8m32s |
| 6 | `yolo11s` | 9m48s |
| 7 | `yolo26n` | 10m05s |
| 8 | `yolo11s-seg` | 12m02s |
| 9 | `yolo26s-pose` | 12m16s |
| 10 | `yolox_s` | 16m02s |

**Total ≈ 2 h**, serial. The compile step dominates; the download and export steps are a minute or two
each. Times scale with host CPU — treat them as ratios, not promises.

---

## Model 1 — `resnet50` (classification, no surgery)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id resnet50      # downloads torchvision weights -> 98 MB ONNX
python compile/graph_surgery.py   --model-id resnet50      # prints "kind=none ... skipping"
python compile/compiler.py        --model-id resnet50   # the long step; whole block ≈ 4m32s
python compile/test_model.py      --model-id resnet50 --validate-only
```

**Result (host):**

```text
[compile] resnet50: rc=0            ...  A65 : 0
[PASS] resnet50   elf=1 so=0  (resnet50_mpk.tar.gz)
```

**Result (DevKit — `python compile/test_model.py --model-id resnet50`):**

```text
[test] resnet50: imagenet_topk on real image(s)
   000000000885.jpg   racket 0.59, tennis ball 0.02
```

---

## Model 2 — `convnext_tiny` (classification, no surgery)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id convnext_tiny   # -> 110 MB ONNX
python compile/graph_surgery.py   --model-id convnext_tiny   # skipped (surgery: none)
python compile/compiler.py        --model-id convnext_tiny   # the long step; whole block ≈ 9m59s
python compile/test_model.py      --model-id convnext_tiny --validate-only
```

**Result (host):**

```text
[compile] convnext_tiny: rc=0       ...  A65 : 0
[PASS] convnext_tiny   elf=1 so=0
```

**Result (DevKit):**

```text
   000000000139.jpg   home theater 0.29, television 0.16
   000000000885.jpg   racket 0.57, tennis ball 0.05
```

---

## Model 3 — `densenet169` (classification, no surgery)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id densenet169   # -> 55 MB ONNX
python compile/graph_surgery.py   --model-id densenet169   # skipped
python compile/compiler.py        --model-id densenet169   # the long step; whole block ≈ 19m27s
python compile/test_model.py      --model-id densenet169 --validate-only
```

**Result (host):**

```text
[compile] densenet169: rc=0         ...  A65 : 0
[PASS] densenet169   elf=1 so=0
```

**Result (DevKit):**

```text
   000000000885.jpg   racket 0.99, tennis ball 0.01
```

---

## Model 4 — `efficientnet_v2_s` (classification, no surgery, 384×384 input)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id efficientnet_v2_s   # -> 82 MB ONNX
python compile/graph_surgery.py   --model-id efficientnet_v2_s   # skipped
python compile/compiler.py        --model-id efficientnet_v2_s   # the long step; whole block ≈ 19m43s
python compile/test_model.py      --model-id efficientnet_v2_s --validate-only
```

**Result (host):**

```text
[compile] efficientnet_v2_s: rc=0   ...  A65 : 0
[PASS] efficientnet_v2_s   elf=1 so=0
```

**Result (DevKit):**

```text
   000000000885.jpg   racket 0.75, tennis ball 0.04
```

---

## Model 5 — `yolo11n` (detection, surgery)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id yolo11n   # downloads yolo11n.pt -> 11 MB ONNX
python compile/graph_surgery.py   --model-id yolo11n   # cuts the decode tail, exposes 6 raw heads
python compile/compiler.py        --model-id yolo11n   # the long step; whole block ≈ 8m32s
python compile/test_model.py      --model-id yolo11n --validate-only
```

**Result (host):**

```text
[surgery] yolo11n: OK  outputs=['bbox_0','bbox_1','bbox_2','class_logit_0','class_logit_1','class_logit_2']
[compile] yolo11n: rc=0             ...  A65 : 0
[PASS] yolo11n   elf=1 so=0
```

**Result (DevKit)** — the surgery contract, **6 tensors, NHWC**:

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

bbox = 4 ch × 3 scales · class = 80 ch × 3 scales.

---

## Model 6 — `yolo11s` (detection, surgery)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id yolo11s   # -> 37 MB ONNX
python compile/graph_surgery.py   --model-id yolo11s
python compile/compiler.py        --model-id yolo11s   # the long step; whole block ≈ 9m48s
python compile/test_model.py      --model-id yolo11s --validate-only
```

**Result (host):**

```text
[compile] yolo11s: rc=0             ...  A65 : 0
[PASS] yolo11s   elf=1 so=0
```

**Result (DevKit)** — identical contract to `yolo11n` (head node names are scale-invariant):

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

---

## Model 7 — `yolo26n` (detection, surgery — **no DFL rebuild**)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id yolo26n   # -> 9.5 MB ONNX
python compile/graph_surgery.py   --model-id yolo26n   # one2one_cv* heads; DFL step skipped
python compile/compiler.py        --model-id yolo26n   # the long step; whole block ≈ 10m05s
python compile/test_model.py      --model-id yolo26n --validate-only
```

**Result (host):**

```text
[compile] yolo26n: rc=0             ...  A65 : 0
[PASS] yolo26n   elf=1 so=0
```

**Result (DevKit):**

```text
   6 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                     (1,80,80,80) (1,40,40,80) (1,20,20,80)
```

YOLO26's heads are already 4-channel, so the DFL reconstruction that YOLO11 needs is skipped.

---

## Model 8 — `yolo11s-seg` (segmentation, surgery)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id yolo11s-seg   # -> 39 MB ONNX
python compile/graph_surgery.py   --model-id yolo11s-seg   # + mask-coeff heads and the proto head
python compile/compiler.py        --model-id yolo11s-seg   # the long step; whole block ≈ 12m02s
python compile/test_model.py      --model-id yolo11s-seg --validate-only
```

**Result (host):**

```text
[compile] yolo11s-seg: rc=0         ...  A65 : 0
[PASS] yolo11s-seg   elf=1 so=0
```

**Result (DevKit)** — **10 tensors**: the 6 detection heads + 3 mask-coefficient heads + the proto:

```text
   10 head tensor(s): (1,80,80,4) (1,40,40,4) (1,20,20,4)
                      (1,80,80,80) (1,40,40,80) (1,20,20,80)
                      (1,80,80,32) (1,40,40,32) (1,20,20,32)     <- mask coeffs (32 ch)
                      (1,160,160,32)                             <- proto
```

---

## Model 9 — `yolo26s-pose` (pose, surgery — **carries the 209× fix**)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id yolo26s-pose   # -> 40 MB ONNX
python compile/graph_surgery.py   --model-id yolo26s-pose   # + keypoint heads, PADDED 51 -> 64 ch
python compile/compiler.py        --model-id yolo26s-pose   # the long step; whole block ≈ 12m16s
python compile/test_model.py      --model-id yolo26s-pose --validate-only
```

**Result (host):**

```text
[compile] yolo26s-pose: rc=0        ...  A65 : 0
[PASS] yolo26s-pose   elf=1 so=0
```

**Result (DevKit)** — **9 tensors**, note the keypoint heads are **64**, not 51:

```text
   9 head tensor(s): (1,80,80,4)  (1,40,40,4)  (1,20,20,4)      <- bbox
                     (1,80,80,1)  (1,40,40,1)  (1,20,20,1)      <- class (1 = person)
                     (1,80,80,64) (1,40,40,64) (1,20,20,64)     <- keypoints, padded 51 -> 64
```

> **Do not remove the padding.** The natural `{4, 1, 51}` channel mix defeats the compiler's output
> fusion — each of the 9 outputs then gets its own `slice_transform` stage in the post-MLA tail:
> **1782 ms/frame (0.6 fps)**. Padding the keypoint head to a tile-aligned 64 lets the outputs fuse:
> **8.5 ms/frame (117 fps) — a 209× speedup**, with identical weights and information. The host
> decoder slices channels 51..63 (the zero padding) back off.
>
> It is the channel **mix**, not the 51 itself: dropping *either* the class or the keypoint heads also
> makes it fast. And the MLA was never the bottleneck — pose is only 1.21× `yolo11s` in MLA cycles.

---

## Model 10 — `yolox_s` (detection, **different surgery**)

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --model-id yolox_s   # downloads Megvii's official ONNX (no torch)
python compile/graph_surgery.py   --model-id yolox_s   # decoupled anchor-free head -> 3 raw heads
python compile/compiler.py        --model-id yolox_s   # the long step; whole block ≈ 16m02s
python compile/test_model.py      --model-id yolox_s --validate-only
```

**Result (host):**

```text
[surgery] yolox_s: OK  outputs=['yolox_head_0','yolox_head_1','yolox_head_2']
[compile] yolox_s: rc=0             ...  A65 : 0
[PASS] yolox_s   elf=1 so=0
```

**Result (DevKit)** — **3 tensors**, 85 ch each (4 box + 1 obj + 80 class):

```text
   3 head tensor(s): (1,80,80,85) (1,40,40,85) (1,20,20,85)
```

---

## Everything at once

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation

python compile/convert_to_onnx.py --all
python compile/graph_surgery.py   --all
python compile/compiler.py        --all       # serial, ~2 h
python compile/test_model.py      --all --validate-only
```

Expected final line:

```text
all archives: one .elf, zero .so
```

---

## If something goes wrong

| Symptom | Cause |
| --- | --- |
| `[FAIL] ... no _mpk.tar.gz produced` | the compile failed — read `work/<id>/reports/compile.log` |
| `so=1` or more | part of the graph fell back to the host CPU; surgery did not remove everything the MLA cannot place |
| `A65 : <non-zero>` in the log | same thing, visible earlier |
| `[compile] REFUSING: calibration set looks SYNTHETIC` | you pointed `--calib-dir` at generated images. Quantization needs **real** images or the model compiles clean and is quietly wrong |
| compiler killed / OOM | you ran two compiles at once. One at a time |

**A green compile is not a working model.** `rc=0` + one `.elf` + zero `.so` proves the graph is on
the MLA. It proves *nothing* about accuracy — always run the DevKit test on real images.
