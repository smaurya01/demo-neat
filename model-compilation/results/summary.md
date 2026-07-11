# Model Compilation — status

Every model here is built by the same four steps (`compile/`), driven by one registry
(`models.yaml`), calibrated on **real** COCO images (`assets/calibration`) and smoke-tested on
**real** images (`assets/inference`). See [`../README.md`](../README.md).

Target contract: **one `.tar.gz`, one `.elf`, zero `.so`** (`A65: 0` in the compile log).

## Working models

Archive contract verified with `test_model.py --all --validate-only`; behaviour verified by running
each model on the DevKit against real images.

| Model | Task | Surgery | Archive | On-device behaviour |
| --- | --- | --- | --- | --- |
| `resnet50` | classification | none | 1 elf / 0 so | tennis frame → `racket 0.59` |
| `densenet169` | classification | none | 1 elf / 0 so | tennis frame → `racket 0.99` |
| `convnext_tiny` | classification | none | 1 elf / 0 so | living room → `home theater 0.29`, `television 0.16` |
| `efficientnet_v2_s` | classification | none | 1 elf / 0 so | tennis frame → `racket 0.75` |
| `yolo11n` | detection | `yolo_ultralytics` | 1 elf / 0 so | 6 heads: bbox(4)×3 + class(80)×3 |
| `yolo11s` | detection | `yolo_ultralytics` | 1 elf / 0 so | 6 heads (same contract — node names are scale-invariant) |
| `yolo26n` | detection | `yolo_ultralytics` | 1 elf / 0 so | 6 heads (no DFL rebuild — heads already 4-ch) |
| `yolo11s-seg` | segmentation | `yolo_ultralytics` | 1 elf / 0 so | 10 heads: + mask_coeff(32)×3 + proto 160×160×32 |
| `yolo26s-pose` | pose | `yolo_ultralytics` | 1 elf / 0 so | 9 heads: + kpt(51)×3 |
| `yolox_s` | detection | `yolox` | 1 elf / 0 so | 3 heads: 85ch × 3 scales |

Raw-head tensors come back **NHWC** `(1, H, W, C)` — a host decoder must transpose to CHW.

## Known issue: `fastvit_t8` — compiles clean, but INT8 breaks it

**The folder's cautionary case study**, and exactly what the README's "clean archive, quietly wrong"
warning is about.

| Stage | Result |
| --- | --- |
| Archive contract | **PASS** — 1 `.elf`, 0 `.so`, `A65: 0`, `rc=0` |
| FP32 ONNX (onnxruntime, host) | **CORRECT** — tennis frame → `racket 0.83` |
| INT8 compiled archive (DevKit) | **WRONG** — predicts an unrelated class |

The export is fine; **INT8 quantization destroys this model.** More calibration did not rescue it:

- 20 real COCO images (`mse`) → `coil 0.98` on *every* image (degenerate).
- 100 real images (`mse`) → low-confidence garbage (`lampshade 0.08` on a tennis photo).
- 64 real images (`min_max`) → `lampshade 0.53` on a tennis photo.

FastViT uses reparameterized blocks whose activation ranges evidently do not survive 8-bit
quantization. This is a **model-side sensitivity**, not a toolchain or calibration bug.

**Takeaway:** a passing archive contract proves the graph is placed on the MLA. It proves **nothing**
about accuracy. Always run step 4's behavioural test on real images — that is what caught this.

## Removed: ViT, DINOv2, MaxViT, DETR

We cannot compile these **from source** yet. They **are** supported on the hardware — SiMa publishes
archives for DETR and DINOv2 ViT-S/14 that are a clean **1 `.elf` / 0 `.so`** and run fine on the
board — but only because SiMa ships **source-prepared** models. Our stock exports + ONNX surgery
fragment badly (DINOv2: 99 `.elf` / 195 `.so`); `maxvit_t` fragments into 113 stages and OOM-kills
the compiler even with `--any-shape-on-mla`.

**The lever is a source-level model rewrite, not ONNX surgery.** Full write-up, and the commands to
download and run the working DETR/ViT archives today:
[`T7_CORRECTION_transformers_are_supported.md`](T7_CORRECTION_transformers_are_supported.md).
