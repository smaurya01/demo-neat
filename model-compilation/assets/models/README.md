# Prebuilt models — download instead of compiling

All ten models in [`../../models.yaml`](../../models.yaml) have been compiled and bundled, so you can
run the apps **without compiling anything**.

Every model was built from its **upstream original weights** (PyTorch / Ultralytics / Megvii) through
this repo's own export → graph-surgery → INT8 → compile chain. **Nothing here is downloaded
pre-compiled from the SiMa model zoo.**

## Download

> **📦 Drive link: [Models-v1.zip](https://drive.google.com/file/d/10zOUhD56VrZaY3urSHgqpjGZjIY3jEKt/view?usp=sharing)**

Unpack it into this folder (`model-compilation/assets/models/`). Nothing here is committed to git —
`.pt`, `.onnx` and `.tar.gz` are all git-ignored.

## What each model folder contains

```text
assets/models/<model-id>/
    <model-id>.pt                          original weights   (Ultralytics models only)
    <model-id>.onnx                        the ONNX export that was compiled
    <model-id>...._mpk.tar.gz              the compiled SiMa archive   <-- this is what an app loads
```

For the torchvision and Megvii models there is no `.pt`: torchvision weights come from its pretrained
API (`weights="DEFAULT"`), and Megvii ships a pre-exported ONNX. In both cases **the `.onnx` is the
original**.

## Manifest

Every archive is verified: **1 `.elf`, 0 `.so`, `A65: 0`** — the whole graph runs on the MLA.

| Model | Task | Original | ONNX | **Compiled archive** | Size |
| --- | --- | --- | --- | --- | --- |
| `resnet50` | classification | *(torchvision)* | 98 M | `resnet50_mpk.tar.gz` | **23 M** |
| `densenet169` | classification | *(torchvision)* | 55 M | `densenet169_mpk.tar.gz` | **21 M** |
| `convnext_tiny` | classification | *(torchvision)* | 110 M | `convnext_tiny_mpk.tar.gz` | **30 M** |
| `efficientnet_v2_s` | classification | *(torchvision)* | 82 M | `efficientnet_v2_s_mpk.tar.gz` | **26 M** |
| `yolo11n` | detection | 5.4 M `.pt` | 11 M | `yolo11n.compile_ready_mpk.tar.gz` | **12 M** |
| `yolo11s` | detection | 19 M `.pt` | 37 M | `yolo11s.compile_ready_mpk.tar.gz` | **20 M** |
| `yolo26n` | detection | 5.3 M `.pt` | 9.5 M | `yolo26n.compile_ready_mpk.tar.gz` | **14 M** |
| `yolo11s-seg` | segmentation | 20 M `.pt` | 39 M | `yolo11s-seg.compile_ready_mpk.tar.gz` | **21 M** |
| `yolo26s-pose` | pose | 24 M `.pt` | 40 M | `yolo26s-pose.compile_ready_mpk.tar.gz` | **25 M** |
| `yolox_s` | detection | *(Megvii ONNX)* | 35 M | `yolox_s.compile_ready_mpk.tar.gz` | **29 M** |

Bundle total: **~800 MB** (the ten compiled archives alone: **~216 MB** — that is all you need to
*run* the apps; the `.pt`/`.onnx` are there so you can re-export or re-compile).

> **`yolo26s-pose` carries a load-bearing fix.** Its keypoint head is zero-padded 51 → 64 channels.
> Unpadded, the same model runs at **1782 ms/frame**; padded, **8.5 ms/frame** — a **209× speedup**
> for identical weights. The archive above is the padded build (verified: C=64). If you ever recompile
> it, keep `pad_channels_to: 64` in `compile/_surgery_ultralytics.py`.

---

## Which archive does each app need?

**This bundle covers 5 of the 11 archives the apps reference.** The rest come from the SiMa model zoo
or a direct download — they are **not** part of this compile flow.

### ✅ Covered by this bundle

| App | Needs | From this bundle |
| --- | --- | --- |
| [`quad-stream-quad-model`](../../../apps/quad-stream-quad-model/README.md) | `yolo11s.compile_ready_mpk.tar.gz`<br>`yolo11s-seg.compile_ready_mpk.tar.gz`<br>`yolo26s-pose.compile_ready_mpk.tar.gz`<br>`yolox_s.compile_ready_mpk.tar.gz` | **exact filenames — just copy them in** |
| [`single-stream-yolo-yolo11`](../../../apps/single-stream-yolo-yolo11/README.md)<br>[`multi-stream-yolo-yolo11`](../../../apps/multi-stream-yolo-yolo11/README.md)<br>[`detection-vlm-assistant`](../../../apps/detection-vlm-assistant/README.md) | `yolo_11n_mpk.tar.gz` | `yolo11n.compile_ready_mpk.tar.gz` — **must be renamed** ⚠️ |

```bash
cd model-compilation          # all paths below are relative to here

# quad-stream-quad-model — exact names, straight copy
cp assets/models/yolo11s/yolo11s.compile_ready_mpk.tar.gz \
   assets/models/yolo11s-seg/yolo11s-seg.compile_ready_mpk.tar.gz \
   assets/models/yolo26s-pose/yolo26s-pose.compile_ready_mpk.tar.gz \
   assets/models/yolox_s/yolox_s.compile_ready_mpk.tar.gz \
   ../apps/quad-stream-quad-model/assets/models/

# the three YOLO11n apps — note the RENAME
for app in single-stream-yolo-yolo11 multi-stream-yolo-yolo11 detection-vlm-assistant; do
  mkdir -p ../apps/$app/assets/models
  cp assets/models/yolo11n/yolo11n.compile_ready_mpk.tar.gz \
     ../apps/$app/assets/models/yolo_11n_mpk.tar.gz          # <-- renamed
done
```

### ❌ NOT in this bundle — get these elsewhere

| App | Needs | Where it comes from |
| --- | --- | --- |
| `single-stream-yolo-yolov8n`, `benchmark` | `yolo_v8n_mpk.tar.gz` | model zoo |
| `single-stream-yolo-yolov8m` | `yolo_v8m_mpk.tar.gz` | model zoo |
| `single-stream-yolov8n-seg` | `yolo_v8n_seg_mpk.tar.gz` | model zoo |
| `single-stream-open-pose` | `open_pose_mpk.tar.gz` | model zoo |
| `single-stream-yolo26n` | `yolo26n-det-bf16-mla_tess-b1.tar.gz` | direct SiMa download |
| `pcb-defect-detection-yolo26n` | `latest_plc_yolo26n_mpk.tar.gz` | the app's own `compile/` (custom-trained) |

```bash
# the four zoo models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_v8n
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_v8m
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_v8n_seg
sima-cli modelzoo -v 2.1.2 --boardtype modalix get open_pose
```

`single-stream-yolo26n` wants a SiMa-published BF16 archive. Our `yolo26n.compile_ready_mpk.tar.gz`
is an INT8 alternative — usable, but point `model_path` at it and note the decode contract differs.

---

## Want to build these yourself?

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation
./compile_all.sh                    # all ten, one at a time, ~2 hours
```

Or one model at a time with the copy-paste blocks in [`../../REPLICATION.md`](../../REPLICATION.md).

Measured end-to-end time per model (download + export + surgery + INT8 compile + validate), on the
SDK container:

| Model | Time | | Model | Time |
| --- | --- | --- | --- | --- |
| `resnet50` | 4m32s | | `yolo11n` | 8m32s |
| `convnext_tiny` | 9m59s | | `yolo11s` | 9m48s |
| `densenet169` | 19m27s | | `yolo26n` | 10m05s |
| `efficientnet_v2_s` | 19m43s | | `yolo11s-seg` | 12m02s |
| | | | `yolo26s-pose` | 12m16s |
| | | | `yolox_s` | 16m02s |

**Total ≈ 2 hours**, serial. Compile **one at a time** — the compiler is memory-hungry and concurrent
compiles OOM.
