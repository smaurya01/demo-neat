# PCB Defect Detection — YOLO26n

## Introduction

This demo runs a folder of PCB images through the SiMa YOLO26n defect-detection model on a Modalix
DevKit, draws labeled boxes, and writes one annotated JPG per image. Inference runs on the board
through the `pyneat` runtime (on-device YOLO26 box decode).

It detects six PCB manufacturing defects:
`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`.

Dataset and trained model the demo is based on:
**https://platform.ultralytics.com/muhammadrizwanmunawar/datasets/pcb-defects-detection**

## About Project

- Application: `pcb-defect-detection-yolo26n` (`main.py`)
- Model: `plc_yolo26n_mpk.tar.gz` (compiled from the custom-trained `yolo26n.pt`)
- Input: folder of PCB images (`sample_input_images/`)
- Output: annotated JPGs (`output_images/<name>_detected.jpg`)
- Runtime config: `config/default.conf`

> Commands below assume the project lives at `/workspace/demo-neat/pcb-defect-detection-yolo26n`
> (the `/workspace` mount shared with the DevKit). If you placed it elsewhere under `/workspace`,
> substitute your own path. Likewise replace `<board-ip>` / `<user>` / `<password>` with your DevKit's.

## Requirements

Run from a host with the SiMa NEAT SDK installed (provides `sima-cli`). Pair the host with your
board — this opens a NEAT SDK shell where `dk` is available and the `/workspace` mount is shared
with the board:

```bash
sima-cli                       # connect to <board-ip> as <user>/<password>
```

`dk` is a shell **function** defined by that SDK shell, not a binary. Call it directly in the shell;
it will not work wrapped in `timeout`/`env` or run from a sub-shell (`bash some.sh`) that never
sourced the SDK profile. Sanity-check the board connection before running:

```bash
type dk            # should print a shell function (not "not found")
dk status          # should show "SSH status: reachable"
dk shell hostname  # should print the board hostname (e.g. modalix)
```

## Model Download Command

The model is custom-trained, so it is **not** in the SiMa model zoo and is **not** committed to git
(the binaries are large). Create the models folder and download the pack into it from Google Drive:

```bash
mkdir -p /workspace/demo-neat/pcb-defect-detection-yolo26n/assets/models
# Download the files from this folder into the path above:
#   https://drive.google.com/drive/folders/1bIkSADEQWnuZ1D5pyQLz722P5JHdfqA_
```

Only `plc_yolo26n_mpk.tar.gz` is needed to **run**. The `.pt` / `.onnx` files let you inspect or
rebuild the pack (see "How To Build"):

```text
assets/models/
├── plc_yolo26n_mpk.tar.gz          # compiled Modalix MLA pack   ← required to run
├── yolo26n.pt                      # trained checkpoint          (compile input)
├── yolo26n.onnx                    # exported ONNX               (compile intermediate)
├── yolo26n_einsum_raw.onnx         # graph-surgeoned ONNX (C2PSA einsum + raw heads)
└── yolo26n_einsum_raw_prepared.onnx# simplified/shape-fixed ONNX fed to the compiler
```

Expected model path:

```text
/workspace/demo-neat/pcb-defect-detection-yolo26n/assets/models/plc_yolo26n_mpk.tar.gz
```

## Configure

Edit `config/default.conf` before running. Defaults:

```text
model=assets/models/plc_yolo26n_mpk.tar.gz
input_dir=sample_input_images
output_dir=output_images
output_suffix=_detected
infer_size=640
score=0.25
nms=0.45
top_k=300
timeout_ms=8000
labels=missing_hole,mouse_bite,open_circuit,short,spur,spurious_copper
```

CLI flags override config values. For example:

```bash
--score 0.30 --nms 0.50
```

## How To Build

There is no native binary — the deliverable is the compiled **model pack**, built from
`yolo26n.pt`. Run from the SiMa Model SDK environment (the `afe` toolchain + `ultralytics` +
`model_to_pipeline` must be importable):

```bash
sima-cli sdk model "bash /workspace/demo-neat/pcb-defect-detection-yolo26n/scripts/compile_model.sh"
```

`scripts/compile_model.sh` runs export → graph surgery → BF16 quantize + MLA tessellation and stages
the result to `assets/models/plc_yolo26n_mpk.tar.gz`.

## How To Run

Run on the DevKit from the SDK shell, with the model in `assets/models/`:

```bash
dk /workspace/demo-neat/pcb-defect-detection-yolo26n/main.py \
  --config /workspace/demo-neat/pcb-defect-detection-yolo26n/config/default.conf
```

Higher-confidence run:

```bash
dk /workspace/demo-neat/pcb-defect-detection-yolo26n/main.py \
  --config /workspace/demo-neat/pcb-defect-detection-yolo26n/config/default.conf \
  --score 0.30 --nms 0.50
```

A path-independent convenience wrapper is provided. Because `dk` is a shell function, **source** it
from the project root (do not run it with `bash`, which spawns a sub-shell that cannot see `dk`):

```bash
cd /workspace/demo-neat/pcb-defect-detection-yolo26n
source scripts/run_dk.sh                    # uses config/default.conf
source scripts/run_dk.sh --score 0.30       # extra flags forwarded to main.py
```

Exit codes: `0` success (≥1 image) · `2` runtime/IO error · `3` no images found.

## How To See The Output

After the run, the annotated images are in `output_images/`:

```text
output_images/<image-name>_detected.jpg
```

Each shows class-colored boxes with `name score` labels at the original input resolution. Expected
output: 5 annotated JPGs (one per bundled sample image) and a log like:

```text
plc | found 5 images in .../sample_input_images
plc | detector built on MLA (warmup done)
plc | img=pcb_01_missing_hole.jpg dets=3 {'missing_hole': 3} wrote=pcb_01_missing_hole_detected.jpg
plc | img=pcb_02_open_circuit.jpg dets=3 {'open_circuit': 3} wrote=pcb_02_open_circuit_detected.jpg
plc | img=pcb_03_spur.jpg dets=3 {'short': 1, 'spur': 2} wrote=pcb_03_spur_detected.jpg
plc | img=pcb_04_spurious_copper.jpg dets=5 {'missing_hole': 1, 'open_circuit': 1, 'spurious_copper': 3} wrote=pcb_04_spurious_copper_detected.jpg
plc | img=pcb_05_spurious_copper_rotated.jpg dets=2 {'spur': 1, 'spurious_copper': 1} wrote=pcb_05_spurious_copper_rotated_detected.jpg
plc | processed 5 / 5 images in 0.47 s | images_with_detections=5 total_detections=16
plc | per-class totals: {'missing_hole': 4, 'open_circuit': 4, 'short': 1, 'spur': 3, 'spurious_copper': 4}
```
