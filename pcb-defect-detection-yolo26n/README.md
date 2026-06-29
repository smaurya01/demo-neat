# PLC Defect Detection — YOLO26n

## Introduction

This demo runs a folder of PCB images through the SiMa **YOLO26n** defect-detection
model on a **Modalix DevKit**, draws labeled boxes, and writes one annotated JPG
per image. Inference runs **on the board** through the public `pyneat` runtime —
results are produced only by the on-device run.

It detects six PCB manufacturing defects:
`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`.

---

## Quick Start

> **If you just cloned this and want a coding agent to run it for you, give it:**
>
> > *"Run this pipeline, my board ip:`<BOARD_IP>` user:`<USER>` password:`<PASSWORD>`"*
>
> (for this DevKit: ip `192.168.135.116`, user `sima`, password `edgeai`)

Then the four steps below get you to annotated images. They assume you are on a
**host that has the SiMa NEAT SDK installed** (provides `sima-cli`).

```bash
# 1. Pair the host with your board. This opens a NEAT SDK shell that puts `dk`
#    on PATH and NFS-mounts the shared /workspace onto the board.
sima-cli                       # connect to <BOARD_IP> as <USER>/<PASSWORD>
#    (or: sima-cli sdk neat — a NEAT SDK shell already paired with the board)

# 2. Make sure this project lives under the shared /workspace mount, e.g.:
#    /workspace/NEAT/demo-neat/pcb-defect-detection-yolo26n
cd /workspace/NEAT/demo-neat/pcb-defect-detection-yolo26n

# 3. Get the compiled model into assets/models/ (see "Model Download" below).
#    You need at least: assets/models/plc_yolo26n_mpk.tar.gz

# 4. Run the pipeline on the board.
bash scripts/run_dk.sh
```

Annotated images land in [`output_images/`](output_images/) as
`<image-name>_detected.jpg`.

> **Note — `dk` is a shell *function*, not a binary.** It is defined by the NEAT
> SDK shell profile. Call it directly in that shell. It will **not** work inside a
> plain sub-script or wrapped in `timeout`/`env` (those spawn a shell that never
> sourced the profile — you'll get `dk: command not found`). `scripts/run_dk.sh`
> is meant to be invoked from the SDK shell, where the function is in scope.

---

## About Project

| | |
| --- | --- |
| Application | `pcb-defect-detection-yolo26n` (`main.py`) |
| Model | `plc_yolo26n_mpk.tar.gz` (compiled from the trained `yolo26n.pt`) |
| Input | folder of PCB images (`sample_input_images/`) |
| Output | annotated JPGs (`output_images/<name>_detected.jpg`) |
| Runtime config | `config/default.conf` |
| Runtime | `pyneat` on the DevKit MLA (BoxDecodeType.YoloV26, on-device decode) |

---

## Model Download

The model binaries are **not committed** to git (they are large). Download them
into `assets/models/` from Google Drive:

**https://drive.google.com/drive/folders/1bIkSADEQWnuZ1D5pyQLz722P5JHdfqA_**

Place the files so the folder looks like this:

```text
assets/models/
├── plc_yolo26n_mpk.tar.gz          # compiled Modalix MLA pack  ← required to run
├── yolo26n.pt                      # trained checkpoint         (compile input)
├── yolo26n.onnx                    # exported ONNX              (compile intermediate)
├── yolo26n_einsum_raw.onnx         # graph-surgeoned ONNX (C2PSA einsum + raw heads)
└── yolo26n_einsum_raw_prepared.onnx# simplified/shape-fixed ONNX fed to the compiler
```

Only `plc_yolo26n_mpk.tar.gz` is needed to **run** the pipeline. The `.pt` /
`.onnx` files are provided so you can inspect or **rebuild** the pack (see
"How To Build"). The config expects:

```text
assets/models/plc_yolo26n_mpk.tar.gz
```

Prefer building from source instead of downloading? Skip to **How To Build**.

---

## Configure

Edit [`config/default.conf`](config/default.conf) before running. Defaults:

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

CLI flags to `main.py` override config values:

```bash
bash scripts/run_dk.sh --score 0.30 --nms 0.50
```

---

## How To Run

From the NEAT SDK shell (paired with the board), with the model in
`assets/models/`:

```bash
bash scripts/run_dk.sh
```

or call `dk` directly:

```bash
PROJ=/workspace/NEAT/demo-neat/pcb-defect-detection-yolo26n
dk "$PROJ/main.py" --config "$PROJ/config/default.conf"
```

Higher-confidence run:

```bash
bash scripts/run_dk.sh --score 0.30 --nms 0.50
```

`dk` SSHes to the board (`<USER>@<BOARD_IP>`), activates `pyneat`, builds the
graph `input → preprocess (RGB /255 → 640) → MLA → YoloV26 box decode → output`,
runs every image in `sample_input_images/`, and streams logs back. Because the project
lives on the shared `/workspace` mount, the annotated images written on the board
appear directly in this host's `output_images/`.

**Exit codes:** `0` success (≥1 image) · `2` runtime/IO error · `3` no images found.

---

## How To See The Output

After the run, the annotated images are in [`output_images/`](output_images/):

```text
output_images/<image-name>_detected.jpg
```

Each shows class-colored boxes with `name score` labels at the original input
resolution. Expected log (from the bundled 5 sample images):

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

---

## How To Build (rebuild the model pack)

There is no native binary — the deliverable is the compiled **model pack**. The
model is custom-trained, so it is compiled from `yolo26n.pt`. Run this inside the
**SiMa Model SDK** environment (the `afe` toolchain + `ultralytics` +
`model_to_pipeline` must be importable):

```bash
sima-cli sdk model "bash /workspace/NEAT/demo-neat/pcb-defect-detection-yolo26n/scripts/compile_model.sh"
```

[`scripts/compile_model.sh`](scripts/compile_model.sh) runs the **canonical
einsum path** (the one that produced the shipped pack); all artifacts land in
`assets/models/`:

1. **export** `yolo26n.pt → yolo26n.onnx` (ultralytics, `imgsz=640, opset=17`)
2. **graph surgery** [`compile/surgery_einsum_attention.py`](compile/surgery_einsum_attention.py)
   → `yolo26n_einsum_raw.onnx` — rewrites the C2PSA attention (`model.10`,
   `model.22`) as 4D **Einsum** so it stays on the MLA, then exposes the raw
   YOLO26 detection heads (`box_*` / `class_logit_*`) for on-device decode
3. **quantize + compile** [`compile/compile_yolo26_modelsdk.py`](compile/compile_yolo26_modelsdk.py)
   — Model SDK BF16 quantization + MLA tessellation for Modalix → `*_mpk.tar.gz`
   (gated to one MLA ELF, zero `.so`)
4. **stage** the pack → `assets/models/plc_yolo26n_mpk.tar.gz`

The result is a single-MLA pack whose on-device YOLO26 box decoder emits boxes in
original-image space.

---

## Project Structure

```text
pcb-defect-detection-yolo26n/
├── README.md
├── requirements.txt              # numpy, opencv-python (pyneat is on the DevKit)
├── main.py                       # entrypoint: load config -> run on the MLA
├── config/
│   └── default.conf              # runtime config (key=value)
├── assets/
│   └── models/                   # model binaries (downloaded / rebuilt — git-ignored)
│       ├── plc_yolo26n_mpk.tar.gz       # compiled Modalix MLA pack (deployed)
│       ├── yolo26n.pt                   # trained checkpoint (compile input)
│       ├── yolo26n.onnx                 # exported ONNX
│       ├── yolo26n_einsum_raw.onnx      # graph-surgeoned ONNX
│       └── yolo26n_einsum_raw_prepared.onnx
├── sample_input_images/          # sample PCB images to run on (pcb_0N_<defect>.jpg)
├── output_images/                # annotated results (<name>_detected.jpg)
├── compile/
│   ├── surgery_einsum_attention.py   # C2PSA einsum surgery + raw-head exposure
│   └── compile_yolo26_modelsdk.py    # Model SDK BF16 + MLA tessellation compile
├── scripts/
│   ├── compile_model.sh          # export → surgery → quantize/compile → stage
│   └── run_dk.sh                 # run on the DevKit via `dk`
└── src/
    ├── config.py                 # parse config/default.conf
    ├── logging_setup.py          # stdlib logging init
    ├── io_utils.py               # image discovery, output dir
    ├── labels.py                 # 6 defect names + per-class colors
    ├── preprocess.py             # wrap BGR frame as a pyneat Tensor
    ├── inference.py              # pyneat Model (preprocess → MLA → YoloV26 decode)
    ├── postprocess.py            # parse the on-device BBOX payload
    ├── overlay.py                # draw class-colored boxes + labels
    └── pipeline.py               # orchestration
```

---

## DevKit Connection & Prerequisites

| Field | Value |
| --- | --- |
| Host | `<BOARD_IP>` (this DevKit: `192.168.135.116`) |
| User | `<USER>` (this DevKit: `sima`) |
| Password | `<PASSWORD>` (this DevKit: `edgeai`) |
| Workspace mount | host `modalix_workspace` ↔ DevKit `/workspace` (NFS) |
| Runner | `dk` (wraps `devkit-run`, SSH to the board) |

Sanity-check the connection from the NEAT SDK shell before running:

```bash
type dk            # should print a shell function (not "not found")
dk status          # should show "SSH status: reachable"
dk shell hostname  # should print the board hostname (e.g. modalix)
```

If `dk status` says **not reachable**, the board may just be cold (first SSH can
time out on ARP) — retry once or twice. If it stays unreachable, confirm the board
is powered, on the same network, and that `sima-cli` paired successfully.

Host-side Python deps for any local tooling are in `requirements.txt` (`numpy`,
`opencv-python`); the DevKit already provides `pyneat` via the NEAT SDK.
