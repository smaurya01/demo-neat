# demo-neat — NEAT Training Material

The front door for the **NEAT 4-day enablement course** on the SiMa **Modalix DevKit**. If you are a
trainee opening this repo cold, start here, then follow the day-by-day map below.

This repository bundles four things:

- **Runnable demo apps** (`apps/`) — single- and multi-stream YOLO detection, a detection→VLM
  assistant, and a 4-stream/4-model pipeline.
- **A GenAI (LLM/VLM/ASR) track** (`llima/`) — concepts, the `llima` CLI, running models from
  `pyneat.genai`, bring-your-own compilation, and the GenAI server.
- **A model-compilation reference** (`model-compilation/`) — ONNX graph surgery, INT8 quantization,
  and compile for YOLO and harder transformer models.
- **The course itself** (`training/`) plus concept notebooks (`tutorial/`) and setup notes
  (`installation/`).

> **Read this before running anything: [Verified vs documented-but-unrun](#verified-vs-documented-but-unrun).**
> Some of this material is live-validated on the DevKit; a large part (all LLM/VLM/ASR execution and
> all GenAI compilation) is written for you to run manually and has **not** been executed by the
> people who wrote it. Knowing which is which will save you hours.

---

## The 4-day training arc

The full syllabus is [`training/NEAT_4_DAY_TRAINING_PROGRAM.md`](training/NEAT_4_DAY_TRAINING_PROGRAM.md).
At a glance:

| Day | Theme | Primary material in this repo |
| --- | --- | --- |
| **Day 1** | NEAT foundations + first YOLO app | `tutorial/`, `apps/single-stream-yolo-*` |
| **Day 2** | Two-camera pipelines, runtime tuning, detection + VLM | `apps/multi-stream-yolo-yolo11`, `model-compilation/` (surgery walkthrough), `apps/detection-vlm-assistant`, `llima/03-yolo-plus-vlm` |
| **Day 3** | Model prep, unknown-model triage, PCIe | `model-compilation/`, `llima/04-llm-vlm-compilation` |
| **Day 4** | Production apps, GenAI/LLiMa, diagnostics, capstone | `llima/`, `apps/quad-stream-quad-model` |

Days 1–2 are tuned for the Korea team profile (Ubuntu host, Modalix DevKit, YOLOv11, VLM, two
cameras). Days 3–4 broaden to model triage, GenAI, and production support.

---

## Navigation table

| Folder | What you learn there | Entry point |
| --- | --- | --- |
| [`tutorial/`](tutorial/README.md) | Core Neat concepts as notebooks (Model, Graph, Run, Tensor, Sample) | `tutorial/README.md` |
| [`installation/`](installation/README.md) | SDK container, DevKit, Neat Insight, Windows setup | `installation/README.md` |
| [`apps/single-stream-yolo-yolo11/`](apps/single-stream-yolo-yolo11/README.md) | The baseline single-stream YOLO11 detection app | `apps/single-stream-yolo-yolo11/README.md` |
| [`apps/multi-stream-yolo-yolo11/`](apps/multi-stream-yolo-yolo11/README.md) | 2× RTSP → one shared YOLO11 stage → per-stream UDP out | `apps/multi-stream-yolo-yolo11/README.md` |
| [`apps/detection-vlm-assistant/`](apps/detection-vlm-assistant/README.md) | YOLO detection → trigger-gated crops → VLM captions | `apps/detection-vlm-assistant/README.md` |
| [`apps/quad-stream-quad-model/`](apps/quad-stream-quad-model/README.md) | 4 streams × 4 models (det/seg/pose/YOLOX) + teaching doc | `apps/quad-stream-quad-model/README.md` → [`TEACHING.md`](apps/quad-stream-quad-model/TEACHING.md) |
| [`model-compilation/`](model-compilation/README.md) | `.pt` → ONNX → graph surgery → INT8 → archive; YOLO + transformer models | `model-compilation/README.md` |
| [`llima/`](llima/README.md) | LLM/VLM/ASR: concepts, CLI, running, compiling, serving | `llima/README.md` |
| [`training/`](training/NEAT_4_DAY_TRAINING_PROGRAM.md) | The 4-day course syllabus | `training/NEAT_4_DAY_TRAINING_PROGRAM.md` |

The pre-existing single-model apps (`apps/single-stream-yolo-yolov8n`, `-yolov8m`, `-yolo26n`,
`-open-pose`, `apps/single-stream-yolov8n-seg`, `apps/pcb-defect-detection-yolo26n`,
`apps/multi-model-load-probe`, `apps/benchmark`) each keep their own README; open the folder you want.

---

## Verified vs documented-but-unrun

**This is the single most important section of this README.** The training material was built under a
hard rule: **no agent runs any LLM/VLM/ASR model and no agent runs a GenAI compile.** So the material
divides cleanly into three honesty tiers.

### ✅ Live-validated on the DevKit (192.168.135.203)

These were actually executed on the board and observed to work:

- **The YOLO compile chain** — `yolo11n` fresh `.pt` → ONNX → `compile_ready` surgery → INT8 →
  archive validation (one `.elf`, zero `.so`) → still-image smoke test. Re-verified this wave; log in
  [`model-compilation/results/t1_yolo11n_verification.md`](model-compilation/results/t1_yolo11n_verification.md).
- **The 2× RTSP app** (`apps/multi-stream-yolo-yolo11`) — ran live, **~37 fps aggregate** across two
  streams sharing one YOLO11 archive.
- **The detection leg of the VLM assistant** (`apps/detection-vlm-assistant`) — RTSP/image in → boxes
  out, validated live. (The VLM leg is *not* validated — see below.)
- **The quad-stream pipeline** (`apps/quad-stream-quad-model`) — ran live at **~1.7 fps aggregate**
  (4 streams × 4 models). The bottleneck is **A65 host decode, not the MLA**. Detection, segmentation,
  and pose decode correctly.
- **T5 model compiles** — `yolo11s`, `yolo11s-seg`, `yolo26s-pose`, `yolox_s` all compiled INT8 to one
  `.elf` / zero `.so` (`A65:0`). See [`model-compilation/work/T5_MODEL_STATUS.md`](model-compilation/work/T5_MODEL_STATUS.md).

### 📝 NOT executed — documented for manual runs

Written with exact, copy-paste-ready commands and an "expected output" section, but **the authors did
not run them** (hard rule: LLM/VLM/ASR execution is left to you):

- **All LLM/VLM/ASR execution:** `llima pull`, `llima run`, `llima benchmark-server`, and every
  `pyneat.genai` inference call — i.e. all of `llima/02-run-llm-vlm/` and `llima/05-genai-server/`,
  and the **VLM leg** of `apps/detection-vlm-assistant` (run it with `--no-vlm` to exercise the
  detector without touching the VLM).
- **All GenAI compilation** — `llima/04-llm-vlm-compilation/`.

Three models are already pulled on the board, so no `llima pull` is needed in any happy path: LLM
`Qwen3-4B-Instruct-2507-GPTQ-a16w4`, VLM `Qwen3-VL-4B-Instruct-GPTQ-a16w4`, ASR `whisper-small-a16w8`.

### ⚠️ Docs-derived, not source-verified

- **The compilation notebooks (`llima/04-*`) are derived from official docs, not from working code.**
  The real mechanism is a separate **host-side `llima-compile`** (Model Compiler) tool. There is **no
  `llima compile` subcommand** — the on-board CLI is runtime + model-manager only (`run, search, pull,
  list, rm, benchmark-server`). `llima-compile` is not on the board or in `/workspace/core`, and two
  official doc pages returned **HTTP 403**, so the exact flags are unconfirmed. What *is* verified (in
  `core/src/genai/GenAIInternal.cpp`) is the deployed model-directory contract, which is the testable
  part. The notebooks label each fact `[docs]` vs `[core]`.

---

## Prerequisites

| Prereq | Notes |
| --- | --- |
| **SDK container** | The SiMa Modalix SDK/eLxr container. C++ sysroot at `/opt/toolchain/aarch64/modalix`. |
| **Model-compiler venv** | `source /sdk-extensions/model-compiler/bin/activate` → `afe`, `onnx` 1.17.0, `ultralytics` 8.4.90. One env for export + surgery + quantize + compile. |
| **DevKit access** | Board `192.168.135.203`, user `sima`. `pyneat 0.3.0+develop`, Python 3.11.2, `aarch64`. |
| **RTSP source** | `rtsp://192.168.132.129:8555/stream` (H.264 1280×720). Sanity-check first: `ffprobe -hide_banner -rtsp_transport tcp rtsp://192.168.132.129:8555/stream`. |
| **Host viewer** | GStreamer on the host to view UDP/RTP output (see below). |

`/workspace` is **NFS-mounted on the board at the identical path**, so you write files host-side and
run them board-side with **no copying**.

Install host viewer tools:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

### Running on the board: `dk` vs `ssh`

There are two ways to run something on the DevKit, and picking the wrong one wastes time:

- **`dk` — for a human at a real terminal.** This is the intended UX. Source the helper once
  (`source /usr/local/bin/devkit.sh 192.168.135.203 sima 22`), then `dk /workspace/.../main.py ...`.
- **`ssh` — for automation / CI / agents.** `dk` needs a TTY and **hangs** in scripted contexts. Use
  passwordless ssh instead, and always wrap it in `timeout` so a hang cannot stall you:

  ```bash
  timeout 180 ssh -o BatchMode=yes sima@192.168.135.203 \
    'source $HOME/pyneat/bin/activate; python /workspace/demo-neat/apps/<app>/main.py --frames 30'
  ```

The board root filesystem has only **~5.9 GB free of 14 GB** — always `df -h /` before any
`llima pull`.

---

## Typical workflow

1. Open the folder README for the app or track you want.
2. For an app: point `config/default.conf` at your RTSP URL, UDP receiver IP, ports, and thresholds,
   and place/point at the compiled model archive (`assets/models/` is git-ignored).
3. Run on the board with `dk` (human) or `ssh` (automation), per the app README.
4. View UDP output on the host with the README's `gst-launch-1.0` command.

For model preparation, follow [`model-compilation/README.md`](model-compilation/README.md): it walks
the `.pt` → ONNX → surgery → INT8 → archive chain and the output contract, then generalizes it to any
YOLO11/26 variant and to the transformer models in `model-compilation/work/`.

---

## Build artifacts and models

`.gitignore` excludes generated build files and large downloaded model artifacts:

```text
build/  CMakeFiles/  CMakeCache.txt  Makefile
*.o  *.tar.gz  *.mpk  *.onnx  *.log  *.pid
```

After cloning, create/download models by following each app or `model-compilation/` README. Do not
expect model archives or `build/` directories to come from Git. Apps that reference a compiled archive
default to it in place under `model-compilation/work/...` (same NFS path on host and board), or you
drop your own copy in the app's `assets/models/`.

---

## Autonomy / provenance notes

This material was produced across parallel agent waves. Every non-obvious decision is logged in
[`DECISIONS.md`](DECISIONS.md); the plan that drove the work is
[`priority-task-and-implementation-plan.md`](priority-task-and-implementation-plan.md). Cross-project
engineering lessons live in `/workspace/overall-learning.md`.
