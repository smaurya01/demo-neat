# NEAT enablement material for the SiMa Modalix DevKit

Runnable demo apps, concept notebooks, a model-compilation reference, and a GenAI (LLM/VLM/ASR)
track — all for SiMa **NEAT** on the **Modalix DevKit**.

If you are opening this repo cold, do [`installation/`](installation/README.md) first, then pick a
track below.

| Where | What you get |
| --- | --- |
| [`installation/`](installation/README.md) | Set up the SDK container, pair the DevKit, attach VS Code. **Start here.** |
| [`tutorial/`](tutorial/README.md) | Core Neat concepts as runnable notebooks — Tensor, Node, Graph, model options, RTSP, senders. |
| [`apps/`](apps/README.md) | Complete, runnable applications: RTSP in → inference → annotated H.264/RTP UDP out. |
| [`model-compilation/`](model-compilation/README.md) | `.pt` → ONNX → graph surgery → INT8 → a single-`.elf` archive, proven on real images. |
| [`llima/`](llima/README.md) | LLM / VLM / ASR: the `llima` CLI, the `pyneat.genai` API, and the GenAI server. |
| [`appendix.md`](appendix.md) | Operations: RTSP test source, watching the UDP output, un-wedging the DevKit. |

`/workspace` is **NFS-mounted on the DevKit at the same path**, so you edit files host-side and run
them board-side with **no copying**.

## Table of Contents

- [1. Installation](#1-installation)
- [2. Tutorial — learn the concepts](#2-tutorial--learn-the-concepts)
- [3. Apps](#3-apps)
  - [Running one](#running-one)
- [4. Model compilation](#4-model-compilation)
- [5. GenAI — LLM / VLM / ASR](#5-genai--llm--vlm--asr)
- [6. Operations](#6-operations)
- [Prerequisites](#prerequisites)
- [Contributing & Contact](#contributing--contact)

---

## 1. Installation

**[`installation/README.md`](installation/README.md)** — install the SDK, pair the DevKit, attach
VS Code, and run your first thing on the board with `dk`.

- **[`neat_on_windows.md`](installation/neat_on_windows.md)** — the same stack on Windows, via WSL2.
- **[`neat_insight.md`](installation/neat_insight.md)** — Neat Insight: browser-based RTSP sources,
  video viewer, runtime metrics.
- **[`components.md`](installation/components.md)** — what each piece of the stack is (Host, DevKit,
  SDK, Neat Core, PyNeat, Model Compiler, Insight, Apps).

---

## 2. Tutorial — learn the concepts

**[`tutorial/README.md`](tutorial/README.md)** — thirteen runnable notebooks that build up the Neat
object model one piece at a time. Each is a concept cell, a short runnable cell, then an
interpretation. Start at `I-easy/01` and work down.

- **I — Easy:** the core objects. `Tensor`, `Node` and `Graph`, `Sample`, then a first end-to-end
  model (ResNet-50) and a YOLO detection pipeline. Nothing here needs a camera.
- **II — Medium:** the knobs you will actually turn in an app. `ModelOptions`,
  `RtspDecodedInputOptions`, `RunOptions`, `InputOptions`/`OutputOptions`, the video and metadata
  senders — ending with a full RTSP → decode → infer → encode → Neat Insight pipeline.

Run them **on the DevKit**, so the kernel can import `pyneat` — the tutorial README has the
`jupyter notebook` command and the URL to open.

Once the concepts land, move on to a complete application below.

---

## 3. Apps

Complete, runnable applications. Each is self-contained — its own `README.md`,
`config/default.conf`, and `assets/models/`.

Most take an **RTSP** stream, run a model, and publish an annotated **H.264/RTP UDP** stream you can
watch with GStreamer. Two apps differ, and it is worth knowing which before you go hunting for a
config key that does not exist: [`usb-camera-yolo26m`](apps/usb-camera-yolo26m/README.md) reads a
**USB camera on the board** instead of RTSP, and [`benchmark`](apps/benchmark/README.md) takes
**synthetic input** and emits console metrics and JSON — no video at all.

> **[`apps/README.md`](apps/README.md) — the NEAT API reference.** Every NEAT API these apps use,
> its important parameters, and which app to read for a worked example. It also maps the **three
> execution patterns** (Graph pipeline vs `Model::Runner` vs `Model::benchmark()`), the C++ ↔ Python
> naming differences, and the gotchas that cost real debugging time. Read it when you start writing
> your own app.

| App | What it demonstrates |
| --- | --- |
| [`single-stream-yolo-yolo11`](apps/single-stream-yolo-yolo11/README.md) | The baseline: one RTSP stream → YOLO11 → one UDP output. **Read this first.** |
| [`multi-stream-yolo-yolo11`](apps/multi-stream-yolo-yolo11/README.md) | 2× RTSP → **one shared** YOLO11 model stage → per-stream UDP out. Sustains the full 60 fps source rate on both streams. |
| [`quad-stream-quad-model`](apps/quad-stream-quad-model/README.md) | 4 streams × 4 **different** models (detection / segmentation / pose / YOLOX), all decoded on-device. Deep dive: [`LEARNING.md`](apps/quad-stream-quad-model/LEARNING.md). |
| [`detection-vlm-assistant`](apps/detection-vlm-assistant/README.md) | YOLO detection → trigger-gated crops → VLM captions. |
| [`pcb-defect-detection-yolo26n`](apps/pcb-defect-detection-yolo26n/README.md) | A custom-trained YOLO26n on a non-COCO domain, compiled end to end. |
| [`single-stream-yolo-yolov8n`](apps/single-stream-yolo-yolov8n/README.md) · [`-yolov8m`](apps/single-stream-yolo-yolov8m/README.md) · [`-yolo26n`](apps/single-stream-yolo26n/README.md) | The same single-stream shape with other detectors. |
| [`usb-camera-yolo26m`](apps/usb-camera-yolo26m/README.md) | **The one non-RTSP input:** a USB/UVC webcam on the board → YOLO26m → UDP out. 1080p @ 30 fps, camera-limited. C++ and Python. |
| [`single-stream-yolov8n-seg`](apps/single-stream-yolov8n-seg/README.md) | Segmentation masks instead of boxes. |
| [`single-stream-open-pose`](apps/single-stream-open-pose/README.md) | Pose keypoints and skeletons. `(TODO:: Stable Output)`|
| [`multi-model-load-probe`](apps/multi-model-load-probe/README.md) | How many models can the MLA hold and run at once? |
| [`benchmark`](apps/benchmark/README.md) | Throughput / latency measurement harness. |

### Running one

**The app's own README is the authority.** Binary names, config keys, and how the config is passed
all differ between apps. The shape is always the same:

1. **Get the model.** Archives are git-ignored, so after cloning **no models exist**. The app's
   README has a `Model Download Command` — some come from the SiMa model zoo, some from a direct
   URL, some you compile via [`model-compilation/`](model-compilation/README.md).
2. **Check the config.** `config/default.conf` — input (RTSP URL or camera device), UDP receiver IP,
   ports, thresholds.
3. **Build it.** Most apps are C++, so this step is mandatory — `cmake` in the SDK shell.
4. **Run it on the board** with `dk`, pointing at the C++ binary or the Python script.
5. **Watch the output** on the host with GStreamer, on the app's `udp_port`. For an RTSP test source,
   a live FPS readout, or a 2×2 grid, see [`appendix.md`](appendix.md).

Exact commands are in each app's README, under *How To Build*, *How To Run*, and *How To See The
Output*.

---

## 4. Model compilation

**[`model-compilation/README.md`](model-compilation/README.md)** — start here. Download the ten
prebuilt archives, or set up to compile them yourself.

**[`model-compilation/COMPILE-COMMANDS.md`](model-compilation/COMPILE-COMMANDS.md)** — the commands.
`compile_all.sh` for all ten, or a copy-paste block per model with its exact expected result.

**[`model-compilation/MODEL-COMPILATION.md`](model-compilation/MODEL-COMPILATION.md)** — the
reasoning. Why graph surgery is needed, what a good archive looks like (**one `.elf`, zero `.so`**),
the INT8 calibration trap, and what worked and what didn't.

No weights, ONNX graphs or compiled archives are committed — you regenerate them from the recipe.

---

## 5. GenAI — LLM / VLM / ASR

**[`llima/README.md`](llima/README.md)** — the `llima` CLI (which *prepares* models) versus the
`pyneat.genai` API (which *runs* them from your app), bring-your-own compilation, and an
OpenAI-compatible GenAI server.

---

## 6. Operations

**[`appendix.md`](appendix.md)** — the operational bits the apps assume you already have:

- Stand up an **RTSP test source** (`mediamtx` + `ffmpeg`) when you have no camera.
- **Watch the UDP output** — one stream, a live FPS readout, or a 2×2 grid.
- **Un-wedge the DevKit** when the MLA blocks.

---

## Prerequisites

| Prereq | Notes |
| --- | --- |
| **SDK container** | The NEAT SDK / container. |
| **DevKit** | A paired Modalix DevKit with `pyneat` importable. |
| **Model compiler** | `source /sdk-extensions/model-compiler/bin/activate` → `afe`, `onnx`. Plus `pip install ultralytics`. Only needed for [`model-compilation/`](model-compilation/README.md). |
| **RTSP source** | An H.264 stream for the apps. No camera? Serve a video file — [`appendix.md`](appendix.md#1-host-a-local-rtsp-source) has the `mediamtx` + `ffmpeg` recipe.<br>**Check its frame rate first** — it is the hard ceiling on any FPS you can claim:<br>`ffprobe -hide_banner -rtsp_transport tcp rtsp://<host>:8555/stream` |
| **Host viewer** | GStreamer, to watch the UDP/RTP output. Commands (single stream + 2×2 grid) in [`appendix.md`](appendix.md#2-host-viewing-the-udprtp-output). |

Install the host viewer tools:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

---

## Contributing & Contact

This repository is co-developed with [Claude](https://www.anthropic.com/claude).

Found an error? Please report it to [suraj.maurya@sima.ai](mailto:suraj.maurya@sima.ai).

To request an application or feature, suggest a correction, or contribute, contact
[suraj.maurya@sima.ai](mailto:suraj.maurya@sima.ai).
