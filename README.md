# NEAT enablement material for the SiMa Modalix DevKit

Runnable demo apps, concept notebooks, a model-compilation reference, and a GenAI (LLM/VLM/ASR)
track — all for SiMa **NEAT** on the **Modalix DevKit**.

If you are opening this repo cold, work through
[`devkit-quick-start/`](devkit-quick-start/README.md) first — it takes a DevKit from a sealed box
to a running C++ application, installation included — then pick a track below.

| Where | What you get |
| --- | --- |
| [`devkit-quick-start/`](devkit-quick-start/README.md) | Box to working application in 19 chapters: hardware, network, board software, first inference, SDK, Insight, a C++ app, agentic development. **Start here.** |
| [`installation/`](installation/README.md) | Long-form SDK install reference, with screenshots. Covered by the Quick Start Guide. |
| [`tutorial/`](tutorial/README.md) | Core Neat concepts as runnable notebooks — Tensor, Node, Graph, model options, RTSP, senders, GenAI in a graph. |
| [`apps/`](apps/README.md) | Complete, runnable applications, most of them RTSP in → inference → output to Neat Insight. |
| [`model-compilation/`](model-compilation/README.md) | `.pt` → ONNX → graph surgery → INT8 → a single-`.elf` archive, proven on real images. |
| [`llima/`](llima/README.md) | LLM / VLM / ASR: the `llima` CLI, the `pyneat.genai` API, and the GenAI server. |
| [`appendix.md`](appendix.md) | Operations: Neat Insight for RTSP sources and viewing the output, and un-wedging the DevKit. |

`/workspace` is **NFS-mounted on the DevKit at the same path**, so you edit files host-side and run
them board-side with **no copying**.

## Tested on NEAT 0.4.0

Everything here was migrated to and re-verified against **NEAT Library 0.4.0 / SDK 2.1.3**, on a
Modalix DevKit. The 10 C++ apps build with zero warnings, and all 14 apps were run on real
hardware, most against a 1280x720 H.264 60 fps RTSP source. The exceptions are listed in
[`apps/README.md`](apps/README.md#test-source). All 14 tutorial notebooks run on NEAT 0.4.0.

| | Version |
| --- | --- |
| NEAT Library | **0.4.0** (`libsima_neat.so.4`, ABI 4) |
| pyneat | **0.4.0** |
| SDK container | **2.1.3** |
| Board interpreter | `/home/sima/pyneat/bin/python` |

## Table of Contents

- [Tested on NEAT 0.4.0](#tested-on-neat-040)
- [1. Start here — the Quick Start Guide](#1-start-here--the-quick-start-guide)
- [2. Tutorial — learn the concepts](#2-tutorial--learn-the-concepts)
- [3. Apps](#3-apps)
- [4. Model compilation](#4-model-compilation)
- [5. GenAI — LLM / VLM / ASR](#5-genai--llm--vlm--asr)
- [6. Operations](#6-operations)
- [Prerequisites](#prerequisites)
- [Contributing & Contact](#contributing--contact)

---

## 1. Start here — the Quick Start Guide

**[`devkit-quick-start/README.md`](devkit-quick-start/README.md)** — a chapter-by-chapter
walkthrough from a sealed box to a running C++ application. Installation is part of it, so this is
the only thing you need open the first time.

| Part | Chapters | What it covers |
| --- | --- | --- |
| 1 | 1–4 | The hardware, the ports, the serial console |
| 2 | 5–6 | Getting the board on the network |
| 3 | 7–11 | `sima-cli`, the NVMe, board software, `simaai-sentinel`, `pyneat` |
| 4 | 12–13 | First inference on the MLA, then a language model with LLiMa |
| 5 | 14–19 | The Neat stack, the SDK, `dk`, Insight, a full C++ video application, and agentic development |

Work through it in order the first time; afterwards each chapter stands alone.

---

## 2. Tutorial — learn the concepts

**[`tutorial/README.md`](tutorial/README.md)** — fourteen runnable notebooks that build up the Neat
object model one piece at a time. Each is a concept cell, a short runnable cell, then an
interpretation. Start at `I-easy/01` and work down.

- **I — Easy:** the core objects. `Tensor`, `Node` and `Graph`, `Sample`, then a first end-to-end
  model (ResNet-50) and a YOLO detection pipeline. Nothing here needs a camera.
- **II — Medium:** the knobs you will actually turn in an app. `ModelOptions`,
  `RtspDecodedInputOptions`, `RunOptions`, `InputOptions`/`OutputOptions`, the video and metadata
  senders — ending with a full RTSP → decode → infer → encode → Neat Insight pipeline.
- **III — Advanced:** a GenAI model (VLM/LLM) as a stage inside a `Graph`.

Run them **on the DevKit**, so the kernel can import `pyneat` — the tutorial README has the
`jupyter notebook` command and the URL to open.

Once the concepts land, move on to a complete application below.

---

## 3. Apps

Complete, runnable applications for the DevKit. **[`apps/README.md`](apps/README.md)** describes
each one, how to run them, and the known rough edges.

| | | |
| --- | --- | --- |
| [single-stream-yolo-yolo11](apps/single-stream-yolo-yolo11/README.md) | [single-stream-yolo-yolov8m](apps/single-stream-yolo-yolov8m/README.md) | [single-stream-yolo26n](apps/single-stream-yolo26n/README.md) |
| [single-stream-yolov8n-seg](apps/single-stream-yolov8n-seg/README.md) | [single-stream-yolo-insight](apps/single-stream-yolo-insight/README.md) | [single-stream-open-pose](apps/single-stream-open-pose/README.md) |
| [multi-stream-yolo-yolo11](apps/multi-stream-yolo-yolo11/README.md) | [quad-stream-quad-model](apps/quad-stream-quad-model/README.md) | [16stream4model](apps/16stream4model/README.md) |
| [multi-model-load-probe](apps/multi-model-load-probe/README.md) | [usb-camera-yolo26m](apps/usb-camera-yolo26m/README.md) | [pcb-defect-detection-yolo26n](apps/pcb-defect-detection-yolo26n/README.md) |
| [detection-vlm-assistant](apps/detection-vlm-assistant/README.md) | [benchmark](apps/benchmark/README.md) | |

---

## 4. Model compilation

**[`model-compilation/README.md`](model-compilation/README.md)** — start here. Twelve models: download
the ten prebuilt archives, or set up to compile them yourself.

**[`model-compilation/COMPILE-COMMANDS.md`](model-compilation/COMPILE-COMMANDS.md)** — the commands.
`compile_all.sh` for eleven of the twelve (`yolov8s-worldv2` needs bf16 and is built separately), or
a copy-paste block per model with its exact expected result.

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

- Serve **RTSP test sources** from **Neat Insight** when you have no camera.
- **Watch the output** in Insight's browser Video Viewer.
- **Un-wedge the DevKit** when the MLA or a decoder blocks.

---

## Prerequisites

| Prereq | Notes |
| --- | --- |
| **SDK container** | The NEAT SDK / container. |
| **DevKit** | A paired Modalix DevKit with `pyneat` importable. |
| **Model compiler** | `source /sdk-extensions/model-compiler/bin/activate` → `afe`, `onnx`. Plus `pip install ultralytics`. Only needed for [`model-compilation/`](model-compilation/README.md). |
| **Neat Insight** | Bundled with the SDK. It serves the RTSP test sources and shows the apps' output in a browser, with nothing to install. See [`appendix.md`](appendix.md#1-neat-insight-sources-in-video-out).<br>**Check a source's frame rate first**, because it is the hard ceiling on any fps you can claim. |

---

## Contributing & Contact

This repository is co-developed with [Claude](https://www.anthropic.com/claude).

Found an error? Please report it to [suraj.maurya@sima.ai](mailto:suraj.maurya@sima.ai).

To request an application or feature, suggest a correction, or contribute, contact
[suraj.maurya@sima.ai](mailto:suraj.maurya@sima.ai).
