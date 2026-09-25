# SiMa NEAT on the Modalix DevKit: Quick Start, Tutorials and Demo Apps

Runnable demo apps, concept notebooks, a model-compilation reference, and a GenAI (LLM/VLM/ASR)
track — all for SiMa **NEAT** on the **Modalix DevKit**.

If you are opening this repo cold, work through
[1. Start here — the Quick Start Guide](#1-start-here--the-quick-start-guide) first — it takes a
DevKit from a sealed box to a running C++ application, installation included — then pick a track
below.

| Where | What you get |
| --- | --- |
| [`devkit-quick-start/`](devkit-quick-start/README.md) | Box to working application in 19 chapters: hardware, network, board software, first inference, SDK, Insight, a C++ app, agentic development. **Start here.** |
| [`tutorial/`](tutorial/README.md) | Core Neat concepts as runnable notebooks — Tensor, Node, Graph, model options, RTSP, senders, GenAI in a graph. |
| [`apps/`](apps/README.md) | Complete, runnable applications, most of them RTSP in → inference → output to Neat Insight. |
| [`model-compilation/`](model-compilation/README.md) | `.pt` → ONNX → graph surgery → INT8 → a single-`.elf` archive, proven on real images. |
| [`llima/`](llima/README.md) | LLM / VLM / ASR: the `llima` CLI, the `pyneat.genai` API, and the GenAI server. |
| [`troubleshooting`](#6-insight-and-troubleshooting) | Neat Insight for RTSP test sources and viewing app output, DevKit recovery, a command reference, and a symptom-first troubleshooting guide. |

## Prerequisites

- **A Modalix DevKit**, plus an Ethernet cable to put it on your network.
- **A host computer:** Ubuntu 22.04 or 24.04 (Windows 11 through WSL2, or macOS on Apple
  Silicon, also work), with Docker, `sudo` rights, 4 CPU cores, 16 GB RAM and 100 GB free disk.
- **A SiMa developer account**, for downloading the SDK and board software.

---

## 1. Start here — the Quick Start Guide

**[`devkit-quick-start/README.md`](devkit-quick-start/README.md)** — a chapter-by-chapter
walkthrough from a sealed box to a running C++ application. Installation is part of it, so this is
the only thing you need open the first time.

| | | |
| --- | --- | --- |
| [1. Know the hardware](devkit-quick-start/chapters/01-know-the-hardware.md) | [2. The board and its connections](devkit-quick-start/chapters/02-board-and-connections.md) | [3. HDMI, keyboard and mouse](devkit-quick-start/chapters/03-hdmi-and-peripherals.md) |
| [4. The serial console](devkit-quick-start/chapters/04-serial-console.md) | [5. Share the host's internet](devkit-quick-start/chapters/05-internet-sharing.md) | [6. Connect to a router](devkit-quick-start/chapters/06-connect-to-router.md) |
| [7. Install sima-cli](devkit-quick-start/chapters/07-install-sima-cli.md) | [8. Mount the NVMe](devkit-quick-start/chapters/08-mount-nvme.md) | [9. Check and update the board image](devkit-quick-start/chapters/09-check-and-update-image.md) |
| [10. Install simaai-sentinel](devkit-quick-start/chapters/10-install-simaai-sentinel.md) | [11. Install pyneat](devkit-quick-start/chapters/11-install-pyneat.md) | [12. Object detection on images](devkit-quick-start/chapters/12-object-detection.md) |
| [13. LLiMa: search, list, pull](devkit-quick-start/chapters/13-llima.md) | [14. The Neat stack](devkit-quick-start/chapters/14-neat-components.md) | [15. Install the Neat SDK](devkit-quick-start/chapters/15-install-neat-sdk.md) |
| [16. The `dk` command](devkit-quick-start/chapters/16-devkit-tool-dk.md) | [17. Neat Insight](devkit-quick-start/chapters/17-neat-insight.md) | [18. A C++ video application, end to end](devkit-quick-start/chapters/18-cpp-video-app.md) |
| [19. Agentic development](devkit-quick-start/chapters/19-agentic-development.md) |  |  |

Work through it in order the first time; afterwards each chapter stands alone.

---

## 2. Tutorial — learn the concepts

**[`tutorial/README.md`](tutorial/README.md)** — fourteen runnable notebooks that build up the Neat
object model one piece at a time. Each is a concept cell, a short runnable cell, then an
interpretation. Start at `I-easy/01` and work down.

| Level | | | |
| --- | --- | --- | --- |
| **I — Easy** | [1. Neat Tensor](tutorial/I-easy/01_neat_tensor.ipynb) | [2. Node and Graph](tutorial/I-easy/02_node_and_graph.ipynb) | [3. Interpret Model Output Samples](tutorial/I-easy/03_interpret_model_output_samples.ipynb) |
|  | [4. Image Classification with ResNet-50](tutorial/I-easy/04_image_classification_resnet.ipynb) | [5. YOLO CPU Decode](tutorial/I-easy/05_yolo_cpu_decode.ipynb) | [6. YOLOv8 Image Detection Pipeline](tutorial/I-easy/06_yolov8_image_detection_pipeline.ipynb) |
| **II — Medium** | [1. ModelOptions](tutorial/II-medium/01_model_options.ipynb) | [2. RTSP Input And Decode Options](tutorial/II-medium/02_rtsp_input_and_decode_options.ipynb) | [3. RunOptions](tutorial/II-medium/03_run_options.ipynb) |
|  | [4. InputOptions And OutputOptions](tutorial/II-medium/04_input_output_options.ipynb) | [5. VideoSender And VideoSenderOptions](tutorial/II-medium/05_video_sender_options.ipynb) | [6. MetadataSender And MetadataSenderOptions](tutorial/II-medium/06_metadata_sender_options.ipynb) |
|  | [7. RTSP To Insight: Decode, Encode, Annotate, Send](tutorial/II-medium/07_rtsp_decode_encode_metadata_to_insight.ipynb) |  |  |
| **III — Advanced** | [1. GenAI Model In A Graph](tutorial/III-advance/01_genai_model_in_graph.ipynb) |  |  |

Run them **on the DevKit**, so the kernel can import `pyneat` — the tutorial README has the
`jupyter notebook` command and the URL to open.

Once the concepts land, move on to a complete application below.

---

## 3. Apps

Complete, runnable applications for the DevKit. **[`apps/README.md`](apps/README.md)** describes
each one and how to run it. New to the apps? Start with **single-stream-yolo-yolo11**.

| Category | | | |
| --- | --- | --- | --- |
| **Single stream** | [single-stream-yolo-yolo11](apps/single-stream-yolo-yolo11/README.md) | [single-stream-yolo-yolov8m](apps/single-stream-yolo-yolov8m/README.md) | [single-stream-yolo26n](apps/single-stream-yolo26n/README.md) |
| | [single-stream-yolov8n-seg](apps/single-stream-yolov8n-seg/README.md) | [single-stream-yolo-insight](apps/single-stream-yolo-insight/README.md) | [single-stream-open-pose](apps/single-stream-open-pose/README.md) |
| **Multi-stream / multi-model** | [multi-stream-yolo-yolo11](apps/multi-stream-yolo-yolo11/README.md) | [quad-stream-quad-model](apps/quad-stream-quad-model/README.md) | [16stream4model](apps/16stream4model/README.md) |
| | [multi-model-load-probe](apps/multi-model-load-probe/README.md) | | |
| **Other inputs / tasks** | [usb-camera-yolo26m](apps/usb-camera-yolo26m/README.md) | [pcb-defect-detection-yolo26n](apps/pcb-defect-detection-yolo26n/README.md) | [detection-vlm-assistant](apps/detection-vlm-assistant/README.md) |
| | [benchmark](apps/benchmark/README.md) | | |

---

## 4. Model compilation

**[`model-compilation/README.md`](model-compilation/README.md)** — start here. Twelve models: download
the ten prebuilt archives, or set up to compile them yourself.

Models with a compile recipe (each links to its commands):

| Task | | | |
| --- | --- | --- | --- |
| **Classification** | [resnet50](model-compilation/COMPILE-COMMANDS.md#1-resnet50--classification-no-surgery) | [convnext_tiny](model-compilation/COMPILE-COMMANDS.md#2-convnext_tiny--classification-no-surgery) | [densenet169](model-compilation/COMPILE-COMMANDS.md#3-densenet169--classification-no-surgery) |
|  | [efficientnet_v2_s](model-compilation/COMPILE-COMMANDS.md#4-efficientnet_v2_s--classification-no-surgery-384384-input) |  |  |
| **Detection** | [yolov8s](model-compilation/COMPILE-COMMANDS.md#5-yolov8s--detection-surgery-head-at-model22-no-attention) | [yolo11n](model-compilation/COMPILE-COMMANDS.md#6-yolo11n--detection-surgery) | [yolo11s](model-compilation/COMPILE-COMMANDS.md#7-yolo11s--detection-surgery) |
|  | [yolo26n](model-compilation/COMPILE-COMMANDS.md#8-yolo26n--detection-surgery-no-dfl-rebuild) | [yolox_s](model-compilation/COMPILE-COMMANDS.md#11-yolox_s--detection-different-surgery) |  |
| **Segmentation** | [yolo11s-seg](model-compilation/COMPILE-COMMANDS.md#9-yolo11s-seg--segmentation-surgery) |  |  |
| **Pose** | [yolo26s-pose](model-compilation/COMPILE-COMMANDS.md#10-yolo26s-pose--pose-surgery-carries-the-209-fix) |  |  |
| **Open-vocabulary detection** | [yolov8s-worldv2](model-compilation/COMPILE-COMMANDS.md#12-yolov8s-worldv2--open-vocabulary-bf16-not-int8) |  |  |

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

| Topic | | | |
| --- | --- | --- | --- |
| **Basics** | [LLiMa Introduction](llima/01-llima-basics/llima-introduction.ipynb) |  |  |
| **Run LLM / VLM / ASR** | [Run an LLM from Python](llima/02-run-llm-vlm/01_run_llm.ipynb) | [Run a VLM from Python](llima/02-run-llm-vlm/02_run_vlm.ipynb) | [Transcribe audio (ASR)](llima/02-run-llm-vlm/03_audio_input_asr.ipynb) |
| **YOLO + VLM** | [YOLO detection + a VLM](llima/03-yolo-plus-vlm/01_detection_to_vlm.ipynb) |  |  |
| **Compilation** | [Bring-your-own LLM](llima/04-llm-vlm-compilation/01_llm_compilation.ipynb) | [VLM specifics: vision encoder + language model](llima/04-llm-vlm-compilation/02_vlm_compilation.ipynb) |  |
| **GenAI server** | [GenAIServer: one HTTP server for LLM, VLM and ASR](llima/05-genai-server/01_genai_server.ipynb) |  |  |

---

## 6. Insight and troubleshooting

**[`appendix.md`](appendix.md)** — the practical steps behind the apps:

- Serve **RTSP test sources** from **Neat Insight** when you have no camera.
- **Watch the output** in Insight's browser Video Viewer.
- **Un-wedge the DevKit** when the MLA or a decoder blocks.

From the Quick Start Guide:

- **[Troubleshooting](devkit-quick-start/chapters/troubleshooting.md)** — symptom first: find yours
  and work down the list in order.
- **[Miscellaneous](devkit-quick-start/chapters/miscellaneous.md)** — command reference, board
  defaults, the PCIe card, and where to read more.

---

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

---

## Contributing & Contact

This repository is co-developed with [Claude](https://www.anthropic.com/claude).

Found an error? Please report it to [suraj.maurya@sima.ai](mailto:suraj.maurya@sima.ai).

To request an application or feature, suggest a correction, or contribute, contact
[suraj.maurya@sima.ai](mailto:suraj.maurya@sima.ai).
