# NEAT enablement material for the SiMa Modalix DevKit

Runnable demo apps, concept notebooks, a model-compilation reference, and a GenAI (LLM/VLM/ASR)
track — all for SiMa **NEAT** on the **Modalix DevKit**.

If you are opening this repo cold, do [`installation/`](installation/README.md) first, then pick a
track below.

## What is in here

| Where | What you get |
| --- | --- |
| [`installation/`](installation/README.md) | Set up the SDK container, pair the DevKit, install Neat Insight. **Start here.** |
| [`tutorial/`](tutorial/README.md) | Core Neat concepts as runnable notebooks — Tensor, Node, Graph, Run, model options, RTSP, video/metadata senders. |
| [`apps/`](apps/README.md) | Complete, runnable applications: RTSP in → inference → annotated H.264/RTP UDP out. Plus [`apps/README.md`](apps/README.md) — **the NEAT API reference**: every API these apps use, its parameters, and which app to read for it. |
| [`model-compilation/`](model-compilation/README.md) | `.pt` → ONNX → graph surgery → INT8 → a single-`.elf` archive, proven on real images. |
| [`llima/`](llima/README.md) | LLM / VLM / ASR: the `llima` CLI, the `pyneat.genai` API, compilation, and the GenAI server. |
| [`appendix.md`](appendix.md) | **Operations.** Stand up an RTSP test source, watch the UDP output (one stream or a 2×2 grid), and un-wedge the DevKit when the MLA blocks. |

---

## Setup

Read these in order:

1. **[`installation/README.md`](installation/README.md)** — SDK container, DevKit pairing, VS Code,
   the `dk` runner. The platform-neutral guide.
2. **[`installation/neat_on_windows.md`](installation/neat_on_windows.md)** — the same stack on
   Windows, via WSL2.
3. **[`installation/neat_insight.md`](installation/neat_insight.md)** — Neat Insight: browser-based
   RTSP sources, video viewer, runtime metrics.
4. **[`appendix.md`](appendix.md)** — the operational bits the apps assume you already have: an RTSP
   test source (`mediamtx` + `ffmpeg`), the GStreamer commands to watch the UDP output, and DevKit
   recovery when the MLA wedges.

`/workspace` is **NFS-mounted on the DevKit at the same path**, so you edit files host-side and run
them board-side with **no copying**.

---

## Apps

> **[`apps/README.md`](apps/README.md) — the NEAT API reference.** Every NEAT API these apps use, what
> it is for, its important parameters, and which app to read for a worked example. It also maps the
> **three execution patterns** (Graph pipeline vs `Model::Runner` vs `Model::benchmark()`), the
> C++ ↔ Python naming differences, and the gotchas that cost real debugging time. Read it when you
> start writing your own app.

Every app is self-contained — its own `README.md`, `config/default.conf`, and `assets/models/`.

Most take an **RTSP** stream, run a model, and publish an annotated **H.264/RTP UDP** stream you can
watch with GStreamer. Two apps differ, and it is worth knowing which before you go looking for a
config key that does not exist:

- [`usb-camera-yolo26m`](apps/usb-camera-yolo26m/README.md) reads a **USB/UVC camera on the board**,
  not RTSP. It still publishes the same UDP output.
- [`benchmark`](apps/benchmark/README.md) takes **synthetic input** and emits **console metrics and
  JSON reports**. It publishes no video at all.

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

### Running an app

**The app's own README is the authority — follow that.** Binary names, config keys, and how the
config is passed differ between apps. The shape below is the same everywhere; the exact commands are
not.

1. **Open the app's README.** Every app has one, and they all follow the same section order:
   *Requirements → Model Download Command → Configure → Config Parameters → How To Build →
   How To Run → How To See The Output.*

2. **Get the model.** Each README has a `Model Download Command`. Archives are git-ignored, so after
   cloning **no models exist** — you download or build them. Some come from the SiMa model zoo, some
   are a direct URL, and some you compile yourself via
   [`model-compilation/`](model-compilation/README.md). The app's README says which.

3. **Configure.** Edit `config/default.conf`: input (RTSP URL, or camera device), UDP receiver IP,
   ports, thresholds.

4. **Build — most apps are C++ and this step is mandatory.** Run in the SDK shell:

   ```bash
   cd apps/<app>
   cmake -S . -B ./build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
   cmake --build ./build --parallel
   ```

5. **Run it on the board** with `dk`. Most apps ship both a C++ binary and a Python script:

   ```bash
   dk ./build/<binary_name>          # C++  — binary name is in the app's README
   dk ./main.py                      # Python
   ```

   > **How the config is passed is not uniform.** Several apps take `--config ./config/default.conf`;
   > others read `./config/default.conf` implicitly, or take the path as a positional argument and
   > have no `--config` flag at all. Check the app's *How To Run* section rather than assuming — a
   > wrong flag here looks like a broken app.

6. **Watch the output on the host** (match the app's `udp_port`):

   ```bash
   gst-launch-1.0 -v udpsrc port=5206 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
   ```

   For an RTSP test source, a live FPS readout, or a 2x2 multi-stream grid, see
   [`appendix.md`](appendix.md).

---

## Model compilation

**[`model-compilation/README.md`](model-compilation/README.md)** — start here. Download the ten
prebuilt archives, or set up to compile them yourself.

**[`model-compilation/COMPILE-COMMANDS.md`](model-compilation/COMPILE-COMMANDS.md)** — the commands.
`compile_all.sh` for all ten, or a copy-paste block per model with its exact expected result.

**[`model-compilation/MODEL-COMPILATION.md`](model-compilation/MODEL-COMPILATION.md)** — the
reasoning. Why graph surgery is needed, what a good archive looks like (**one `.elf`, zero `.so`**),
the INT8 calibration trap, and what worked and what didn't.

No weights, ONNX graphs or compiled archives are committed — you regenerate them from the recipe.

---

## GenAI (LLM / VLM / ASR)

**[`llima/README.md`](llima/README.md)** — the `llima` CLI (which *prepares* models) versus the
`pyneat.genai` API (which *runs* them from your app), bring-your-own compilation, and an
OpenAI-compatible GenAI server.

---

## Prerequisites

| Prereq | Notes |
| --- | --- |
| **SDK container** | The NEAT SDK / container.|
| **DevKit** | A paired Modalix DevKit with `pyneat` importable. |
| **Model compiler** | `source /sdk-extensions/model-compiler/bin/activate` → `afe`, `onnx`, `ultralytics`. Only needed for [`model-compilation/`](model-compilation/README.md). |
| **RTSP source** | An H.264 stream for the apps. No camera? Serve a video file — [`appendix.md`](appendix.md#1-host-a-local-rtsp-source) has the `mediamtx` + `ffmpeg` recipe.<br>**Check its frame rate first** — it is the hard ceiling on any FPS you can claim:<br>`ffprobe -hide_banner -rtsp_transport tcp rtsp://<host>:8555/stream` |
| **Host viewer** | GStreamer, to watch the UDP/RTP output. Commands (single stream + 2×2 grid) in [`appendix.md`](appendix.md#2-host-viewing-the-udprtp-output). |

Install the host viewer tools:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

---
