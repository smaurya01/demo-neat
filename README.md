# demo-neat — NEAT enablement material for the SiMa Modalix DevKit

Runnable demo apps, concept notebooks, a model-compilation reference, and a GenAI (LLM/VLM/ASR)
track — all for SiMa **NEAT** on the **Modalix DevKit**.

If you are opening this repo cold, do [`installation/`](installation/README.md) first, then pick a
track below.

## What is in here

| Folder | What you get |
| --- | --- |
| [`installation/`](installation/README.md) | Set up the SDK container, pair the DevKit, install Neat Insight. **Start here.** |
| [`tutorial/`](tutorial/README.md) | Core Neat concepts as runnable notebooks — Tensor, Node, Graph, Run, model options, RTSP, video/metadata senders. |
| [`apps/`](#apps) | Complete, runnable applications: RTSP in → inference → annotated H.264/RTP UDP out. |
| [`model-compilation/`](model-compilation/README.md) | `.pt` → ONNX → graph surgery → INT8 → a single-`.elf` archive, proven on real images. |
| [`llima/`](llima/README.md) | LLM / VLM / ASR: the `llima` CLI, the `pyneat.genai` API, compilation, and the GenAI server. |

---

## Setup

Read these in order:

1. **[`installation/README.md`](installation/README.md)** — SDK container, DevKit pairing, VS Code,
   the `dk` runner. The platform-neutral guide.
2. **[`installation/neat_on_windows.md`](installation/neat_on_windows.md)** — the same stack on
   Windows, via WSL2.
3. **[`installation/neat_insight.md`](installation/neat_insight.md)** — Neat Insight: browser-based
   RTSP sources, video viewer, runtime metrics.

`/workspace` is **NFS-mounted on the DevKit at the same path**, so you edit files host-side and run
them board-side with **no copying**.

---

## Apps

Every app is self-contained — its own `README.md`, `config/default.conf`, and `assets/models/`. Each
reads RTSP, runs a model, and publishes an annotated H.264/RTP UDP stream you can watch with
GStreamer.

| App | What it demonstrates |
| --- | --- |
| [`single-stream-yolo-yolo11`](apps/single-stream-yolo-yolo11/README.md) | The baseline: one RTSP stream → YOLO11 → one UDP output. **Read this first.** |
| [`multi-stream-yolo-yolo11`](apps/multi-stream-yolo-yolo11/README.md) | 2× RTSP → **one shared** YOLO11 model stage → per-stream UDP out. Sustains the full 60 fps source rate on both streams. |
| [`quad-stream-quad-model`](apps/quad-stream-quad-model/README.md) | 4 streams × 4 **different** models (detection / segmentation / pose / YOLOX), all decoded on-device. Deep dive: [`LEARNING.md`](apps/quad-stream-quad-model/LEARNING.md). |
| [`detection-vlm-assistant`](apps/detection-vlm-assistant/README.md) | YOLO detection → trigger-gated crops → VLM captions. |
| [`pcb-defect-detection-yolo26n`](apps/pcb-defect-detection-yolo26n/README.md) | A custom-trained YOLO26n on a non-COCO domain, compiled end to end. |
| [`single-stream-yolo-yolov8n`](apps/single-stream-yolo-yolov8n/README.md) · [`-yolov8m`](apps/single-stream-yolo-yolov8m/README.md) · [`-yolo26n`](apps/single-stream-yolo26n/README.md) | The same single-stream shape with other detectors. |
| [`single-stream-yolov8n-seg`](apps/single-stream-yolov8n-seg/README.md) | Segmentation masks instead of boxes. |
| [`single-stream-open-pose`](apps/single-stream-open-pose/README.md) | Pose keypoints and skeletons. `(TODO:: Stable Output)`|
| [`multi-model-load-probe`](apps/multi-model-load-probe/README.md) | How many models can the MLA hold and run at once? |
| [`benchmark`](apps/benchmark/README.md) | Throughput / latency measurement harness. |

### Running an app

1. Open the app's README.
2. Point `config/default.conf` at your RTSP URL, UDP receiver IP, ports and thresholds.
3. Put the compiled archive in the app's `assets/models/` (git-ignored — see
   [`model-compilation/`](model-compilation/README.md) to build it).
4. Run it on the board:

   ```bash
   cd apps/<app>
   dk ./main.py --config ./config/default.conf
   ```

5. Watch the output on the host (match the app's `udp_port`):

   ```bash
   gst-launch-1.0 -v udpsrc port=5206 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
   ```

---

## Model compilation

**[`model-compilation/README.md`](model-compilation/README.md)** — the reference. Why graph surgery
is needed, what a good archive looks like (**one `.elf`, zero `.so`**), and how INT8 calibration will
silently ruin a model if you get it wrong.

**[`model-compilation/REPLICATION.md`](model-compilation/REPLICATION.md)** — a copy-paste block for
each of the ten models: the exact four commands and the exact expected result.

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
| **SDK container** | The SiMa Modalix SDK / eLxr container. C++ sysroot at `/opt/toolchain/aarch64/modalix`. |
| **DevKit** | A paired Modalix DevKit with `pyneat` importable (Python 3.11, `aarch64`). |
| **Model compiler** | `source /sdk-extensions/model-compiler/bin/activate` → `afe`, `onnx`, `ultralytics`. Only needed for [`model-compilation/`](model-compilation/README.md). |
| **RTSP source** | An H.264 stream for the apps. **Check its frame rate first** — it is the hard ceiling on any FPS you can claim:<br>`ffprobe -hide_banner -rtsp_transport tcp rtsp://<host>:8555/stream` |
| **Host viewer** | GStreamer, to watch the UDP/RTP output. |

Install the host viewer tools:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

---

## What has been run, and what has not

Not everything here has been executed on hardware. Knowing which is which will save you time.

**Validated on the DevKit**

- **The YOLO compile chain** — `.pt` → ONNX → surgery → INT8 → archive (one `.elf`, zero `.so`) →
  inference on real images, for all ten models in `models.yaml`.
  See [`model-compilation/REPLICATION.md`](model-compilation/REPLICATION.md).
- **[`multi-stream-yolo-yolo11`](apps/multi-stream-yolo-yolo11/README.md)** — sustains the **full
  source rate on both streams** (~59 fps each against a 59.94 fps camera, zero dropped frames).
- **[`quad-stream-quad-model`](apps/quad-stream-quad-model/README.md)** — all four models exceed
  **60 fps model rate**. Its *overlay* path is gated by host-side (A65) decode, and it has known
  run-to-run variance; both are documented in its README.
- **The detection leg of [`detection-vlm-assistant`](apps/detection-vlm-assistant/README.md)** —
  RTSP/image in, boxes out.

**Written but NOT executed** — copy-paste ready with expected output, but run them yourself:

- **All LLM / VLM / ASR execution** — `llima pull`, `llima run`, and every `pyneat.genai` inference
  call. That is all of `llima/02-run-llm-vlm/` and `llima/05-genai-server/`, plus the **VLM leg** of
  `detection-vlm-assistant` (pass `--no-vlm` to exercise just the detector).
- **All GenAI compilation** — `llima/04-llm-vlm-compilation/`.

**Docs-derived, not source-verified**

- The GenAI compilation notebooks are written from official documentation, not from working code.
  There is **no `llima compile` subcommand** — the on-board CLI is runtime + model-manager only
  (`run, search, pull, list, rm, benchmark-server`). Compilation is a separate **host-side
  `llima-compile`** tool; confirm its flags against the official SiMa docs. The part that *is*
  verified against `core/` is the deployed model-directory contract.

---

## What is committed

The repo holds only what you cannot regenerate. `.gitignore` excludes build output and model
artifacts:

```text
build/  CMakeFiles/  CMakeCache.txt  Makefile
*.o  *.tar.gz  *.mpk  *.onnx  *.pt  *.elf  *.so  *.log
```

So after cloning: **no model archives and no `build/` directories exist.** Build them by following
[`model-compilation/README.md`](model-compilation/README.md), then drop the archive into the app's
`assets/models/`.

The one deliberate exception is `model-compilation/assets/` — the calibration images are an *input*,
not an artifact. Quantization is not reproducible without them, so they are tracked.
