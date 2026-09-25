# `apps/`: SiMa Neat demo applications

Runnable reference applications for the Modalix DevKit, built on the SiMa Neat Library. Each app
is self-contained: its own `main.cpp` and/or `main.py`, `config/default.conf`, `assets/models/`,
and a README with the exact build and run commands.

For the API these apps use (`Graph`/`Run`, nodes, node groups, `Model` options, enums, and the
C++ ↔ Python naming map), see **[NEAT-API-REFERENCE.md](NEAT-API-REFERENCE.md)**.

## Contents

- [Applications](#applications)
- [Running an app](#running-an-app)
- [Test source](#test-source)
- [C++ vs Python](#c-vs-python)
- [Migrating from NEAT 0.3.0](#migrating-from-neat-030)

---

## Applications

Most apps take an **RTSP** stream, run a model, and send annotated **H.264/RTP** video over UDP
to **Neat Insight**, which shows it in a browser. The exceptions are marked in the tables.

### Single stream, one model

| app | languages | what it does |
| --- | --- | --- |
| [**single-stream-yolo-yolo11**](single-stream-yolo-yolo11/README.md) | C++, Python | RTSP in → YOLO11 detection → annotated H.264/RTP out over UDP. The baseline. **Read this first.** |
| [**single-stream-yolo-yolov8m**](single-stream-yolo-yolov8m/README.md) | C++, Python | Same shape with YOLOv8m. The heaviest single-stream model here, and the one that shows the C++/Python gap most clearly. |
| [**single-stream-yolo26n**](single-stream-yolo26n/README.md) | C++, Python | Same shape with YOLO26n (`bf16` tessellated archive), which uses the `YoloV26` on-device box-decode family. |
| [**single-stream-yolov8n-seg**](single-stream-yolov8n-seg/README.md) | C++, Python | Instance segmentation. Neat's on-device `YoloV8Seg` decode assembles the per-instance masks; the app blends them onto the NV12 frame. |
| [**single-stream-yolo-insight**](single-stream-yolo-insight/README.md) | C++, Python | **Output to [Insight](https://developer.sima.ai/software/tools/insight/)** instead of burned-in overlays: encoded video on one UDP port, detection metadata on another, drawn by the Insight viewer. |
| [**single-stream-open-pose**](single-stream-open-pose/README.md) | C++ | OpenPose multi-person keypoints: heatmap peaks, PAF limb matching, skeleton assembly. Three decode defects were fixed in the 0.4.0 migration; the skeleton output has not yet been visually re-verified. |

### Multi-stream / multi-model

| app | languages | what it does |
| --- | --- | --- |
| [**multi-stream-yolo-yolo11**](multi-stream-yolo-yolo11/README.md) | Python | Two RTSP streams through **one shared** YOLO11 model stage, each with its own annotated UDP output. Sustains **~119 fps aggregate (≈60 fps per stream)**, the full source rate, with overlay on. |
| [**quad-stream-quad-model**](quad-stream-quad-model/README.md) | C++, Python | Four RTSP streams, four *different* models (detection, segmentation, pose, YOLOX) on one MLA, all decoded on-device. **C++ ~236 fps aggregate**, Python ~79 fps. Pick C++ for throughput, Python to read and modify. |
| [**16stream4model**](16stream4model/README.md) | C++ | **Sixteen 720p30 RTSP streams, four model groups** of four streams each: YOLO26n, YOLO11n and YOLOv8n detection plus YOLO26n pose. **Output to Insight**: the video passes through untouched, and detections and poses are sent as metadata. Holds **480 / 480 fps**, every stream at the source rate. |
| [**multi-model-load-probe**](multi-model-load-probe/README.md) | C++ | How many models can the MLA hold and run at once? Loads four model graphs in one process off a single RTSP source. A probe, not a product pipeline. |

### Other inputs / other tasks

| app | languages | what it does |
| --- | --- | --- |
| [**usb-camera-yolo26m**](usb-camera-yolo26m/README.md) | C++, Python | **USB/UVC webcam on the board**, not RTSP → YOLO26m → annotated H.264/RTP out. MJPEG capture is mandatory; raw YUYV at 1080p is USB-bandwidth-capped at ~5 fps. Verified on a Logitech Brio 100 at 1920x1080@30: **30.0 fps C++, 28.4 fps Python**, camera-limited. |
| [**pcb-defect-detection-yolo26n**](pcb-defect-detection-yolo26n/README.md) | Python | **Still images, not video.** A custom-trained YOLO26n on a non-COCO domain, compiled end to end, run over a folder of PCB photos to write annotated JPEGs. |
| [**detection-vlm-assistant**](detection-vlm-assistant/README.md) | Python | Detector always on; a vision-language model captions trigger-gated crops. Shows how to gate an expensive VLM behind a cheap detector. Verified on NEAT 0.4.0 with **Qwen3-VL-4B-Instruct-Autoround-a16w4**: detector at **46.4 fps** for 3 min while the VLM captioned at **~13.8 tok/s, 0.6–0.7 s TTFT**. |
| [**benchmark**](benchmark/README.md) | Python | **Synthetic input, no video.** Wraps `pyneat.Model.benchmark()` to measure one compiled model package in isolation and emit console metrics and JSON. Use it to get a model's ceiling before building a pipeline around it. |

---

## Running an app

**The app's own README is the authority.** Binary names, config keys and how the config is
passed differ between apps. The shape is always the same:

1. **Get the model.** Archives are git-ignored, so a fresh clone has **no models**. Each app's
   README has the download command. Some come from the SiMa model zoo, some from a direct URL,
   and some you compile with [`model-compilation/`](../model-compilation/README.md).
2. **Fill in the config.** `config/default.conf` ships with placeholders, such as `<rtsp-url>`
   and `<host-ip>`, for the input, the output host and ports. Replace them before the first run.
3. **Build it.** C++ apps need a build in the SDK shell:

   ```bash
   cd <app>
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
   cmake --build build -j"$(nproc)"
   ```

   A few apps need extra CMake flags (for example `16stream4model`); the app's README has the
   exact command. Python apps need no build step. They run on the board's pyneat interpreter
   (`/home/sima/pyneat/bin/python`).
4. **Run it on the board** with `dk`, pointing at the C++ binary or the Python script.
5. **Watch the output** in Insight's Video Viewer. Set the app's `udp_host` (or `insight_host`)
   to the Insight host; UDP port `9000 + N` is viewer channel `N`. [`appendix.md`](../appendix.md)
   covers serving RTSP test sources from Insight and opening the viewer.

---

## Test source

Unless stated otherwise, every app was exercised against the same input: **1280×720 H.264,
60 fps**, served over RTSP.

> **Test video:** [1280x720@60FPS](https://drive.google.com/file/d/10Bmi_a_6zA_dyRV-GGPuYkqwj3Llbl7O/view?usp=sharing)

Upload it to Insight, serve it as an RTSP source, and put that URL and the Insight host in the
app's `config/default.conf`. **Check the source's frame rate first** (Insight's `/api/media-info`,
see [`appendix.md`](../appendix.md#13-confirm-the-source-before-blaming-the-app)), because it is
the hard ceiling on any fps you can claim.

The exceptions:

- `usb-camera-yolo26m` uses a live UVC webcam.
- `pcb-defect-detection-yolo26n` uses still images.
- `benchmark` uses synthetic input.
- `16stream4model` uses sixteen 1280×720 30 fps Insight catalog clips, which its
  `scripts/start_sources.sh` sets up.

---

## C++ vs Python

**Where both exist, prefer the C++ build for performance.** The two implementations express the
same pipeline, but the Python one carries roughly **+5 ms of host work per frame**: model
push/pull marshalling, the NumPy→Tensor copy for the video sender, and overlay drawing under the
GIL.

That overhead is invisible when there is headroom and decisive when there is not. Measured on
NEAT 0.4.0 / SDK 2.1.3 against the 720p60 source:

| app | C++ | Python | note |
| --- | ---: | ---: | --- |
| `single-stream-yolo-yolo11` | **59.3 fps** | 59.3 fps | both at the source rate |
| `single-stream-yolo-yolov8m` | **58.9 fps** | 48.1 fps | heaviest model; Python runs out of budget |
| `quad-stream-quad-model` | **~236 fps** agg | ~79 fps agg | four models on one MLA; Python is overlay-bound |

The frame budget at 60 fps is 16.7 ms. Light models leave enough slack to absorb Python's
overhead. `yolov8m` spends 13.2 ms in inference alone, so Python's extra cost pushes the total
well past it. Use Python to read and modify, C++ to ship.

---

## Migrating from NEAT 0.3.0

Every app here was migrated to NEAT 0.4.0 / SDK 2.1.3 and re-verified on hardware.

- **The C++ ABI is now 4.** Binaries built against 0.3.0 link `libsima_neat.so.3` and fail to
  start. Delete `build/` and rebuild; the ABI break itself needs no source change.
- **A non-dropping source sink can stall the hardware decoder permanently.**
  `OutputOptions::EveryFrame()` / `every_frame()` leave `drop = false`. On 0.4.0, the moment the
  consumer drains slower than the source, the sink fills, back-pressures the decoder, and the
  stream goes to zero and stays there. 0.3.0 tolerated the same setting. Set `drop = true` on a
  live source sink that the application drains itself:

  ```python
  src_out = pyneat.OutputOptions.every_frame(4)
  src_out.drop = True
  ```

  The one deliberate exception is the C++ build of `quad-stream-quad-model`. Its source sink does
  not drop, because the app sheds load in its own drop-oldest mailbox and drains at the full
  source rate.
- **Config keys changed in these apps:** `fallback_width` / `fallback_height` / `fallback_fps`
  are now `width` / `height` / `fps`. Most apps reject unknown keys, so an older config fails
  loudly instead of silently doing the wrong thing. (`16stream4model` warns instead.)
