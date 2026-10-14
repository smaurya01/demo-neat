# `apps/`: SiMa Neat demo applications

Runnable reference applications for the Modalix DevKit, built on the SiMa Neat Library. Each app
is self-contained: its own `main.cpp` and/or `main.py`, `config/default.conf`, `assets/models/`,
and a README with the exact build and run commands.

All apps are tested on **Neat 0.5.0 / SDK 2.1.3.1** (DevKit software 2.1.3).

For the API these apps use (`Graph`/`Run`, nodes, node groups, `Model` options, enums, and the
C++ ↔ Python naming map), see **[NEAT-API-REFERENCE.md](NEAT-API-REFERENCE.md)**.

## Contents

- [Applications](#applications)
- [Running an app](#running-an-app)
- [Test source](#test-source)
- [C++ vs Python](#c-vs-python)

---

## Applications

Most apps take an **RTSP** stream, run a model, and send annotated **H.264/RTP** video over UDP
to **Neat Insight**, which shows it in a browser. The exceptions are marked in the tables.

### Single stream, one model

| app | languages | what it does |
| --- | --- | --- |
| [**single-stream-yolo-yolo11**](single-stream-yolo-yolo11/README.md) | C++, Python | RTSP in → YOLO11 detection → annotated H.264/RTP out over UDP. The baseline. **Read this first.** |
| [**single-stream-yolo-yolov8m**](single-stream-yolo-yolov8m/README.md) | C++, Python | Same shape with YOLOv8m, the heaviest single-stream model here. |
| [**single-stream-yolo26n**](single-stream-yolo26n/README.md) | C++, Python | Same shape with YOLO26n (`bf16` tessellated archive), which uses the `YoloV26` on-device box-decode family. |
| [**single-stream-yolov8n-seg**](single-stream-yolov8n-seg/README.md) | C++, Python | Instance segmentation. Neat's on-device `YoloV8Seg` decode assembles the per-instance masks; the app blends them onto the NV12 frame. |
| [**single-stream-yolo-insight**](single-stream-yolo-insight/README.md) | C++, Python | **Output to [Insight](https://developer.sima.ai/software/tools/insight/)** instead of burned-in overlays: encoded video on one UDP port, detection metadata on another, drawn by the Insight viewer. |

### Multi-stream / multi-model

| app | languages | what it does |
| --- | --- | --- |
| [**multi-stream-yolo-yolo11**](multi-stream-yolo-yolo11/README.md) | Python | Two RTSP streams through **one shared** YOLO11 model stage, each with its own annotated UDP output, both at the full source rate. |
| [**quad-stream-quad-model**](quad-stream-quad-model/README.md) | C++, Python | Four RTSP streams, four *different* models (detection, segmentation, pose, YOLOX) on one MLA, all decoded on-device. Pick C++ for throughput, Python to read and modify. |
| [**16stream4model**](16stream4model/README.md) | C++ | **Sixteen 720p30 RTSP streams, four model groups** of four streams each: YOLO26n, YOLO11n and YOLOv8n detection plus YOLO26n pose. **Output to Insight**: the video passes through untouched, and detections and poses are sent as metadata. |

### Other inputs / other tasks

| app | languages | what it does |
| --- | --- | --- |
| [**usb-camera-yolo26m**](usb-camera-yolo26m/README.md) | C++, Python | **USB/UVC webcam on the board**, not RTSP → YOLO26m → annotated H.264/RTP out. Captures MJPEG at 1080p30 and holds the camera's rate. |
| [**pcb-defect-detection-yolo26n**](pcb-defect-detection-yolo26n/README.md) | Python | **Still images, not video.** A custom-trained YOLO26n on a non-COCO domain, compiled end to end, run over a folder of PCB photos to write annotated JPEGs. |
| [**detection-vlm-assistant**](detection-vlm-assistant/README.md) | Python | Detector always on; a vision-language model captions trigger-gated crops. Shows how to gate an expensive VLM behind a cheap detector on one MLA. |
| [**benchmark**](benchmark/README.md) | Python | **Synthetic input, no video.** Wraps `pyneat.Model.benchmark()` to measure one compiled model package in isolation and emit console metrics and JSON. Use it to get a model's ceiling before building a pipeline around it. |

---

## Running an app

**The app's own README is the authority.** Binary names, config keys and how the config is
passed differ between apps. The shape is always the same:

0. **Clone this repo inside the SDK workspace** (`~/workspace` on the host, which is `/workspace`
   in the SDK container and on the board). `dk` only runs files under `/workspace`.
1. **Get the model.** Archives are git-ignored, so a fresh clone has **no models**. Each app's
   README has the download command. Some come from the SiMa model zoo, some from a direct URL,
   and some you compile with [`model-compilation/`](../model-compilation/README.md).
2. **Fill in the config.** `config/default.conf` ships with placeholders, such as `<rtsp-url>`
   and `<host-ip>`, for the input, the output host and ports. Replace them before the first run.
   `<host-ip>` is the address of the Insight host, as the DevKit can reach it.
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

Unless stated otherwise, the figures in these READMEs use the same input: **1280×720 H.264 at
59.94 fps (60000/1001)**, served over RTSP.

> **Test video:** [1280x720 @ 59.94 fps](https://drive.google.com/file/d/10Bmi_a_6zA_dyRV-GGPuYkqwj3Llbl7O/view?usp=sharing)

Upload it to Insight, serve it as an RTSP source, and put that URL and the Insight host in the
app's `config/default.conf`. **Check the source's frame rate first** (Insight's `/api/media-info`,
see [`appendix.md`](../appendix.md#13-confirm-the-source-before-blaming-the-app)), because it is
the hard ceiling on any fps you can claim.

The exceptions:

- `usb-camera-yolo26m` uses a live UVC webcam.
- `pcb-defect-detection-yolo26n` uses still images.
- `benchmark` uses synthetic input.
- `detection-vlm-assistant` uses a 1280×720 30 fps RTSP source.
- `16stream4model` uses sixteen 1280×720 30 fps Insight catalog clips, which its
  `scripts/start_sources.sh` sets up.

---

## C++ vs Python

**Where both exist, prefer the C++ build for performance.** The two implementations express the
same pipeline, but the Python one carries roughly **+5 ms of host work per frame**: model
push/pull marshalling, the NumPy→Tensor copy for the video sender, and overlay drawing under the
GIL.

That overhead is invisible when there is headroom and decisive when there is not. Measured on
Neat 0.5.0 / SDK 2.1.3.1 against the 59.94 fps test source:

| app | C++ | Python | note |
| --- | ---: | ---: | --- |
| `single-stream-yolo-yolo11` | **59.94 fps** | 59.94 fps | both at the source rate |
| `single-stream-yolo-yolov8m` | **59.94 fps** | 53–57 fps | heaviest model; C++ at the source rate, Python still runs short of budget |
| `quad-stream-quad-model` | **~240 fps** agg | ~75–88 fps agg | four models on one MLA; Python is overlay-bound |

The frame budget at 59.94 fps is 16.7 ms. Light models leave enough slack to absorb Python's
overhead. `yolov8m` spends 13.2 ms in inference alone, so Python's extra cost pushes the total
well past it. Use Python to read and modify, C++ to ship.
