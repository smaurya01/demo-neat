# `apps/` — SiMa Neat demo applications

Runnable reference applications for the Modalix DevKit, built on the SiMa Neat Library.
Each app is self-contained: its own `main.cpp` and/or `main.py`, `config/default.conf`,
`assets/models/`, and a README with build and run instructions.

For the API surface these apps use — `Graph`/`Run`, nodes, node groups, `Model` options,
enums, and the C++ ↔ Python naming map — see **[NEAT-API-REFERENCE.md](NEAT-API-REFERENCE.md)**.

---

## Test source

Unless stated otherwise, every application here was exercised against the same input:

**1280×720 H.264, 60 fps**, served over RTSP.

> **Test video:** [1280x720@60FPS](https://drive.google.com/file/d/10Bmi_a_6zA_dyRV-GGPuYkqwj3Llbl7O/view?usp=sharing)

Point an app at it by setting the RTSP URL and UDP output host in that app's
`config/default.conf`. Two apps are exceptions and are marked in the table:
`usb-camera-yolo26m` (live UVC webcam) and `pcb-defect-detection-yolo26n` (still images).

## C++ vs Python

**Where both exist, prefer the C++ build for performance.** The two implementations express
the same pipeline, but the Python one carries roughly **+5 ms of host work per frame**
(model push/pull marshalling, the NumPy→Tensor copy for the video sender, and overlay
drawing under the GIL).

That overhead is invisible when there is headroom and decisive when there is not. Measured
on this DevKit against the 720p60 source:

| app | C++ | Python | note |
| --- | ---: | ---: | --- |
| `single-stream-yolo-yolo11` | **60.4 fps** | 59.7 fps | both at the source rate |
| `single-stream-yolo-yolov8m` | **60.1 fps** | 45.5 fps | heaviest model; Python runs out of budget |
| `quad-stream-quad-model` | **~235 fps** agg | ~95 fps agg | four models on one MLA; Python is overlay-bound |

The frame budget at 60 fps is 16.7 ms. Light models leave enough slack to absorb Python's
overhead; `yolov8m` spends 13.2 ms in inference alone, so Python's extra cost pushes the
total to 22.1 ms — exactly the 45 fps measured. Use Python to read and modify, C++ to ship.

---

## Applications

### Single stream, one model

| app | languages | what it does |
| --- | --- | --- |
| **single-stream-yolo-yolo11** | C++, Python | RTSP in → YOLO11 detection → annotated H.264/RTP out over UDP. The cleanest starting point for reading the API. |
| **single-stream-yolo-yolov8m** | C++, Python | Same shape with YOLOv8m. The heaviest single-stream model here, and the one that shows the C++/Python gap most clearly. |
| **single-stream-yolo26n** | C++, Python | Same shape with YOLO26n (`bf16` tessellated archive), which uses the `YoloV26` on-device box-decode family. |
| **single-stream-yolov8n-seg** | C++, Python | Instance segmentation. Decodes mask coefficients plus the prototype tensor and blends per-instance masks onto the NV12 frame. |
| **single-stream-yolo-insight** | C++, Python | Detection with output routed to [Insight](https://developer.sima.ai/software/tools/insight/): encoded video on one UDP port, detection metadata on another, drawn by the Insight viewer instead of burned in. |
| **single-stream-open-pose** | C++ | OpenPose multi-person keypoints — heatmap peaks, PAF limb matching, skeleton assembly. ⚠️ **See "Known rough edges" below.** |

### Multi-stream / multi-model

| app | languages | what it does |
| --- | --- | --- |
| **multi-stream-yolo-yolo11** | Python | Two RTSP streams through **one shared** YOLO11 model stage, each with its own annotated UDP output. Sustains **~119 fps aggregate (≈60 fps per stream)** with overlay on. |
| **quad-stream-quad-model** | C++, Python | Four RTSP streams, four *different* models (detection, segmentation, pose, YOLOX) on one MLA. **C++ delivers ~235 fps aggregate** with overlay; **Python ~95 fps**, stable over a 3-minute run. Both are usable — pick C++ for throughput, Python to read and modify. |
| **high-density-multi-stream-insight** | C++ | Many streams (16/24/48 profiles) sharing one detector, publishing video plus metadata to Insight. The reference for how Neat scales stream count. |
| **multi-model-load-probe** | C++ | Loads four different model graphs in one process off a single RTSP source, to check they coexist on the MLA. A probe, not a product pipeline. |

### Other inputs / other tasks

| app | languages | what it does |
| --- | --- | --- |
| **usb-camera-yolo26m** | C++, Python | **UVC webcam input** (not RTSP) → YOLO26m → annotated H.264/RTP out. MJPEG capture is mandatory; raw YUYV at 1080p is USB-bandwidth-capped at ~5 fps. |
| **pcb-defect-detection-yolo26n** | Python | **Still images, not video.** Runs a folder of PCB photos through a custom YOLO26n defect model and writes annotated JPEGs. |
| **detection-vlm-assistant** | Python | Detector always-on; a vision-language model is triggered on selected detections to describe them. Demonstrates gating an expensive VLM behind a cheap detector. |
| **benchmark** | Python | Not a pipeline — wraps `pyneat.Model.benchmark()` to measure one compiled model package in isolation. Use it to get a model's ceiling before building a pipeline around it. |

---

## Known rough edges

Recording these honestly so nobody loses time rediscovering them.

- **`single-stream-open-pose` is not in good shape.** It builds and runs, but the **pose
  output is not reliable** — skeleton assembly produces poor/incorrect results. Treat it as
  a work-in-progress reference for the heatmap/PAF decode path, not as a working pose demo.
  Do not use it as a correctness baseline.

- **`quad-stream-quad-model`: both builds work; C++ is ~2.5x faster.** `main.py` now sustains
  **~95 fps aggregate** over a 3-minute run against the C++ twin's ~235. Two fixes got it
  there, and both are worth knowing before you write a Python Neat app:
  - **The source sink must drop.** `OutputOptions.every_frame()` leaves `drop=False`, and a
    non-dropping sink back-pressures the *admitted* hardware decoder into a permanent stall
    the moment a Python consumer falls behind — every stream went to 0 fps and stayed there.
    `main.cpp` survives the identical setting only because it drains at 61 fps/stream.
  - **Batch the OpenCV calls in the overlay.** Per-object `cv2.line`/slice-write drawing cost
    3.5x more than the batched equivalent on the pose stream.

  The remaining gap to C++ is entirely the host-side overlay (GIL-serialised drawing);
  `infer` is at parity (24 ms vs 21.7 ms).

- **`usb-camera-yolo26m` is inference-bound, not camera-bound.** YOLO26m is a large model;
  the app's own diagnostic reports the MLA topping out near 16 fps for it. If you see
  `boxes=0` on a live camera, check the scene before the pipeline — COCO has 80 classes and
  an empty desk matches none of them. Set `source_override` to a still image with a known
  answer to separate "model wrong" from "nothing to detect"; the app's `config/default.conf`
  documents the exact GStreamer fragment to use.

- **RTP output cannot be viewed on the DevKit.** There is no `avdec_h264` on the board. Run
  the `gst-launch-1.0` receiver each app prints on your desktop, not over SSH.

---

## Building and running

C++ apps use CMake against the installed Neat toolchain:

```bash
cd <app>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
cmake --build build -j"$(nproc)"
```

Python apps need no build step and use the board's pyneat interpreter
(`/home/sima/pyneat/bin/python`).

Every app reads `config/default.conf` from its own directory. The shipped configs contain
placeholders — `<rtsp-server-ip>`, `<host-ip-that-receives-video>` — that you must replace
before the first run. See each app's own README for its full option list.
