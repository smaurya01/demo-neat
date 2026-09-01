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
| `single-stream-yolo-yolo11` | **59.3 fps** | 59.3 fps | both at the source rate |
| `single-stream-yolo-yolov8m` | **58.9 fps** | 48.1 fps | heaviest model; Python runs out of budget |
| `quad-stream-quad-model` | **~235 fps** agg | ~79 fps agg | four models on one MLA; Python is overlay-bound |

Re-measured on NEAT 0.4.0 / SDK 2.1.3. C++ is unchanged within noise. `yolov8m` Python is
slightly *up* (45.5 → 48.1); `quad-stream-quad-model` Python is down from ~95 to ~79 and the
cause is not resolved — see [NEAT-0.4.0-MIGRATION.md](NEAT-0.4.0-MIGRATION.md).

The frame budget at 60 fps is 16.7 ms. Light models leave enough slack to absorb Python's
overhead; `yolov8m` spends 13.2 ms in inference alone, so Python's extra cost pushes the
total well past it. Use Python to read and modify, C++ to ship.

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
| **multi-model-load-probe** | C++ | Loads four different model graphs in one process off a single RTSP source, to check they coexist on the MLA. A probe, not a product pipeline. |

### Other inputs / other tasks

| app | languages | what it does |
| --- | --- | --- |
| **usb-camera-yolo26m** | C++, Python | **UVC webcam input** (not RTSP) → YOLO26m → annotated H.264/RTP out. MJPEG capture is mandatory; raw YUYV at 1080p is USB-bandwidth-capped at ~5 fps. Verified on a Logitech BRIO at 1920x1080@30: **30.0 fps C++, 28.4 fps Python**. |
| **pcb-defect-detection-yolo26n** | Python | **Still images, not video.** Runs a folder of PCB photos through a custom YOLO26n defect model and writes annotated JPEGs. |
| **detection-vlm-assistant** | Python | Detector always-on; a vision-language model is triggered on selected detections to describe them. Demonstrates gating an expensive VLM behind a cheap detector. Verified with **Qwen3-VL-4B-Instruct-Autoround-a16w4**: detector **46.4 fps** sustained for 3 min while the VLM captioned triggered detections at **~13.8 tok/s, 0.6–0.7 s TTFT**. |
| **benchmark** | Python | Not a pipeline — wraps `pyneat.Model.benchmark()` to measure one compiled model package in isolation. Use it to get a model's ceiling before building a pipeline around it. |

---

## Known rough edges

Recording these honestly so nobody loses time rediscovering them.

- **A non-dropping source sink stalls the decoder permanently on NEAT 0.4.0.**
  `OutputOptions::EveryFrame()` / `every_frame()` leave `drop = false`. The moment the
  consumer drains slower than the source produces, the sink fills, back-pressures the
  admitted hardware decoder, and the stream goes to zero **and stays there** — a burst of
  frames at the expected rate, then nothing. It is not language-specific: it hit C++
  (`single-stream-yolov8n-seg`, `multi-model-load-probe`) as readily as Python.

  The fix is to keep the buffer count and set `drop`:

  ```python
  src_out = pyneat.OutputOptions.every_frame(4)
  src_out.drop = True
  ```

  **Every source sink in this tree now sets `drop`**, including the apps that were passing
  because they drain at the full 60 fps. Setting it there measured free — same fps within
  noise, equal or higher frame counts — so treat `drop = true` on a live source sink as the
  default when writing a new app, not as a fix applied only after something stalls. Full
  detail in [NEAT-0.4.0-MIGRATION.md](NEAT-0.4.0-MIGRATION.md).

- **`single-stream-open-pose` had three defects; all are now fixed, but the output has not been
  visually re-verified.** The app fed the model `[0,1]` when its archive declares an input range
  of `[-0.5, 0.496]`, so every pixel ≥ 127 saturated to int8 127 and half the input domain went
  unused; the peak-suppression radius was ~10x too large (6 heatmap cells ≈ 128 source pixels, so
  nearby people deleted each other's joints); and person-merging had no conflict check, fusing
  adjacent people into one skeleton. Detections went from 2-5 persons / 21-44 keypoints to
  **4-10 / 38-86**, which is the right direction — but confirm the skeletons look correct before
  treating this as a working pose demo. Detail in
  [NEAT-0.4.0-MIGRATION.md](NEAT-0.4.0-MIGRATION.md).

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

- **`usb-camera-yolo26m`: set `camera_device` to your webcam's own node.** The shipped value
  is `/dev/video96`, where a Logitech BRIO enumerated on this DevKit. **Most `/dev/video*`
  nodes on Modalix are not cameras** — they are ISP and codec devices (`modalix-isp-*`), and
  pointing the app at one gets you nothing. Find the real one with:

  ```bash
  for d in /dev/video*; do echo "$d = $(cat /sys/class/video4linux/$(basename $d)/name)"; done
  ```

  A UVC camera usually claims several consecutive nodes; the capture one is the node whose
  `v4l2-ctl -d <dev> --list-formats` reports `MJPG`.

- **`usb-camera-yolo26m` is camera-bound on 0.4.0, not inference-bound.** On NEAT 0.4.0
  YOLO26m infers in ~31 ms, so the MLA ceiling is ~32 fps and the app delivers the BRIO's
  full **30.0 fps (C++) / 28.4 fps (Python)** — its own diagnostic prints
  `bottleneck: THE CAMERA`. (An earlier note here claimed the MLA topped out near 16 fps for
  this model; that is no longer what the board measures.) A faster model will not help; a
  faster camera mode would.

- **A killed `usb-camera-yolo26m` keeps holding the camera, and the next run fails with a
  misleading error.** If the process is killed rather than stopped cleanly (an SSH timeout,
  a dropped connection), it keeps `/dev/videoN` open. The next launch then reports
  `Diagnostic ID: gstreamer.caps_incompatible` / `error_code: misconfig.media_caps`, which
  points at your caps rather than at the busy device. Check and clear before re-running:

  ```bash
  fuser -v /dev/video96          # who holds it
  pkill -f usb_camera_yolo26m    # release it
  ```

- **`boxes=0` on a live camera is almost always the scene, not the pipeline.** COCO has 80
  classes and an empty desk matches none of them. Set `source_override` to a still image with
  a known answer to separate "model wrong" from "nothing to detect" — the app's
  `config/default.conf` documents the exact GStreamer fragment. Doing that here turned
  `boxes=0` into `13 boxes — person(0.95) person(0.91) person(0.88)` from the same binary.

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
placeholders — `<rtsp-url>`, `<host-ip>` — that you must replace
before the first run. See each app's own README for its full option list.
