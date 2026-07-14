# USB Camera YOLO26m Detection

## Introduction

This demo runs a USB (UVC) webcam through the SiMa YOLO26m detection model, draws labeled boxes, and
publishes one annotated H.264/RTP UDP stream. It is the only app in this repo whose input is a
**camera on the board**, not an RTSP stream.

It sustains **1920x1080 @ 30 fps** — the camera's full rate. The MLA is not the bottleneck; the
camera is.

## About Project

- Application: `usb_camera_yolo26m` (C++) / `main.py` (Python) — identical topology, identical results
- Model: `yolo26m-det-bf16-mla_tess-b1.tar.gz` (an INT8 build is also supported — see the appendix)
- Input: USB/UVC camera, MJPEG 1920x1080@30 (`/dev/video16`)
- Output: one UDP/RTP H.264 stream with labeled boxes; optional detection JSON to Neat Insight
- Runtime config: `./config/default.conf`
- Validated on: Modalix DevKit, Neat runtime `0.2.2`, Logitech Brio 100 (`046d:094c`)

Pipeline:

```text
v4l2src (MJPEG 1080p30) ! jpegparse ! jpegdec ! videoconvert (I420->NV12)
  -> appsrc -+-> neatprocesscvu (letterbox NV12->RGB 640x640)
             |     -> neatprocessmla (YOLO26m) -> neatobjectdecode (YoloV26) -> boxes
             +-> box overlay -> H264EncodeSima -> RTP -> UDP
```

MJPEG is decoded on the **CPU** on purpose — the SiMa hardware MJPEG decoder is ~7x slower on this
camera. See [`LEARNING.md`](LEARNING.md) for the measurements.

## Requirements

Run build commands from the Modalix SDK/eLxr environment where the Modalix SDK sysroot and `dk` are
available. Run the final binary on the DevKit with `dk`.

You also need a **UVC camera that offers MJPEG at 1080p**, plugged into the DevKit.

Run the commands below from this app folder:

```bash
cd /path/to/demo-neat/apps/usb-camera-yolo26m
```

Find your camera's capture node on the DevKit before you start:

```bash
v4l2-ctl --list-devices                       # which /dev/videoN is the camera
v4l2-ctl -d /dev/video16 --list-formats-ext   # confirm MJPEG 1920x1080 @ 30
```

A UVC camera registers **two** nodes: a *Video Capture* node (use this one) and a *Metadata Capture*
node. On this DevKit they are `/dev/video16` and `/dev/video17`. Opening the metadata node as a
camera yields nothing. See [Appendix: Camera Setup](#appendix-camera-setup).

## Model Download Command

The yolo26 family is **not** in `sima-cli modelzoo list` — it is published as a direct download.
Run this in the SDK shell:

```bash
mkdir -p ./assets/models
cd ./assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz
```

Expected model path:

```text
./assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
```

Optionally also fetch the INT8 build — faster on the MLA, but its confidence scores are capped at
0.50. Read [Appendix: bf16 vs INT8](#appendix-bf16-vs-int8--which-model) before using it.

```bash
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26m-det-int8-b1.tar.gz
```

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
camera_device=/dev/video16
model_path=./assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
udp_host=<host-ip-that-receives-video>
udp_port=5205
```

`udp_host` is the only value you **must** change — it is the machine you want to watch the stream on.

If your camera is mounted upside-down, set `flip=rotate-180`. COCO models lose real confidence on
inverted scenes.

For a bounded smoke test, set `frames=200` in `./config/default.conf`.

## Config Parameters

`camera_device`: The camera's *Video Capture* node, e.g. `/dev/video16`. Not the metadata node.

`width`, `height`: Capture resolution. `1920x1080` is the Brio 100's maximum.

`fps`: Capture rate. The camera sustains a true 30 fps at 1080p in MJPEG.

`flip`: Correct an upside-down camera mount before inference. One of `none`, `rotate-180`,
`horizontal-flip`, `vertical-flip`.

`model_path`: Model archive loaded by the Neat model node.

`model_width`, `model_height`: Model input size. YOLO26m is compiled for `640x640`; changing this
requires recompiling the model.

`score_threshold`: Detection score threshold used by YOLO26 box decode. Use `0.30` for the bf16
model. **Halve it (~0.15) for the INT8 model**, whose scores are capped at 0.50.

`nms_iou`: NMS IoU threshold used by Neat decode.

`top_k`: Maximum decoded detections per frame.

`num_classes`: Number of classes in the model output (80 for stock COCO).

`udp_host`: Host/IP that receives the annotated UDP/RTP output stream. **Required.**

`udp_port`: UDP/RTP output port used by the H.264 video sender.

`bitrate_kbps`: H.264 output encoder bitrate in kbps.

`overlay`: Draw detection boxes onto the streamed video. Costs ~4 ms/frame on the CPU. Set `false`
if you render overlays downstream instead.

`metadata_host`: Optional host running Neat Insight, to receive detection JSON so Insight can render
overlays itself. Leave empty to disable.

`metadata_port`: UDP port for detection metadata.

`pipeline_mode`: `push` (default) or `graph`. **Keep `push`** — `graph` returns zero detections on
runtime 0.2.2. See [Appendix: Known Limitations](#appendix-known-limitations).

`camera_bridge`: Append the `neatcamerabridge` element. Requires a runtime that ships it; runtime
0.2.2 does not.

`frames`: Number of frames to process. Use `0` to run until interrupted.

`profile_interval`: Seconds between live time-profile lines (default `1.0`). `0` turns the live
profile off; the exit summary still prints. See [Appendix: Reading The Time Profile](#appendix-reading-the-time-profile).

`queue_depth`: Pipeline queue depth. `3` keeps latency low.

`print_backend`: Print generated backend pipelines when set to `true`.

`verbose_planner`: Dump the model's MPK contract and the planner's routing/fusion decisions. Verbose,
but it is the tool that explains what a model package is actually doing.

`source_override`: Replace the camera with any GStreamer fragment that ends producing NV12 at
`width x height`. Used to validate the model against a known image — see
[Appendix: Verifying It Works](#appendix-verifying-it-works).

## How To Build

Run from the SDK shell:

```bash
cmake -S . \
  -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
cmake --build ./build --parallel
```

## How To Run

Run on the DevKit from the SDK shell. The C++ demo takes the config path as an optional **positional**
argument; it does not use a `--config` flag. With no argument it reads `./config/default.conf`.

```bash
dk ./build/usb_camera_yolo26m
```

Or point it at a specific config:

```bash
dk ./build/usb_camera_yolo26m ./config/default.conf
```

For a bounded smoke test, set `frames=200` in `./config/default.conf`, then run the same command.

> **The first start takes ~48 seconds.** The YOLO26m package is unpacked and its MLA ELF copied into
> `/tmp/simaai/` before the pipeline runs. Nothing is streamed until the app prints its first
> `frame=` line — do not conclude the video output is broken before then.

## How To Run With Python

Run the Python version on the DevKit from the SDK shell. It also takes the config path as a
**positional** argument, not `--config`:

```bash
dk ./main.py ./config/default.conf
```

Bounded smoke test: set `frames=200` in the config, then run the same command.

Expected output — **identical in C++ and Python**:

```text
frame=776 fps=30.1 boxes=1 ms(capture=2.3 infer=29.4 overlay=1.5 encode=0.1 total=33.3) person(0.93)
frame=806 fps=30.1 boxes=2 ms(capture=2.3 infer=29.4 overlay=1.5 encode=0.1 total=33.3) person(0.93) laptop(0.31)
^C
── time profile ──────────────────────────────
stage        mean ms    p95 ms
capture         2.31      3.10
infer          29.41     31.02
overlay         1.52      2.01
encode          0.09      0.14
total          33.33     35.20

frames 6207   elapsed 206.2s   steady-state 30.10 fps
bottleneck: THE CAMERA. Inference takes 29.4 ms, so the MLA could sustain ~34.0 fps;
you are getting 30.1. A smaller/faster model will not help.
```

See [Appendix: Reading The Time Profile](#appendix-reading-the-time-profile).

## How To See The Output

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

Run this on the host machine receiving UDP. Use the same port configured by `udp_port`.

```bash
gst-launch-1.0 -v udpsrc port=5205 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtpjitterbuffer latency=100 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Expected output: live video with YOLO26m labeled detection boxes.

> **Watch it on a desktop, not on the DevKit.** The board has no `avdec_h264`, and its `openh264dec`
> cannot decode an RTP-depayed H.264 stream — it fails a synthetic control stream the same way, so
> this is a board decoder limitation, not a problem with the output. The stream itself is verified
> well-formed (SPS + PPS + IDR keyframes, ~3.1 Mbps at 30 fps).

## Appendix: Camera Setup

### Identify the right node

The `video0*`–`video3*` nodes with `m2m` / `meta` / `out` / `raw` suffixes are SiMa's own ISP and
codec devices (`arm-isp-*`), unrelated to USB. A USB camera appears as plain numeric nodes:

```text
Brio 100 (usb-0003:01:00.0-3.1):
        /dev/video16      <- Video Capture     (use this)
        /dev/video17      <- Metadata Capture  (UVC headers, no image data)
```

### MJPEG is mandatory at 1080p

| Format | 1920x1080 | 1280x720 | 640x480 |
| --- | --- | --- | --- |
| **MJPG** | 30 fps | 30 fps | 30 fps |
| **YUYV** (raw) | **5 fps** | 5 fps | 30 fps |

Raw YUYV collapses above 640x360 — uncompressed 4:2:2 at 1080p does not fit in USB 2.0 bandwidth
(the link negotiates at 480 Mbps). Measured: 4.96 fps, matching the advertised cap exactly.

**The trap:** if you ask GStreamer for `video/x-raw` at 1080p, `v4l2src` silently hands you the 5 fps
YUYV mode and the pipeline looks mysteriously broken. This app pins `image/jpeg` in its caps.

## Appendix: bf16 vs INT8 — which model

Both packages can sit in `./assets/models/`; the `model_path` line switches between them.

| | **bf16** (default) | **INT8** |
| --- | --- | --- |
| Package | `yolo26m-det-bf16-mla_tess-b1` | `yolo26m-det-int8-b1` |
| Size | 66 MB | 35 MB |
| Inference | 30.3 ms | **15.9 ms** (~2x faster) |
| End-to-end | 30.9 fps | 32.9 fps |
| Boxes / classes | correct | correct — identical to bf16 |
| Confidence scores | **0.0 – 1.0** | **capped at 0.50** |
| Stages | `CVU preproc -> MLA` | `CVU quanttess -> MLA -> CVU detess+dequant` |

**Use bf16 unless you have a specific reason not to.** It is the only one with usable confidence
scores, and it is *not* the slower choice in practice: this pipeline is **camera-limited at 30 fps**,
and bf16 already infers in 30 ms (~33 fps of headroom). INT8's 2x faster inference buys essentially
**zero end-to-end fps** — it just leaves the MLA idle longer.

**Use INT8** when you want that MLA headroom for something else — a second model, more camera
streams, a faster sensor — and can live with the score limitation.

### The INT8 score ceiling

This is a property of the published package, not of this app or of Neat.

Detections, classes and box geometry are correct and match bf16 exactly. But **every confidence score
is capped at 0.50**: objects bf16 scores at 0.88–0.90 come back as exactly 0.50, and setting
`score_threshold=0.51` returns nothing at all, ever.

The cause is in the package's own quantization. Its class-score heads carry `dq_zp = [127, 127, 108]`
— the zero-point sits at the *top* of the int8 range. Since `q <= 127`, the dequantized class logit
`(q - zp)` can never be positive, so `sigmoid(0) = 0.50` is a hard ceiling.

Neat handles the INT8 path correctly — `verbose_planner=true` shows it fusing the required stages
(`post_fusion=user_boxdecode(cast+detess+dequant)->boxdecode`). The math downstream is simply working
from a class head that cannot express a positive logit.

**If you switch to INT8, roughly halve `score_threshold`** (0.30 -> 0.15). Its scores are compressed
into 0.0–0.50 and are *not* comparable with bf16's.

There is **no** tessellated INT8 build — `yolo26m-det-int8-mla_tess-b1` returns HTTP 403.

## Appendix: Reading The Time Profile

Both the C++ and the Python version print the same profile. `profile_interval` controls the cadence
(default `1.0` s; `0` = live profile off, exit summary still prints).

### The live line

```text
frame=776 fps=30.1 boxes=2 ms(capture=2.3 infer=29.4 overlay=1.5 encode=0.1 total=33.3) person(0.93)
```

| Stage | What it measures | Runs on |
| --- | --- | --- |
| `capture` | Blocking wait for the next camera frame. **High here = the camera is pacing you** (which is the healthy state — see below). | — |
| `infer` | CVU letterbox/colour-convert + MLA + EV74 box decode. The only stage on the accelerator. | CVU / MLA / EV74 |
| `overlay` | NV12 copy + drawing the boxes. | CPU |
| `encode` | Pushing the annotated frame into the H.264 encoder graph. Async — this is the *push*, not the encode itself. | HW encoder |
| `total` | The whole per-frame loop. | |

**Every number is the mean over that window, not since start.** A cumulative mean drifts so slowly
that by frame 6000 a bad minute is invisible. `fps` is likewise the rate over that window, so a stall
shows up on the next line instead of being averaged away.

### The exit summary

```text
── time profile ──────────────────────────────
stage        mean ms    p95 ms
capture         2.31      3.10
infer          29.41     31.02
overlay         1.52      2.01
encode          0.09      0.14
total          33.33     35.20

frames 6207   elapsed 206.2s   steady-state 30.10 fps
bottleneck: THE CAMERA. Inference takes 29.4 ms, so the MLA could sustain ~34.0 fps;
you are getting 30.1. A smaller/faster model will not help.
```

**p95 is the number a mean will never tell you.** A stage that is fine on average but occasionally
stalls shows up here and nowhere else. (p95 is computed over the most recent 20 000 frames — about
11 minutes at 30 fps — so memory stays flat on an open-ended run.)

### The bottleneck line

`infer` is the only stage on the accelerator, so `1000 / infer_ms` is the MLA's ceiling. The app
compares that against what you are actually delivering and tells you which one is the constraint:

- **`bottleneck: THE CAMERA`** — the MLA has headroom and the sensor is pacing the pipeline. This is
  the expected state here: inference is ~29 ms while the camera delivers a frame every 33 ms.
  **A smaller or faster model will not gain you a single frame.** (This is exactly why the INT8
  model — 2× faster inference — buys ~0 end-to-end fps. See the bf16-vs-INT8 appendix.)
- **`bottleneck: INFERENCE`** — the MLA is the constraint. Now a smaller model *would* help.

### Tuning

- `overlay=false` frees ~1.5 ms/frame of CPU. Point `metadata_host` at Neat Insight and let it render
  the boxes instead.
- `profile_interval=0` silences the live lines for a clean run; the exit summary still prints.
- Python measures within ~1 fps of C++ and returns identical boxes — the language choice is free.

## Appendix: Verifying It Works

`source_override` replaces the camera with a still image of known content, through the exact same
graph. This separates *"is the model right"* from *"is the camera path right"* in one run.

```bash
cp /workspace/calibration_images/000000000139.jpg /tmp/coco139.jpg
```

Then in `./config/default.conf`:

```text
source_override=multifilesrc location=/tmp/coco139.jpg loop=true caps=image/jpeg,framerate=30/1 ! jpegparse ! jpegdec ! videoscale ! videoconvert ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! queue
```

COCO image `000000000139` is a living room. Both C++ and Python return the same 15 boxes:

```text
boxes=15 person(0.88) chair(0.90) chair(0.89) chair(0.82) potted plant(0.57)
```

**If you get high-confidence boxes here but few from the camera, the pipeline is fine** — your scene
is just hard (dim, or no COCO objects in view). Run this before debugging weak detections.

## Appendix: Known Limitations

- **`pipeline_mode=graph` returns zero detections** on runtime 0.2.2. The zero-copy branch topology
  builds and runs at full speed, but the CVU reads system-memory buffers as black frames. The fix
  (`neatcamerabridge`) exists in the `core` source tree but is **not** in the shipped library
  (`strings /usr/lib/libsima_neat.so.2.1.2 | grep -c neatcamerabridge` -> `0`). Keep
  `pipeline_mode=push`.
- **The stream cannot be viewed on the DevKit itself.** No `avdec_h264`; `openh264dec` fails on
  RTP-depayed H.264 (including a synthetic control). View on a desktop.
- **~48 s startup** while the model is unpacked and staged into `/tmp/simaai/`. This happens per run.
- **`jpegparse` logs `Failed to parse app0 segment`** repeatedly. Harmless — the Brio 100 writes a
  non-standard JPEG APP0 header. Decoding is unaffected.
- **nanobind leak warnings** print at Python exit. Cosmetic, from the pyneat bindings.

## Appendix: Learnings

[`LEARNING.md`](LEARNING.md) records how this app was built: what was tried, what the numbers said,
and what did not work — including why the hardware MJPEG decoder was rejected (4 fps vs 27 fps for
CPU `jpegdec`), why the zero-copy graph silently detects nothing, and the measurement mistakes that
cost the most time.
