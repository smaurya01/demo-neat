# LEARNING — building a USB-camera YOLO26 pipeline on Modalix

How this app was built, in the order it actually happened: what was tried, what the
numbers said, what worked, and what didn't. Written so someone can rebuild it from
scratch — or, more usefully, so the next person doesn't burn a day on the same three
dead ends.

Environment: Modalix DevKit, Neat runtime **0.2.2** (`libsima_neat.so.2.1.2`),
SDK 2.1.2, Logitech Brio 100 USB camera.

## Table of Contents

- [The short version](#the-short-version)
- [Step 1 — Identify the camera](#step-1--identify-the-camera)
  - [Resolution and frame rate — the format decides everything](#resolution-and-frame-rate--the-format-decides-everything)
  - [A measurement lesson](#a-measurement-lesson)
- [Step 2 — Find the model (it is not in the model zoo)](#step-2--find-the-model-it-is-not-in-the-model-zoo)
- [Step 3 — Decode MJPEG: the hardware decoder loses, badly](#step-3--decode-mjpeg-the-hardware-decoder-loses-badly)
  - [What worked](#what-worked)
- [Step 4 — The zero-copy graph that returns zero detections](#step-4--the-zero-copy-graph-that-returns-zero-detections)
  - [Two bugs found on the way](#two-bugs-found-on-the-way)
  - [The root cause](#the-root-cause)
- [Step 5 — The push path (what actually ships)](#step-5--the-push-path-what-actually-ships)
- [Step 6 — "It only detects a TV at 0.33" — bug, or just a boring room?](#step-6--it-only-detects-a-tv-at-033--bug-or-just-a-boring-room)
- [Step 7 — "The encoder emits no video" (it did; I was measuring the wrong 48 seconds)](#step-7--the-encoder-emits-no-video-it-did-i-was-measuring-the-wrong-48-seconds)
- [Step 8 — Where the time actually goes](#step-8--where-the-time-actually-goes)
- [Step 9 — int8: 2× faster inference, worth nothing, and it breaks the scores](#step-9--int8-2-faster-inference-worth-nothing-and-it-breaks-the-scores)
  - [Reading the planner instead of guessing](#reading-the-planner-instead-of-guessing)
  - [The actual cause is in the package's quantization](#the-actual-cause-is-in-the-packages-quantization)
  - [The optimisation was pointless anyway](#the-optimisation-was-pointless-anyway)
- [Replicating from scratch](#replicating-from-scratch)
- [Things that did not work — quick index](#things-that-did-not-work--quick-index)

---

## The short version

Three things cost the most time, and none were obvious up front:

1. **The SiMa hardware MJPEG decoder is 7× slower than the CPU one** for this camera.
   Measured: 4 fps vs 27 fps. The "obvious" hardware-accelerated design is the wrong one.
2. **`neatcamerabridge` — the element that makes a zero-copy camera graph work — does not
   exist in the shipped runtime.** It is in the `core` source tree, so it reads as
   available, but `libsima_neat.so.2.1.2` does not contain it. Without it the CVU silently
   reads black frames and you get *zero detections at full frame rate* — a failure mode
   that looks like a model bug.
3. **`sima-cli modelzoo list` does not contain yolo26.** The model exists, but is published
   as a direct download outside the zoo index.

Final working design: CPU MJPEG decode → appsrc push → CVU → MLA → EV74 box decode.
**1920×1080 @ 30.4 fps**, camera-limited.

---

## Step 1 — Identify the camera

Two new nodes appeared after plugging the camera in:

```
/dev/video16   /dev/video17
```

The `video0*`–`video3*` nodes with `m2m`/`meta`/`out`/`raw` suffixes are SiMa's own ISP
and codec devices (`arm-isp-*` on the `isp-v4l2-*` platform bus), unrelated to USB.

```bash
v4l2-ctl --list-devices
```

```
Brio 100 (usb-0003:01:00.0-3.1):
        /dev/video16
        /dev/video17
```

**A UVC camera registers two nodes, not one.** `video16` is *Video Capture*; `video17` is
*Metadata Capture* — per-frame UVC header info, not image data. Opening `video17` as a
camera gives you nothing. Confirm with `v4l2-ctl -d /dev/videoN --info` and read the
`Device Caps` block.

### Resolution and frame rate — the format decides everything

```bash
v4l2-ctl -d /dev/video16 --list-formats-ext
```

| Format | 1920×1080 | 1280×720 | 640×480 |
| --- | --- | --- | --- |
| **MJPG** | 30 fps | 30 fps | 30 fps |
| **YUYV** (raw) | **5 fps** | 5 fps | 30 fps |

Raw YUYV collapses to 5 fps above 640×360 — uncompressed 4:2:2 at 1080p does not fit in
USB 2.0 bandwidth (the link negotiates at 480 Mbps). Streaming raw confirmed it: **4.96
fps**, matching the advertised cap exactly.

**So MJPEG is mandatory at 1080p.** This is the single most important constraint in the
whole design, and it is what forces the decode question in Step 3.

The trap: if you ask GStreamer for `video/x-raw` at 1080p, `v4l2src` will happily give you
the 5 fps YUYV mode and the pipeline will look mysteriously broken. Always pin
`image/jpeg` in the caps.

### A measurement lesson

An early short run measured 25.8 fps and I nearly designed around "the camera can only do
~26". A longer run with `fpsdisplaysink` showed the truth:

```
current: 30.08   current: 29.78   current: 29.98   current: 29.85
```

The camera does a **true 30 fps**. The first number was diluted by pipeline startup and
the sensor's auto-exposure ramp. *Measure steady state, not the first two seconds.*

---

## Step 2 — Find the model (it is not in the model zoo)

`sima-cli modelzoo list` is interactive and paginated; scripting it needs a pty. Easier to
read the index directly — the URL pattern is in `sima_cli/model_zoo/model.py`:

```
https://docs.sima.ai/pkg_downloads/SDK<version>/model_zoo/metadata_gen2.json
```

The GA index (2.1.2, 136 models) contains `yolo_11*`, `yolo_v8*`, `yolo_v9*`, `yolo_v3*`
— **and no yolo26 at all**.

But Neat clearly supports it: `core/include/pipeline/BoxDecodeType.h` has

```cpp
YoloV26 = 17,     ///< YOLO26 detection (raw l/t/r/b distance heads).
```

The resolution is in the reference app's README (`apps/examples/object-detection/
single-stream-object-detector`): yolo26 is published as a **direct download, outside the
zoo index**:

```bash
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz
```

**Lesson: "not in the model zoo" does not mean "not available".** Check the example apps'
READMEs before concluding a model doesn't exist or needs compiling from scratch.

The package's `0_preproc.json` confirms what to configure: 640×640 output, NV12→RGB,
`aspect_ratio: true` + `padding_type: CENTER` (i.e. letterbox), mean 0 / stddev 1 (the
`COCO_YOLO` preset).

---

## Step 3 — Decode MJPEG: the hardware decoder loses, badly

Neat has **no V4L2 source node**. `CameraInput` is hard-wired to `libcamerasrc` (MIPI
only) and takes a libcamera name, not a device path. So the camera has to come in through
the `Custom()` escape hatch as a raw GStreamer fragment.

That left the decode question. Neat's `SimaDecode` has an explicit `MJPEG` mode backed by
the `neatdecoder` hardware element, and `nodes/groups/HttpMjpegDecodedInput.h` is a
ready-made MJPEG→HW-decode template. This looked like the obvious right answer: decode on
silicon, keep the CPU free.

It negotiated perfectly:

```
neatdecoder0.GstPad:src: caps = video/x-raw, format=(string)NV12, width=1920, height=1080, framerate=30/1
```

Then I measured throughput, and it fell over:

| Decode path | fps @ 1080p |
| --- | --- |
| Camera only, no decode | 28 |
| **CPU `jpegdec`** | **27** |
| **HW `neatdecoder`** | **4** |

I did not accept that at face value — `dec-type` had not been set, so it might have been
running some default path. I retried it properly configured, across every knob the element
exposes:

| Configuration | fps |
| --- | --- |
| `dec-type=mjpeg num-buffers=7 next-element=CVU` | 4 |
| `dec-type=mjpeg num-buffers=10 next-element=APU` | 4 |
| `dec-type=jpeg num-buffers=7 next-element=CVU` | 4 |
| CPU `jpegdec` (baseline) | 27 |

**~4 fps regardless.** The hardware MJPEG decoder is simply not viable for this camera's
JPEG stream (4:2:2 sampling, non-standard APP0 header). I did not chase the root cause
further — CPU decode meets the camera's full rate, so the hardware path buys nothing even
if it could be fixed.

### What worked

```
v4l2src io-mode=mmap ! image/jpeg,1920x1080,30/1
  ! queue leaky=downstream ! jpegparse ! jpegdec ! videoconvert ! NV12
```

- `io-mode=mmap` — zero-copy DMA from the UVC driver. `io-mode=rw` memcpys every frame.
- `jpegdec` outputs **I420**; the CVU preprocessor and the H.264 encoder both want NV12,
  so one `videoconvert` feeds both. Measured 26–27 fps — free, since the camera caps at 30.
- `queue leaky=downstream` — drop stale frames instead of back-pressuring the camera.

`jpegparse` spams `Failed to parse app0 segment` on every frame. It is harmless: the Brio
writes a non-standard APP0 header, and decoding is unaffected.

**Lesson: benchmark the hardware path before designing around it.** "Hardware accelerated"
is a claim, not a measurement.

---

## Step 4 — The zero-copy graph that returns zero detections

With decode settled, the natural topology is a single graph with a fan-out — the frame
never touches the CPU:

```
Custom(camera) → Branch ─┬→ VideoSender (H264 → RTP → UDP)
                         └→ Model(yolo26m) → Output("detections")
```

This **built and ran at full speed**. The backend plan was exactly right:

```
segment 2  [CapsRaw, VideoConvert, H264EncodeSima, H264Parse, H264Packetize, UdpOutput]
segment 3  [Preproc, ModelFragment, SimaBoxDecode, Output]
Configuring for the decoding type: 8:yolo26
```

And it detected **nothing**. `boxes=0`, every frame, at 30+ fps.

This is the nastiest failure mode in the whole exercise, because *nothing errors*. A
pipeline running at full frame rate and emitting zero boxes reads as a model or
threshold problem, and you will go looking in the wrong place.

### Two bugs found on the way

Worth recording, because both produce confusing errors:

**A trailing caps string does not parse.** Ending a `Custom()` fragment with
`! video/x-raw,format=NV12,...` fails with:

```
gst_parse_launch failed: no element "video"
```

`gst_parse_launch` parses a trailing `video/x-raw,...` as an *element name*. Terminate the
fragment on a real element (`! queue`) and the caps become a capsfilter as intended.

**`add()` + `connect()` duplicates the source.** `graph.add(source)` followed by
`graph.connect(source, branch)` registers the node twice — the plan showed two
`CustomNode` segments, i.e. two `v4l2src` elements fighting over one camera. `connect()`
alone is sufficient.

### The root cause

`jpegdec`/`videoconvert` emit **system memory**. The CVU cannot read system memory — it
needs SiMa DMA memory with a `GstSimaMeta` stamp. The element that bridges the two is
`neatcamerabridge`, and `CameraInput.cpp` appends it for exactly this reason.

Adding it failed:

```
gst_parse_launch failed: no element "neatcamerabridge"
```

Not a plugin-path problem — it is a *private* element that `GstInit.cpp` registers inside
the Neat process, and the registration is unconditional. So it should have been there.

Checking the actual shipped library settles it:

```bash
$ strings /usr/lib/libsima_neat.so.2.1.2 | grep -c neatcamerabridge
0
$ strings /usr/lib/libsima_neat.so.2.1.2 | grep -c latest_by_stream
1
```

`latest_by_stream` is registered in the *same block* of `GstInit.cpp` and is present;
`neatcamerabridge` is not. **The `/workspace/core` source checkout is newer than the
installed runtime.** The camera bridge was added after the 0.2.2 release.

**Lesson: the source tree is not the runtime.** When a private element is missing, check
the shipped `.so` with `strings` before assuming a configuration mistake. And a
neighbouring symbol from the same init block is a good control.

---

## Step 5 — The push path (what actually ships)

The workaround is the structure the existing reference apps use, and it is proven on this
runtime: pull the frame, **push it through `appsrc`** into the model graph. The appsrc
push is what lands the buffer in SiMa DMA memory, gated by:

```cpp
setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);
```

```
source graph:  Custom(camera)      → Output("frame")
model  graph:  Input("image")      → Model(yolo26m) → Output("detections")
udp    graph:  Input("video")      → VideoSender (H264 → RTP → UDP)
```

It costs one NV12 copy per frame. Measured cost: **~0 ms** — it disappears into the
camera's 33 ms frame interval. The theoretical purity of zero-copy bought nothing here.

First run, and the TV on the wall appeared:

```
frame=53 fps=31.9 boxes=1 tv(0.33)
```

Both modes are kept in the app (`pipeline_mode=push|graph`), so `graph` mode can be
switched on with `camera_bridge=true` once the runtime ships the element.

---

## Step 6 — "It only detects a TV at 0.33" — bug, or just a boring room?

Detections were sparse and low-confidence. At `score_threshold=0.05` the model was seeing
*something* sensible — `tv(0.33)`, `tv(0.27)`, some low-confidence `bottle` — but nothing
convincing. Was preprocessing subtly wrong (letterbox? colour? normalization?), or was the
scene just bad?

Guessing here is expensive. **Test against known ground truth instead.** I added a
`source_override` config key that swaps the camera for any NV12-producing fragment, and
looped a COCO validation image with a known answer through the *exact same graph*:

```ini
source_override=multifilesrc location=/tmp/coco139.jpg loop=true caps=image/jpeg,framerate=30/1 ! jpegparse ! jpegdec ! videoscale ! videoconvert ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! queue
```

COCO `000000000139` is a living room — person, chairs, TV, potted plants:

```
boxes=15 person(0.88) chair(0.90) chair(0.89) chair(0.82) potted plant(0.57)
```

**The pipeline is correct.** Preprocessing, letterbox, normalization and YOLO26 decode all
work; C++ and Python agree box-for-box. The camera scene was simply a dim, near-empty room
whose only COCO object was a dark TV on a white wall.

A hypothesis I tested and rejected on the way: the camera is mounted upside-down, and COCO
models do degrade on inverted scenes. Adding `videoflip method=rotate-180` moved `tv` from
0.12 to 0.25 — real, but not the main effect. The `flip` option stayed in as a genuinely
useful feature for inverted mounts; it was not the answer.

**Lesson: when detection quality is in doubt, feed the pipeline something whose answer you
already know.** It splits "is the model right" from "is the input path right" in one run,
and it is far cheaper than reasoning about it.

---

## Step 7 — "The encoder emits no video" (it did; I was measuring the wrong 48 seconds)

With detection working, the H.264 output looked dead. A receiver on the video port got
**zero RTP packets**, run after run — while `push("video")` returned success every time.

I chased this hard, and produced three confident, wrong diagnoses:

1. *"`build(TensorList{seed}, opts)` runs the graph once and closes the input."* Plausible.
   Wrong — an isolated VideoSender graph emits RTP fine either way.
2. *"The model graph starves the encoder."* A bisect seemed to confirm it: `camera` emitted
   RTP, `camera+model` did not. Wrong — the `camera+model` binary had not finished
   **loading the model** before the receiver timed out. It never reached the push loop.
3. *"`OutputMemory::ZeroCopy` breaks the encoder"* and *"graph build order matters."* Both
   wrong. Both reverted.

The actual cause: **the yolo26m model takes ~48 seconds to load on first run** (it unpacks
and copies the MLA ELF into `/tmp/simaai/...`). Every one of my receiver windows opened and
closed inside that load. The encoder was not broken; it had not started yet.

Once the receiver is started only after the app prints its first `frame=` line, the stream
is there and it is well-formed:

```
udpsrc -> fakesink            : 100 RTP packets received
udpsrc -> rtph264depay        : 4,691,531 bytes in 12 s (~3.1 Mbps, matches bitrate=4000)
NAL counts                    : {1: 596 slices, 5: 8 IDR, 6: 302 SEI, 7: 12 SPS, 8: 12 PPS, 9: 306 AUD}
```

SPS, PPS and IDR keyframes all present — a valid H.264 elementary stream at ~30 fps.

Two traps worth naming, because both cost time on top of the first one:

- **`udpsrc num-buffers=N` truncates the depayloader.** `rtph264depay` buffers until it has
  a complete access unit; cutting the source off after N packets can yield *zero bytes* from
  a perfectly good stream. Bound the receiver by **time**, not packet count.
- **`h264parse ! filesink` writes AVC, not Annex-B.** The captured file had no start codes
  at all, so nothing could parse it back. Pin
  `video/x-h264,stream-format=byte-stream,alignment=au` before `filesink`.

Finally: `openh264dec` **cannot decode this stream on the DevKit** — but it fails a
synthetic `videotestsrc → neatencoder → RTP` control exactly the same way, so it is a
board-side decoder limitation, not a defect in the app's output. The board has no
`avdec_h264`. **View the stream on a desktop**, where `avdec_h264` handles it.

**Lesson: before concluding a stage is broken, prove your measurement window overlaps the
thing you are measuring.** A 48-second model load invalidated three experiments in a row,
and each one produced a plausible story that sent me editing correct code.

---

## Step 8 — Where the time actually goes

900 frames, 1920×1080, overlay on:

```
Processed 900 frames, steady-state 30.39 fps
Avg ms: capture=0.02  infer=28.74  overlay=4.39  encode=0.09
```

| Stage | ms | Runs on |
| --- | --- | --- |
| Inference (CVU + MLA + EV74 decode) | 28.74 | on-chip |
| Box overlay | 4.39 | CPU |
| H.264 encode push | 0.09 | HW encoder |
| Camera capture | 0.02 | blocks on camera |

Inference at 28.7 ms is ~35 fps of headroom. The camera delivers 30. **The camera is the
bottleneck, not the SoC** — so a smaller model (yolo26s/n) buys nothing here unless you
also get a faster camera. That is a good place to end up: the accelerator is not the
constraint.

Python measures 31.5 fps and returns identical boxes, so the language choice is free.

---

---

## Step 9 — int8: 2× faster inference, worth nothing, and it breaks the scores

The obvious next optimisation is int8. It is not in the model zoo either, but the same
direct-download path has it — note there is **no int8 tessellated build**
(`yolo26m-det-int8-mla_tess-b1` → 403), only `yolo26m-det-int8-b1`.

The two packages are structurally different pipelines:

| | Stages |
| --- | --- |
| `bf16-mla_tess` | `CVU preproc` → `MLA` |
| `int8` | `CVU quanttess` → `MLA` → `CVU detess+dequant` |

It ran, and it was fast: **inference 15.9 ms vs 30.3 ms.** It also found exactly the same
objects as bf16 on the COCO ground-truth image — same 15 boxes, same classes, same order.

And every confidence score was **exactly 0.50**.

```
bf16 :  person(0.88) chair(0.90) chair(0.89) chair(0.82)  ... person(0.13)
int8 :  person(0.50) chair(0.50) chair(0.50) chair(0.50)  ... person(0.21)
```

Not a threshold artefact: with `score_threshold=0.51` the model returns **nothing, ever**.
0.50 is a hard ceiling.

### Reading the planner instead of guessing

The first instinct was "Neat isn't inserting the detess/dequant stage" — the built graph is
`preproc → mla → boxdecode`, the same shape as the bf16 model, with no visible dequant
node. Plausible, and wrong.

`Model::Options.verbose.planner = true` (exposed here as `verbose_planner=true`) prints the
MPK contract and the routing decisions, and it says plainly:

```
[route-debug] mpk_summary  pre=quanttess post=detess  has_pre=1 has_post=1  detess=1 dequant=1
route: fusion_boxdecode_supported=1
route: session-route: post_fusion=user_boxdecode(cast+detess+dequant)->boxdecode
route: session-route: final_post_chain=boxdecode
```

Neat **fuses** cast+detess+dequant *into* the boxdecode kernel. The dequant is happening;
there is no missing stage. The pipeline is correct.

### The actual cause is in the package's quantization

`0_boxdecoder.json`, class-score heads:

```json
"dq_zp":    [-111, -124, -128,  127,  127,  108],
"dq_scale": [22.28, 13.91, 14.54, 8.15, 5.75, 4.88],
"input_depth": [4, 4, 4, 80, 80, 80]
```

The last three tensors are the 80-class score heads. Their zero-points are **127 and 108 —
at the top of the int8 range**. Since `q ≤ 127`, the dequantized class logit `(q − zp)` can
never be positive. Feed a non-positive logit to a sigmoid and the best you can get is
`sigmoid(0) = 0.50`.

That predicts precisely what we measured: strong detections pinned at 0.50, weaker ones
below it (`0.21` = a negative logit), nothing above. The *relative ordering* of logits is
preserved, which is why the boxes and classes are still right — it is an affine offset
problem in the class head, not a broken pipeline.

### The optimisation was pointless anyway

End-to-end fps: **30.9 (bf16) → 32.9 (int8)**. Two frames per second, mostly startup noise.

Because we are **camera-limited**. bf16 already infers in 30 ms while the camera delivers a
frame every 33 ms. Halving inference to 16 ms just leaves the MLA idle longer. int8 buys
MLA *headroom* — useful if you want a second model or more streams — but it buys no frames.

Both packages ship in `assets/models/`; `model_path` switches between them. bf16 stays the
default. If you do use int8, halve `score_threshold` (0.30 → 0.15), because its scores live
in 0.0–0.50 and are not comparable to bf16's.

**Two lessons.** First: **profile before optimising.** A 2× faster model stage is worth zero
when a different stage is the bottleneck — the win existed only on the spreadsheet.
Second: **when a model misbehaves, read the planner before editing the pipeline.** The graph
*looked* like it was missing a dequant stage. Ten seconds of `verbose_planner` output showed
the dequant was fused in and sent me to the model's own quantization parameters, where the
answer actually was.

---

## Replicating from scratch

1. Plug in the camera. `v4l2-ctl --list-devices` → find the *Video Capture* node (not the
   metadata one). `v4l2-ctl -d /dev/videoN --list-formats-ext` → confirm MJPEG 1080p30.
2. Download the model (it is **not** in `sima-cli modelzoo list`):
   `sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz`
3. Build the camera fragment with `nodes::Custom(..., InputRole::Source)`. Pin
   `image/jpeg` in the caps. Use `jpegdec`, **not** `SimaDecode`/`neatdecoder`. End the
   fragment on a real element, never on a bare caps string.
4. Use `pipeline_mode=push`. Set `SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY=1`.
5. Configure the model: NV12→RGB, `ResizeMode::Letterbox`, `NormalizePreset::COCO_YOLO`,
   `BoxDecodeType::YoloV26`, 80 classes.
6. Before trusting detections, run a known COCO image through `source_override` and check
   you get `person(0.88) chair(0.90) …`. If that works, the pipeline is right.

## Things that did not work — quick index

| Attempt | Result |
| --- | --- |
| `SimaDecode`/`neatdecoder` MJPEG hardware decode | **4 fps.** Unusable, at every `dec-type`/buffer/next-element setting |
| Raw YUYV capture at 1080p | **5 fps.** USB 2.0 bandwidth limit; MJPEG is mandatory |
| Zero-copy `graph` mode with Branch | Builds, runs at 30 fps, **detects nothing** — CVU reads system memory as black |
| `neatcamerabridge` to fix the above | **Not in runtime 0.2.2**, despite being in the `core` source tree |
| Fragment ending in `! video/x-raw,...` | `no element "video"` — trailing caps parse as an element name |
| `graph.add(source)` + `graph.connect(source, ...)` | Source emitted twice; two `v4l2src` on one camera |
| `sima-cli modelzoo list` for yolo26 | Not in the index; published as a direct download |
| `videoflip rotate-180` to fix weak detections | Real but minor (tv 0.12→0.25); not the cause. Kept as a feature |
| Python `tensor.to_numpy()` on an NV12 frame | `__dlpack__ only supports dense tensors` — use `copy_payload_bytes()` |
| "The encoder emits no RTP" | **Measurement artifact.** Receiver ran inside the ~48 s model load. Encoder was always fine |
| `udpsrc num-buffers=N` to bound a capture | Truncates `rtph264depay` mid-access-unit → 0 bytes from a valid stream. Bound by time |
| `h264parse ! filesink` to save a stream | Writes AVC, no start codes. Pin `stream-format=byte-stream` |
| `openh264dec` on the DevKit to view output | Fails on RTP-depayed H.264 — *including a synthetic control*. Board limitation; view on a desktop |
| int8 yolo26m for more fps | **2× faster inference, ~0 extra fps** — pipeline is camera-limited, not MLA-limited |
| int8 yolo26m at all | Confidence scores **capped at 0.50** (`dq_zp=127` on the class heads). Boxes/classes fine; scores unusable |
| `yolo26m-det-int8-mla_tess-b1` | Does not exist — 403. There is no tessellated int8 build |
| "Neat forgot the dequant stage" | Wrong. `verbose_planner` shows it fused: `post_fusion=user_boxdecode(cast+detess+dequant)->boxdecode` |
