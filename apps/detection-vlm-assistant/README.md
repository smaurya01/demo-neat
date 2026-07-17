# Detection-to-VLM Assistant (YOLO11 + Vision-Language Model)

## Table of Contents

- [Introduction](#introduction)
- [About Project](#about-project)
- [Pipeline Shape](#pipeline-shape)
- [Full Architecture](#full-architecture)
- [Split Validation — read this](#split-validation--read-this)
- [Requirements](#requirements)
- [Model Setup](#model-setup)
- [Configure](#configure)
  - [Config parameters](#config-parameters)
- [How To Run (human UX)](#how-to-run-human-ux)
- [How To Run (CI / automation fallback)](#how-to-run-ci--automation-fallback)
- [Expected Output](#expected-output)
- [Visualization — boxes over UDP (H.264/RTP)](#visualization--boxes-over-udp-h264rtp)
  - [How the VLM box is chosen (what the red box means)](#how-the-vlm-box-is-chosen-what-the-red-box-means)
- [Colour Correctness (a real trap)](#colour-correctness-a-real-trap)
- [Direct `VisionLanguageModel` vs `GenAIServer`](#direct-visionlanguagemodel-vs-genaiserver)
- [Verified](#verified)
- [Notes](#notes)

---

## Introduction

An **always-on YOLO11 detector** watches a video stream (or a folder of images).
Selected detections are cropped and handed to a **vision-language model (VLM)**, which
returns a natural-language description of the crop. Output = detections **plus** a caption.

This is **trigger-based multimodal**: the detector is cheap and runs on every frame
(~37 fps on the sibling 2-stream YOLO11 app; ~54 fps single-stream measured here); the
VLM is expensive (seconds per call) and fires **only** on interesting, de-duplicated
events. A bounded background worker absorbs VLM latency so the detection loop never blocks.

Adapted from `apps/examples/genai/detection-to-vlm-assistant` (crop-to-VLM shape) and
`apps/multi-stream-yolo-yolo11` (Agent A's live-validated YOLO11 detection idioms).

## About Project

- Application: `detection-vlm-assistant` (`main.py` + `src/vlm_commenter.py`)
- Detector model: `yolo_11n_mpk.tar.gz` (the verified T1 compiled YOLO11n archive)
- VLM model: `Qwen3-VL-4B-Instruct-GPTQ-a16w4` (already on the board)
- Input: one RTSP H.264 stream **or** still image(s)
- Output: per-frame detection log + VLM caption per selected crop
- Runtime config: `./config/default.conf`

<details>
<summary><h2>Pipeline Shape</h2></summary>

<br>

```
frame (RTSP NV12 -> BGR, or still image BGR)
        |
        v
  YOLO11 detector (Neat model-managed box decode, BoxDecodeType.YoloV26)   <- cheap, every frame
        |
        v
  trigger gate (class allow-list + min score + min area) + dedup (IoU/cooldown) + rate limit
        |
        v
  bounded background VLM worker  ->  pyneat.genai.VisionLanguageModel  ->  caption   <- expensive, gated
```

References: `apps/examples/genai/detection-to-vlm-assistant` (crop-to-VLM),
`apps/multi-stream-yolo-yolo11/main.py` (detector, `push`+`pull("detections")`,
`pyneat.decode_bbox`), `core/include/genai/VisionLanguageModel.h`,
`core/include/genai/GenAITypes.h`, `llima/02-run-llm-vlm/02_run_vlm.ipynb`.

</details>

<details>
<summary><h2>Full Architecture</h2></summary>

<br>

Two decoupled legs share each frame. The **detector** runs every frame (cheap) and feeds
both the **VLM leg** (gated / dedup'd / rate-limited, bounded background worker, never
blocks the loop) and the **video leg** (overlay + UDP). The VLM leg and video leg agree on
which box is "the subject" through the same `_passes_gate` logic — `on_frame` uses it for
the real send, `gate_candidate` uses it (non-mutating) for the red highlight.

```
                          RTSP H.264 stream (camera / Insight RTSP source)
                                        |
                                        v
             +----------------------------------------------------+
             |  make_rtsp_source: pyneat.groups.rtsp_decoded_input |  NV12 out, Realtime/KeepLatest
             +----------------------------------------------------+
                                        |  source_run.pull_tensors()
                                        v
                        decoded_tensor_to_bgr()  -> clean BGR frame ----------------+
                                        |                                           | (clean copy)
                                        v                                           |
     +-------------------------------------------------------+                      |
     |  detect(): push([BGR]) -> pull("detections")          |  YOLO11n             |
     |  -> pyneat.decode_bbox() -> boxes[x1,y1,x2,y2,score,c] |  (YoloV26 head)      |
     +-------------------------------------------------------+                      |
                                        |                                           |
                 +----------------------+-----------------------+                   |
                 v (every frame, cheap)                         v                   |
   +------------------------------+          +-------------------------------------+-----------+
   | VlmCommenter.on_frame(boxes) |          | gate_candidate(boxes) -> the ONE red box (viz)  |
   |  rate-limit -> gate -> dedup |          +----------------------+--------------------------+
   |  -> crop CLEAN frame -> queue|                                 v
   +--------------+---------------+          +------------------------------------------------+
                  v (bounded, background)    | draw_boxes(): white = all, red = gate_candidate |
   +------------------------------+          +----------------------+--------------------------+
   | worker thread:               |                                 v
   |  dry-run -> log crop+prompt  |          +------------------------------------------------+
   |  real    -> VisionLanguageModel         | bgr_to_nv12() -> make_nv12_tensor() -> push     |
   |            .run(GenReq) caption          | VideoSender: H.264 encode -> RTP -> UDP :port  |
   +------------------------------+          +----------------------+--------------------------+
        stdout captions                                             v
                                                     UDP H.264/RTP -> your gst viewer
```

Key point: the VLM crop is taken from the **clean** BGR frame inside `on_frame(...)`
*before* `draw_boxes(...)` runs, so the white/red overlay never leaks into the image
handed to the VLM. The video leg only runs in RTSP mode when `udp_host` is set; image
mode is stdout-only.

</details>

<details>
<summary><h2>Split Validation — read this</h2></summary>

<br>

The two legs are validated differently on purpose:

- **Detection leg — validated LIVE on the DevKit** (this app). RTSP or still image in ->
  real detections out. See "Verified" below.
- **VLM leg — code-complete and API-checked, but NOT executed by this app's authors.**
  The owner runs the real VLM manually. Until then, run in **dry-run** mode: the app logs
  the selected crop and the **exact prompt that WOULD be sent**, and never loads or calls
  the VLM.

Dry-run turns on automatically when `--no-vlm` is passed, when `vlm_enabled=false`, or when
the configured `vlm_model_dir` is missing. This is also how you validate the detection leg
without a VLM present, and it is genuinely useful for tuning the trigger/prompt.

</details>

## Requirements

Run on the DevKit. `pyneat` (0.3.0+), `numpy`, and OpenCV (`cv2`) must be importable there.
`/workspace` is NFS-mounted on the board at the same path, so write host-side and run
board-side — no copying.

```bash
cd /path/to/demo-neat/apps/detection-vlm-assistant
```

## Model Setup

**Detector.** The verified T1 YOLO11n archive is already copied into
`./assets/models/yolo_11n_mpk.tar.gz` (`./assets/models/` is git-ignored). To rebuild it
yourself, follow `../../model-compilation/README.md`; the compiled archive is:

```text
../../model-compilation/work/yolo11n/compile_int8/*/*_mpk.tar.gz
```

A `compile_ready` yolo11n surgery exposes the YoloV26 grouped-tensor head, so this archive
decodes with `BoxDecodeType.YoloV26` (NOT YoloV8). Keep `model_name=yolo11`.

**VLM (for the real, owner-run path only).** `Qwen3-VL-4B-Instruct-GPTQ-a16w4` is already
pulled on the board at `/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4`. No pull
is needed. If you must pull a fresh one, the disk is tight (~5.9 GB free) — the smallest
viable VLM is `LFM2.5-VL-450M-a16w4`.

## Configure

Edit `./config/default.conf`. At minimum for the RTSP demo:

```text
source=rtsp
rtsp_url=rtsp://<rtsp-server-ip>:8555/stream
model_path=./assets/models/yolo_11n_mpk.tar.gz
vlm_trigger_classes=person
```

<details>
<summary><h3>Config parameters</h3></summary>

<br>

- `source`: `rtsp` (always-on demo) or `image` (still file / directory / glob).
- `rtsp_url`, `rtsp_transport`, `latency_ms`, `fallback_*`: RTSP input.
- `image_path`: file, directory, or glob for `source=image`.
- `model_path` / `models_dir` / `model_name`: detector archive and decode family
  (`yolo11`/`yolo26n` => `BoxDecodeType.YoloV26`).
- `model_width`, `model_height`: compiled model input size (Neat letterbox-resizes to it).
- `score_threshold`, `nms_iou`, `top_k`: box-decode controls.
- `frames`: frames to process; `0` runs until interrupted (rtsp) / processes all images.
- **VLM trigger / commenter:**
  - `vlm_enabled`: master switch; `false` => dry-run.
  - `vlm_model_dir`: VLM directory on the board (real mode only).
  - `vlm_trigger_classes`: comma-separated COCO allow-list (empty = any class).
  - `vlm_trigger_min_score`, `vlm_trigger_min_area_frac`: confidence / size gating.
  - `vlm_interval_seconds`: minimum **wall-clock** seconds between VLM calls (rate limit).
  - `vlm_max_pending`: bounded in-flight/queued crops; a full queue drops the trigger.
  - `vlm_dedup_iou`, `vlm_dedup_cooldown_s`: treat a box as the same object at IoU ≥
    threshold; re-trigger only after the cooldown. Stops one person re-firing every frame.
  - `vlm_max_new_tokens`, `vlm_prompt`: generation controls. `{label}` in the prompt is
    substituted with the detected class.

Every value is overridable on the command line; run `python main.py --help`.

</details>

## How To Run (human UX)

On the DevKit, from a real terminal, use the `dk` helper (source it once:
`source /usr/local/bin/devkit.sh <devkit-ip> sima 22`).

Detection leg, dry-run VLM (validate live without a VLM):

```bash
dk ./main.py --config ./config/default.conf --no-vlm --frames 40
```

Still-image dry-run over a folder:

```bash
dk ./main.py --source image --image ../../model-compilation/assets/inference --no-vlm
```

Full pipeline with the real VLM (owner runs this after confirming the VLM dir):

```bash
dk ./main.py --config ./config/default.conf     # vlm_enabled=true, model dir present
```

## How To Run (CI / automation fallback)

`dk` needs a TTY and hangs in non-interactive/agent contexts. For CI use passwordless ssh
(the sima-neat skill's documented fallback). `/workspace` is NFS-mounted, so run the same
on-disk file:

```bash
timeout 200 ssh -o BatchMode=yes sima@<devkit-ip> \
  'source $HOME/pyneat/bin/activate; \
   cd apps/detection-vlm-assistant; \
   python main.py --no-vlm --frames 40'
```

## Expected Output

Detection (live) — per-frame log lines and, when a person clears the gate, a dry-run block:

```text
detector=yolo_11n_mpk.tar.gz decode=yolo11 vlm=DRY-RUN (crop+prompt logged, VLM not called)
rtsp=rtsp://<rtsp-server-ip>:8555/stream stream=1280x720@60
frame=1 detections=13 fps=14.57
vlm[dry-run] WOULD send crop -> VLM
  class   : PERSON score=0.81
  bbox    : (927, 429, 1019, 666)
  crop    : shape=(237, 92, 3) (BGR; converted to RGB before the request)
  model   : /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4
  prompt  : 'You are watching a security camera. This image is a crop of one detected person. ...'
frame=30 detections=12 fps=53.65
```

VLM (real, owner-run) — each dry-run block is replaced by a caption line:

```text
vlm[PERSON score=0.81 bbox=(927, 429, 1019, 666)]: A person in a dark jacket is walking
across the platform carrying a bag.  (58 tok, 22.4 tok/s, ttft=0.34s)
```

Exact wording depends on the image and the model's sampling; the shape is what to verify.

<details>
<summary><h2>Visualization — boxes over UDP (H.264/RTP)</h2></summary>

<br>

Like the sibling single-stream apps, RTSP mode can draw the detections on each frame
and stream the result as H.264/RTP over UDP. Colour convention:

- **every detection is drawn white**, and
- **the one box the VLM would caption this frame is drawn red** (highest-score box that
  clears the trigger gate — see "How the VLM box is chosen" below).

Enable it by pointing `udp_host` at your viewer machine (video is on by default; leave
`udp_host` empty or pass `--no-video` for stdout-only detection):

```bash
dk ./main.py --config ./config/default.conf --udp-host 192.168.1.50 --udp-port 9000 --no-vlm
```

The app prints a ready-to-paste `gst-launch-1.0` viewer command on startup, e.g.:

```bash
gst-launch-1.0 -v udpsrc port=9000 \
  caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
  ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Config keys: `video_enabled`, `udp_host`, `udp_port` (or `udp_port_base`), `bitrate_kbps`.
Implementation mirrors `single-stream-yolo-yolo11` (`VideoSenderOptions.h264_rtp_udp_from_raw`
+ `pyneat.groups.video_sender`); the frame is drawn in BGR, converted to NV12, and pushed.
The VLM crop is always taken from the **clean** frame *before* the overlay is drawn, so the
red/white boxes never leak into the image handed to the VLM.

### How the VLM box is chosen (what the red box means)

The red box is the highest-score detection that passes the trigger gate: class in
`vlm_trigger_classes` (default `person`), `score >= vlm_trigger_min_score`, and area
`>= vlm_trigger_min_area_frac` of the frame. The *actual* VLM send additionally applies
the rate limit (`vlm_interval_seconds`), IoU/cooldown dedup (`vlm_dedup_iou`,
`vlm_dedup_cooldown_s`), and the bounded queue (`vlm_max_pending`); the red overlay
(`VlmCommenter.gate_candidate`) is non-mutating and shows the current subject every frame.

</details>

<details>
<summary><h2>Colour Correctness (a real trap)</h2></summary>

<br>

VLM images must be **uint8 HWC RGB**. Crops here are OpenCV-native **BGR**, so
`src/vlm_commenter.py` calls `cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)` **at the VLM request
boundary**. Skipping this silently feeds the VLM channel-swapped images and quietly degrades
every answer. The dry-run log prints the crop shape and reminds you of the conversion.

</details>

<details>
<summary><h2>Direct `VisionLanguageModel` vs `GenAIServer`</h2></summary>

<br>

This app calls the VLM **in-process** via `pyneat.genai.VisionLanguageModel` — lowest
latency, no network hop, one model handle holding LM weights + vision encoder resident.
Use `GenAIServer` (OpenAI-compatible HTTP) instead only when the boundary is a network: a
browser UI or a separate service that should not link the Neat runtime. The upstream
`apps/examples/genai/detection-to-vlm-assistant` uses the HTTP path; see `llima/05-genai-server`.

</details>

<details>
<summary><h2>Verified</h2></summary>

<br>

- **Detection leg, live on the DevKit (<devkit-ip>)** via ssh, `--no-vlm`:
  - Still-image mode over `model-compilation/assets/yolo_calibration`: real detections,
    e.g. `000000000885.jpg detections=4 [PERSON:0.92, PERSON:0.85, PERSON:0.66, TENNIS RACKET:0.81]`;
    the dry-run trigger logged the PERSON crop + exact prompt.
  - RTSP mode `rtsp://<rtsp-server-ip>:8555/stream` (1280x720@60): `frame=1 detections=13`,
    `frame=30 detections=12 fps=53.65`; dry-run trigger fired on `PERSON:0.81`.
- **VLM leg: code-complete, API-checked, NOT executed.** Owner runs it manually after
  confirming the VLM directory.

</details>

<details>
<summary><h2>Notes</h2></summary>

<br>

- The detector uses `push([tensor])` + `pull("detections", ...)` (Agent A's pattern). The
  synchronous `run([...])` helper does **not** surface this archive's model-managed
  box-decode output — push + named pull does. Verified on the DevKit.
- `vlm_interval_seconds` is wall-clock: over a fast batch of still images only the first
  qualifying crop fires. That is intended for the always-on RTSP use case; lower it (or set
  it to 0) if you want a caption per image.

</details>
