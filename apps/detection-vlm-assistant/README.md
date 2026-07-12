# Detection-to-VLM Assistant (YOLO11 + Vision-Language Model)

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

## Pipeline Shape

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

## Split Validation — read this

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

### Config parameters

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

## Colour Correctness (a real trap)

VLM images must be **uint8 HWC RGB**. Crops here are OpenCV-native **BGR**, so
`src/vlm_commenter.py` calls `cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)` **at the VLM request
boundary**. Skipping this silently feeds the VLM channel-swapped images and quietly degrades
every answer. The dry-run log prints the crop shape and reminds you of the conversion.

## Direct `VisionLanguageModel` vs `GenAIServer`

This app calls the VLM **in-process** via `pyneat.genai.VisionLanguageModel` — lowest
latency, no network hop, one model handle holding LM weights + vision encoder resident.
Use `GenAIServer` (OpenAI-compatible HTTP) instead only when the boundary is a network: a
browser UI or a separate service that should not link the Neat runtime. The upstream
`apps/examples/genai/detection-to-vlm-assistant` uses the HTTP path; see `llima/05-genai-server`.

## Verified

- **Detection leg, live on the DevKit (<devkit-ip>)** via ssh, `--no-vlm`:
  - Still-image mode over `model-compilation/assets/yolo_calibration`: real detections,
    e.g. `000000000885.jpg detections=4 [PERSON:0.92, PERSON:0.85, PERSON:0.66, TENNIS RACKET:0.81]`;
    the dry-run trigger logged the PERSON crop + exact prompt.
  - RTSP mode `rtsp://<rtsp-server-ip>:8555/stream` (1280x720@60): `frame=1 detections=13`,
    `frame=30 detections=12 fps=53.65`; dry-run trigger fired on `PERSON:0.81`.
- **VLM leg: code-complete, API-checked, NOT executed.** Owner runs it manually after
  confirming the VLM directory.

## Notes

- The detector uses `push([tensor])` + `pull("detections", ...)` (Agent A's pattern). The
  synchronous `run([...])` helper does **not** surface this archive's model-managed
  box-decode output — push + named pull does. Verified on the DevKit.
- `vlm_interval_seconds` is wall-clock: over a fast batch of still images only the first
  qualifying crop fires. That is intended for the always-on RTSP use case; lower it (or set
  it to 0) if you want a caption per image.
