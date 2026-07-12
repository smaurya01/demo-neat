# Quad Stream Quad Model (4x RTSP, 4 different models)

## Introduction

Four RTSP streams → **four different** compiled INT8 models → four independent annotated
H.264/RTP UDP sinks, in one process.

Stream identity is preserved end to end: the frame pulled from stream *i*'s source is the exact
frame pushed into stream *i*'s model, decoded for stream *i*'s task, annotated in place, and
published on stream *i*'s own UDP port. Each frame carries a burned-in `S<i> <TASK> :<port>` banner
so you can tell the four windows apart.

## About Project

- Application: `quad_stream_quad_model` (`main.py`, Python)
- Input: 4x RTSP H.264 streams
- Output: 4x UDP/RTP H.264 streams, one port per stream
- Runtime config: `./config/default.conf`

| slot | task | model (compiled INT8 archive) | on-device decode? |
| --- | --- | --- | --- |
| 0 | detection | `yolo11s` | **yes** — Neat `BoxDecodeType.YoloV26` |
| 1 | segmentation | `yolo11s-seg` | no — raw heads → host decode |
| 2 | pose | `yolo26s-pose` | no — raw heads → host decode |
| 3 | detection (YOLOX) | `yolox_s` | no — raw heads → host decode |

Why only one of them decodes on-device — and what it costs the other three — is the core lesson of
this app: see [Appendix: Why three of the four models decode on the host](#appendix-why-three-of-the-four-models-decode-on-the-host).

## Requirements

Run on the DevKit with `dk`. `pyneat` must be importable there. `/workspace` is NFS-mounted on the
board at the same path, so you edit host-side and run board-side — no copying.

```bash
cd /path/to/demo-neat/apps/quad-stream-quad-model
```

Sanity-check your RTSP source first — its frame rate is the hard ceiling on any FPS you can claim:

```bash
ffprobe -hide_banner -rtsp_transport tcp rtsp://<rtsp-server-ip>:8555/stream
```

## Model

**None of these four archives is in the model zoo** — you compile all of them. They are large and
**not committed**; `assets/models/` is git-ignored.

Build them with the graph-surgery flow in [`model-compilation/`](../../model-compilation/README.md).
See [`REPLICATION.md`](../../model-compilation/REPLICATION.md) for the exact commands per model
(`yolo11s`, `yolo11s-seg`, `yolo26s-pose`, `yolox_s`), then copy the four archives here:

```text
./assets/models/yolo11s.compile_ready_mpk.tar.gz
./assets/models/yolo11s-seg.compile_ready_mpk.tar.gz
./assets/models/yolo26s-pose.compile_ready_mpk.tar.gz
./assets/models/yolox_s.compile_ready_mpk.tar.gz
```

> **The pose archive must be the padded build.** Its keypoint head is zero-padded 51 → 64 channels
> (`pad_channels_to: 64` in `model-compilation/compile/_surgery_ultralytics.py`). That padding is a
> **209x performance fix**, not cosmetics — see
> [Appendix: Known limitations](#appendix-known-limitations). The compile flow does this for you.

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
rtsp_default=rtsp://<rtsp-server-ip>:8555/stream
udp_host=<host-ip-that-receives-video>
udp_port_base=5206
udp_port_stride=2
```

With those defaults stream *i* publishes on `5206 + 2*i` → `5206`, `5208`, `5210`, `5212`.

For a bounded smoke test, set `frames=30`.

## Config Parameters

`rtsp_default`: RTSP URL used by any stream slot that does not set its own `stream<i>_rtsp`.

`stream0_rtsp` … `stream3_rtsp`: Per-slot RTSP URL. Each slot builds its **own** source graph, so
four streams means four real H.264 decodes — not one decode fanned out. All four default to the same
source; point them at four different cameras and stream identity still holds.

`rtsp_transport`: RTSP transport mode. Use `tcp` for reliability or `udp` for lower latency.

`num_streams`: How many stream slots to run (1–4). Drop to 2 for a lighter, higher-FPS pipeline.

`stream0_task` … `stream3_task`: Task for each slot — `detection` | `segmentation` | `pose` | `yolox`.

`stream0_model` … `stream3_model`: Compiled archive for each slot. Relative to this app folder.

`udp_host`: Host/IP that receives all four annotated UDP/RTP output streams.

`udp_port_base`: UDP/RTP output port for stream 0.

`udp_port_stride`: Port spacing. Stream `i` publishes on `udp_port_base + i * udp_port_stride`.

`model_width`, `model_height`: Model input size used by Neat preprocessing (all four are 640×640).

`fallback_width`, `fallback_height`, `fallback_fps`: Used when RTSP caps are incomplete.

`latency_ms`: RTSP receiver jitter buffer, in milliseconds.

`score_threshold`, `nms_iou`, `top_k`: Decode thresholds — used by the on-device box decode and by
the host decoders alike.

`queue_depth`: Bounded per-graph queue depth (Realtime preset, KeepLatest overflow).

`bitrate_kbps`: H.264 output encoder bitrate.

`frames`: Frames to process **per stream**. Use `0` to run until interrupted.

`print_backend`: Print the generated GStreamer backends when `true`.

Every value is also overridable on the command line. Run `python main.py --help`.

Useful flags: `--num-streams {1..4}`, `--tasks a,b,c` (custom model set), `--task <t>` (run ONE model
solo), `--no-overlay` (isolate the model rate from host-decode cost), `--duration <seconds>` (measure
over a wall-clock window — the correct stop condition for a shared-MLA benchmark), `--frames N`,
`--rtsp URL` (override all sources), `--print-backend`.

## How To Run

Run on the DevKit from the SDK shell:

```bash
dk ./main.py \
  --config ./config/default.conf
```

Bounded smoke test:

```bash
dk ./main.py \
  --config ./config/default.conf \
  --frames 30
```

Measure the model rate without the host-decode/overlay cost:

```bash
dk ./main.py --no-overlay --duration 20
```

## How To See The Output

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

Run this on the machine at `udp_host` — one viewer per stream:

```bash
# stream 0 detection :5206   stream 1 segmentation :5208
# stream 2 pose      :5210   stream 3 yolox        :5212
for P in 5206 5208 5210 5212; do
  gst-launch-1.0 -v udpsrc port=$P caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false &
done
```

Expected output: four live windows — boxes, masks, skeletons and YOLOX boxes respectively — each
with an `S<i> <TASK> :<port>` banner burned into the top-left.

---

# Appendix

## Appendix: Why three of the four models decode on the host

The `compile_ready` surgery for all four models deliberately exposes **raw per-scale head tensors**
and cuts the data-dependent decode/NMS tail, so the whole graph stays on the MLA (`A65:0`, one
`.elf`, zero `.so`). Neat's built-in fused `BoxDecode` covers only the plain **detection** family,
so:

* stream 0 (detection) uses the on-device `BoxDecodeType.YoloV26` decode and `pyneat.decode_bbox`;
* streams 1–3 pull the raw heads and decode them on the A65 in NumPy (`src/decoders.py`):
  anchor-grid + stride geometry, sigmoid/exp, NMS, letterbox-inverse, and (seg) prototype-mask
  assembly.

That difference dominates the delivered FPS — see the measured numbers below.
[`TEACHING.md`](TEACHING.md) has the full design discussion; `src/decoders.py` has the math.

## Appendix: Time profile

Every run prints a per-stage breakdown and two different FPS numbers. They answer different
questions and must not be conflated:

- **`model fps`** = `1000 / mean(infer)`. The **model stage alone** — EV74 preprocess + MLA + any
  on-device decode. No host decode, no overlay. This is the "can this model do 60 fps" number.
- **`delivered fps`** = frames actually published to UDP per second of wall clock. Includes host
  decode + overlay + encode.

Useful flags:

```bash
--no-overlay          # skip host decode + annotation; isolates the MODEL rate
--pipeline-depth 1    # lock-step push/pull, so `infer` is the TRUE per-frame model cost
                      # (with depth > 1, `infer` measures graph latency, not service time)
--task <t>            # run ONE model solo, MLA uncontended
--tasks a,b,c         # custom model set, one per stream slot
--duration <seconds>  # measure over a fixed wall-clock window (see the warning below)
--warmup-frames N     # exclude the first N frames per stream from the means
```

> **Measure with `--duration`, not `--frames`, whenever streams share the MLA.** A per-stream frame
> cap only stops the run when *all* streams reach it, so a fast stream keeps running — and keeps
> consuming the one MLA — while slow streams starve. That reports rates no steady state ever
> produced (we saw detection run 7934 frames against a 250-frame cap while its peers were starved to
> ~0.1 fps).

## Appendix: Measured behaviour (Modalix DevKit; RTSP 1280x720 H.264 @ 59.94 fps)

### Model rate, each model solo (`--task <t> --no-overlay --pipeline-depth 1`)

| stream-model | infer ms | **model fps** |
| --- | --- | --- |
| detection `yolo11s` (on-device decode) | 7.37 | **135.6** |
| segmentation `yolo11s-seg` | 9.54 | **104.8** |
| yolox `yolox_s` | 8.80 | **113.6** |
| pose `yolo26s-pose` | **1821.92** | **0.5** ← broken, see below |

### Three models concurrently, `--no-overlay`, 25 s window

All three hold the full source rate; one MLA sustains ~167 inferences/s in aggregate.

| stream-model | model fps | delivered fps |
| --- | --- | --- |
| detection `yolo11s` | 96.2 | 55.81 |
| segmentation `yolo11s-seg` | 87.3 | 55.73 |
| yolox `yolox_s` | 90.5 | 55.77 |
| **aggregate** | | **167.32** |

Note this is **far more than `1 / Σ(service times)`** would predict — the EV74-preprocess / MLA /
dequantize stages of *different* streams overlap. Do not predict multi-model capacity by adding
service times; measure it.

### Three models concurrently, WITH overlay — only detection survives

| stream-model | infer ms | decode ms | overlay ms | delivered fps |
| --- | --- | --- | --- | --- |
| detection `yolo11s` | 8.11 | **0.62** | 4.31 | **56.25** |
| segmentation `yolo11s-seg` | 34.53 | **337.16** | **77.43** | 0.08 |
| yolox `yolox_s` | 147.39 | **142.59** | 2.07 | 0.00 |

This is the core lesson of the app, quantified: detection uses Neat's **fused on-device**
`BoxDecodeType.YoloV26` and decodes in **0.62 ms**. Segmentation and yolox pay an **A65 host NumPy
decode** of the raw heads (`src/decoders.py`) costing **337 ms** and **143 ms** per frame. The host
decode — not the MLA — is what destroys their delivered FPS.

## Appendix: `tools/pose_probe.py` — study one model on its own

The quad app runs four models, three graphs each, across a dozen threads. When one model misbehaves
that is the worst possible place to debug it. `tools/pose_probe.py` strips everything away: ONE model
graph (`input -> model -> output`), lock-step push/pull, no threads, no overlay, no UDP.

```bash
# what does this model actually cost, and are its outputs right?
python tools/pose_probe.py --task pose --iters 20 --save-out /tmp/pose.jpg

# control: the same probe on a model known to be fast
python tools/pose_probe.py --task segmentation --iters 20

# can ANY runtime option move the number?
python tools/pose_probe.py --sweep

# empirically derive the correct keypoint decode formula (see below)
python tools/pose_probe.py --kpt-scan
```

It prints every raw output tensor's shape/dtype, the true per-frame model cost, the host-decode cost,
and a sanity check of the decoded result — so you can tell *slow but correct* from *slow and wrong*.
Because it is lock-step, its `infer` is the real model service time, not graph latency.

### It caught a real bug: the YOLO26 keypoint decode was 2x too big

`src/decoders.py` originally decoded keypoints with the **Ultralytics v8/v11** pose formula
`(k * 2.0 + i) * stride`. **YOLO26's `one2one_cv4_kpts` head does not use that `2x`.** The correct
decode is `(k + anchor) * stride` with `anchor = i + 0.5` — the *same* anchor point the box decode
already uses.

This was a silent bug: no error, and the person **boxes were perfect**, so the detection counts looked
completely healthy (7 people, 17 keypoints each, all in-frame). Only a rendered frame showed the
skeletons sprawling at exactly twice their true size.

`--kpt-scan` settles it by measurement rather than guesswork. The boxes decode correctly and
independently, so they are free ground truth: score each candidate formula by how many visible
keypoints land inside their own person box. **But that check alone is gameable** — a sigmoid-bounded
variant collapses every keypoint onto the anchor and scores 100% while being useless. So it also
scores *span* (skeleton height / box height) and *anatomy* (nose above ankles):

| formula | inside% | span | nose<ankle |
| --- | --- | --- | --- |
| **B `(k + i + 0.5) * s`  ← correct** | **99.9%** | **0.83** | **100%** |
| C `(k + i) * s` | 99.6% | 0.83 | 100% |
| D `(2k + i + 0.5) * s` | 52.2% | 1.67 | 100% |
| A `(2k + i) * s`  ← the old v8 formula | 44.2% | **1.67** ← 2x too big | 100% |
| E `(2*sigmoid(k) - 0.5 + i) * s` | 100% ← gamed | **0.25** ← collapsed | 100% |

## Appendix: Known limitations

1. **`yolo26s-pose` is unusable, and all four models together will not run.** Pose costs **1.82 s per
   frame inside the model graph** — measured with `--no-overlay`, so this is *not* host decode. With
   pose in the mix, it holds the MLA so long that every other graph backs up and the run dies with
   `model push failed`. Root cause is a **compile artifact**, not an app bug:

   - Its MLA is healthy: 1,455,681 cycles vs yolo11s's 1,205,496 (**1.21x**) — it *should* run ~9 ms.
     (Per-layer cycles are in `*_mla_stats.yaml` inside the `_mpk.tar.gz`.)
   - Output volume is a red herring: seg moves 1.79 M floats through its tail in 9.5 ms; pose moves
     only 0.47 M in 1822 ms.
   - The **post-MLA tail** is the cost. Dumping the MPK contract
     (`ModelOptions.verbose.planner = True`) shows pose routes **every one of its 9 outputs through
     its own `slice_MLA_0/tuple_get_item_N_slice_transform` stage**, whereas `yolo11s-seg` emits most
     outputs from a **single fused `MLA_0_ofm_unpack_transform`**. Nine serial per-output slice
     transforms is the 1.8 s.
   - `ModelOptions.processcvu.post_run_target = "EV74"` does **not** help — it is the stage
     count/shape, not the execution target.

   **Fix: recompile `yolo26s-pose` so its post-MLA tail fuses into one `ofm_unpack`.** Not yet done.
   Meanwhile run a pose-free set: `--tasks detection,segmentation,yolox`.

2. **Teardown heap abort.** The process intermittently dies at exit with
   `malloc(): mismatching next->prev_size` / `double free` (exit 134/139). It fires **after** the
   profile prints, so the numbers are valid, but it is a real defect in the teardown path.

3. Set `QSQM_DEBUG=1` to print the raw tensor shapes each model delivers.
