# Multi Stream YOLO11 Detection (2x RTSP)

## Table of Contents

- [Introduction](#introduction)
- [About Project](#about-project)
- [Requirements](#requirements)
- [Model Download Command](#model-download-command)
- [Configure](#configure)
- [Config Parameters](#config-parameters)
- [How To Run](#how-to-run)
- [How To See The Output](#how-to-see-the-output)
- [Appendix](#appendix)
- [Appendix: Model Build](#appendix-model-build)
- [Appendix: Pipeline Shape](#appendix-pipeline-shape)
- [Appendix: Threading — how it sustains 60 fps per stream](#appendix-threading--how-it-sustains-60-fps-per-stream)
- [Appendix: Reading The Time Profile](#appendix-reading-the-time-profile)
- [Appendix: Learnings](#appendix-learnings)

---

## Introduction

This demo runs two RTSP streams through one shared SiMa Neat YOLO11 object detection model, draws
decoded detections on NV12 frames, and publishes one annotated H.264/RTP UDP stream per input.

Both streams sustain the full 60 fps source rate with zero dropped frames. Stream identity is
preserved end to end: each stream owns its RTSP source, its UDP port, and an on-frame `STREAM <id>`
banner.

## About Project

- Application: `multi_stream_yolo_yolo11` (`main.py`, Python)
- Model: `yolo_11n_mpk.tar.gz` (one archive, shared by both streams)
- Input: 2x RTSP H.264 streams
- Output: 2x UDP/RTP H.264 streams with bounding boxes, one port per stream
- Runtime config: `./config/default.conf`

## Requirements

Run on the DevKit with `dk`. `pyneat` must be importable there. `/workspace` is NFS-mounted on the
board at the same path, so edit host-side and run board-side — no copying.

Run the commands below from this app folder:

```bash
cd /path/to/demo-neat/apps/multi-stream-yolo-yolo11
```

## Model Download Command

YOLO11 is published in the SiMa model zoo, so just download it:

```bash
mkdir -p ./assets/models
cd ./assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_11n
cd ../..
```

The zoo asset is named `yolo_11n_mpk.tar.gz`, which is exactly the path `./config/default.conf`
already expects — no config edit needed.

Expected model path:

```text
./assets/models/yolo_11n_mpk.tar.gz
```

The zoo publishes the whole detection family (`yolo_11n`, `yolo_11s`, `yolo_11m`, `yolo_11l`,
`yolo_11x`). To run a different size, `get` that name and point `model_path` at it. Both streams
share this one archive.

Only compile YOLO11 yourself for a variant the zoo does not publish — and note that a self-compiled
archive needs a different `model_name`. See [Appendix: Model Build](#appendix-model-build).

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
rtsp_url_0=rtsp://<rtsp-server-ip>:8555/stream
rtsp_url_1=rtsp://<rtsp-server-ip>:8555/stream
model_path=./assets/models/yolo_11n_mpk.tar.gz
model_name=yolov8
udp_host=<host-ip-that-receives-video>
udp_port_base=5206
udp_port_stride=2
```

Leave `model_name=yolov8` alone when running the zoo archive — despite the name, it is the decode
family, and it is the right one for zoo YOLO11. See `model_name` under
[Config Parameters](#config-parameters).

With those defaults stream 0 publishes on `5206` and stream 1 on `5208`.

For a bounded smoke test, set `frames=30` in `./config/default.conf`.

<details>
<summary><h2>Config Parameters</h2></summary>

<br>

`rtsp_url_0`, `rtsp_url_1`: The two RTSP H.264 input streams. Both default to the same source; set
them to distinct cameras when you have them.

`rtsp_transport`: RTSP transport mode. Use `tcp` for reliability or `udp` for lower latency.

`udp_host`: Host/IP that receives both annotated UDP/RTP output streams.

`udp_port_base`: UDP/RTP output port for stream 0.

`udp_port_stride`: Port spacing. Stream `i` publishes on `udp_port_base + i * udp_port_stride`.

`model_path`: Shared model archive loaded by the Neat model stage.

`model_name`: Decode family, chosen by the shape of the archive's detection head — **not** by the
model's version number. `yolov8` selects `BoxDecodeType.YoloV8` (raw 64-channel DFL bbox heads),
which is what the zoo `yolo_11n` archive ships, so it is the default. `yolo11` / `yolo26n` select
`BoxDecodeType.YoloV26` (4-channel l/t/r/b distance heads), which is only correct for a
self-compiled archive. Setting this wrong still runs, but decodes boxes from the wrong channels.

`model_width`: Model input width used by Neat preprocessing.

`model_height`: Model input height used by Neat preprocessing.

`fallback_width`: Fallback decoded frame width used when RTSP caps are incomplete.

`fallback_height`: Fallback decoded frame height used when RTSP caps are incomplete.

`fallback_fps`: Fallback decoded stream FPS used when RTSP caps are incomplete.

`latency_ms`: RTSP receiver latency buffer in milliseconds.

`score_threshold`: Detection score threshold used by YOLO box decode.

`nms_iou`: NMS IoU threshold used by Neat decode.

`top_k`: Maximum decoded detections per frame.

`num_classes`: Number of classes in the model output.

`frames`: Number of frames to process PER stream. Use `0` to run until interrupted.

`warmup_frames`: Frames per stream excluded from the reported FPS and stage means. Graph build,
model load and RTSP jitter-buffer fill all land on the first few frames.

`model_queue_depth`: Frames the shared model Run keeps in flight. This is what pipelines the MLA.

`stream_queue_depth`: Bounded hand-off depth per stream (input queue and result queue).

`bitrate_kbps`: H.264 output encoder bitrate in kbps.

`print_backend`: Print generated backend pipelines when set to `true`.

Every value is also overridable on the command line (`--rtsp0`, `--rtsp1`, `--udp-port-base`,
`--frames`, ...). Run `python main.py --help`.

</details>

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

The app prints a per-stage time profile and per-stream FPS at exit:

```text
=== time profile (ms/frame, mean | p95) ===
stream frames           rtsp           prep          qwait           push          infer         decode        overlay           send        latency
     0    500  16.36|39.02     0.50|0.66      0.69|1.38      0.10|0.14      7.34|9.47      0.37|0.48      6.62|11.02     0.56|0.72     36.09|59.32
     1    507  16.10|39.25     0.51|0.70      0.71|1.62      0.10|0.15      7.39|9.68      0.36|0.43      6.43|10.66     0.54|0.66     35.67|57.91

=== fps ===
  stream 0: delivered  58.81 fps  (500 frames, 0 dropped)
  stream 1: delivered  59.63 fps  (507 frames, 0 dropped)
  aggregate:          118.44 fps across 2 streams in 8.5s
```

See [Appendix: Reading The Time Profile](#appendix-reading-the-time-profile) for what each column
means and which ones are easy to misread.

## How To See The Output

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

Run these on the host machine receiving UDP. Use the same ports configured by `udp_port_base` and
`udp_port_stride` — one viewer per stream.

```bash
# Stream 0
gst-launch-1.0 -v udpsrc port=5206 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false

# Stream 1
gst-launch-1.0 -v udpsrc port=5208 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Expected output: two live windows, each with YOLO11 detection boxes and a `STREAM 0` / `STREAM 1`
banner burned into the top-left, confirming per-stream identity.

---

# Appendix

<details>
<summary><h2>Appendix: Model Build</h2></summary>

<br>

You do **not** need this for `yolo11n` — the SDK 2.1.2 Modalix zoo publishes `yolo_11n` (and
`yolo_11s/m/l/x`), so use [Model Download Command](#model-download-command). Compile only for a
variant the zoo does not ship, or to change the surgery.

Compile it with the graph-surgery flow. See `../../model-compilation/README.md` for the full
walkthrough. The compiled archive lands at:

```text
../../model-compilation/work/yolo11n/<...>/compile_int8/.../yolo11n.compile_ready_mpk.tar.gz
```

Copy or symlink it to `./assets/models/yolo_11n_mpk.tar.gz`, or point `model_path` at it directly.
`./assets/models/` is git-ignored.

**Then set `model_name=yolo11`.** A self-compiled archive is not interchangeable with the zoo one:
the surgery folds the DFL into the graph and exposes 3x 4-channel l/t/r/b distance heads
(`BoxDecodeType.YoloV26`), whereas the zoo archive keeps 3x 64-channel raw DFL heads
(`BoxDecodeType.YoloV8`, the `model_name=yolov8` default). The class heads are 80-channel in both.
Leaving `model_name=yolov8` on a self-compiled archive still runs, but decodes boxes from the wrong
channels.

</details>

<details>
<summary><h2>Appendix: Pipeline Shape</h2></summary>

<br>

```text
RTSP stream 0 ──> source graph 0 ─┐
                                  ├─> SHARED YOLO11 model stage ─> Neat box decode ─┬─> annotate ─> video sender 0 ─> udp:PORT
RTSP stream 1 ──> source graph 1 ─┘   (one compiled archive, one Run handle)        └─> annotate ─> video sender 1 ─> udp:PORT+stride
```

The shared model stage needs one input geometry, so both streams must decode at the same resolution.
Since both default to the same source this holds. If your two cameras differ, the app raises a clear
error; to support mixed resolutions, build one model graph per stream instead of one shared handle —
the rest of the engine is unchanged.

</details>

<details>
<summary><h2>Appendix: Threading — how it sustains 60 fps per stream</h2></summary>

<br>

The thread topology is ported from a proven C++ 4-stream reference implementation
(4 streams x 60 fps). It is not a design invented here:

```text
source thread per stream -> input queue [4]
   -> ONE round-robin PUSHER thread -> shared model Run -> ONE PULLER thread
   -> result queue [4] -> output worker per stream (box decode + overlay + UDP push)
```

Two settings do all the work, and both are counter-intuitive:

**1. The shared model Run uses `RunPreset.Reliable` + `OverflowPolicy.Block` + `queue_depth=4`** —
NOT `Realtime` + `KeepLatest`, which looks like the obvious choice for a live camera and is wrong
here.

- `Block` keeps push and pull **strictly paired**, so a plain FIFO deque of `(stream, frame)` is
  enough to route each result back to its own stream. `KeepLatest` silently drops results and
  destroys that pairing.
- `Block` + `queue_depth` is also what **pipelines the MLA**: 4 frames stay in flight instead of the
  accelerator waiting on the host every frame.
- Frames are dropped at the **source queue** instead (drop-oldest, since a live camera does not
  wait). Drop at the source, never at the model.

**2. The pusher is ROUND-ROBIN over the per-stream input queues.** That is the only thing providing
fairness — without it, one stream can monopolise the shared model.

Measured against a 1280x720 H.264 @ 59.94 fps source:

| | fps/stream | aggregate | dropped |
| --- | --- | --- | --- |
| before (single-threaded round-robin) | 32.5 | 65 | 0 |
| **now** | **58.6 - 61.2** (3 runs) | **118 - 122** | **0** |

The old loop ran every stage of every stream on one thread and left the MLA idle ~60% of the time:
only 6.1 of 14.8 ms/frame was the model; the rest was host marshalling, overlay and encode push.

</details>

<details>
<summary><h2>Appendix: Reading The Time Profile</h2></summary>

<br>

The stages run on four different threads and **overlap**, so they deliberately do NOT sum to the
frame period. Read `delivered fps` as throughput; read the columns as cost attribution.

| column | meaning |
| --- | --- |
| `rtsp` | source thread: wait for + copy one decoded NV12 frame out of the RTSP graph |
| `prep` | source thread: NV12 -> `pyneat.Tensor` for the model input |
| `qwait` | time the frame sat in its stream's input queue waiting for the pusher |
| `push` | pusher thread: `model_run.push()`. **Blocking** time — see below |
| `infer` | push returns -> that frame's result is pulled. Model-graph **latency** — see below |
| `decode` | output thread: `pyneat.decode_bbox` on the BBOX tensor |
| `overlay` | output thread: NV12 Y/UV annotation. Usually the largest host cost |
| `send` | output thread: NV12 -> Tensor + push into the H.264/RTP UDP sender |
| `latency` | end to end: RTSP pull started -> annotated frame handed to the encoder |

Two columns are easy to misread:

- **`push` is the saturation tell.** Under `OverflowPolicy.Block` a push only blocks once the MLA
  already holds `queue_depth` frames. `push ≈ 0` means the model always had room — it is **not** the
  bottleneck, and there is headroom for more streams. If `push` grows, the model is back-pressuring
  and **is** the bottleneck.
- **`infer` is latency, not service time.** It includes queueing behind the other in-flight frames,
  so `1000 / infer` is *not* the model's throughput.

The app also prints the shared model stage's inter-result interval. **That is the observed output
rate, not a capacity ceiling** — when the model is unsaturated it simply mirrors the arrival rate.
The app says which case you are in.

**`rtsp ≈ 16.4 ms` is the good sign.** That is the source thread waiting on a 59.94 fps camera
(16.7 ms frame period): the pipeline is **source-limited, not compute-limited**. You cannot exceed
the source rate — if you ever see a number above it, the measurement window was too short (queue
drain at the boundary), not real throughput.

</details>

<details>
<summary><h2>Appendix: Learnings</h2></summary>

<br>

**Do not "modernise" this into an in-graph `graphs.branch` / `combine` pipeline.** Tried and rejected
on measurement. The official example `apps/examples/object-detection/multi-stream-object-detector` is
fast **because it never pulls frames to the host** — it pulls only a small BBOX payload and sends
clean video in-graph, letting Insight draw the overlay from `MetadataSender` JSON. The moment you
need the overlay **burned into the stream**, you must bring the frame back to the CPU. Adding a
full-frame `Output` node to the branch (a 1.4 MB NV12 copy per frame) collapsed throughput to
**1-3 fps**, measured four ways. Neat's proper answer for in-graph burned-in overlay is
`nodes.sima_render()`, which is not wired here yet.

**`RealtimeLatestByStream` breaks `combine(ByFrame)`.** It drops each combine leg *independently*, so
the legs' frame IDs diverge and the join almost never matches. The official example never hits this —
its `combine` is only on an occasional debug-save path, never the hot path.

**Known defect:** the process can abort at **exit** with `malloc(): mismatching next->prev_size`,
*after* the summary has printed. It is pre-existing — the original single-threaded version did it
too — and it does not affect the reported numbers, but it is still open.

References: a C++ 4-stream reference implementation (the thread topology this app copies),
`apps/single-stream-yolo-yolo11` (app conventions),
`core/tutorials/018_consume_rtsp_stream` (RTSP source fragment).

</details>
