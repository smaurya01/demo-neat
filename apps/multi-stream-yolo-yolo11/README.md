# Multi-Stream YOLO11 Detection (2x RTSP)

## Introduction

This demo runs **two** RTSP streams through **one shared** SiMa Neat YOLO11 object
detection stage, draws decoded detections on each stream's NV12 frames, and
publishes one annotated H.264/RTP UDP output per input stream. Stream identity is
preserved end to end: each stream owns its own RTSP source, its own UDP output
port, and an on-frame `STREAM <id>` banner.

It is the two-stream sibling of `apps/single-stream-yolo-yolo11` and reuses the
same NV12 shuttle conventions.

## About Project

- Application: `multi-stream-yolo-yolo11` (`main.py`)
- Model: `yolo_11n_mpk.tar.gz` (shared by both streams)
- Input: 2x RTSP H.264 streams
- Output: 2x UDP/RTP H.264 streams with bounding boxes (one port per stream)
- Runtime config: `./config/default.conf`

## Pipeline Shape

```
RTSP stream 0 ──> source graph 0 ─┐
                                  ├─> SHARED YOLO11 model stage ─> Neat box decode ─┬─> annotate ─> video sender 0 ─> udp:PORT
RTSP stream 1 ──> source graph 1 ─┘   (one compiled archive, one Run handle)        └─> annotate ─> video sender 1 ─> udp:PORT+stride
```

- The shared model stage needs one input geometry, so both streams must decode at
  the same resolution. Since both default to the same source this holds; if you
  point the two inputs at cameras of different resolutions, run two model stages
  instead (see "Different-resolution inputs" below).

### Threading: why this app sustains the full 60 fps source rate

Both streams sustain the **full source rate with zero dropped frames**. The thread topology is
ported from the proven C++ demo `neat_demo_elxr/demo-yolo-4-stream` (4 streams x 60 fps) — it is not
a design invented here:

```
source thread per stream -> input queue [4]
   -> ONE round-robin PUSHER thread -> shared model Run -> ONE PULLER thread
   -> result queue [4] -> output worker per stream (box decode + overlay + UDP push)
```

Two settings do all the work, and both are counter-intuitive:

**1. The shared model Run uses `RunPreset.Reliable` + `OverflowPolicy.Block` + `queue_depth=4`** —
NOT `Realtime` + `KeepLatest`, which looks like the obvious choice for a live camera and is wrong
here.

* `Block` keeps push and pull **strictly paired**, so a plain FIFO deque of `(stream, frame)` is
  enough to route each result back to its own stream. `KeepLatest` silently drops results and
  destroys that pairing.
* `Block` + `queue_depth` is also what **pipelines the MLA**: 4 frames stay in flight instead of the
  accelerator waiting on the host every frame.
* Frames are dropped at the **source queue** instead (drop-oldest, since a live camera does not
  wait). Drop at the source, never at the model.

**2. The pusher is ROUND-ROBIN over the per-stream input queues.** That is the only thing providing
fairness — without it one stream can monopolise the shared model.

Measured against a 1280x720 H.264 @ 59.94 fps source:

| | fps/stream | aggregate | dropped |
| --- | --- | --- | --- |
| before (single-threaded round-robin) | 32.5 | 65 | 0 |
| **now** | **58.6 - 61.2** (3 runs) | **118 - 122** | **0** |

The old loop ran every stage of every stream on one thread and left the MLA idle ~60% of the time
(only 6.1 of 14.8 ms/frame was the model; the rest was host marshalling, overlay and encode push).

> **Do not "modernise" this into an in-graph `graphs.branch` / `combine` pipeline.** That was tried
> and rejected on measurement. The official example
> `apps/examples/object-detection/multi-stream-object-detector` is fast **because it never pulls
> frames to the host** — it pulls only a small BBOX payload and sends clean video in-graph, letting
> Insight draw the overlay from `MetadataSender` JSON. The moment you need the overlay **burned into
> the stream**, you must bring the frame back to the CPU; adding a full-frame `Output` node to the
> branch (a 1.4 MB NV12 copy per frame) collapsed throughput to **1-3 fps**. Neat's proper answer for
> in-graph burned-in overlay is `nodes.sima_render()`, which is not wired here yet.

References: `neat_demo_elxr/demo-yolo-4-stream/main.cpp` (the thread topology this app copies),
`apps/single-stream-yolo-yolo11/main.py` (app conventions),
`core/tutorials/018_consume_rtsp_stream` (RTSP source fragment).

## Requirements

Run on the DevKit. `pyneat` (0.3.0+) must be importable there. `/workspace` is
NFS-mounted on the board at the same path, so write host-side and run board-side —
no copying.

```bash
cd /path/to/demo-neat/apps/multi-stream-yolo-yolo11
```

## Model Download / Build

> **There is no prebuilt YOLO11n in the model zoo.** The SDK 2.1.2 Modalix zoo exposes only
> `yolo_v8n`, `yolo_v8n_seg`, and `open_pose` — no yolo11 variant. (As of 2026-07-09 the zoo
> metadata URL also 302-redirects to `auth.sima.ai`, so it can no longer be listed anonymously.)
> A `sima-cli modelzoo ... get yolo_11n` command will not give you this model. Compile it yourself.

Compile it with the graph-surgery flow (this is the flow T1 verifies). See
`../../model-compilation/README.md` for the full walkthrough. The compiled archive is:

```text
../../model-compilation/work/yolo11n/<...>/compile_int8/.../yolo11n.compile_ready_mpk.tar.gz
```

Copy or symlink it to `./assets/models/yolo_11n_mpk.tar.gz`, or point
`model_path` at it directly. `./assets/models/` is git-ignored.

Expected model path:

```text
./assets/models/yolo_11n_mpk.tar.gz
```

## Configure

Edit `./config/default.conf`. At minimum:

```text
rtsp_url_0=rtsp://<rtsp-server-ip>:8555/stream
rtsp_url_1=rtsp://<rtsp-server-ip>:8555/stream
model_path=./assets/models/yolo_11n_mpk.tar.gz
udp_host=<host-ip-that-receives-video>
udp_port_base=5206
udp_port_stride=2
```

With those defaults stream 0 publishes on `5206` and stream 1 on `5208`.

## Config Parameters

- `rtsp_url_0`, `rtsp_url_1`: the two RTSP H.264 input streams. Both default to
  the same source by design; set them to distinct cameras when you have them.
- `rtsp_transport`: `tcp` (reliable) or `udp` (lower latency).
- `udp_host`: host/IP that receives both annotated UDP/RTP outputs.
- `udp_port_base`: UDP/RTP port for stream 0.
- `udp_port_stride`: port spacing; stream `i` uses `udp_port_base + i*stride`.
- `model_path`: shared model archive loaded by the Neat model stage.
- `model_name`: decode family (`yolo11`/`yolo26n` => `BoxDecodeType.YoloV26`).
- `model_width`, `model_height`: model input size for Neat preprocessing.
- `fallback_width`/`fallback_height`/`fallback_fps`: used when RTSP caps are
  incomplete.
- `latency_ms`: RTSP receiver jitter buffer.
- `score_threshold`, `nms_iou`, `top_k`, `num_classes`: box-decode controls.
- `frames`: frames PER stream to process; `0` runs until interrupted.
- `bitrate_kbps`: H.264 output bitrate.
- `print_backend`: print generated GStreamer backends when `true`.

Every value is also overridable on the command line (`--rtsp0`, `--rtsp1`,
`--udp-port-base`, `--frames`, ...); run `python main.py --help`.

## How To Run (human UX)

On the DevKit, from a real terminal, use the `dk` helper (source the helper once:
`source /usr/local/bin/devkit.sh 192.168.2.103 sima 22`):

```bash
dk ./main.py --config ./config/default.conf
```

Bounded smoke test (30 frames per stream):

```bash
dk ./main.py --config ./config/default.conf --frames 30
```

## How To Run (CI / automation fallback)

`dk` needs a TTY and hangs in non-interactive/agent contexts. For CI use
passwordless ssh. `/workspace` is NFS-mounted on the DevKit, so ssh runs the very
same on-disk file — no copying.

Put the timeout on the **board side**: `timeout ... ssh` only kills the local ssh
client and leaves the remote python running, holding `/dev/rpmsg*` channels, which
makes the *next* run fail with `neatdecoder ... Input buffer allocation failed`.

```bash
APP=/workspace/demo-neat/apps/multi-stream-yolo-yolo11

ssh sima@192.168.2.103 "cd $APP && timeout -s INT 120 /home/sima/pyneat/bin/python -u main.py \
    --rtsp0 rtsp://192.168.2.105:8555/stream \
    --rtsp1 rtsp://192.168.2.105:8555/stream \
    --udp-host 192.168.2.105 --udp-port-base 5206 \
    --frames 500 --warmup-frames 80"
```

## Verify It Hits 60 FPS

Progress lines (every 60 frames) prove identity is preserved per output — `stream=`
and `port=` always travel together:

```text
stream=0 port=5206 frame=420 detections=11 visible=11 dropped=0
stream=1 port=5208 frame=420 detections=10 visible=10 dropped=0
```

At exit it prints the steady-state summary — this is the number to read:

```text
=== steady-state ===
  stream 0: delivered  59.31 fps  (499 frames, 0 dropped)
  stream 1: delivered  59.91 fps  (504 frames, 0 dropped)
aggregate: 119.22 fps across 2 streams in 8.4s
```

**How to read it.** The source is 1280x720 H.264 @ **59.94 fps**, so ~59-60 fps per
stream with `dropped=0` means the pipeline is keeping up with the camera exactly —
it is **source-limited, not compute-limited**, which is the goal state. You cannot
exceed the source rate; if you see a number above it, the measurement window was too
short (queue drain at the boundary), not real throughput.

Check the source rate first, so you know what you are aiming at:

```bash
ffprobe -hide_banner -rtsp_transport tcp rtsp://192.168.2.105:8555/stream
```

`--warmup-frames N` excludes the first N frames per stream (graph build, model load,
RTSP jitter-buffer fill) from the reported FPS. `--model-queue-depth` sets how many
frames the shared model Run keeps in flight (default 4).

> Known: the process can abort at **exit** with `malloc(): mismatching next->prev_size`
> after the summary has printed. Pre-existing (the original single-threaded version did
> it too) and it does not affect the numbers above, but it is still open.

## How To See The Output

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

Open one viewer per stream on the host that receives UDP (match `udp_port_base`
and `udp_port_stride`):

```bash
# Stream 0
gst-launch-1.0 -v udpsrc port=5206 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
  ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false

# Stream 1
gst-launch-1.0 -v udpsrc port=5208 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
  ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

`ffplay` alternative:

```bash
ffplay -fflags nobuffer -flags low_delay -rtsp_transport udp \
  -i "udp://@:5206"   # and :5208 for stream 1
```

Expected output: two live windows, each with YOLO11 detection boxes and a
`STREAM 0` / `STREAM 1` banner burned into the top-left, confirming per-stream
identity.

## Different-Resolution Inputs

The shared model stage assumes both inputs decode at the same size. If your two
cameras differ, the app raises a clear error. To support mixed resolutions, build
one model graph per stream (call `build_model_graph` per context) instead of one
shared handle — the rest of the loop is unchanged.
