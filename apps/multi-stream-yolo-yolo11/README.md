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

- Each stream is serviced round-robin: pull one decoded frame, push it into the
  **shared** model `Run`, pull that stream's result, annotate, and publish. The
  bbox result always belongs to the frame just pushed, so identity is preserved.
- The shared model stage needs one input geometry, so both streams must decode at
  the same resolution. Since both default to the same source this holds; if you
  point the two inputs at cameras of different resolutions, run two model stages
  instead (see "Different-resolution inputs" below).

References: `apps/single-stream-yolo-yolo11/main.py` (app conventions),
`core/tutorials/018_consume_rtsp_stream` (RTSP source fragment),
`core/tutorials/015_run_multiple_streams` (multi-stream graph / `combine` join).

## Requirements

Run on the DevKit. `pyneat` (0.3.0+) must be importable there. `/workspace` is
NFS-mounted on the board at the same path, so write host-side and run board-side —
no copying.

```bash
cd /path/to/demo-neat/apps/multi-stream-yolo-yolo11
```

## Model Download / Build

Option A — download the prebuilt YOLO11n archive from the model zoo (SDK shell):

```bash
mkdir -p ./assets/models
cd ./assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_11n
```

Option B — compile it yourself with the graph-surgery flow (recommended, this is
the flow T1 verifies). See `../../model-compilation/README.md` for the full
walkthrough. The compiled archive is:

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
`source /usr/local/bin/devkit.sh 192.168.135.203 sima 22`):

```bash
dk ./main.py --config ./config/default.conf
```

Bounded smoke test (30 frames per stream):

```bash
dk ./main.py --config ./config/default.conf --frames 30
```

## How To Run (CI / automation fallback)

`dk` needs a TTY and hangs in non-interactive/agent contexts. For CI use
passwordless ssh (the sima-neat skill's documented fallback). `/workspace` is
NFS-mounted, so run the same on-disk file:

```bash
timeout 180 ssh -o BatchMode=yes sima@192.168.135.203 \
  'source $HOME/pyneat/bin/activate; \
   python /workspace/demo-neat/apps/multi-stream-yolo-yolo11/main.py \
     --config /workspace/demo-neat/apps/multi-stream-yolo-yolo11/config/default.conf \
     --frames 30'
```

Per-frame log lines look like:

```text
stream=0 port=5206 frame=1 detections=12 visible=7 agg_fps=8.30
stream=1 port=5208 frame=1 detections=12 visible=7 agg_fps=9.10
```

`stream=` and `port=` prove identity is preserved per output.

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
