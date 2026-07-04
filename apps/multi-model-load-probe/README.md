# Multi Model Load Probe

## Introduction

This demo loads multiple Neat model pipelines in one process, feeds all selected models from one
RTSP stream, overlays each model result on a separate NV12 output frame, and publishes one
H.264/RTP UDP stream per model.

## About Project

- Application: `multi_model_load_probe`
- Models: `yolo_v8n`, `yolo_v8n_seg`, `yolo26n`, and `open_pose`
- Input: one RTSP H.264 stream
- Output: one UDP/RTP H.264 stream per selected model
- Runtime config: `./config/default.conf`

Default output order when all four models run:

```text
udp_port_base + 0  yolov8n      detection boxes with COCO labels
udp_port_base + 1  yolov8n-seg  class-colored masks plus COCO labels
udp_port_base + 2  yolo26n      detection boxes with COCO labels
udp_port_base + 3  open_pose    grouped skeleton overlay
```

## Requirements

Run build commands from the Modalix SDK/eLxr environment where the Modalix SDK sysroot
and `dk` are available. Run the final binary on the DevKit with `dk`.

Run all commands below from this app folder:

```bash
cd /path/to/demo-neat/apps/multi-model-load-probe
```

## Model Download Commands

Run these in the SDK shell:

```bash
mkdir -p ./assets/models
(
  cd ./assets/models
  sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_v8n
  sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_v8n_seg
  sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz
  sima-cli modelzoo -v 2.1.2 --boardtype modalix get open_pose
)
```

Expected model files:

```text
./assets/models/yolo_v8n_mpk.tar.gz
./assets/models/yolo_v8n_seg_mpk.tar.gz
./assets/models/yolo26n-det-bf16-mla_tess-b1.tar.gz
./assets/models/open_pose_mpk.tar.gz
```

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
rtsp_url=rtsp://<rtsp-server-ip>:8555/stream
udp_host=<host-ip-that-receives-video>
udp_port_base=5201
models_dir=./assets/models
```

Relative paths such as `./assets/models` are intended to be used from this app folder.

To run one model from this app, set `only` in config or pass `--only`:

```text
only=yolov8n
```

Valid model names are:

```text
yolov8n
yolov8n-seg
yolo26n
open_pose
```

CLI flags override config values. For example:

```bash
--only yolo26n --score 0.30 --nms 0.50 --frames 30
```

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

Run all four models in one DevKit process:

```bash
dk ./build/multi_model_load_probe \
  --config ./config/default.conf
```

Bounded smoke test:

```bash
dk ./build/multi_model_load_probe \
  --config ./config/default.conf \
  --frames 30
```

Load/build graph test without RTSP:

```bash
dk ./build/multi_model_load_probe \
  --config ./config/default.conf \
  --load-only
```

Run one model only:

```bash
dk ./build/multi_model_load_probe \
  --config ./config/default.conf \
  --only yolov8n-seg \
  --udp-port-base 5202
```

## How To See The Output

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

Run one receiver per output on the host machine receiving UDP. If `udp_port_base=5201`, all four
receivers are:

```bash
gst-launch-1.0 -v udpsrc port=5201 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

```bash
gst-launch-1.0 -v udpsrc port=5202 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

```bash
gst-launch-1.0 -v udpsrc port=5203 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

```bash
gst-launch-1.0 -v udpsrc port=5204 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Expected mapping:

```text
5201 yolov8n
5202 yolov8n-seg
5203 yolo26n
5204 open_pose
```

## Notes

- Each model gets its own output frame and UDP stream. This app does not create a single combined
  mosaic frame.
- Overlay drawing is done on NV12 frames.
- A single multi-model process may run more models than separate processes because separate
  processes can consume all `/dev/rpmsg*` dispatcher channels independently.
- To inspect current DevKit rpmsg ownership:

```bash
dk shell 'for f in /tmp/rpmsg_lock_rpmsg*.owner; do [ -e "$f" ] || continue; echo "==== $f"; cat "$f"; done'
```
