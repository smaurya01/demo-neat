# Single Stream YOLOv8n Detection

## Introduction

This demo runs one RTSP stream through the SiMa Neat YOLOv8n object detection model, draws decoded
detections on NV12 frames, and publishes the annotated video as H.264/RTP over UDP.

## About Project

- Application: `single_stream_yolo_yolov8n`
- Model: `yolo_v8n_mpk.tar.gz`
- Input: RTSP H.264 stream
- Output: one UDP/RTP H.264 stream with bounding boxes
- Runtime config: `config/default.conf`

## Requirements

Run build commands from the Modalix SDK/eLxr environment where `/opt/toolchain/aarch64/modalix`
and `dk` are available. Run the final binary on the DevKit with `dk`.

Host tools for viewing output:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

## Model Download Command

Run this in the SDK shell:

```bash
mkdir -p /workspace/demo-neat/single-stream-yolo-yolov8n/assets/models
cd /workspace/demo-neat/single-stream-yolo-yolov8n/assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_v8n
```

Expected model path:

```text
/workspace/demo-neat/single-stream-yolo-yolov8n/assets/models/yolo_v8n_mpk.tar.gz
```

## Configure

Edit `config/default.conf` before running. At minimum, set:

```text
rtsp_url=rtsp://<rtsp-server-ip>:8555/stream
udp_host=<host-ip-that-receives-video>
udp_port=5201
```

CLI flags override config values. For example:

```bash
--score 0.30 --nms 0.50 --frames 30
```

## How To Build

Run from the SDK shell:

```bash
cmake -S /workspace/demo-neat/single-stream-yolo-yolov8n \
  -B /workspace/demo-neat/single-stream-yolo-yolov8n/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/toolchain/aarch64/modalix/usr;/opt/toolchain/aarch64/modalix/usr/lib/cmake;/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/cmake"
cmake --build /workspace/demo-neat/single-stream-yolo-yolov8n/build --parallel
```

## How To Run

Run on the DevKit from the SDK shell:

```bash
dk /workspace/demo-neat/single-stream-yolo-yolov8n/build/single_stream_yolo_yolov8n \
  --config /workspace/demo-neat/single-stream-yolo-yolov8n/config/default.conf
```

Bounded smoke test:

```bash
dk /workspace/demo-neat/single-stream-yolo-yolov8n/build/single_stream_yolo_yolov8n \
  --config /workspace/demo-neat/single-stream-yolo-yolov8n/config/default.conf \
  --frames 30
```

## How To See The Output

Run this on the host machine receiving UDP. Use the same port configured by `udp_port`.

```bash
gst-launch-1.0 -v udpsrc port=5201 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Expected output: live video with YOLOv8n detection boxes.
