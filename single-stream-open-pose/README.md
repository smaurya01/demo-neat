# Single Stream OpenPose

## Introduction

This demo runs one RTSP stream through the SiMa Neat OpenPose model, decodes heatmaps and PAFs,
assembles multi-person skeletons, and publishes one annotated H.264/RTP UDP stream.

## About Project

- Application: `single_stream_open_pose`
- Model: `open_pose_mpk.tar.gz`
- Input: RTSP H.264 stream
- Output: one UDP/RTP H.264 stream with skeleton overlay
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
mkdir -p /workspace/demo-neat/single-stream-open-pose/assets/models
cd /workspace/demo-neat/single-stream-open-pose/assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get open_pose
```

Expected model path:

```text
/workspace/demo-neat/single-stream-open-pose/assets/models/open_pose_mpk.tar.gz
```

## Configure

Edit `config/default.conf` before running. At minimum, set:

```text
rtsp_url=rtsp://<rtsp-server-ip>:8555/stream
udp_host=<host-ip-that-receives-video>
udp_port_base=5204
only=open_pose
```

OpenPose uses packaged 480x480 model geometry internally; the shared `model_width` and
`model_height` config values are used by YOLO models and can remain at 640 here.

## How To Build

Run from the SDK shell:

```bash
cmake -S /workspace/demo-neat/single-stream-open-pose \
  -B /workspace/demo-neat/single-stream-open-pose/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/toolchain/aarch64/modalix/usr;/opt/toolchain/aarch64/modalix/usr/lib/cmake;/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/cmake"
cmake --build /workspace/demo-neat/single-stream-open-pose/build --parallel
```

## How To Run

Run on the DevKit from the SDK shell:

```bash
dk /workspace/demo-neat/single-stream-open-pose/build/single_stream_open_pose \
  --config /workspace/demo-neat/single-stream-open-pose/config/default.conf
```

Bounded smoke test:

```bash
dk /workspace/demo-neat/single-stream-open-pose/build/single_stream_open_pose \
  --config /workspace/demo-neat/single-stream-open-pose/config/default.conf \
  --frames 30
```

## How To See The Output

Run this on the host machine receiving UDP. Use the same port configured by `udp_port_base`.

```bash
gst-launch-1.0 -v udpsrc port=5204 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Expected output: live video with grouped OpenPose skeletons.
