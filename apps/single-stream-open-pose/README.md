# Single Stream OpenPose

## Introduction

This demo runs one RTSP stream through the SiMa Neat OpenPose model, decodes heatmaps and PAFs,
assembles multi-person skeletons, and publishes one annotated H.264/RTP UDP stream.

## About Project

- Application: `single_stream_open_pose`
- Model: `open_pose_mpk.tar.gz`
- Input: RTSP H.264 stream
- Output: one UDP/RTP H.264 stream with skeleton overlay
- Runtime config: `./config/default.conf`

## Requirements

Run build commands from the Modalix SDK/eLxr environment where the Modalix SDK sysroot
and `dk` are available. Run the final binary on the DevKit with `dk`.


Run the commands below from this app folder:

```bash
cd /path/to/demo-neat/apps/single-stream-open-pose
```

## Model Download Command

Run this in the SDK shell:

```bash
mkdir -p ./assets/models
cd ./assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get open_pose
```

Expected model path:

```text
./assets/models/open_pose_mpk.tar.gz
```

If runtime reports `model file not found`, the file above is missing from this app folder on the
same `/workspace` path used by `dk`.

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
rtsp_url=rtsp://<rtsp-server-ip>:8555/stream
model_path=./assets/models/open_pose_mpk.tar.gz
udp_host=<host-ip-that-receives-video>
udp_port_base=5204
```

OpenPose uses packaged 480x480 model geometry internally. Download the model before running,
then keep `model_path` pointed at the archive.

## Config Parameters

`rtsp_url`: RTSP H.264 input stream consumed by the source graph.

`rtsp_transport`: RTSP transport mode. Use `tcp` for reliability or `udp` for lower latency.

`udp_host`: Host/IP that receives the annotated UDP/RTP output stream.

`udp_port_base`: UDP/RTP output port used by the H.264 video sender.

`model_path`: OpenPose model archive loaded by the Neat model node.

`fallback_width`: Fallback decoded frame width used when RTSP caps are incomplete.

`fallback_height`: Fallback decoded frame height used when RTSP caps are incomplete.

`fallback_fps`: Fallback decoded stream FPS used when RTSP caps are incomplete.

`latency_ms`: RTSP receiver latency buffer in milliseconds.

`frames`: Number of frames to process. Use `0` to run until interrupted.

`bitrate_kbps`: H.264 output encoder bitrate in kbps.

`print_backend`: Print generated backend pipelines when set to `true`.

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

Run on the DevKit from the SDK shell. This demo reads `./config/default.conf`; it does not use
`--config` or `--frames` command-line flags.

```bash
dk ./build/single_stream_open_pose
```

For a bounded smoke test, set `frames=30` in `./config/default.conf`, then run the same command.

## How To See The Output

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```


Run this on the host machine receiving UDP. Use the same port configured by `udp_port_base`.

```bash
gst-launch-1.0 -v udpsrc port=5204 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Expected output: live video with grouped OpenPose skeletons.

## TODO

- Add a sample input/output image after running the demo.
