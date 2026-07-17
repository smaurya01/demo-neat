# Single Stream YOLO26n Detection

## Table of Contents

- [Introduction](#introduction)
- [About Project](#about-project)
- [Requirements](#requirements)
- [Model Download Command](#model-download-command)
- [Configure](#configure)
- [Config Parameters](#config-parameters)
- [How To Build](#how-to-build)
- [How To Run](#how-to-run)
- [How To Run With Python](#how-to-run-with-python)
- [How To See The Output](#how-to-see-the-output)

---

## Introduction

This demo runs one RTSP stream through the SiMa YOLO26n detection model, draws labeled boxes, and
publishes one annotated H.264/RTP UDP stream.

## About Project

- Application: `single_stream_yolo26n`
- Model: `yolo26n-det-bf16-mla_tess-b1.tar.gz`
- Input: RTSP H.264 stream
- Output: one UDP/RTP H.264 stream with labeled boxes
- Runtime config: `./config/default.conf`

## Requirements

Run build commands from the Modalix SDK/eLxr environment where the Modalix SDK sysroot
and `dk` are available. Run the final binary on the DevKit with `dk`.


Run the commands below from this app folder:

```bash
cd /path/to/demo-neat/apps/single-stream-yolo26n
```

## Model Download Command

Run this in the SDK shell:

```bash
mkdir -p ./assets/models
cd ./assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz
```

Expected model path:

```text
./assets/models/yolo26n-det-bf16-mla_tess-b1.tar.gz
```

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
rtsp_url=rtsp://<rtsp-server-ip>:8555/stream
model_path=./assets/models/yolo26n-det-bf16-mla_tess-b1.tar.gz
udp_host=<host-ip-that-receives-video>
udp_port=5203
```

For a bounded C++ smoke test, set `frames=30` in `./config/default.conf`.

<details>
<summary><h2>Config Parameters</h2></summary>

<br>

`rtsp_url`: RTSP H.264 input stream consumed by the source graph.

`rtsp_transport`: RTSP transport mode. Use `tcp` for reliability or `udp` for lower latency.

`udp_host`: Host/IP that receives the annotated UDP/RTP output stream.

`udp_port`: UDP/RTP output port used by the H.264 video sender.

`model_path`: Model archive loaded by the Neat model node.

`model_width`: Model input width used by Neat preprocessing.

`model_height`: Model input height used by Neat preprocessing.

`fallback_width`: Fallback decoded frame width used when RTSP caps are incomplete.

`fallback_height`: Fallback decoded frame height used when RTSP caps are incomplete.

`fallback_fps`: Fallback decoded stream FPS used when RTSP caps are incomplete.

`latency_ms`: RTSP receiver latency buffer in milliseconds.

`score_threshold`: Detection score threshold used by YOLO26 box decode.

`nms_iou`: NMS IoU threshold used by Neat decode.

`top_k`: Maximum decoded detections or instances per frame.

`num_classes`: Number of classes in the model output.

`frames`: Number of frames to process. Use `0` to run until interrupted.

`bitrate_kbps`: H.264 output encoder bitrate in kbps.

`print_backend`: Print generated backend pipelines when set to `true`.

</details>

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

Run on the DevKit from the SDK shell. The C++ demo reads `./config/default.conf`; it does not use
`--config` or `--frames` command-line flags.

```bash
dk ./build/single_stream_yolo26n
```

For a bounded C++ smoke test, set `frames=30` in `./config/default.conf`, then run the same command.

## How To Run With Python

Run the Python version on the DevKit from the SDK shell:

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

## How To See The Output

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```


Run this on the host machine receiving UDP. Use the same port configured by `udp_port_base`.

```bash
gst-launch-1.0 -v udpsrc port=5203 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Expected output: live video with YOLO26n labeled detection boxes.
