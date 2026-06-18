# demo-neat

## Introduction

This repository contains SiMa Neat C++ demo applications for running RTSP video through compiled
Modalix model packages and streaming annotated output over H.264/RTP UDP.

## About Project

Each project is a standalone CMake application with its own README and `config/default.conf`.
Follow the README inside the project directory you want to run.

```text
single-stream-yolo-yolov8n   YOLOv8n detection
single-stream-yolo-yolov8m   YOLOv8m detection
single-stream-yolov8n-seg    YOLOv8n instance segmentation
single-stream-yolo26n        YOLO26n detection
single-stream-open-pose      OpenPose skeleton overlay
multi-model-load-probe       One process running selected models with separate UDP outputs
```

Each demo directory keeps only source, build instructions, config, and an empty model directory in
Git. Build outputs and downloaded model archives are intentionally ignored.

```text
<demo>/
  CMakeLists.txt
  main.cpp
  README.md
  config/default.conf
  assets/models/.gitkeep
```

## Requirements

- Modalix SDK/eLxr development environment
- SiMa Neat C++ SDK available through `/opt/toolchain/aarch64/modalix`
- `dk` configured for the DevKit
- RTSP H.264 input stream reachable from the DevKit
- GStreamer on the host machine for viewing UDP/RTP output

Install host viewer tools:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

## Typical Workflow

1. Open the demo folder README.
2. Download that demo's model into its `assets/models` directory.
3. Edit that demo's `config/default.conf` for RTSP URL, UDP receiver IP, ports, and thresholds.
4. Build from the SDK shell with CMake.
5. Run the ARM64 binary on the DevKit with `dk`.
6. View UDP output on the host with the README's `gst-launch-1.0` command.

## Build Artifacts And Models

The repository `.gitignore` excludes generated build files and large downloaded model artifacts:

```text
build/
CMakeFiles/
CMakeCache.txt
Makefile
*.o
*.tar.gz
*.mpk
*.onnx
*.log
*.pid
```

After cloning, create/download models by following each demo README. Do not expect model archives
or `build/` directories to come from Git.

## Project READMEs

- [single-stream-yolo-yolov8n](single-stream-yolo-yolov8n/README.md)
- [single-stream-yolo-yolov8m](single-stream-yolo-yolov8m/README.md)
- [single-stream-yolov8n-seg](single-stream-yolov8n-seg/README.md)
- [single-stream-yolo26n](single-stream-yolo26n/README.md)
- [single-stream-open-pose](single-stream-open-pose/README.md)
- [multi-model-load-probe](multi-model-load-probe/README.md)

## Runtime Config

Every app supports:

```bash
--config /path/to/config/default.conf
```

CLI options override config values, so you can keep a checked-in default and still run quick
experiments:

```bash
--frames 30 --score 0.30 --nms 0.50
```
