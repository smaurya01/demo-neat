# demo-neat

## Introduction

This repository contains SiMa Neat tutorials, installation notes, and runnable demo applications
for Modalix DevKit workflows.

## Repository Layout

```text
apps/          Runnable Neat pipelines and benchmark utilities
tutorial/      Notebook tutorials for learning Neat concepts
installation/  SDK, DevKit, and Neat Insight setup notes
video/         Local media used by demos and tests
```

## Applications

Each app is standalone with its own README and runtime assets. Follow the README inside the app
directory you want to run.

```text
apps/benchmark                     Model benchmark utility
apps/single-stream-yolo-yolov8n    YOLOv8n detection
apps/single-stream-yolo-yolov8m    YOLOv8m detection
apps/single-stream-yolo-yolo11     YOLO11 detection
apps/single-stream-yolov8n-seg     YOLOv8n instance segmentation
apps/single-stream-yolo26n         YOLO26n detection
apps/single-stream-open-pose       OpenPose skeleton overlay
apps/multi-model-load-probe        One process running selected models with separate UDP outputs
apps/pcb-defect-detection-yolo26n  Image-folder PCB defect detection
```

Each app directory keeps its own source, run instructions, config, and assets. Build outputs and
downloaded model archives are intentionally ignored.

```text
apps/<app>/
  CMakeLists.txt        # C++ apps only
  main.cpp              # C++ apps only
  main.py               # Python apps/utilities only
  README.md
  config/default.conf   # RTSP/UDP/model settings where applicable
  assets/models/        # Downloaded model archives where applicable
```

## Requirements

- Modalix SDK/eLxr development environment
- SiMa Neat C++ SDK sysroot available at `/opt/toolchain/aarch64/modalix`
- `dk` configured for the DevKit
- RTSP H.264 input stream reachable from the DevKit
- GStreamer on the host machine for viewing UDP/RTP output

Install host viewer tools:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

## Typical Workflow

1. Open the app folder README.
2. Download that app's model into its `assets/models` directory.
3. Edit that app's `config/default.conf` for RTSP URL, UDP receiver IP, ports, and thresholds.
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

After cloning, create/download models by following each app README. Do not expect model archives
or `build/` directories to come from Git.

## Project READMEs

- [benchmark](apps/benchmark/README.md)
- [single-stream-yolo-yolov8n](apps/single-stream-yolo-yolov8n/README.md)
- [single-stream-yolo-yolov8m](apps/single-stream-yolo-yolov8m/README.md)
- [single-stream-yolo-yolo11](apps/single-stream-yolo-yolo11/README.md)
- [single-stream-yolov8n-seg](apps/single-stream-yolov8n-seg/README.md)
- [single-stream-yolo26n](apps/single-stream-yolo26n/README.md)
- [single-stream-open-pose](apps/single-stream-open-pose/README.md)
- [multi-model-load-probe](apps/multi-model-load-probe/README.md)
- [pcb-defect-detection-yolo26n](apps/pcb-defect-detection-yolo26n/README.md)
- [tutorial notebooks](tutorial/README.md)
- [installation notes](installation/README.md)

## Runtime Config

The streaming demos read their local config file from the app folder:

```bash
./config/default.conf
```

Edit values like `rtsp_url`, `model_path`, `models_dir`, `udp_host`, `udp_port_base`, and `frames`
in `./config/default.conf`, then run the documented `dk ./build/...` or `dk ./main.py` command from
the app folder. Some utilities, such as benchmark and image-folder apps, expose additional command
line flags; follow each app README for the exact command.
