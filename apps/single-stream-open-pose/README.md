# Single Stream OpenPose (TODO:: Stable Output)

## Table of Contents

- [Introduction](#introduction)
- [About Project](#about-project)
- [Requirements](#requirements)
- [Model Download Command](#model-download-command)
- [Configure](#configure)
- [Config Parameters](#config-parameters)
- [How To Build](#how-to-build)
- [How To Run](#how-to-run)
- [How To See The Output](#how-to-see-the-output)
- [TODO](#todo)

---

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
rtsp_url=<rtsp-url>
model_path=./assets/models/open_pose_mpk.tar.gz
udp_host=<host-ip>
udp_port_base=5204
```

OpenPose uses packaged 480x480 model geometry internally. Download the model before running,
then keep `model_path` pointed at the archive.

<details>
<summary><h2>Config Parameters</h2></summary>

<br>

`rtsp_url`: RTSP H.264 input stream consumed by the source graph.

`rtsp_transport`: RTSP transport mode. Use `tcp` for reliability or `udp` for lower latency.

`udp_host`: Host/IP that receives the annotated UDP/RTP output stream.

`udp_port_base`: UDP/RTP output port used by the H.264 video sender.

`model_path`: OpenPose model archive loaded by the Neat model node.

`width`: Fallback decoded frame width used when RTSP caps are incomplete.

`height`: Fallback decoded frame height used when RTSP caps are incomplete.

`fps`: Fallback decoded stream FPS used when RTSP caps are incomplete.

`latency_ms`: RTSP receiver latency buffer in milliseconds.

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

Run on the DevKit from the SDK shell. This demo reads `./config/default.conf`; it does not use
`--config` or `--frames` command-line flags.

```bash
dk ./build/single_stream_open_pose
```

For a bounded smoke test, set `frames=30` in `./config/default.conf`, then run the same command.

## How To See The Output

### Neat Insight (recommended)

**Neat Insight** decodes and displays the stream in a browser — nothing to install on your machine,
and it works from any device that can reach the host.

1. Open **`https://192.168.131.12:9900`** in a browser.
   *It is **HTTPS**, not HTTP. The SDK uses a local mkcert certificate, so accept the browser
   warning the first time.* Replace the IP with your own host if Insight runs elsewhere.
2. Go to the **Video Viewer** tab.
3. Set **Port** to **9000** — the same value as `udp_port_base` in `./config/default.conf`.
   Insight ingests video on UDP `9000 + channel`, so channel 0 is port `9000`.

Make sure `udp_host` in `./config/default.conf` points at the machine running Insight — that is
where the app sends the RTP stream. Insight in the SDK exposes **4 video channels (ports
9000-9003)**; if the defaults are already taken, read the real ports from `neat --json`
(`exposedPorts[*].hostPortStart`) rather than assuming.

### gst-launch (alternative, no Insight needed)

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

Run this on the machine at `udp_host`:

```bash
gst-launch-1.0 -v udpsrc port=9000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

> **Not on the DevKit.** There is no `avdec_h264` on the board — run this on your desktop, not
> over SSH.

<details>
<summary><h2>TODO</h2></summary>

<br>

- Add a sample input/output image after running the demo.

</details>
