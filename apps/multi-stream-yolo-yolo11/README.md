# Multi Stream YOLO11 Detection (2x RTSP)

## Table of Contents

- [Introduction](#introduction)
- [About Project](#about-project)
- [Requirements](#requirements)
- [Model Download Command](#model-download-command)
- [Configure](#configure)
- [Config Parameters](#config-parameters)
- [How To Run](#how-to-run)
- [How To See The Output](#how-to-see-the-output)
- [Appendix](#appendix)
- [Appendix: Model Build](#appendix-model-build)
- [Appendix: Pipeline Shape](#appendix-pipeline-shape)
- [Appendix: Threading — how it sustains 60 fps per stream](#appendix-threading--how-it-sustains-60-fps-per-stream)
- [Appendix: Reading The Time Profile](#appendix-reading-the-time-profile)
- [Appendix: Learnings](#appendix-learnings)

---

## Introduction

This demo runs two RTSP streams through one shared SiMa Neat YOLO11 object detection model, draws
decoded detections on NV12 frames, and publishes one annotated H.264/RTP UDP stream per input.

Both streams sustain the full 60 fps source rate with zero dropped frames. Stream identity is
preserved end to end: each stream owns its RTSP source, its UDP port, and an on-frame `STREAM <id>`
banner.

## About Project

- Application: `multi_stream_yolo_yolo11` (`main.py`, Python)
- Model: `yolo_11n_mpk.tar.gz` (one archive, shared by both streams)
- Input: 2x RTSP H.264 streams
- Output: 2x UDP/RTP H.264 streams with bounding boxes, one port per stream
- Runtime config: `./config/default.conf`

## Requirements

Run on the DevKit with `dk`. `pyneat` must be importable there. `/workspace` is NFS-mounted on the
board at the same path, so edit host-side and run board-side — no copying.

Run the commands below from this app folder:

```bash
cd /path/to/demo-neat/apps/multi-stream-yolo-yolo11
```

## Model Download Command

YOLO11 is published in the SiMa model zoo, so just download it:

```bash
mkdir -p ./assets/models
cd ./assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_11n
cd ../..
```

The zoo asset is named `yolo_11n_mpk.tar.gz`, which is exactly the path `./config/default.conf`
already expects — no config edit needed.

Expected model path:

```text
./assets/models/yolo_11n_mpk.tar.gz
```

The zoo publishes the whole detection family (`yolo_11n`, `yolo_11s`, `yolo_11m`, `yolo_11l`,
`yolo_11x`). To run a different size, `get` that name and point `model_path` at it. Both streams
share this one archive.

Only compile YOLO11 yourself for a variant the zoo does not publish — and note that a self-compiled
archive needs a different `model_name`. See [Appendix: Model Build](#appendix-model-build).

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
rtsp_url_0=<rtsp-url>
rtsp_url_1=<rtsp-url>
model_path=./assets/models/yolo_11n_mpk.tar.gz
model_name=yolov8
udp_host=<host-ip>
udp_port_base=9000
udp_port_stride=1
```

Leave `model_name=yolov8` alone when running the zoo archive — despite the name, it is the decode
family, and it is the right one for zoo YOLO11. See `model_name` under
[Config Parameters](#config-parameters).

With those defaults stream 0 publishes on `9000` and stream 1 on `9001`, Insight viewer channels 0
and 1.

For a bounded smoke test, set `frames=30` in `./config/default.conf`.

<details>
<summary><h2>Config Parameters</h2></summary>

<br>

`rtsp_url_0`, `rtsp_url_1`: The two RTSP H.264 input streams. Both default to the same source; set
them to distinct cameras when you have them.

`rtsp_transport`: RTSP transport mode. Use `tcp` for reliability or `udp` for lower latency.

`udp_host`: Host/IP that receives both annotated UDP/RTP output streams.

`udp_port_base`: UDP/RTP output port for stream 0.

`udp_port_stride`: Port spacing. Stream `i` publishes on `udp_port_base + i * udp_port_stride`.

`model_path`: Shared model archive loaded by the Neat model stage.

`model_name`: Decode family, chosen by the shape of the archive's detection head — **not** by the
model's version number. `yolov8` selects `BoxDecodeType.YoloV8` (raw 64-channel DFL bbox heads),
which is what the zoo `yolo_11n` archive ships, so it is the default. `yolo11` / `yolo26n` select
`BoxDecodeType.YoloV26` (4-channel l/t/r/b distance heads), which is only correct for a
self-compiled archive. Setting this wrong still runs, but decodes boxes from the wrong channels.

`model_width`: Model input width used by Neat preprocessing.

`model_height`: Model input height used by Neat preprocessing.

`width`: Fallback decoded frame width used when RTSP caps are incomplete.

`height`: Fallback decoded frame height used when RTSP caps are incomplete.

`fps`: Fallback decoded stream FPS used when RTSP caps are incomplete.

`latency_ms`: RTSP receiver latency buffer in milliseconds.

`score_threshold`: Detection score threshold used by YOLO box decode.

`nms_iou`: NMS IoU threshold used by Neat decode.

`top_k`: Maximum decoded detections per frame.

`num_classes`: Number of classes in the model output.

`frames`: Number of frames to process PER stream. Use `0` to run until interrupted.

`warmup_frames`: Frames per stream excluded from the reported FPS and stage means. Graph build,
model load and RTSP jitter-buffer fill all land on the first few frames.

`model_queue_depth`: Frames the shared model Run keeps in flight. This is what pipelines the MLA.

`stream_queue_depth`: Bounded hand-off depth per stream (input queue and result queue).

`bitrate_kbps`: H.264 output encoder bitrate in kbps.

`print_backend`: Print generated backend pipelines when set to `true`.

Every value is also overridable on the command line (`--rtsp0`, `--rtsp1`, `--udp-port-base`,
`--frames`, ...). Run `python main.py --help`.

</details>

## How To Run

Run on the DevKit from the SDK shell:

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

The app prints a per-stage time profile and per-stream FPS at exit:

```text
=== time profile (ms/frame, mean | p95) ===
stream frames           rtsp           prep          qwait           push          infer         decode        overlay           send        latency
     0    500  16.36|39.02     0.50|0.66      0.69|1.38      0.10|0.14      7.34|9.47      0.37|0.48      6.62|11.02     0.56|0.72     36.09|59.32
     1    507  16.10|39.25     0.51|0.70      0.71|1.62      0.10|0.15      7.39|9.68      0.36|0.43      6.43|10.66     0.54|0.66     35.67|57.91

=== fps ===
  stream 0: delivered  58.81 fps  (500 frames, 0 dropped)
  stream 1: delivered  59.63 fps  (507 frames, 0 dropped)
  aggregate:          118.44 fps across 2 streams in 8.5s
```

See [Appendix: Reading The Time Profile](#appendix-reading-the-time-profile) for what each column
means and which ones are easy to misread.

## How To See The Output

### Neat Insight (recommended)

**Neat Insight** decodes and displays the stream in a browser — nothing to install on your machine,
and it works from any device that can reach the host.

1. Open the Insight UI, **`https://<sdk-host-ip>:9900`** (`neat --json` shows it as
   `insight.webUiUrl`), in a browser. *It is **HTTPS**, not HTTP. The SDK uses a local mkcert
   certificate, so accept the browser warning the first time.*
2. Go to the **Video Viewer** tab.
3. This app publishes **2 streams**, so open one viewer channel per stream.
   `udp_port_base` sets the first port and each later stream takes the next one:

   | channel | port | stream |
   |---|---|---|
   | 0 | `9000` | stream 0 |
   | 1 | `9001` | stream 1 |

Make sure `udp_host` in `./config/default.conf` points at the machine running Insight — that is
where the app sends the RTP stream. The number of video channels is set by the SDK's port map;
read the range from `neat --json` (`exposedPorts`, `videoUDP`) rather than assuming.

</details>

<details>
<summary><h2>Appendix: Learnings</h2></summary>

<br>

**Do not "modernise" this into an in-graph `graphs.branch` / `combine` pipeline.** Tried and rejected
on measurement. The official example `apps/examples/object-detection/multi-stream-object-detector` is
fast **because it never pulls frames to the host** — it pulls only a small BBOX payload and sends
clean video in-graph, letting Insight draw the overlay from `MetadataSender` JSON. The moment you
need the overlay **burned into the stream**, you must bring the frame back to the CPU. Adding a
full-frame `Output` node to the branch (a 1.4 MB NV12 copy per frame) collapsed throughput to
**1-3 fps**, measured four ways. Neat's proper answer for in-graph burned-in overlay is
`nodes.sima_render()`, which is not wired here yet.

**`RealtimeLatestByStream` breaks `combine(ByFrame)`.** It drops each combine leg *independently*, so
the legs' frame IDs diverge and the join almost never matches. The official example never hits this —
its `combine` is only on an occasional debug-save path, never the hot path.

**Known defect:** the process can abort at **exit** with `malloc(): mismatching next->prev_size`,
*after* the summary has printed. It is pre-existing — the original single-threaded version did it
too — and it does not affect the reported numbers, but it is still open.

References: a C++ 4-stream reference implementation (the thread topology this app copies),
`apps/single-stream-yolo-yolo11` (app conventions),
`core/tutorials/018_consume_rtsp_stream` (RTSP source fragment).

</details>
