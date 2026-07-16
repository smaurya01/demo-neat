# Single Stream YOLO11 Detection — Insight Metadata Output

## Introduction

This demo runs one RTSP stream through the SiMa Neat YOLO11 object detection model and publishes
**two separate streams**: the clean decoded video as H.264/RTP over UDP via `VideoSender`, and the
decoded detections as JSON over UDP via `MetadataSender`. **No boxes are drawn on the frames** —
the viewer (Neat Insight) overlays them client-side from the metadata.

This is the metadata-overlay counterpart of
[`single-stream-yolo-yolo11`](../single-stream-yolo-yolo11/README.md), which burns the boxes into
the pixels on the CPU. Keeping the frames untouched has two consequences worth understanding:

- **The video leg never leaves the graph.** Decode → encode happens entirely in-graph
  (`RtspDecodedInput` → branch → `VideoSender`), so no NV12 frame is ever copied to the CPU. The
  Python loop only handles the tiny detection tensors.
- **Overlay follows the metadata, not the pixels.** Boxes can be restyled, filtered, or hidden in
  the viewer without re-encoding video, and downstream consumers get machine-readable detections
  instead of burned-in pixels.

The pipeline shape (one graph, `branch` to video + model, detections pulled from a named output,
metadata sent from the app loop) follows the upstream example
[`single-stream-object-detector`](https://github.com/sima-neat/apps/tree/main/examples/object-detection/single-stream-object-detector).

## About Project

- Application: `single_stream_yolo_insight`
- Model: `yolo_11n_mpk.tar.gz`
- Input: RTSP H.264 stream
- Output 1: UDP/RTP H.264 **clean** video on `video_port_base + channel` (default 9000)
- Output 2: UDP JSON `object-detection` metadata on `metadata_port_base + channel` (default 9100)
- Runtime config: `./config/default.conf`

```text
RtspDecodedInput ── branch ──> VideoSender ──────────> Insight video (ch N)
      (NV12)          └──────> Model ──> "detections" output
                                              │
                              app loop: decode_bbox ──> MetadataSender ──> Insight metadata (ch N)
```

## Requirements

Run build commands from the Modalix SDK/eLxr environment where the Modalix SDK sysroot
and `dk` are available. Run the final binary or Python script on the DevKit with `dk`.

The receiving host should run **Neat Insight** (bundled with the NEAT Development Environment) —
its Video Viewer pairs the video and metadata ports by channel and draws the boxes. See
[`installation/neat_insight.md`](../../installation/neat_insight.md).

Run the commands below from this app folder:

```bash
cd /path/to/demo-neat/apps/single-stream-yolo-insight
```

## Model Download Command

YOLO11 is published in the SiMa model zoo, so just download it:

```bash
mkdir -p ./assets/models
cd ./assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_11n
cd ../..
```

Expected model path (`assets/models/` is git-ignored):

```text
./assets/models/yolo_11n_mpk.tar.gz
```

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
rtsp_url=rtsp://<rtsp-server-ip>:8555/stream
model_path=./assets/models/yolo_11n_mpk.tar.gz
insight_host=<insight-host-ip>
channel=0
```

Do not assume the default Insight ports — read the real ones from `neat --json` on the Insight
host and set `video_port_base` / `metadata_port_base` if they differ.

For a bounded smoke test, set `frames=30`.

## Config Parameters

`rtsp_url`: RTSP H.264 input stream consumed by the source graph.

`rtsp_transport`: RTSP transport mode. Use `tcp` for reliability or `udp` for lower latency.

`insight_host`: Host/IP running Neat Insight; receives both video and metadata streams.

`channel`: Insight channel. `video_port = video_port_base + channel` and
`metadata_port = metadata_port_base + channel`. Video and metadata **must share the channel**
or overlays land on the wrong viewer tile.

`video_port_base`: UDP video port base used by the H.264 video sender (Insight default 9000).

`metadata_port_base`: UDP metadata port base used by the metadata sender (Insight default 9100).

`model_path`: Model archive loaded by the Neat model node.

`model_width` / `model_height`: Model input size used by Neat preprocessing.

`fallback_width` / `fallback_height` / `fallback_fps`: Decoded frame geometry used when RTSP caps
are incomplete. The C++ demo also sizes the encoder contract from these, so set them to match
your stream.

`latency_ms`: RTSP receiver latency buffer in milliseconds.

`score_threshold`: Minimum decoded detection score to publish as metadata.

`nms_iou`: NMS IoU threshold used by Neat box decode.

`top_k`: Maximum decoded detections per frame.

`num_classes`: Number of detection classes in the model output.

`frames`: Number of frames to process. Use `0` to run until interrupted.

`bitrate_kbps`: H.264 output encoder bitrate in kbps.

`print_backend`: Print the generated backend pipeline when set to `true`.

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
command-line flags.

```bash
dk ./build/single_stream_yolo_insight
```

Both variants print per-window stats:

```text
frame=<n> detections=<decoded> published=<sent as metadata> fps=<output fps> avg_ms(pull=..., metadata_send=...)
```

`pull` is the wait for the next detection sample (dominated by the source frame period on a
healthy pipeline); `metadata_send` should be well under a millisecond — it is one UDP datagram.

## How To Run With Python

Run the Python version on the DevKit from the SDK shell:

```bash
dk ./main.py --config ./config/default.conf
```

Bounded smoke test:

```bash
dk ./main.py --config ./config/default.conf --frames 30
```

## How To See The Output

Open Neat Insight on the receiving host (`https://localhost:9900`) → **Video Viewer** → select the
channel configured as `channel` (default 0). The viewer decodes the video stream and draws the
boxes from the `object-detection` metadata arriving on the paired metadata port.

Verify delivery in order (both are UDP — fire-and-forget):

1. `https://<insight-host>:9900` → Stats, or `GET /api/ingest/stats` — packets arriving on the
   video port (`packets_received`, `seen_sps`)?
2. `GET /api/egress/stats` — `frames_decoded` climbing?
3. Video Viewer tile for the channel — boxes appear over the live video.

To sanity-check the raw video leg without Insight (no boxes — the frames are clean by design):

```bash
gst-launch-1.0 -v udpsrc port=9000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

Expected output: live video in the Insight viewer with YOLO11 boxes drawn from metadata. If you
see video but no boxes, video and metadata are usually on mismatched channels, or every detection
is below `score_threshold`.

---

# Appendix

## Appendix: Metadata Wire Format

Each frame sends one `send_metadata("object-detection", data_json, timestamp_ms, frame_id)` call.
An empty `{"objects":[]}` is sent when nothing is detected, so stale boxes never linger in the
viewer. `data_json` looks like:

```json
{
  "objects": [
    {
      "id": "obj_1",
      "label": "PERSON",
      "confidence": 0.91,
      "bbox": [412.0, 187.0, 164.0, 339.0]
    }
  ]
}
```

`bbox` is `[x, y, width, height]`, top-left origin, in encoded-frame pixels — note that Neat's
`decode_bbox` returns **corner** coordinates (`x1, y1, x2, y2`), so the app converts on every
frame. Insight renders the types `object-detection`, `classification`, `pose-estimation`,
`segmentation` and `tracking`; other types are delivered but not drawn.

## Appendix: Decode Family

The zoo `yolo_11n` archive exposes raw 64-channel DFL heads and decodes with
`BoxDecodeType.YoloV8`. A **self-compiled** YOLO11 archive (via
[`model-compilation/`](../../model-compilation/README.md)) folds the DFL into the model and emits
4-channel l/t/r/b heads — switch to `BoxDecodeType.YoloV26` in `make_model()`. Getting this wrong
still runs and still produces boxes; they are just decoded from the wrong channels.
