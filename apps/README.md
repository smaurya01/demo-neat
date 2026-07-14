# `apps/` — the NEAT API, as these apps actually use it

A reference for every NEAT API used by the applications in this folder: what each one is for, its
important parameters, and which app to read for a worked example.

This is an **API map, not a tutorial**. For the concepts, work through
[`../tutorial/`](../tutorial/README.md). For running an app, see that app's own `README.md`.

← Back to the [repo README](../README.md)

---

## Contents

- [The three execution patterns](#the-three-execution-patterns) — pick one before you write anything
- [Graph & Run](#1-graph--run) · [Nodes](#2-nodes) · [Node groups](#3-node-groups)
- [Model](#4-model) · [Graph fragments](#5-graph-fragments-multi-branch-topology)
- [Tensor & Sample](#6-tensor--sample) · [Detection decode helpers](#7-detection-decode-helpers)
- [Senders](#8-senders-out-of-band-egress) · [GenAI](#9-genai-llm--vlm--asr)
- [Enums you will actually set](#10-enums-you-will-actually-set)
- [Which app uses what](#which-app-uses-what)
- [C++ ↔ Python naming](#c--python-naming)
- [Gotchas](#gotchas)

---

## Boilerplate

**C++** — one header, three namespaces:

```cpp
#include <neat.h>

namespace neat   = simaai::neat;             // Graph, Run, Model, Tensor, Sample, enums
namespace nodes  = simaai::neat::nodes;      // Input, Output, Custom
namespace groups = simaai::neat::nodes::groups;  // RtspDecodedInput, VideoSender
namespace graphs = simaai::neat::graphs;     // Branch, Combine
```

CMake: `find_package(SimaNeat REQUIRED CONFIG)` → `target_link_libraries(app PRIVATE SimaNeat::sima_neat)`

**Python** — one module:

```python
import pyneat        # pyneat.Graph, pyneat.nodes.*, pyneat.groups.*, pyneat.graphs.*, pyneat.genai.*
```

---

## The three execution patterns

Every app in this folder is one of these three. **Choosing the wrong one is the most expensive
mistake you can make**, so pick deliberately.

| Pattern | Use when | How you run it | Example app |
| --- | --- | --- | --- |
| **Graph pipeline** | You have a live source (RTSP / camera) and want streaming video out. **The default.** | `Graph` → `build(RunOptions)` → `Run` → `push()` / `pull()` | [`single-stream-yolo-yolo11`](single-stream-yolo-yolo11/README.md) |
| **`Model::Runner`** | You have images in hand (files, a folder) and want request→response. No video, no source. | `Model` → `model.build()` → `runner.run([tensor])` | [`pcb-defect-detection-yolo26n`](pcb-defect-detection-yolo26n/README.md) |
| **`Model::benchmark()`** | You only want numbers. Synthetic inputs, no source, no graph. | `Model` → `model.benchmark(n)` | [`benchmark`](benchmark/README.md) |

---

## 1. Graph & Run

`Graph` composes the pipeline. `build()` turns it into a `Run` — the live handle you push to and
pull from.

| API | Use | Important parameters |
| --- | --- | --- |
| `Graph("name")` <br> `pyneat.Graph("name")` | The pipeline container. The name shows up in diagnostics. | — |
| `graph.add(node)` | Append a node **linearly** — each `add` links to the previous one. | — |
| `graph.connect(from, to, opts)` | Wire two nodes/subgraphs explicitly. Needed for **non-linear** (branching) topologies. | `GraphLinkOptions.policy` — set `RealtimeLatestByStream` so a slow branch drops stale frames instead of back-pressuring the source. |
| `graph.build(RunOptions)` → `Run` | Compile and start the pipeline. | see `RunOptions` below |
| `graph.describe_backend()` | Print the generated GStreamer pipeline. **The single best debugging tool in NEAT** — it shows you what NEAT actually built, not what you think you asked for. | — |
| `graph.validate()` | Check caps/plugins **without running**. | returns a `GraphReport` |
| `run.push("port", TensorList{t})` | Feed a frame into an `Input` node. | port name must match `nodes::Input("port")` |
| `run.pull("port", timeout_ms, …)` | Read a result from an `Output` node. | **C++:** `pull(name, ms, sample, &err)` → `PullStatus`. **Python:** `pull(name, ms)` → `Optional[Sample]` (`None` = timeout/closed). |
| `run.close()` | Tear down. Always call it. | — |

### `RunOptions` — the four knobs that matter

| Field | Default | What it does |
| --- | --- | --- |
| `preset` | `Balanced` | `Realtime` (drop to stay current) vs `Reliable` (never drop). |
| `queue_depth` | `4` | Frames in flight. Too low starves the MLA; too high adds latency. |
| `overflow_policy` | — | `KeepLatest` = drop old frames (live video). `Block` = never drop, keeps push/pull strictly paired. |
| `output_memory` | — | `ZeroCopy` = a view into pipeline memory (fast). `Owned` = your own copy (safe to hold/mutate). |

> **Live streaming preset used by nearly every app here:**
> `preset=Realtime`, `overflow_policy=KeepLatest`, `queue_depth=3`, `output_memory=ZeroCopy`.
> The multi-stream apps deliberately use `Reliable` + `Block` instead — see
> [`multi-stream-yolo-yolo11`](multi-stream-yolo-yolo11/README.md) for why that is what gets 60 fps.

---

## 2. Nodes

| API | Use | Important parameters |
| --- | --- | --- |
| `nodes::Input("name", InputOptions)` <br> `pyneat.nodes.input(...)` | An **appsrc** — the port you `push()` frames into. | see `InputOptions` |
| `nodes::Output("name", OutputOptions)` <br> `pyneat.nodes.output(...)` | An **appsink** — the port you `pull()` results from. | see `OutputOptions` |
| `nodes::Custom(fragment, InputRole)` <br> `pyneat.nodes.custom(...)` | **The escape hatch.** Emit an arbitrary GStreamer fragment when NEAT has no node for what you need. | `InputRole::Source` if the fragment produces its own data (e.g. `v4l2src`). Used by [`usb-camera-yolo26m`](usb-camera-yolo26m/README.md), because NEAT has **no V4L2 source node**. |

### `InputOptions` — describing what you will push

| Field | What it does |
| --- | --- |
| `payload_type` | `PayloadType::Image` for frames. |
| `format` | `FormatTag::NV12` etc. |
| `width` / `height` / `depth` | The frame you push. |
| `max_width` / `max_height` / `max_depth` | Upper bounds for variable input. |
| `fps_n` / `fps_d` | Framerate as a fraction (`30/1`). |
| `caps_override` | Pin caps exactly, e.g. `"video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1"`. Disables renegotiation — use when the format is genuinely fixed. |
| `use_simaai_pool` | Allocate from the SiMa pool. |

### `OutputOptions` — three presets cover everything

| Preset | Behaviour | Use for |
| --- | --- | --- |
| `Latest()` | `max_buffers=1, drop=true` — always the freshest frame. | Live video, lowest latency. |
| `EveryFrame(n)` | Queue up to `n`, drop nothing. | Detections you must not miss. |
| `Clocked(n)` | `sync=true` — paced to the pipeline clock. | Playback-rate output. |

---

## 3. Node groups

Prebuilt multi-element fragments. **Prefer these over hand-rolling nodes** — they handle the caps
negotiation you would otherwise get wrong.

| API | Use | Important parameters |
| --- | --- | --- |
| `groups::RtspDecodedInput(opts)` <br> `pyneat.groups.rtsp_decoded_input(...)` | RTSP H.264 in → hardware-decoded **NV12** out. The input side of almost every app here. | `url`, `tcp` (true = reliable), `latency_ms` (jitter buffer, 200 typical), `out_format` (`NV12`), `auto_caps_from_stream`, `fallback_h264_width/height/fps` (used when the stream's caps are incomplete) |
| `groups::VideoSender(opts)` <br> `pyneat.groups.video_sender(...)` | Raw NV12 → **H.264 encode → RTP → UDP**. The output side of almost every app here. | Build with `VideoSenderOptions::H264RtpUdpFromRaw(w, h, fps)`, then set `host`, `video_port_base`, `channel` (the actual port is `video_port_base + channel`), `encoder.bitrate_kbps` |

---

## 4. Model

`Model` wraps a compiled `.tar.gz` and **plans the whole CVU → MLA → box-decode chain for you** from
your intent. You describe *what* you want; the planner picks the stages.

| API | Use |
| --- | --- |
| `Model(path, Options)` <br> `pyneat.Model(path, opt)` | Load a compiled model package. |
| `graph.add(model)` | Drop the whole model chain into a `Graph`. |
| `model.build()` → `Runner` | Standalone request/response, no `Graph`. `runner.run([tensor])`. |
| `model.benchmark(n)` | Synthetic-input benchmark. Returns a `BenchmarkReport`. |

### `Model::Options` / `pyneat.ModelOptions`

**Preprocess** (`opt.preprocess.*`) — runs on the **CVU**, not the CPU:

| Field | Typical value | Notes |
| --- | --- | --- |
| `kind` | `InputKind::Image` | |
| `enable` | `AutoFlag::On` | Master switch. |
| `input_max_width` / `input_max_height` | `1920` / `1080` | Bounds for the incoming frame. |
| `resize.width` / `resize.height` | `640` / `640` | The model's input size. |
| `resize.mode` | `ResizeMode::Letterbox` | Aspect-preserving pad. What YOLO expects. |
| `color_convert.input_format` | `PreprocessColorFormat::NV12` | What the decoder hands you. |
| `color_convert.output_format` | `PreprocessColorFormat::RGB` | What the model wants. |
| `preset` | `NormalizePreset::COCO_YOLO` | Shorthand for the YOLO mean/stddev. |

**Postprocess / box decode:**

| Field | Typical value | Notes |
| --- | --- | --- |
| `decode_type` | `BoxDecodeType::YoloV8` / `YoloV26` | **Must match the model's head layout, not its name.** See [Gotchas](#gotchas). |
| `score_threshold` | `0.30` | |
| `nms_iou_threshold` | `0.50` | |
| `top_k` | `100` | Max detections/frame. |
| `num_classes` | `80` | COCO. |

**Diagnostics:**

| Field | Notes |
| --- | --- |
| `verbose.planner = true` | Dumps the MPK contract and **the planner's routing decisions** — which packaged stages map to which CVU/MLA nodes, and what gets fused into box decode. Reach for this *before* editing a pipeline you think is mis-planned. |

---

## 5. Graph fragments (multi-branch topology)

For anything that is not a straight line.

| API | Use | Important parameters |
| --- | --- | --- |
| `graphs::Branch("in", {"a","b"})` <br> `pyneat.graphs.branch(...)` | **Fan-out**: one source → N consumers (e.g. one decoded frame → video encoder *and* model). | Pair with `GraphLinkOptions.policy = RealtimeLatestByStream` so a slow branch cannot stall the source. |
| `pyneat.graphs.combine([...], "out", policy)` | **Fan-in**: merge branches back into one sample. | `CombinePolicy.ByFrame` — join on `frame_id`. |

---

## 6. Tensor & Sample

`Tensor` is data + metadata. `Sample` is what comes out of a `pull()`.

| API | Use |
| --- | --- |
| `Tensor.from_numpy(arr, …)` | NumPy → Tensor. Params: `layout`, `memory` (`TensorMemory.CPU` / `EV74`), `image_format` (`PixelFormat.NV12` / `BGR`). |
| `tensor.to_numpy(copy=True)` | Tensor → NumPy. **Fails on planar NV12** — see [Gotchas](#gotchas). |
| `tensor.copy_payload_bytes()` | Raw bytes. The reliable way to get NV12 out. |
| `make_cpu_owned_storage(n)` (C++) | Allocate a CPU-owned buffer to build a `Tensor` by hand. |
| `Plane` / `PlaneRole::Y` / `PlaneRole::UV` | Describe the two planes of an NV12 tensor. **Required** — a bare buffer is not a valid NV12 tensor. |
| `tensors_from_sample(sample, flat)` (C++) | Pull the tensors out of a `Sample`, whatever its kind. |
| `sample.kind` | `SampleKind::Tensor` / `TensorSet` / `Bundle`. **Always check before reading** — a Bundle has `fields`, not `tensor`. |
| `sample.frame_id` / `stream_id` | Routing and joins in multi-stream apps. |

`PullStatus`: `Ok` / `Timeout` / `Closed` / `Error`. **Handle all four** — a timeout is not an error, and
`Closed` means the source ended.

---

## 7. Detection decode helpers

The MLA emits raw tensors. These turn them into boxes/masks/keypoints. NEAT's box-decode stage runs
on the **EV74**, so this is cheap.

| API | Returns | Important parameters |
| --- | --- | --- |
| `decode_bbox_tensor(t, w, h, top_k, …)` (C++) <br> `pyneat.decode_bbox(tensors, clamp_to=(w,h), top_k=…)` | `Box{x1,y1,x2,y2,score,class_id}` in **original-image pixels** | `clamp_to` = the source frame size (undoes the letterbox). |
| `decode_segmentation_tensor(…)` <br> `pyneat.decode_segmentation(…)` | boxes + mask tensors | masks come back at `kDecodedMaskWidth` × `kDecodedMaskHeight` (160×160) — you upscale. |
| pose decode (`PoseDecodeTensors`) | boxes + keypoints | See [`quad-stream-quad-model`](quad-stream-quad-model/README.md). |

---

## 8. Senders (out-of-band egress)

| API | Use | Important parameters |
| --- | --- | --- |
| `MetadataSender(opts)` | Ship detections as **JSON over UDP** to Neat Insight, alongside the video. Not a graph node — you call it from your loop. | `host`, `metadata_port_base`, `channel`; then `send_metadata(type, json, ts_ms, frame_id)` |

Sending metadata instead of burning boxes into the frame **frees the CPU overlay cost entirely** —
Insight renders them.

---

## 9. GenAI (LLM / VLM / ASR)

Used by [`detection-vlm-assistant`](detection-vlm-assistant/README.md). Full treatment in
[`../llima/`](../llima/README.md).

| API | Use |
| --- | --- |
| `pyneat.genai.VisionLanguageModel` | Image + prompt → text. |
| `pyneat.genai.GenerationRequest` | The prompt/params object you hand it. |

---

## 10. Enums you will actually set

| Enum | Values you will use |
| --- | --- |
| `BoxDecodeType` | `YoloV8` (also **YOLO11**), `YoloV26`, `YoloV8Seg`, `YoloV26Pose`, `YoloX` |
| `RunPreset` | `Realtime`, `Reliable`, `Balanced` |
| `OverflowPolicy` | `KeepLatest` (live), `Block` (never drop) |
| `OutputMemory` | `ZeroCopy`, `Owned` |
| `ResizeMode` | `Letterbox` |
| `NormalizePreset` | `COCO_YOLO`, `None` |
| `PreprocessColorFormat` | `NV12`, `RGB`, `BGR`, `I420`, `GRAY8` |
| `FormatTag` / `pyneat.Format` | `NV12` |
| `PayloadType` | `Image` |
| `InputRole` | `Source` (a `Custom` node that generates its own data), `Push`, `None` |
| `CapsMemory` | `Any`, `SystemMemory` |
| `PullStatus` | `Ok`, `Timeout`, `Closed`, `Error` |

---

## Which app uses what

Read the app in the rightmost column that matches what you are building.

| App | RTSP in | Custom src | VideoSender | Metadata | Branch | Combine | bbox | seg | pose |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [`single-stream-yolo-yolo11`](single-stream-yolo-yolo11/README.md) | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | |
| [`single-stream-yolo-yolov8n`](single-stream-yolo-yolov8n/README.md) · [`-yolov8m`](single-stream-yolo-yolov8m/README.md) · [`-yolo26n`](single-stream-yolo26n/README.md) | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | |
| [`single-stream-yolov8n-seg`](single-stream-yolov8n-seg/README.md) | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | |
| [`single-stream-open-pose`](single-stream-open-pose/README.md) | ✅ | | ✅ | | | | | | |
| [`multi-stream-yolo-yolo11`](multi-stream-yolo-yolo11/README.md) | ✅ | | ✅ | | | | ✅ | | |
| [`quad-stream-quad-model`](quad-stream-quad-model/README.md) | ✅ | | ✅ | | | | ✅ | ✅ | ✅ |
| [`multi-model-load-probe`](multi-model-load-probe/README.md) | ✅ | | ✅ | | | | ✅ | ✅ | |
| [`usb-camera-yolo26m`](usb-camera-yolo26m/README.md) | | ✅ | ✅ | ✅ | ✅ | | ✅ | | |
| [`detection-vlm-assistant`](detection-vlm-assistant/README.md) | ✅ | | | | | | ✅ | | + GenAI |
| [`pcb-defect-detection-yolo26n`](pcb-defect-detection-yolo26n/README.md) | | | | | | | ✅ | | *`Model::Runner`, no Graph* |
| [`benchmark`](benchmark/README.md) | | | | | | | | | *`Model::benchmark()`, no Graph* |

---

## C++ ↔ Python naming

Same objects, different conventions. The Python side is **not** a literal transliteration.

| C++ | Python |
| --- | --- |
| `neat::Graph` | `pyneat.Graph` |
| `neat::nodes::Input` / `Output` / `Custom` | `pyneat.nodes.input` / `.output` / `.custom` |
| `neat::nodes::groups::RtspDecodedInput` | `pyneat.groups.rtsp_decoded_input` |
| `neat::nodes::groups::VideoSender` | `pyneat.groups.video_sender` |
| `neat::graphs::Branch` | `pyneat.graphs.branch` |
| `neat::Model::Options` | `pyneat.ModelOptions` |
| `OutputOptions::EveryFrame(4)` | `pyneat.OutputOptions.every_frame(4)` |
| `VideoSenderOptions::H264RtpUdpFromRaw(w,h,fps)` | `pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(w,h,fps)` |
| `decode_bbox_tensor(t, w, h, k, …)` | `pyneat.decode_bbox(tensors, clamp_to=(w,h), top_k=k)` |
| `run.pull(name, ms, sample, &err)` → `PullStatus` | `run.pull(name, ms)` → `Optional[Sample]` |

**The `pull()` difference is the one that bites.** In C++ you get a status and out-params. In Python
you get `None` on timeout *or* close, with no way to tell them apart from the return value alone.

---

## Gotchas

Each of these cost real debugging time. They are load-bearing.

**`BoxDecodeType` must match the model's *head layout*, not its name.** A model-zoo **YOLO11**
archive decodes with **`YoloV8`** (raw 64-channel DFL bbox heads). A YOLO11 you compile yourself with
the graph-surgery flow in [`../model-compilation/`](../model-compilation/README.md) produces
4-channel l/t/r/b heads and needs **`YoloV26`**. Same model name, different decoder.

**An enum value does not imply an implemented decoder.** `BoxDecodeType` defines families
(`Detr = 13`, `Centernet = 16`, …) that the runtime does **not** raw-decode. Check before assuming.

**`describe_backend()` before you theorise.** It prints the GStreamer pipeline NEAT actually built.
Most "NEAT is broken" moments are "NEAT built something other than what I meant."

**`verbose.planner = true` before you edit a mis-planned pipeline.** The generated graph can *look*
like it is missing a stage when the planner has quietly fused it in (e.g.
`post_fusion=user_boxdecode(cast+detess+dequant)->boxdecode` for INT8 models). Read the plan first.

**Python `tensor.to_numpy()` fails on an NV12 frame** — `__dlpack__ only supports dense tensors`,
because NV12 is semi-planar. Use `copy_payload_bytes()` and reshape.

**Building an NV12 `Tensor` needs explicit `Plane` metadata.** A bare buffer plus a shape is not
enough; the encoder cannot interpret it. Set `PlaneRole::Y` and `PlaneRole::UV` with their offsets.

**A `Custom` fragment must not end on a bare caps string.** `gst_parse_launch` parses a trailing
`video/x-raw,...` as an *element name* and fails with `no element "video"`. End on a real element
(`! queue`).

**`graph.add(src)` *and* `graph.connect(src, …)` registers the source twice** — two `v4l2src`
elements fighting over one camera. `connect()` alone is enough.

**Handle all four `PullStatus` values.** `Timeout` is not an error. `Closed` means the source ended
and you should stop, not retry.
