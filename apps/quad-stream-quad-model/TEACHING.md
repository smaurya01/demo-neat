# Designing a complex Neat pipeline: 4 streams × 4 models on one MLA

This app is a teaching vehicle for the hard parts of a real multi-stream,
multi-model Neat pipeline. Every API below is cited to its source in
`https://github.com/sima-neat/core`. Nothing here is invented — where an option does not exist, we
say so.

---

## 1. Graph topology: one shuttle per stream, not one mega-graph

There are two ways to run many streams in Neat:

* **Join inside one graph** with `pyneat.graphs.combine([...], "combined",
  CombinePolicy.ByFrame)` — see `core/tutorials/015_run_multiple_streams`. This
  fuses streams into a single bundle you pull once. It is the right tool when the
  streams feed the *same* downstream stage and you want a synchronized join.

* **One independent graph shuttle per stream** — what this app does, following
  `apps/multi-stream-yolo-yolo11/main.py`. Each stream owns three graphs:

  ```
  RTSP source graph ──NV12──▶ model graph (its own archive) ──raw/bbox──▶ video sink graph
  ```

  We use the shuttle form because the four streams run **four different models
  and four different UDP sinks** — there is nothing to join. Keeping them
  independent also means each stream's stats (FPS, drops) are separable, and a
  slow stream cannot silently starve a `combine` bundle.

**Stream identity** is a property of *how you drive the graphs*, not of a field
you set. In `service_stream()` we pull one frame from source *i*, push that exact
tensor into model *i*, and push the annotated result into sink *i* before moving
on. The result therefore always belongs to the frame we just pulled. (Neat also
carries `sample.stream_id` / `sample.frame_id` — see
`core/tutorials/015` — useful when you *do* fan-in.)

## 2. Named endpoints: how you pull the right thing out

A graph's output node is a **named endpoint**. `run.pull(name, timeout_ms)`
returns the sample published on that name.

```python
graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(1)))
...
sample = run.pull("detections", 20000)
```

* For the detection stream we name the endpoint `"detections"` and the fused
  BoxDecode stage publishes one decoded BBOX tensor there.
* For the raw-head streams we name it `"heads"`; the model publishes **all** its
  raw output tensors into that one sample. `extract_tensors(sample)` walks
  `SampleKind.Tensor` / `SampleKind.TensorSet` / nested `fields` (see the helper
  in `main.py`) and hands you the list. **Route by tensor shape, not by output
  order** — the compiler's output ordering is not a contract, but the shapes
  (`C==4` bbox, `C==80` class, `C==32` mask-coeff, `C==51` keypoints, `C==85`
  YOLOX head, `160×160` proto) are unambiguous.

`OutputOptions.every_frame(1)` (from `core/include/neat/nodes.h`) publishes every
frame; a larger stride or a different policy lets you sub-sample expensive sinks.

## 3. The load-bearing lesson: what the MLA gives you vs. what the A65 must do

This is the most valuable insight in the app.

The four `compile_ready` archives were surgered (see
`model-compilation/compile/_surgery_*.py`) to expose **raw per-scale
head tensors** and to *cut* the data-dependent decode/NMS tail. That is what
keeps them 100% on the MLA: `A65:0`, one `.elf`, zero `.so`. The MLA hands you,
per head, a **calibrated FP32 NHWC tensor** — detessellation and dequantization
are done for you by a `dequantize`/`detessdequant` CVU stage (visible in
`--print-backend`: `neatprocessmla ! neatprocesscvu name=dequantize ! appsink`).

But **Neat's built-in `BoxDecode` only covers the detection family.**
`core/include/pipeline/BoxDecodeType.h` *does* define `YoloV26Seg (19)`,
`YoloV26Pose (18)` and `YoloX (21)`, and `core/include/pipeline/DetectionTypes.h`
*does* expose host helpers `decode_pose` / `decode_segmentation`. However those
helpers parse a **BoxDecode wire payload** (`uint32 count + top_k×BBOX +
top_k×PoseOut/masks`) — i.e. they require the model to have run a BoxDecode
*pose/seg* stage that produced that payload. Our raw-head archives deliberately
do **not** contain that stage, so on these archives the seg/pose/YOLOX decode is a
genuine **host-side (A65, NumPy) step you write yourself**. That is the design
tension:

> The MLA is a fixed-function dataflow engine. Everything that is a *dense conv
> graph* (including the seg prototype FCN with its stride-2 `ConvTranspose`, and
> YOLO26's attention `Einsum` blocks) runs on it beautifully. Everything that is
> *data-dependent and dynamic* — "how many objects are there, which anchors
> survive NMS, gather their mask coefficients" — cannot be expressed as a static
> tensor program and falls to the host CPU. Surgery draws that line explicitly;
> the app pays for it on the A65.

`src/decoders.py` is the concrete host side of that line. Notice the
**NHWC→CHW** transpose in `_squeeze_batch` (the MLA emits channel-last
`(1,H,W,C)`), the anchor-grid conventions (`+0.5` for boxes, integer grid for
keypoints/YOLOX), and the letterbox-inverse back to frame pixels — none of which
the MLA does for a raw-head model, all of which the fused detection path did for
free.

## 4. `ModelOptions` — preprocess and decode intent

`core/include/model/Model.h`, `Options` struct. Setting these "upgrades the
framework's basic dtype-bridge stages into fused Generic Preproc / BoxDecode
stages" (Model.h line ~222).

```python
opt.preprocess.kind   = pyneat.InputKind.Image
opt.preprocess.enable = pyneat.AutoFlag.On
opt.preprocess.resize.enable = pyneat.AutoFlag.On
opt.preprocess.resize.width  = 640;  opt.preprocess.resize.height = 640
opt.preprocess.resize.mode   = pyneat.ResizeMode.Letterbox   # aspect-preserving
opt.preprocess.resize.pad_value = 114
opt.preprocess.color_convert.input_format  = pyneat.PreprocessColorFormat.NV12  # RTSP frames are NV12
opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
```

* **Colour format follows the source, not the model.** RTSP-decoded frames are
  **NV12**; OpenCV frames are BGR. Using the wrong one is a silent correctness
  bug: the image/BGR route needs an explicit letterbox resize.
* **Decode intent** (detection only):
  `opt.decode_type = pyneat.BoxDecodeType.YoloV26` — a `compile_ready` YOLO11/26
  archive decodes with **YoloV26**, not YoloV8 (YoloV8 rejects the grouped head
  at build time). Also set `score_threshold`, `nms_iou_threshold`, `top_k`,
  `num_classes`. For the raw-head models leave `decode_type` **`Unspecified`** —
  that is exactly what tells Neat to publish the raw tensors.
* **Never** set `boxdecode_original_width/height` — deprecated in Model.h; box
  decode reads geometry from preprocess metadata.

**Pre/post target (EV74 vs A65).** The MPK contract decides the plugin placement;
you can inspect it via `Model.info()` / `ModelInfo::RouteNeeds` and
`RouteCapabilities` (Model.h ~line 109–147: `needs_detess`, `needs_dequant`,
`has_post_boxdecode`, `selected_post_kind`). In this app the compiler placed all
inference on the MLA and the detess/dequant on an EV74 CVU (A65:0). The *only*
work that lands on the A65 is our Python host decode — which is precisely why it
dominates the loop (§6).

## 5. `RunOptions`, `InputOptions`, `OutputOptions`

**`RunOptions`** (`core/include/pipeline/Run.h`, struct at line 164):

```python
ro = pyneat.RunOptions()
ro.preset          = pyneat.RunPreset.Realtime          # low-latency; small queues; KeepLatest
ro.queue_depth     = 3                                   # bounded buffers
ro.overflow_policy = pyneat.OverflowPolicy.KeepLatest    # drop-oldest under pressure
ro.output_memory   = pyneat.OutputMemory.ZeroCopy
```

* `RunPreset` (Run.h ~90): `Realtime` (low latency, KeepLatest), `Balanced`
  (default), `Reliable` (deeper queues, `Block` overflow — never drop, but
  back-pressures the producer).
* `OverflowPolicy` (Run.h ~74): `Block` waits (reliable, but a stalled consumer
  stalls the source); `KeepLatest` drops the oldest queued frame (realtime — a
  slow consumer just skips frames instead of building latency). For live RTSP you
  almost always want `KeepLatest`.
* `queue_depth` (default 4): deeper for jittery sources, shallower for latency.
  `core/tutorials/016_tune_throughput_and_queues` is the reference for trading
  queue depth against latency; `015`'s note is explicit that growing queues to
  hold a whole batch demonstrates `Combine`, not something production live code
  should do — pull concurrently instead.

**`InputOptions`** (`core/include/neat/nodes/io/Input.h`) describe the graph input
boundary/caps. Here every model and video graph takes an NV12 image input with an
explicit `caps_override` (`video/x-raw,format=NV12,width=…,height=…,framerate=…`)
so the compiler negotiates the appsrc correctly. (`use_simaai_pool` is
deprecated — the board warns and points at `memory_policy`; harmless here.)

**`OutputOptions`** — `every_frame(1)` publishes every frame; raise the stride to
throttle a heavy sink.

## 6. Efficient output streaming: per-stream encoder sinks

Each stream owns one H.264/RTP/UDP sink built from
`pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(w, h, fps)`
(`pyneat.groups.video_sender`, `core/include/neat/node_groups.h`):

```python
so = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(w, h, fps)
so.host = udp_host
so.video_port_base = 5206 + 2*stream_id      # distinct port per stream = identity on the wire
so.encoder.bitrate_kbps = 4000               # encoder knob: raise for quality, lower for bandwidth
```

Per-stream ports are the cheapest, most robust way to keep identity end to end —
the receiver picks a stream by port. The encoder runs on the DevKit's hardware
video encoder, off the MLA and off the A65, so it is essentially free relative to
the host decode. We annotate directly on the NV12 **Y plane** (draw box edges,
labels, keypoints, and a translucent mask tint) before the encoder, so no extra
colour conversion is needed.

## 7. Measuring: per-stream FPS, drops, latency — and the real bottleneck

The loop records, per stream, cumulative service time and frame count, and prints
`stream_fps` (that stream's own compute rate) and `agg_fps` (whole-loop rate).
Measured on the DevKit (20 frames/stream):

| stream | task | decode site | service FPS |
| --- | --- | --- | --- |
| 0 | detection | MLA fused BoxDecode | ~15.7 |
| 3 | yolox | A65 host | ~12.7 |
| 1 | segmentation | A65 host + mask assembly | ~3.6 |
| 2 | pose | A65 host | ~0.5 |

**Aggregate ≈ 1.7 FPS.** The lesson is in *why*: this is a single Python thread
doing round-robin, so `agg_fps ≈ 1 / Σ(per-stream frame times)` and the **slowest
host decoder gates everything**. The MLA is not the bottleneck — inference is a
small slice; the A65 NumPy decode (and, for seg, the prototype-mask matmul +
resize) dominates. Three independent levers, in order of impact:

1. **Parallelize.** The four graph shuttles are independent — give each stream its
   own worker thread so their host-decode and MLA time overlap. The MLA still
   serializes at the hardware gatekeeper, but detection/yolox would no longer wait
   behind pose's 1.9 s/frame host decode.
2. **Move decode off the A65.** Where a fused `BoxDecode` family fits (detection
   already does), on-device decode is ~4× the host path. For seg/pose, either
   vectorize harder or accept a lower publish rate on those sinks
   (`OutputOptions` stride) while detection/yolox stay realtime.
3. **Bound the work.** `top_k`, `score_threshold`, and (seg) `max_masks` directly
   size the host loop. Mask assembly is `O(instances × 160²)`.

**Drops** are observable through `OverflowPolicy.KeepLatest`: with realtime
queues a slow consumer skips frames rather than growing latency, so a stream's
`stream_fps` below the source FPS *is* the drop signal. For end-to-end latency,
timestamp at source-pull and at sink-push per frame (`sample` carries PTS); the
gap is dominated by the same host-decode term.

### Honest conclusion

**A single-threaded round-robin does not sustain 4 realtime streams × 4 models on
one MLA** — measured ~1.7 agg FPS, gated by A65 host decode, not the MLA. A
2-stream configuration (detection + seg, `--num-streams 2`) sustains ~8 agg FPS.
The fix is per-stream threading plus keeping decode on the MLA wherever a
`BoxDecodeType` family fits. That measured limit — and *where* the time actually
goes — is the point of the exercise.
