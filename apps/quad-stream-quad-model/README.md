# Quad Stream Quad Model (4x RTSP, 4 different models)

## Introduction

Four RTSP streams → **four different** INT8 models → four independent annotated H.264/RTP UDP
sinks, in one process. Every model decodes **on-device** with Neat's fused `BoxDecode` — nothing is
decoded on the host CPU.

Stream identity is preserved end to end: the frame pulled from stream *i*'s source is the exact
frame pushed into stream *i*'s model, decoded for stream *i*'s task, annotated in place, and
published on stream *i*'s own UDP port. Each frame carries a burned-in `S<i> <TASK> :<port>` banner
so you can tell the four windows apart.

## About Project

- Application: `quad_stream_quad_model` — **C++ (`main.cpp`) and Python (`main.py`)**
- Input: 4x RTSP H.264 streams
- Output: 4x UDP/RTP H.264 streams, one port per stream
- Runtime config: `./config/default.conf` — **shared by both implementations**

Both do exactly the same thing. The C++ build is roughly **2x faster end to end**
(**~171 fps** aggregate delivered vs **~77 fps**), because Python's overlay is serialised by
the GIL. Prefer C++ for throughput; the Python version is the more readable reference.

| slot | task | model | source | Neat on-device decode |
| --- | --- | --- | --- | --- |
| 0 | detection | `yolo_11s` | **model zoo** | `BoxDecodeType.YoloV8` |
| 1 | segmentation | `yolo_11s_seg` | **model zoo** | `BoxDecodeType.YoloV8Seg` |
| 2 | pose | `yolo26s-pose` | self-compiled | `BoxDecodeType.YoloV26Pose` |
| 3 | detection (YOLOX) | `yolox_s` | self-compiled | `BoxDecodeType.YoloX` |

The decode family is chosen by the **shape of the archive's detection head**, not by the model's
version number — this is the thing most worth understanding before you change anything here. There
is no `YoloV11` family; zoo YOLO11 decodes as `YoloV8`. See [`LEARNING.md`](LEARNING.md).

## Requirements

Run on the DevKit with `dk`. `pyneat` must be importable there. `/workspace` is NFS-mounted on the
board at the same path, so you edit host-side and run board-side — no copying.

```bash
cd /path/to/demo-neat/apps/quad-stream-quad-model
```

Sanity-check your RTSP source first — its frame rate is the hard ceiling on any FPS you can claim:

```bash
ffprobe -hide_banner -rtsp_transport tcp rtsp://<rtsp-server-ip>:8555/stream
```

## Model Download Command

Two of the four are published in the SiMa model zoo. Download them:

```bash
mkdir -p ./assets/models
cd ./assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_11s
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_11s_seg
cd ../..
```

The zoo publishes **no YOLO-pose and no YOLOX**, so those two are built with the graph-surgery flow
in [`model-compilation/`](../../model-compilation/README.md):

```bash
source /sdk-extensions/model-compiler/bin/activate
cd ../../model-compilation
for M in yolo26s-pose yolox_s; do
  python compile/convert_to_onnx.py --model-id $M
  python compile/graph_surgery.py   --model-id $M
  python compile/compiler.py        --model-id $M
done
cd ../apps/quad-stream-quad-model
cp ../../model-compilation/work/yolo26s-pose/compile_int8/*/*_mpk.tar.gz ./assets/models/
cp ../../model-compilation/work/yolox_s/compile_int8/*/*_mpk.tar.gz      ./assets/models/
```

Expected model paths (`assets/models/` is git-ignored — the archives are large and not committed):

```text
./assets/models/yolo_11s_mpk.tar.gz
./assets/models/yolo_11s_seg_mpk.tar.gz
./assets/models/yolo26s-pose.compile_ready_mpk.tar.gz
./assets/models/yolox_s.compile_ready_mpk.tar.gz
```

Two things about the self-compiled pair are load-bearing, and the compile flow handles both for you:

> **`yolo26s-pose` must be the padded build.** Its keypoint head is zero-padded 51 → 64 channels.
> That padding is a **209x performance fix** (1782 ms/frame → 8.5 ms/frame for identical weights),
> and it is still decodable on-device — `YoloV26Pose` requires a *slice* depth of 51 but allows a
> larger *input* depth.

> **`yolox_s` must be the split-head build, compiled with `std = 1/255`.** Neat's `YoloX` decoder
> needs three separate tensors per scale — `(bbox, obj, cls)` = `(4, 1, 80)` — not a packed 85-channel
> head. And YOLOX is trained on **raw 0-255 pixels**, unlike the Ultralytics models. Get the
> normalization wrong and YOLOX detects **nothing**, silently, at full speed. See
> [`LEARNING.md`](LEARNING.md) Lesson 3.

## Configure

Edit `./config/default.conf` before running. At minimum, set:

```text
rtsp_default=rtsp://<rtsp-server-ip>:8555/stream
udp_host=<host-ip-that-receives-video>
udp_port_base=5206
udp_port_stride=2
```

With those defaults stream *i* publishes on `5206 + 2*i` → `5206`, `5208`, `5210`, `5212`.

For a bounded smoke test, set `frames=30`.

## Config Parameters

`rtsp_default`: RTSP URL used by any stream slot that does not set its own `stream<i>_rtsp`.

`stream0_rtsp` … `stream3_rtsp`: Per-slot RTSP URL. Each slot builds its **own** source graph, so
four streams means four real H.264 decodes — not one decode fanned out. All four default to the same
source; point them at four different cameras and stream identity still holds.

`rtsp_transport`: RTSP transport mode. Use `tcp` for reliability or `udp` for lower latency.

`num_streams`: How many stream slots to run (1–4). Drop to 2 for a lighter, higher-FPS pipeline.

`stream0_task` … `stream3_task`: Task for each slot — `detection` | `segmentation` | `pose` | `yolox`.
The task also selects the on-device decode family and the input normalization.

`stream0_model` … `stream3_model`: Model archive for each slot. Relative to this app folder.

`udp_host`: Host/IP that receives all four annotated UDP/RTP output streams.

`udp_port_base`: UDP/RTP output port for stream 0.

`udp_port_stride`: Port spacing. Stream `i` publishes on `udp_port_base + i * udp_port_stride`.

`model_width`, `model_height`: Model input size used by Neat preprocessing (all four are 640×640).

`fallback_width`, `fallback_height`, `fallback_fps`: Used when RTSP caps are incomplete.

`latency_ms`: RTSP receiver jitter buffer, in milliseconds.

`score_threshold`, `nms_iou`, `top_k`: Passed into the **on-device** BoxDecode stage, so the NMS runs
on the accelerator and the host never sees a sub-threshold box.

`queue_depth`: Bounded per-graph queue depth (Realtime preset, KeepLatest overflow).

`cvu_pre_target`, `cvu_post_target`: Where the model's pre/post CVU stages run — `AUTO` | `EV74` |
`A65`. Defaults to `EV74`; `AUTO` measured ~12% slower (see [`LEARNING.md`](LEARNING.md) Lesson 6).

`bitrate_kbps`: H.264 output encoder bitrate.

`frames`: Frames to process **per stream**. Use `0` to run until interrupted.

`print_backend`: Print the generated GStreamer backends when `true`.

Every value is also overridable on the command line. Run `python main.py --help`.

Useful flags: `--num-streams {1..4}`, `--tasks a,b,c` (custom model set), `--task <t>` (run ONE model
solo), `--no-overlay` (isolate the model rate from the host overlay cost), `--duration <seconds>`,
`--frames N`, `--rtsp URL` (override all sources), `--pre-target/--post-target`, `--print-backend`.

## How To Build (C++)

Run from the SDK shell:

```bash
cmake -S . \
  -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
cmake --build ./build --parallel
```

## How To Run

**C++** (recommended — ~2x the throughput):

```bash
dk ./build/quad_stream_quad_model --duration 20
```

**Python**:

```bash
dk ./main.py --config ./config/default.conf
```

Both read `./config/default.conf` and take the same flags. Bounded smoke test:

```bash
dk ./build/quad_stream_quad_model --frames 30
dk ./main.py --config ./config/default.conf --frames 30
```

Measure the model rate with the overlay taken out:

```bash
dk ./build/quad_stream_quad_model --no-overlay --duration 20
```

## How To See The Output

Install host viewer tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

Run this on the machine at `udp_host` — one viewer per stream:

```bash
# stream 0 detection :5206   stream 1 segmentation :5208
# stream 2 pose      :5210   stream 3 yolox        :5212
for P in 5206 5208 5210 5212; do
  gst-launch-1.0 -v udpsrc port=$P caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false &
done
```

Expected output: four live windows — boxes, masks, skeletons and YOLOX boxes respectively — each
with an `S<i> <TASK> :<port>` banner burned into the top-left.

## Time Profile

Both `main.cpp` and `main.py` print a **live** time profile to the terminal every
`--profile-interval` seconds (default 5; `0` turns it off), plus a fuller summary at exit. There is
no per-frame log line — one reporter prints the whole table, so the terminal shows stage timings
instead of four threads racing to print frame counters at each other.

Every number is the mean over **that window**, not a cumulative average — a cumulative mean hides a
stream that degrades halfway through the run.

Real output (Modalix DevKit, 4 × RTSP 1280×720 @ 59.94 fps, overlay on):

```text
── t=170.3s ─── ms/frame, mean over this window ───
stream task            decode   infer  postproc  overlay  encode  latency   dec fps   mdl fps  deliv fps   objs
0      detection         0.95   18.25      0.05     0.45    0.09    18.84      59.9      54.8       44.5     17
1      segmentation      0.97   22.17      0.93    11.36    0.10    34.55      59.9      45.1       28.2     13
2      pose              0.94   16.62      0.09     0.98    0.09    17.78      59.9      60.2       44.5      8
3      yolox             0.96   16.16      0.05     0.19    0.08    16.48      59.9      61.9       47.7      3
                                                    aggregate delivered 164.9 fps
```

The columns are named for **what they actually measure**, which is not always what you would guess:

| Column | What it is |
| --- | --- |
| `decode` | memcpy of the decoded NV12 frame out of the zero-copy pool (host CPU). The **H.264 decode itself runs on the hardware decoder and is not visible from the host** — use `dec fps` to see whether it is keeping up. |
| `infer` | model `push` + `pull`. **This is a round trip, not MLA time** — see below. |
| `postproc` | host-side read of the already-decoded payload + instance build. |
| `overlay` | host-side NV12 draw. `0.00` under `--no-overlay`. |
| `encode` | video `push`. This **enqueues** to the encoder and returns, so it is encoder **headroom**, not encode latency — it sits near 0 until the encoder falls behind, and only then does backpressure show up here. |
| `latency` | frame in hand → frame handed to the encoder. |
| `dec fps` | frames the decoder produced, **including** ones the worker was too busy to take. The source's true rate. |
| `mdl fps` | `1000 / infer` — what this stream could sustain back-to-back. |
| `deliv fps` | frames that actually reached the encoder. **The number that matters.** |

> Two of these are **pipelined stages**, so their host-side timing is not the device's internal cost:
> `decode` and `encode`. Reading them as "the decoder takes 0.95 ms" or "the encoder takes 0.09 ms"
> is wrong — they are a memcpy and a queue push. The device work happens off-thread.

### Why `infer` is ~16–22 ms, and why that is *not* the MLA

The models are small, so 16–22 ms looks alarming. **Almost none of it is MLA compute.**

`infer` brackets `model_run.push()` → `model_run.pull()`, and that window contains the whole
model graph, not just the accelerator:

1. **CPU → EV74 copy** of the 1.4 MB NV12 frame. The app pushes a *host* tensor (it needs the NV12 on
   the CPU for the overlay), so the runtime inserts a compatibility copy —
   `SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY`, set in `main()`.
2. **EV74 preprocess** — NV12→RGB, letterbox 1280×720 → 640×640, normalize, quantize, tessellate.
3. **MLA** — the actual model.
4. **EV74 postprocess** — detessellate + dequantize.
5. **On-device BoxDecode** — anchor grids, sigmoid, NMS (+ DFL for the zoo models, + 32×160×160 proto
   mask assembly for seg, + keypoint decode for pose).
6. **`pull`** of the decoded payload.

Steps 1, 2, 4 and 5 all run on the **EV74**, and **all four streams share one EV74** (`cvu_pre_target`
and `cvu_post_target` are both pinned to it). Step 3 is the only part that is "the model".

**The proof that the MLA is not the driver** comes from the compiled archives themselves — every
`*_mpk.tar.gz` ships a `*_stage1_mla_stats.yaml` with exact per-frame cycle counts:

| model | MLA cycles/frame | rel. | measured `infer` | rel. |
| --- | --- | --- | --- | --- |
| `yolo_11s` (detection) | 1,222,587 | 1.00× | 18.25 ms | 1.00× |
| `yolo_11s_seg` | 1,661,007 | 1.36× | 22.17 ms | 1.21× |
| `yolo26s-pose` | 1,444,005 | **1.18×** | 16.62 ms | **0.91×** |
| `yolox_s` | 1,080,648 | 0.88× | 16.16 ms | 0.89× |

**Pose does 18% *more* MLA work than detection, yet finishes 9% *faster*.** If `infer` tracked MLA
compute, that could not happen. The ranking is set by the **decode contract**, not the model: the zoo
`yolo_11s`/`yolo_11s_seg` archives keep raw 64-channel DFL heads, so BoxDecode must run the DFL
softmax+conv on device; the self-compiled `yolo26s-pose` and `yolox_s` fold the DFL away and emit
4-channel heads, which decode far more cheaply.

That argument needs no clock. For scale: at the delivered rates the four models together consume
~217 M MLA cycles/s. Even at a conservative 0.5 GHz the MLA is only ~43% busy (~22% at 1 GHz), and
per-frame MLA time is on the order of **1–2 ms** — under 10% of `infer`. **The MLA has headroom; the
EV74 and the host loop do not.**

### Why `deliv fps` < `mdl fps`

The worker is **fully serial** — it infers, then post-processes, then draws, then encodes, all on one
thread, with nothing overlapping. The numbers say so exactly:

```
latency  ==  infer + postproc + overlay + encode        (all four streams, to 0.01 ms)
segmentation:  22.17 + 0.93 + 11.36 + 0.10  =  34.56   (reported: 34.55)
```

So `deliv fps ≈ 1000 / latency`, and segmentation's 11.4 ms overlay is charged in full against its
frame rate: 45 fps of model capacity delivers only 28 fps.

### What is left on the table

Three known, unexploited wins — **none of these are applied**, this is analysis only:

1. **Decouple `push` from `pull`.** The worker pushes one frame and immediately blocks on `pull`, so
   only ever **one frame is in flight** and `queue_depth=3` does nothing. Nothing in the graph
   overlaps: frame *i+1*'s EV74 preprocess cannot start until frame *i*'s decode has finished. Both
   reference pipelines avoid this — [`multi-stream-yolo-yolo11`](../multi-stream-yolo-yolo11/README.md)
   and the C++ 4-stream demo both use a **separate pusher and puller thread** with
   `Reliable` + `OverflowPolicy::Block` + `queue_depth=4`, which is what keeps the MLA pipelined.
   That app went from 32 → 118 fps aggregate on exactly this change.
2. **Overlap overlay+encode with the next frame's inference** (a separate output thread, as `main.py`
   already has). Segmentation would rise from ~28 to ~45 fps — its model rate — because 11.4 ms of
   drawing would stop being charged against the frame period.
3. **Use the self-compiled `yolo11s.compile_ready` archive for detection** instead of the zoo
   `yolo_11s`. Same weights, but the DFL is folded into the graph, so it decodes as `YoloV26` (4-ch
   heads) rather than `YoloV8` (64-ch DFL) — the cheaper contract that already makes pose and yolox
   the two fastest streams here.

> **Caveat on the reference comparison.** The C++ 4-stream demo hits 4 × 60 fps with **one model
> shared by four streams** — one `.elf` resident, one `Run`. This app runs **four different models**,
> each with its own `Run`. They are not the same workload, and the demo's numbers are not a target
> this app should be expected to match without qualification.

---

# Appendix

## Appendix: Measured behaviour

Modalix DevKit, RTSP 1280×720 H.264 @ 59.94 fps, all four streams concurrent, 20 s window,
`cvu_*_target=EV74`, overlay on. Three runs of each implementation.

### C++ (`./build/quad_stream_quad_model`)

| stream | model | infer ms | postproc ms | overlay ms | **model fps** | **delivered fps** |
| --- | --- | --- | --- | --- | --- | --- |
| 0 detection | `yolo_11s` (zoo) | 18.7 | 0.05 | **0.50** | **55** | **44** |
| 1 segmentation | `yolo_11s_seg` (zoo) | 22.7 | 0.91 | **11.9** | **44** | **30 – 35** |
| 2 pose | `yolo26s-pose` | 16.5 | 0.08 | **1.10** | **64** | **46** |
| 3 yolox | `yolox_s` | 16.1 | 0.04 | **0.29** | **63** | **48** |
| | | | | | | **aggregate 165 – 174 fps** |

### Python (`./main.py`) — same models, same config

| stream | model | infer ms | postproc ms | overlay ms | **model fps** | **delivered fps** |
| --- | --- | --- | --- | --- | --- | --- |
| 0 detection | `yolo_11s` (zoo) | 24 – 27 | 0.7 | ~63 | 37 – 41 | ~15 |
| 1 segmentation | `yolo_11s_seg` (zoo) | 28 – 35 | 6.6 | ~138 | 29 – 35 | ~7 |
| 2 pose | `yolo26s-pose` | 24 – 31 | 0.8 | ~60 | 32 – 42 | ~15 – 18 |
| 3 yolox | `yolox_s` | 24 – 31 | 0.5 | ~26 | 32 – 41 | ~34 – 39 |
| | | | | | | **aggregate 71 – 80 fps** |

**C++ delivers ~2.2x the aggregate throughput for identical models and identical config.** The whole
difference is host-side: the overlay drops **~63 ms → 0.50 ms** (detection) and **~138 ms → 11.9 ms**
(segmentation). Python's NumPy overlay holds the GIL, so the four stream threads serialise; the C++
overlay is a plain memory write on four real OS threads. See [`LEARNING.md`](LEARNING.md) Lesson 8.

Read the two FPS columns differently:

- **`model fps`** = `1000 / mean(infer)` — the model stage alone (preprocess + MLA + on-device
  decode). The "can this model do 60 fps" number.
- **`delivered fps`** = frames actually published to UDP per second of wall clock, including the
  overlay and the encoder push.

In C++ the two columns are close, which is the point: the host is no longer the bottleneck. What is
left is the shared MLA (four models, `infer` 16 – 23 ms each) and the 60 fps source ceiling.

**Post-processing is not a cost in either.** It used to be: segmentation, pose and YOLOX were decoded
in NumPy on the A65, costing **~340 ms** and **~143 ms** per frame. Every model now box-decodes
on-device — so the host `postproc` stage is only **0.04 – 0.91 ms** in C++ and **0.5 – 6.6 ms** in
Python (there it is just the cost of reshaping the payload through NumPy).

> **`postproc` was called `decode` in earlier builds.** It was always the *host-side read of the
> already-decoded payload*, never a decode. The column is now named for what it measures, and
> `decode` now means the RTSP/H.264 frame decode — a genuinely different stage that earlier builds
> never measured at all.

> **Measure with `--duration`, not `--frames`, whenever streams share the MLA.** A per-stream frame
> cap only stops the run when *all* streams reach it, so a fast stream keeps running — and keeps
> consuming the one MLA — while slow streams starve. That reports rates no steady state ever
> produced.

## Appendix: `tools/pose_probe.py` — study one model on its own

The quad app runs four models across a dozen threads. When one model misbehaves that is the worst
possible place to debug it. `tools/pose_probe.py` strips everything away: ONE model graph
(`input -> model -> output`), lock-step push/pull, no threads, no overlay, no UDP.

```bash
# what does this model actually cost, and are its outputs right?
python tools/pose_probe.py --task pose --iters 20 --save-out /tmp/pose.jpg

# control: the same probe on a different model
python tools/pose_probe.py --task segmentation --iters 20

# can ANY runtime option move the number?
python tools/pose_probe.py --sweep
```

Because it is lock-step, its `infer` is the real model service time, not graph latency.

## Appendix: Known limitations

1. **The host overlay gates the delivered rate.** See the measured table above and `LEARNING.md`.
   Burning masks and skeletons into the stream requires the frame on the CPU; Neat's in-graph
   `sima_render` draws boxes only.

2. **Teardown heap abort.** The process intermittently dies at exit with
   `malloc(): mismatching next->prev_size` (exit 134/139). It fires **after** the profile prints, so
   the numbers are valid, but it is a real defect in the teardown path.

3. Set `QSQM_DEBUG=1` to print the tensor shapes each model delivers.
