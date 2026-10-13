# yolo26x INT8 on Modalix: compile and runtime study

Ultralytics **yolo26x** (the largest YOLO26 detector), compiled to INT8 for Modalix and measured
on a DevKit with Neat 0.5.0. Two kinds of runtime numbers, side by side:

1. **Image in → image out**, step by step, on 60 real 1280×720 frames: `compile/profile_model.py`.
2. **`model-benchmark`**, the Neat Apps benchmark (`pyneat.Model.benchmark()`, synthetic input),
   run unmodified from `apps/examples/benchmarking/model-benchmark`.

Board: DevKit (Modalix), Neat 0.5.0, SDK 2.1.3.1 (DevKit software 2.1.3). Compiler: Model Compiler
2.1.3, on a 24-core x86 host with 78 GB of RAM.

## Summary

| | value |
|---|---|
| Compile, INT8, batch 1, 20 real calibration images | **16 min 36 s**, peak 8.1 GB RAM, everything on the MLA (`A65: 0`), 83.4 MB archive |
| Compile at batch 6 | **not possible**: the first conv needs large-tensor splitting |
| `model-benchmark` latency / throughput | **19.17 ms / 73.2 fps** (default route), **19.03 ms / 73.5 fps** (`yolo26-det` box-decode route) |
| Neat `run()` per frame, image route | **18.34 ms**: preprocess 2.30, MLA 14.12, box decode 0.87 |
| Image in → image out, one frame at a time | **65.6 ms (15.3 frames/s)**, of which JPEG decode + encode take 43.8 ms |
| Pipelined (4 frames in flight) | **73.4 frames/s** (13.6 ms per frame), limited by the MLA |

The two methods agree:
- Throughput: 73.2–73.5 fps from `model-benchmark` against 73.4 frames/s pipelined.
- Latency: 19.0–19.2 ms against an 18.3 ms `run()`.
- The MLA is about 77 % of the device time per frame.
- One yolo26x stream fits at 60 fps, or two at 30 fps. A JPEG-based app reaches only 15 frames/s
  per thread, because of the host JPEG work.

---

## 1. Model and compile

**Model:**
- `yolo26x.pt` from Ultralytics, exported to ONNX (223 MB) with the demo-neat
  `model-compilation` flow.
- Graph surgery: raw one-to-one heads, `bbox_{0,1,2}` (4 ch) + `class_logit_{0,1,2}` (80 ch),
  decoded by Neat `YoloV26`.

**Surgery note:**
- yolo26x has **three** attention blocks: `/model.10/m/m.0/attn`, `/model.10/m/m.1/attn` and
  `/model.22/m.0/m.0.1/attn`. yolo26n/s/m have two.
- With yolo26n's surgery spec the second layer-10 block keeps its MatMuls. Those land on the EV74
  and APU, and the compile fails the flexible-batch check.
- The spec used here lists all three. This is needed if yolo26l/x is ever added to `models.yaml`.

**Compile:**
- Command: `quantize_compile.py`, INT8, `mse` calibration, `sima` requant, 20 real COCO images,
  input `1,3,640,640`, Modalix.
- Where: an x86 host, in a Docker container with Model Compiler 2.1.3.

| phase | time |
|---|---|
| ONNX prepare, load, quantization setup | ≈ 3 min 0 s |
| calibration, 20 images (≈ 6 s each) | ≈ 2 min 0 s |
| quantize finish | ≈ 16 s |
| MLA compile (layouts, large-tensor tiling, code generation) | ≈ 11 min 22 s |
| **total** (phases rounded) | **16 min 36 s** |

- **Resources:** peak RSS 8.1 GB. Average CPU 212 %, because the compile is mostly single-threaded;
  more cores barely help.
- **Saturation warning:** one INT8 warning on the layer-10 attention `qkv` conv (`/model.10/m/m.0/attn/qkv/conv/Conv`).
  The compiler zeroed weights to avoid saturation there.
- **Large-tensor note:** `Trigger large tensor support due to: MLA_0/conv2d_add_0`. The first conv
  does not fit MLA memory and is tiled spatially.

**Batch 6:**
- Same `quantize_compile.py` flags plus `--batch_size 6` (with `compiler.py` it would be
  `--batch-size 6`). It fails after 5 min 47 s:
  `CannotSplitLargeTensorError: MLA_0/conv2d_add_0: Cannot be segmented due to batch size`.
- The large-tensor tiling above cannot be combined with batching.
- See [`model-compilation/MULTI-BATCH.md`](MULTI-BATCH.md#1-can-this-model-be-batched).

## 2. Runtime with `model-benchmark` (synthetic input)

**Tool:** `examples/benchmarking/model-benchmark/src/python/main.py` from the
[Neat Apps](https://github.com/sima-neat/apps) repository, run on the DevKit as published, with the
apps repo cloned into the workspace (`git clone https://github.com/sima-neat/apps /workspace/apps`):

```bash
dk /workspace/apps/examples/benchmarking/model-benchmark/src/python/main.py \
  --model <yolo26x.compile_ready_mpk.tar.gz> --frames 1000 [--decode-type yolo26-det] \
  --output-json model_benchmark_<route>.json
```

| route | resolved postprocess | latency (one frame) | throughput |
|---|---|---|---|
| package default | `detessdequant` (raw heads, dequantized on the EV74) | **19.17 ms** | **73.2 fps** |
| `--decode-type yolo26-det` | `boxdecode` (YoloV26, top-k 100, on the A65) | **19.03 ms** | **73.5 fps** |

Power reads 0 W: model-benchmark reports 0 W when board power telemetry is unavailable.

**What it measures:**
- It does: the model package on synthetic input. Latency is one frame through the graph;
  throughput is with frames queued.
- It does not: image decode, preprocessing on the host, drawing or output. The route's own EV74
  preprocess and decode stages are included.

## 3. Runtime image in → image out (60 real images)

**Tool:** `compile/profile_model.py`, the demo-neat `model-compilation` profiler, on 60 real
1280×720 frames. These are 15 frames from each of four Insight catalog clips: street, highway,
parking garage, gym.

**Path:** Neat's image route, BGR frame → EV74 preprocess (resize, letterbox, quantize) → MLA →
YoloV26 box decode. Each call is one frame; 120 calls, 10 of them warm-up. `frames/` holds the 60
frames; see [MULTI-BATCH.md](MULTI-BATCH.md#5-profile-on-50-images) for how to cut them from your
own clips.

```bash
dk compile/profile_model.py --model-id yolo26x --precision int8 --images frames \
   --out profile_out --json profile_b1.json
```

`yolo26x` is not in this repo's `models.yaml`, so this ran in the study setup (section 7): the
repo's `profile_model.py` and `common.py` with a one-model `models.yaml` holding a `yolo26x` entry.

| step (per frame) | where | mean ms | p50 ms | p95 ms |
|---|---|---|---|---|
| read + JPEG decode | A65 | 19.97 | 19.82 | 24.47 |
| to Neat tensor (copy into EV74 memory) | A65 | 1.93 | 1.53 | 3.59 |
| **Neat `run()`** | device | **18.34** | 18.34 | 18.49 |
| parse boxes (`decode_bbox`) | A65 | 0.18 | 0.18 | 0.19 |
| draw boxes and labels | A65 | 1.28 | 1.30 | 2.14 |
| JPEG encode (q 90) | A65 | 23.86 | 23.85 | 25.82 |
| **total** | | **65.57** | 64.77 | 72.89 |

One frame at a time that is 15.3 frames/s; the frames had 1–17 boxes each.

**Inside Neat `run()`** (Neat's own counters, `SIMA_RUNTIME_PROFILE=1`, 120 calls):

| stage | where | ms per frame |
|---|---|---|
| `preproc`: resize 1280×720 → 640, letterbox, normalize, quantize, tessellate | EV74 | 2.30 |
| `MLA_0`: inference | MLA | **14.12** |
| `boxdecode`: YoloV26, top-k 100 | A65 | 0.87 |
| hand-offs, push / pull | | ≈ 1.0 |

**Pipelined:** 4 frames in flight, inputs prepared in advance: **73.4 frames/s**.

## 4. The two methods side by side

| | `model-benchmark` | `profile_model.py` |
|---|---|---|
| input | synthetic tensor | 60 real 1280×720 JPEG frames |
| covers | the model package's graph | image in → image out, plus Neat's per-stage counters |
| latency, one frame through Neat | 19.17 ms (default) / 19.03 ms (box decode) | 18.34 ms `run()` (p50 18.34) |
| throughput, frames queued | 73.2 / 73.5 fps | 73.4 frames/s pipelined |
| MLA time per frame | (not split out) | 14.12 ms |
| host work per frame | not measured | 47.2 ms (JPEG decode 20.0, tensor 1.9, boxes 0.2, draw 1.3, JPEG encode 23.9) |

**Why the latency differs by ~0.8 ms:** `Model.benchmark()` measures from graph entry to the public
pull, including the queue. `profile_model.py` times the Python `run()` call on an already-built
runner. Throughput matches because both are limited by the MLA: 73.4 frames/s is 13.6 ms per
frame, a little under the 14.12 ms the `MLA_0` counter reports when calls run one at a time (the
counter includes per-call overhead that overlaps when calls are pipelined).

## 5. For context

- **yolo26m INT8** (same flow and board): 9.10 ms / 189.4 img/s from `model-benchmark`; 5.74 ms
  MLA. yolo26x costs about 2.5× the MLA time of yolo26m.
- **SiMa Model Zoo `yolo26x-det-bf16-mla_tess-b1`:** 39.77 ms / 27.98 fps in the Apps'
  [`BENCHMARK_RESULTS.md`](https://github.com/sima-neat/apps/blob/main/examples/benchmarking/model-benchmark/BENCHMARK_RESULTS.md).
  That is bf16, on Neat 0.4.0, so it is not a like-for-like comparison. INT8 here is
  about 2.6× the throughput; the accuracy trade-off is not measured.

## 6. Detections and accuracy

**Checked:**
- The annotated frames from `profile_model.py --out` show correctly placed boxes on the 1280×720
  frames: cars, people, the cyclist, traffic lights.
- On the five COCO test images, scores range from 0.24 to 0.97.

**Watch:**
- Same-class pairs overlapping at IoU > 0.7 per image: 0 / 1 / 5 / 1 / 0.
- The crowded image (16 people) has 5, more than the 0–3 seen for yolo26n/s. YOLO26's one-to-one
  head runs without NMS. INT8 and the saturation warning above may add near-duplicates.
- A light class-aware NMS after decode removes them if an app needs that.

**Not measured:** COCO mAP of this INT8 build.
`model-compilation/accuracy/` can run that study once `yolo26x` is in `models.yaml` with the
three-attention-block surgery spec from section 1.

## 7. Study setup

The study ran outside this repo's normal flow, because `yolo26x` is not in `models.yaml`:

- **Compile:** Model Compiler 2.1.3 in Docker, with `quantize_compile.py`
  called directly on `yolo26x.compile_ready.onnx` (the surgery spec in section 1) and the 20
  images in `assets/calibration/`.
- **Profile:** the repo's `profile_model.py` and `common.py`, with a one-model `models.yaml` holding
  a `yolo26x` entry and the archive under `work/yolo26x/compile_int8/`.
- **Inputs:** the 60 frames described in section 3, and the five images in `assets/inference/`.
