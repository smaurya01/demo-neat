# Multi-batch INT8 compilation and runtime profiling

How to compile a model so one MLA call processes **N images**, check it, and profile it on the
DevKit against the batch-1 build on 50+ real images. The worked example is `yolo26m` in INT8 at
batch 6, measured on Neat 0.5.0 / SDK 2.1.3.1.

**Read the result before you start:**
- For `yolo26m`, batch 6 was **slower per image and less accurate** than batch 1. For `yolo11n` it
  was slower too (accuracy not measured).
- Neat 0.5.0 cannot box-decode a batched archive on the device.

The flow below is the way to find out whether batching pays off for *your* model.

## Table of Contents

- [1. Can this model be batched?](#1-can-this-model-be-batched)
- [2. Compile batch N and batch 1](#2-compile-batch-n-and-batch-1)
- [3. Check the archive on the host](#3-check-the-archive-on-the-host)
- [4. Check the outputs on the DevKit](#4-check-the-outputs-on-the-devkit)
- [5. Profile on 50+ images](#5-profile-on-50-images)
- [6. Benchmark with `model-benchmark`](#6-benchmark-with-model-benchmark)
- [7. Results: `yolo26m` INT8, batch 1 vs batch 6](#7-results-yolo26m-int8-batch-1-vs-batch-6)
- [8. What Neat 0.5.0 can do with a batched archive](#8-what-neat-050-can-do-with-a-batched-archive)

---

## 1. Can this model be batched?

At batch 1 the compiler splits any layer too large for MLA memory into spatial tiles. Its log
records this as `Trigger large tensor support due to: <layer>`. It cannot also split that layer by
batch, so every batch above 1 fails with `CannotSplitLargeTensorError … Cannot be segmented due to
batch size`.

Compile batch 1 first (section 2), then look in its log (`compile_int8.log` or `compile_bf16.log`):

```bash
grep "Trigger large tensor" work/<ID>/reports/compile_int8.log   # no output = can be batched
```

| model | large-tensor splitting at batch 1 | batch 6 |
|---|---|---|
| `yolo11n` INT8 | no | ✅ |
| `yolo26m` INT8 | no | ✅ |
| `yolo26m` bf16 | yes (first conv; bf16 doubles its tensors) | ❌ |
| `yolo26x` INT8 | yes (first conv) | ❌ |

INT8 halves every tensor against bf16, so a model that cannot batch in bf16 may still batch in INT8.
`yolo26m` is the example.

## 2. Compile batch N and batch 1

Steps 1 and 2 (export and surgery) are the same as for any model. Then build both batch sizes, one
compile at a time:

```bash
python compile/convert_to_onnx.py --model-id yolo26m
python compile/graph_surgery.py   --model-id yolo26m
python compile/compiler.py        --model-id yolo26m --precision int8                  # batch 1
python compile/compiler.py        --model-id yolo26m --precision int8 --batch-size 6   # batch 6
```

- **Output folders:** batch 1 goes to `work/yolo26m/compile_int8/`, batch 6 to
  `work/yolo26m/compile_int8_b6/`. Reports are `reports/compile_int8.*` and `reports/compile_int8_b6.*`.
- **Calibration is unchanged.** The same 20 real images, and the ONNX input stays `[1,3,640,640]`.
  `--batch-size` only changes the compiled MLA program, so both builds get identical quantization
  parameters.
- **Achieved batch is checked.** `compiler.py` fails unless the compiler reports
  `Achieved batch size: 6`. If the model needs large-tensor splitting, it explains that instead.
- **Making it the default:** `batch_size: 6` in the model's `models.yaml` entry makes batch 6 the
  default for `compiler.py`, `test_model.py` and `profile_model.py`. For `yolo26m` also set
  `precision: int8` (bf16 cannot batch), and build the batch-1 reference with `--batch-size 1`.

Measured on a 24-core x86 host:

| build | time | archive |
|---|---|---|
| `yolo26m` INT8 batch 1 | 8 min 17 s | 35 MB |
| `yolo26m` INT8 batch 6 | 18 min | 72 MB |

```text
[compile] yolo26m: precision=int8 batch=6 onnx=yolo26m.compile_ready.onnx calib=calibration (20 real imgs, using 20)
    ... Plugin distribution per backend:
    ...   A65 : 0
    desired batch 6, achieved 6
[compile] yolo26m (int8_b6): rc=0  -> .../model-compilation/work/yolo26m/compile_int8_b6
```

## 3. Check the archive on the host

```bash
python compile/test_model.py --model-id yolo26m --precision int8 --batch-size 6 --validate-only
```

```text
[PASS] yolo26m          elf=1 so=0 batch=6  (compile_int8_b6/yolo26m.compile_ready_mpk.tar.gz)
```

`batch=6` is read from the archive itself (its plugins' `actual_batch_size`), not from the folder
name.

## 4. Check the outputs on the DevKit

```bash
dk compile/test_model.py --model-id yolo26m --precision int8 --batch-size 6
```

- The five images in `assets/inference/`, with the first one repeated to fill the sixth slot, go in
  as **one** `[6, 640, 640, 3]` tensor, tagged N/H/W/C.
- Every head comes back with the batch first.
- Each item is compared with the batch-1 build's output for the same image, and the worst cosine
  is reported. The two builds share their quantization, so anything below 0.999 means the batch-6
  program computes different numbers. The script reports it as a NOTE: compare accuracy and
  latency before choosing a batch size.

```text
[test] yolo26m: batch 6 archive, 6 image(s) as one [6, 640, 640, 3] tensor
   6 output tensor(s): (6, 80, 80, 4) (6, 40, 40, 4) (6, 20, 20, 4) (6, 80, 80, 80) (6, 40, 40, 80) (6, 20, 20, 80)
   item 0 000000000139.jpg       vs batch-1: cosine min 0.95118 over 6 heads
   item 1 000000000885.jpg       vs batch-1: cosine min 0.92421 over 6 heads
   ...
   NOTE: batch items differ from the batch-1 build (worst cosine 0.92421; ...). Compare the accuracy and latency of batch 1 and batch 6 before choosing one: MULTI-BATCH.md

all archives: one .elf, zero .so
```

**The difference is real, and it costs accuracy.** Against the FP32 ONNX:

- **Raw heads:** batch 1 stays at a worst-head cosine of 0.97–0.99, and batch 6 drops to 0.92–0.97,
  mostly in the box heads.
- **Not data mixing:** the batch-6 program is deterministic, and an item's output does not change
  with the other images in its batch.
- **Detections:** decoded on the host and matched to FP32's boxes at IoU > 0.5.

  | image | batch 1 | batch 6 |
  |---|---|---|
  | `000000000139` | 16/18 | 15/18 |
  | `000000000885` | 14/15 | **10/15** |
  | `000000001000` | 17/17 | 16/17 |
  | `000000001268` | 9/9 | 9/9 |
  | `000000001296` | 6/6 | **3/6** |

`test_box_decode.py` cannot check a batched build (see [section 8](#8-what-neat-050-can-do-with-a-batched-archive)) and
says so instead of failing.

## 5. Profile on 50+ images

`compile/profile_model.py` times **image in → image out**, step by step, on the DevKit:

1. JPEG decode
2. host preprocessing (tensor route only)
3. tensor creation
4. Neat `run()`, split into its stages by Neat's own counters
5. box decode (on the tensor route, after copying the raw heads out of Neat: `outputs to numpy`)
6. drawing
7. JPEG encode

It then times a pipelined run, with 4 calls in flight. It refuses fewer than 50 images, and
supports only YOLO detection models (`box_decode` YoloV8 or YoloV26).

**The images.** Use real images at the camera's resolution, all the same size. The Neat runner is
built for one frame size, and other sizes are resized in the read step. To cut 60 frames from four
of your own videos:

```bash
mkdir -p frames
for v in clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$v")
  for i in $(seq 0 14); do
    ffmpeg -v error -y -ss "$(python3 -c "print(($i+0.5)*$d/15)")" -i "$v" -frames:v 1 -q:v 2 \
      "frames/$(basename "$v" .mp4)_$i.jpg"
  done
done
```

**Run it:**

```bash
# batch 1: Neat's image route (preprocess and box decode on the device)
dk compile/profile_model.py --model-id yolo26m --precision int8 --images frames \
   --out profile_out --json profile_b1.json

# batch 6: the tensor route for batch 6 AND batch 1, so the two compare like for like
dk compile/profile_model.py --model-id yolo26m --precision int8 --batch-size 6 --images frames \
   --out profile_out --json profile_b6.json
```

- **Calls:** each route makes 120 calls, and the first 10 are warm-up. Neat prints its per-stage
  counters every 120 calls; the script sets `SIMA_RUNTIME_PROFILE=1` and reads them.
- **Batch 6:** 120 calls are 720 frames.
- **Output:** Markdown tables, plus a JSON file with every number. The first 6 annotated frames of
  each route go to `--out`.

## 6. Benchmark with `model-benchmark`

The Neat Apps [`model-benchmark`](https://github.com/sima-neat/apps/tree/main/examples/benchmarking/model-benchmark)
(`examples/benchmarking/model-benchmark/src/python/main.py`) runs `pyneat.Model.benchmark()`.

- **Input:** synthetic, so no image decode, preprocessing on the host, or drawing.
- **Latency:** one call.
- **Throughput:** images per second (calls per second × batch).

Run it on the DevKit, from `model-compilation/`, with the apps repo cloned into the workspace
(`git clone https://github.com/sima-neat/apps /workspace/apps`):

```bash
dk /workspace/apps/examples/benchmarking/model-benchmark/src/python/main.py \
  --model work/yolo26m/compile_int8_b6/yolo26m.compile_ready/yolo26m.compile_ready_mpk.tar.gz \
  --frames 1000 [--decode-type yolo26-det] --output-json report.json
```

## 7. Results: `yolo26m` INT8, batch 1 vs batch 6

**Setup:**
- Board: DevKit (Modalix), Neat 0.5.0.
- Images: 60 real 1280×720 frames, 15 from each of four clips: street, highway, parking garage, gym.
- Settings: score > 0.25, top 100.

**`model-benchmark` (1000 synthetic frames):**

| build | route | latency (one call) | throughput |
|---|---|---|---|
| batch 1 | package default (`detessdequant`) | 9.10 ms | **189.4 img/s** |
| batch 1 | `--decode-type yolo26-det` | 8.87 ms | **191.2 img/s** |
| batch 6 | package default | 61.24 ms per 6 | 156.5 img/s |
| batch 6 | `--decode-type yolo26-det` | 57.61 ms per 6 | 156.8 img/s |

The batch-6 box-decode row measures time only. With real images that route returns one result per
call (section 8).

**Inside Neat `run()`, per call (`profile_model.py`, Neat's counters):**

| stage | batch 1, image route | batch 1, tensor route | batch 6, tensor route | batch 6, per image |
|---|---|---|---|---|
| preprocess / quantize (EV74) | 2.55 ms (`preproc`: resize + letterbox + quantize) | 1.48 ms (`quanttess`: quantize only) | 12.31 ms | 2.05 ms |
| MLA inference | 5.74 ms | 5.75 ms | 43.09 ms | **7.18 ms** |
| box decode / dequantize | 0.87 ms (`boxdecode`) | 1.34 ms (`detessdequant`) | 5.16 ms | 0.86 ms |
| Neat `run()` total | 10.17 ms | 10.17 ms | 62.30 ms | 10.38 ms |

**The MLA itself is slower per image at batch 6:** 7.18 ms against 5.74 ms.

**Image in → image out, per frame (mean ms):**

| step | batch 1, image route | batch 1, tensor route | batch 6, tensor route |
|---|---|---|---|
| read + JPEG decode (A65) | 18.98 | 19.04 | 18.81 |
| host preprocess: letterbox, normalize (A65) | — (done on the EV74) | 29.42 | 29.11 |
| to Neat tensor | 1.57 | 3.09 | 3.10 (18.61 per call) |
| Neat `run()` | 10.17 | 10.17 | 10.38 (62.30 per call) |
| boxes (`decode_bbox` / host head decode) | 0.19 | 3.54 | 3.49 |
| draw | 1.26 | 1.23 | 1.22 |
| JPEG encode (A65) | 23.71 | 23.88 | 23.69 |
| **total** | **55.89** | 90.37 | 89.81 |
| one call at a time | 17.9 frames/s | 11.1 frames/s | 11.1 frames/s |
| pipelined, 4 calls in flight | **190.5 frames/s** | 188.5 frames/s | 155.0 frames/s |

**What the numbers say:**

1. **Batch 6 lowers device throughput.** Pipelined, batch 6 reaches 155 frames/s against 188–191
   for batch 1, and every frame waits for five others.
2. **The image route is the fastest way in.** The EV74 does resize, letterbox and quantize in
   2.55 ms; numpy on the A65 needs 29 ms to letterbox and normalize, and the EV74 still quantizes. Batch N cannot use this route on
   Neat 0.5.0.
3. **The host, not the device, limits a single stream.** JPEG decode and encode take 43 ms of the
   56 ms. A camera app replaces both with the hardware video decoder and encoder.
4. **Batch 6 is less accurate** (section 4).

**Recommendation for Neat 0.5.0:**
- Use **batch 1** on the image route, with several frames in flight (`runner.push` / `runner.pull`).
- Re-measure a batched build when Neat supports batched preprocessing and box decode, using
  sections 4 and 5.

## 8. What Neat 0.5.0 can do with a batched archive

| | batch 1 | batch N |
|---|---|---|
| `Model.compiled_batch_size()` | 1 | N |
| Tensor route (raw heads, `Model.benchmark()`) | works | works. Input is one `[N,H,W,3]` tensor with `tensor.axis_semantics = [N, H, W, C]`, otherwise it is rejected ("batched packed video must use HWC layout or N/H/W/C axis semantics"). Heads come back `[N, …]`. |
| Image route (Neat preprocess + box decode) | works | **times out** in the preprocess stage at build |
| On-device box decode | one result per image | **one result per call**: the other N−1 images are lost |

`yolo11n` INT8 shows the same speed pattern (`Model.benchmark()`; throughput with frames kept in
flight, latency for one call):

| build | latency (one call) | throughput |
|---|---|---|
| batch 1 | 5.7 ms | **614.7 img/s** |
| batch 6 | 31.3 ms per 6 images | 507.6 img/s |

So a batched YOLO archive needs host preprocessing (letterbox, RGB, `x/255`) and host decoding of
the raw heads. `profile_model.py` contains both: see `letterbox()` and `decode_heads()`. That host
work costs more than the device time it would save.
