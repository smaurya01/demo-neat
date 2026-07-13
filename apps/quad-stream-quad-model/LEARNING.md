# Learning from this app: 4 streams × 4 models on one MLA

This app exists to be *studied*, not just run. It puts four different models — detection,
segmentation, pose and YOLOX — on one shared MLA, from four RTSP streams, and burns the
results into four independent H.264/UDP outputs.

It ships **twice**: `main.cpp` and `main.py`, same models, same `config/default.conf`, same
output. That is deliberate — the pair is itself one of the lessons (Lesson 8: C++ delivers
~2.2× the throughput, and the entire difference is the GIL, not the accelerator).

Everything below was measured on a Modalix DevKit against a live 1280×720 H.264 stream.
Where a number appears, it came from a run, not from an estimate. Where something is still
slow, it says so.

Every API is cited to <https://github.com/sima-neat/core>.

---

## Lesson 1: The decode family follows the HEAD SHAPE, not the model's name

This is the single most useful thing in this app, and the easiest to get wrong.

Neat's `BoxDecodeType` (`core/include/pipeline/BoxDecodeType.h`) lists 21 families —
`YoloV5`, `YoloV8`, `YoloV9`, `YoloV10`, `YoloV26`, `YoloX`, `Detr`, `Centernet`, … —
and **there is no `YoloV11`**. That is not an oversight. You do not pick the family by
reading the model's version number; you pick it by looking at the shape of the tensors its
detection head actually emits.

The same YOLO11 weights ship in two different head shapes here:

| archive | bbox head | what it means | family |
| --- | --- | --- | --- |
| model-zoo `yolo_11s` | 3 × **64**-channel | raw DFL distributions (4 sides × 16 bins) | `YoloV8` |
| self-compiled `yolo11s` | 3 × **4**-channel | DFL already folded into the graph → l/t/r/b distances | `YoloV26` |

Both are "YOLO11". They need *different* decoders. The zoo build keeps the DFL as raw
distributions, so it decodes with the YOLOv8 contract; our graph surgery
(`model-compilation/compile/_surgery_ultralytics.py`) rebuilds the DFL as
`Split → Softmax → Conv(arange) → Concat` inside the model, so it emits 4 distance channels
and decodes with the YOLO26 contract.

Pick the wrong one and **the app still runs, still draws boxes, and reports healthy FPS** —
the boxes are just decoded from the wrong channels. There is no error. That is why this is
worth internalising rather than looking up.

The mapping this app uses is in `DECODE_FAMILY` in `main.py`, with the head shape in a
comment beside each entry. Read that before changing a model.

## Lesson 2: Neat decodes segmentation, pose and YOLOX on-device too

An earlier version of this app decoded segmentation, pose and YOLOX **on the host**, in
~380 lines of NumPy (`src/decoders.py`, now deleted): anchor grids, sigmoid/exp, NMS,
letterbox-inverse, prototype-mask assembly. The reasoning was that "Neat's built-in
BoxDecode only covers the detection family."

**That was false**, and it cost the app most of its throughput. Neat has:

```python
pyneat.decode_bbox(tensors,         clamp_to=(w, h), top_k=k)   # -> boxes [N, 6]
pyneat.decode_pose(tensors,         clamp_to=(w, h), top_k=k)   # -> boxes + keypoints [N, 17, 3]
pyneat.decode_segmentation(tensors, clamp_to=(w, h), top_k=k)   # -> boxes + masks [N, 160, 160]
```

These are not host decoders. They **read a payload that the on-device BoxDecode stage
already produced** — the MLA/EV74 did the grids, the activations, the NMS and the mask
assembly. All you do is reshape and threshold.

The catch — and the reason the mistake was easy to make — is that a decoder only exists for
your model if the **archive was compiled with a head that matches a family**. Set
`ModelOptions.decode_type` and Neat appends the BoxDecode stage; leave it `Unspecified` and
the model publishes raw heads and you are on your own. The old app left it unspecified for
three of four streams, then wrote the NumPy to cope. The fix was not to write better NumPy.
The fix was to set `decode_type`.

Measured effect on the segmentation stream's decode stage: **~340 ms → 6.6 ms**.

## Lesson 3: YOLOX is not normalized like the Ultralytics models

This one is a *silent, total* failure, and it is the best bug in the app.

Ultralytics models (YOLO11, YOLO26) are trained on pixels scaled to `[0,1]` — you feed them
`x/255`. **Megvii YOLOX is trained on raw 0-255 pixels.** Feed YOLOX the `/255` input that
its neighbours want and it sees an image 255× too dark. Its objectness logits pin negative,
every score collapses below threshold, and it detects **nothing** — while running at full
speed, reporting a healthy ~40 model fps, and raising no error anywhere.

Measured on COCO val `000000000139`, same surgered ONNX:

| input | obj logit max | sigmoid | detections |
| --- | --- | --- | --- |
| raw 0-255 (correct) | **+5.90** | 0.997 | yes |
| `/255` (what COCO_YOLO gives it) | **−8.39** | 0.0002 | **none** |

Fixing it takes **two** changes that must agree, and either one alone does nothing:

1. **Compile time** (`model-compilation/models.yaml`): `std: [1/255, 1/255, 1/255]`.
   The compiler's mean/std are applied in `[0,1]` space — it always divides by 255 first,
   then computes `(x - mean)/std`. So `std = 1/255` cancels the division and hands the model
   the raw pixel back. You can verify it landed by looking at `0_preproc.json` inside the
   archive: `q_scale ≈ 1.0` means the model expects 0-255, `q_scale ≈ 255` means `[0,1]`.
2. **Runtime** (`NORMALIZE_PRESET` in `main.py`, `uses_coco_yolo_normalize` in `main.cpp`):
   `NormalizePreset::None` for YOLOX, `NormalizePreset::COCO_YOLO` for the other three.
   Set the preset and **nothing else** — do not also pass explicit normalize stats of
   `mean=0, stddev=1`. Those are interpreted in `[0,1]` space, so `stddev=1` re-applies the
   very `/255` you are trying to avoid. (This bit the C++ port: it detected nothing again
   until the explicit stats were removed.)

The general lesson: **normalization is a property of the model family, not of the
pipeline.** A four-model app cannot have one preprocess setting. And a model that silently
detects nothing looks exactly like a model that is working — check that objects actually
come out, not just that frames do.

## Lesson 4: YOLOX's head must be exposed SPLIT, not packed

Neat's `YoloX` decoder does not accept a packed `[1,85,H,W]` head. Its contract is three
*separate* tensors per scale, interleaved scale-major — `(bbox, obj, cls)` with depths
`(4, 1, 80)`. `infer_yolox_interleaved_class_depth`
(`core/src/pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.cpp`) rejects the
contract outright unless the tensor count is a multiple of 3 and the depths match exactly.

The original surgery exposed the per-scale `Concat` output (85 = 4 + 1 + 80 packed), which is
precisely why YOLOX *could not* use the on-device decoder and had to be done in NumPy.
`_surgery_yolox.py` now cuts one node earlier and exposes the three branches.

It also exposes the **pre-sigmoid** obj/cls logits, because Neat forces
`score_activation = Sigmoid` for the YoloX family
(`apply_raw_yolov6_yolox_compiled_payload_overrides`). Hand it the exported Sigmoid outputs
and the sigmoid is applied twice — another silent score corruption.

## Lesson 5: A padded head can be faster *and* still decodable

`yolo26s-pose` compiles its keypoint head zero-padded from 51 → 64 channels. This is a
**209× performance fix**, not cosmetics: with the natural 51 channels the same weights run at
1782 ms/frame (0.6 fps); at 64 they run at 8.5 ms/frame. Unpadded, pose holds the shared MLA
so long that the other three streams back up and the quad cannot run at all.

The happy part: the padding does **not** cost you the on-device decoder. `YoloV26Pose`
requires the keypoint head's **slice** depth to be 51 but allows its **input** depth to be
larger (`keypoint_depth != 51` check in `BoxDecodeStageSemantics.cpp`), so a 51-of-64 padded
head is accepted as-is and Neat ignores the 13 pad channels.

Two constraints that look like they must conflict — pad for speed, don't pad for decode —
turn out not to. Read the validator before assuming a trade-off is real.

## Lesson 6: Pin the CVU stages; don't trust AUTO

`ModelOptions.processcvu.pre_run_target` / `post_run_target` choose where the model's
tessellate/quantize and detessellate/dequantize stages run: `AUTO`, `EV74` or `A65`.

`AUTO` lets Neat's planner choose, and it does not always pick the accelerator. Measured on
this pipeline (4 streams, 15 s window, identical otherwise):

| CVU target | aggregate delivered |
| --- | --- |
| `AUTO` | 76.7 fps |
| **`EV74`** | **86.5 fps** |

Historically `AUTO` was far worse — it once placed `yolo26s-pose`'s post stage on the A65 at
~1.8 s/frame, which starved every other stream. `EV74` is the default in
`config/default.conf` for this reason.

## Lesson 7: `1000 / infer` is NOT a frame rate — and this once fooled us badly

> **This lesson previously said "on a saturated MLA, deeper pipelining makes things worse."
> That was wrong, and the error is worth preserving.** The MLA was never saturated (Lesson 12),
> and deeper pipelining later took this app from 165 → 236 fps. What actually happened is that
> we read a *latency* as if it were a *rate*.

`infer` measures `push` → `pull` for one frame. With N frames in flight, a frame waits behind
the others, so `infer` is that frame's **in-graph latency**, not its service period. Little's
law is the whole story:

```
frames in flight  =  throughput  ×  latency
```

So when pipelining is switched on, `infer` **rises** and throughput **rises too**. Computing
`1000 / infer` and calling it "model fps" makes the model look like it collapsed:

| | infer | 1000/infer ("model fps") | pull fps (TRUE throughput) |
| --- | --- | --- | --- |
| serial (1 in flight) | 22.2 ms | 45.1 | 45.1 |
| pipelined (~1.8 in flight) | 32.3 ms | **30.9 — looks 30% worse** | **57.9 — is 28% better** |

Same model, same board. The rate did not fall; the *latency* rose because we deliberately put
more frames in flight. **The app therefore reports `pull fps` — frames the model actually
completed per second — and has no `model fps` column at all.** A rate you computed from a
latency is not a rate.

## Lesson 8: The GIL is a real pipeline stage

Four streams means four Python overlay threads. Two facts collide:

* **NumPy fancy indexing holds the GIL.** `region[m] = np.clip(region[m] + 60, 0, 255)`
  serialises all four overlay threads.
* **OpenCV releases it.** `cv2.add(region, 60, dst=region, mask=m)` does the same work in C
  and lets the other three threads run.

Micro-benchmarked on the DevKit (13 objects, 1280×720 Y-plane), single-threaded:

| operation | cost |
| --- | --- |
| 4 box edges × 13 | 0.40 ms |
| `putText` LINE_AA × 13 | 0.72 ms |
| `putText` LINE_8 × 13 | 0.39 ms |
| mask blend, fancy-index | **27.75 ms** |
| mask blend, `cv2.add(mask=)` | **5.15 ms** |

The single-thread win is 5.4×. The multi-thread win is larger, because the NumPy version was
also blocking the other three streams while it ran. **When you profile a threaded Python
pipeline, a stage's wall-clock includes time it spent waiting for the GIL** — which is why
the overlay stage reports ~60 ms for work that costs under 1 ms in isolation.

**The C++ port proves it.** `main.cpp` runs the identical models, the identical config and the
identical overlay, but on four real OS threads with no interpreter lock:

| overlay stage | Python | C++ |
| --- | --- | --- |
| detection (boxes + labels) | ~63 ms | **0.50 ms** |
| segmentation (+ masks) | ~138 ms | **11.9 ms** |
| pose (+ skeletons) | ~60 ms | **1.10 ms** |
| **aggregate delivered** | **71 – 80 fps** | **167 – 174 fps** |

Same MLA, same models, ~2.2× the throughput. The gap was never the accelerator — it was the
GIL. If a Python pipeline's per-stage cost is wildly larger than the same work measured in
isolation, suspect lock contention before you suspect the hardware.

## Lesson 9: A live source must be drained, or it kills itself

Three separate bugs in the C++ port had the same root cause, and all three presented as
`GraphRun: sink backpressure timeout (edge_queue=256)` — the source graph's internal queue
filling up. A 60 fps camera does not wait for you. If you are not taking frames out of the
source at the rate it produces them, its queue fills and the stream dies.

**1. Do not hold a ZeroCopy Sample across downstream work.** The pulled `Sample`'s buffer
belongs to the source graph's pool. The obvious straight-line loop —

```cpp
auto sample = source.pull(...);              // holds a pool buffer
model.push(tensors_from_sample(sample));     // ... for the whole
model.pull(...);                             // ... model +
annotate(...);                               // ... overlay +
video.push(...);                             // ... encode
```

— means the source cannot recycle that buffer for the entire chain. Copy the frame out,
let the `Sample` die, and work on your own memory.

**2. Drop BEFORE the copy, not after.** Once a source thread drains at the full 60 fps, the
naive next step is to copy every frame and hand the newest to the worker. That is 1.4 MB per
frame per stream — ~336 MB/s of `memcpy` across four streams — and it starves the workers of
the CPU they need. Pull always (that is what drains the queue); copy only when a worker is
actually waiting for a frame.

**3. `Graph::build()` STARTS the pipeline.** This one was the most instructive. Building each
stream's graphs in a loop —

```cpp
for (spec : specs) {
  rt.source_run = source_graph.build(...);   // camera starts NOW
  rt.model_run  = model_graph.build(...);    // ... and this takes seconds
  rt.video_run  = video_graph.build(...);
}
```

— starts stream 0's camera while streams 1-3 are still loading their MLA archives. By the
time the worker threads start, stream 0's source has been free-running for ~10 seconds with
nobody pulling: 256 frames queued, dead on arrival. Streams 0, 1 and 2 died; **stream 3 —
built last, and so idle for the shortest time — was the only survivor.** That asymmetry is
the whole diagnosis: when N-1 of your N identical streams fail and the last one lives, the
bug is in the *order* you started them, not in the streams.

Build every model first; start the sources last, immediately before the threads that drain
them.

## Lesson 10: Measure with `--duration`, never `--frames`, on a shared resource

A per-stream frame cap only stops the run when *every* stream reaches it. So a fast stream
keeps running — and keeps consuming the one MLA — while the slow streams starve. We once saw
detection run 7934 frames against a 250-frame cap while its peers were pinned near 0.1 fps.
That reports a steady state that never existed.

Use a wall-clock window (`--duration 20`). Every stream gets the same slice of contended
time, which is the thing you are actually trying to measure.

## Lesson 11: The three rates, and the one that is a lie

The profile prints `dec fps`, `pull fps` and `deliv fps`. Read them as a chain:

* **`dec fps`** — what the decoder produced, *including* frames the pusher was too busy to
  take. The source's true rate.
* **`pull fps`** — frames the model actually completed. **The model's true throughput.**
* **`deliv fps`** — frames that reached the encoder. The number that matters.

When all three converge on the camera rate (~60), the pipeline is **source-limited** — the
goal state. That is where this app now sits: ~236 fps aggregate against a 239.8 ceiling.

The lie is any rate you *derive* from `infer` (Lesson 7). There is deliberately no such column.

**And a rate above the source rate is always a bug.** A 59.94 fps camera makes 239.8 the hard
ceiling for four streams. We once printed **261 fps aggregate**, with one stream "delivering"
85 fps from a 60 fps source. That was not throughput — it was a backlog draining (Lesson 13).
The impossibility is what exposed the bug. Always sanity-check against the source rate.

---

## What is still slow, and why

Be honest about the remaining ceiling — it is the most instructive part.

**1. In C++, the host is no longer the bottleneck — the shared MLA is.** `infer` is
16 – 23 ms per model and the four contend for one accelerator. `model fps` (44 – 64) and
`delivered fps` (30 – 48) are now close, which is exactly what "host-limited" stops looking
like. Do not predict multi-model capacity by adding service times: the preprocess / MLA /
dequantize stages of *different* streams overlap, so the aggregate beats `1 / Σ(service)`.
Measure it.

**2. In Python, the overlay round-trip still dominates.** Burning boxes/masks/skeletons into
the video means pulling the full NV12 frame to the A65 and drawing on it — and NumPy holds
the GIL while it does. That is the ~2.2× gap, and it is why the C++ build exists.

If you need in-graph rendering (no host frame at all), Neat has
`nodes::SimaRender` (`core/include/nodes/sima/SimaRender.h`): it consumes the BoxDecode
output and draws **inside the graph**. It is not wired here because it renders **bounding
boxes only** — adopting it would cost the segmentation masks and pose skeletons, which are
the whole point of those two streams. In C++ the host overlay is cheap enough (0.5 – 12 ms)
that the trade is not worth making. That is a real design choice, not an oversight.

**3. Teardown heap abort (Python only).** The Python process intermittently dies at exit with
`malloc(): mismatching next->prev_size`. It fires **after** the profile prints, so the
numbers are valid, but it is a real defect in the teardown path and it is still open.

## Where to look next

* `main.py` — `DECODE_FAMILY`, `NORMALIZE_PRESET`, `make_model()`, `decode_sample()`.
* `config/default.conf` — every knob, with the measurement that justifies its default.
* `tools/pose_probe.py` — study **one** model on its own: one graph, lock-step push/pull, no
  threads, no overlay, no UDP. When a model misbehaves inside a four-model app, that is the
  worst possible place to debug it.
* `model-compilation/compile/_surgery_yolox.py` — the split-head surgery, and why.
