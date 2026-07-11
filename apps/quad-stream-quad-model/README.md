# quad-stream-quad-model

Four RTSP streams → **four different** compiled INT8 models → four independent
annotated H.264/RTP UDP sinks, in one process. Stream identity is preserved end
to end: the frame pulled from stream *i*'s source is the exact frame pushed into
stream *i*'s model, decoded for stream *i*'s task, annotated in place, and
published on stream *i*'s own UDP port.

| slot | task | model (compiled INT8 archive) | on-device decode? |
| --- | --- | --- | --- |
| 0 | detection | `yolo11s` | **yes** — Neat `BoxDecodeType.YoloV26` |
| 1 | segmentation | `yolo11s-seg` | no — raw heads → host decode |
| 2 | pose | `yolo26s-pose` | no — raw heads → host decode |
| 3 | detection (YOLOX) | `yolox_s` | no — raw heads → host decode |

The archives are large and are **not committed**. The app references them in
place under `model-compilation/work/<model>/compile_int8/...` by default (both
the host and the DevKit see `/workspace` at the same NFS path, so nothing is
copied). To use your own copies, drop archives into `assets/models/` and set
`stream<i>_model=...` in `config/default.conf`.

## Why three of the four models decode on the host (the core lesson)

The `compile_ready` surgery for all four models deliberately exposes **raw
per-scale head tensors** and cuts the data-dependent decode/NMS tail so the whole
graph stays on the MLA (`A65:0`, one `.elf`, zero `.so`). Neat's built-in fused
`BoxDecode` covers only the plain **detection** family, so:

* stream 0 (detection) uses the on-device `BoxDecodeType.YoloV26` decode and
  `pyneat.decode_bbox`;
* streams 1–3 pull the raw heads and decode them on the A65 in NumPy
  (`src/decoders.py`): anchor-grid + stride geometry, sigmoid/exp, NMS,
  letterbox-inverse, and (seg) prototype-mask assembly.

See `TEACHING.md` for the full design discussion and `src/decoders.py` for the math.

## Run it

### Human UX (a real terminal on your workstation)

```bash
# from the SDK container host, with the DevKit helper sourced:
source /usr/local/bin/devkit.sh 192.168.135.203 sima 22
dk /workspace/demo-neat/apps/quad-stream-quad-model/main.py --frames 100
```

`dk` gives you the nice interactive DevKit UX. It needs a TTY.

### CI / non-interactive fallback (ssh)

`dk` hangs without a TTY, so scripted/agent runs use ssh and wrap the board
command in `timeout`:

```bash
timeout 300 ssh -o BatchMode=yes sima@192.168.135.203 \
  'source /media/nvme/pyneat/bin/activate; \
   cd /workspace/demo-neat/apps/quad-stream-quad-model; \
   python main.py --num-streams 4 --frames 100 --score 0.25'
```

Useful flags: `--num-streams {1..4}` (drop to 2 for a lighter, higher-FPS
pipeline), `--frames N` (0 = forever), `--rtsp URL` (override all sources),
`--score`, `--nms`, `--top-k`, `--queue-depth`, `--print-backend`.

## View the four annotated outputs

Each stream publishes to `udp_host:port`. With the defaults stream *i* → port
`5206 + 2*i`. On the machine at `udp_host`, one viewer per port:

```bash
# stream 0 detection  :5206   stream 1 segmentation :5208
# stream 2 pose       :5210   stream 3 yolox        :5212
for P in 5206 5208 5210 5212; do
  gst-launch-1.0 -v udpsrc port=$P \
    caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
    ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false &
done
```

Each frame carries a burned-in banner `S<i> <TASK> :<port>` so you can tell the
four windows apart at a glance.

## Sanity-check the RTSP source first

```bash
ffprobe -hide_banner -rtsp_transport tcp rtsp://192.168.132.129:8555/stream
```

## Time profile

Every run prints a per-stage breakdown and two different FPS numbers. They answer different
questions and must not be conflated:

- **`model fps`** = `1000 / mean(infer)`. The **model stage alone** — EV74 preprocess + MLA + any
  on-device decode. No host decode, no overlay. This is the "can this model do 60 fps" number.
- **`delivered fps`** = frames actually published to UDP per second of wall clock. Includes host
  decode + overlay + encode.

Useful flags:

```bash
--no-overlay          # skip host decode + annotation; isolates the MODEL rate
--pipeline-depth 1    # lock-step push/pull, so `infer` is the TRUE per-frame model cost
                      # (with depth > 1, `infer` measures graph latency, not service time)
--task <t>            # run ONE model solo, MLA uncontended
--tasks a,b,c         # custom model set, one per stream slot
--duration <seconds>  # measure over a fixed wall-clock window (see the warning below)
--warmup-frames N     # exclude the first N frames per stream from the means
```

> **Measure with `--duration`, not `--frames`, whenever streams share the MLA.** A per-stream frame
> cap only stops the run when *all* streams reach it, so a fast stream keeps running — and keeps
> consuming the one MLA — while slow streams starve. That reports rates no steady state ever
> produced (we saw detection run 7934 frames against a 250-frame cap while its peers were starved to
> ~0.1 fps).

## Measured behaviour (DevKit 192.168.2.103, RTSP 1280x720 H.264 @ 59.94 fps)

### Model rate, each model solo (`--task <t> --no-overlay --pipeline-depth 1`)

| stream-model | infer ms | **model fps** |
| --- | --- | --- |
| detection `yolo11s` (on-device decode) | 7.37 | **135.6** |
| segmentation `yolo11s-seg` | 9.54 | **104.8** |
| yolox `yolox_s` | 8.80 | **113.6** |
| pose `yolo26s-pose` | **1821.92** | **0.5** ← broken, see below |

### Three models concurrently, `--no-overlay`, 25 s window

All three hold the full source rate; one MLA sustains ~167 inferences/s in aggregate.

| stream-model | model fps | delivered fps |
| --- | --- | --- |
| detection `yolo11s` | 96.2 | 55.81 |
| segmentation `yolo11s-seg` | 87.3 | 55.73 |
| yolox `yolox_s` | 90.5 | 55.77 |
| **aggregate** | | **167.32** |

Note this is **far more than `1 / Σ(service times)`** would predict — the EV74-preprocess / MLA /
dequantize stages of *different* streams overlap. Do not predict multi-model capacity by adding
service times; measure it.

### Three models concurrently, WITH overlay — only detection survives

| stream-model | infer ms | decode ms | overlay ms | delivered fps |
| --- | --- | --- | --- | --- |
| detection `yolo11s` | 8.11 | **0.62** | 4.31 | **56.25** |
| segmentation `yolo11s-seg` | 34.53 | **337.16** | **77.43** | 0.08 |
| yolox `yolox_s` | 147.39 | **142.59** | 2.07 | 0.00 |

This is the core lesson of the app, quantified: detection uses Neat's **fused on-device**
`BoxDecodeType.YoloV26` and decodes in **0.62 ms**. Segmentation and yolox pay an **A65 host NumPy
decode** of the raw heads (`src/decoders.py`) costing **337 ms** and **143 ms** per frame. The host
decode — not the MLA — is what destroys their delivered FPS.

## `tools/pose_probe.py` — study one model on its own

The quad app runs four models, three graphs each, across a dozen threads. When one model misbehaves
that is the worst possible place to debug it. `tools/pose_probe.py` strips everything away: ONE model
graph (`input -> model -> output`), lock-step push/pull, no threads, no overlay, no UDP.

```bash
# what does this model actually cost, and are its outputs right?
python tools/pose_probe.py --task pose --iters 20 --save-out /tmp/pose.jpg

# control: the same probe on a model known to be fast
python tools/pose_probe.py --task segmentation --iters 20

# can ANY runtime option move the number?
python tools/pose_probe.py --sweep

# empirically derive the correct keypoint decode formula (see below)
python tools/pose_probe.py --kpt-scan
```

It prints every raw output tensor's shape/dtype, the true per-frame model cost, the host-decode cost,
and a sanity check of the decoded result — so you can tell *slow but correct* from *slow and wrong*.
Because it is lock-step, its `infer` is the real model service time, not graph latency.

### It caught a real bug: the YOLO26 keypoint decode was 2x too big

`src/decoders.py` originally decoded keypoints with the **Ultralytics v8/v11** pose formula
`(k * 2.0 + i) * stride`. **YOLO26's `one2one_cv4_kpts` head does not use that `2x`.** The correct
decode is `(k + anchor) * stride` with `anchor = i + 0.5` — the *same* anchor point the box decode
already uses.

This was a silent bug: no error, and the person **boxes were perfect**, so the detection counts looked
completely healthy (7 people, 17 keypoints each, all in-frame). Only a rendered frame showed the
skeletons sprawling at exactly twice their true size.

`--kpt-scan` settles it by measurement rather than guesswork. The boxes decode correctly and
independently, so they are free ground truth: score each candidate formula by how many visible
keypoints land inside their own person box. **But that check alone is gameable** — a sigmoid-bounded
variant collapses every keypoint onto the anchor and scores 100% while being useless. So it also
scores *span* (skeleton height / box height) and *anatomy* (nose above ankles):

| formula | inside% | span | nose<ankle |
| --- | --- | --- | --- |
| **B `(k + i + 0.5) * s`  ← correct** | **99.9%** | **0.83** | **100%** |
| C `(k + i) * s` | 99.6% | 0.83 | 100% |
| D `(2k + i + 0.5) * s` | 52.2% | 1.67 | 100% |
| A `(2k + i) * s`  ← the old v8 formula | 44.2% | **1.67** ← 2x too big | 100% |
| E `(2*sigmoid(k) - 0.5 + i) * s` | 100% ← gamed | **0.25** ← collapsed | 100% |

## Known limitations

1. **`yolo26s-pose` is unusable, and all four models together will not run.** Pose costs **1.82 s per
   frame inside the model graph** — measured with `--no-overlay`, so this is *not* host decode. With
   pose in the mix, it holds the MLA so long that every other graph backs up and the run dies with
   `model push failed`. Root cause is a **compile artifact**, not an app bug:

   - Its MLA is healthy: 1,455,681 cycles vs yolo11s's 1,205,496 (**1.21x**) — it *should* run ~9 ms.
     (Per-layer cycles are in `*_mla_stats.yaml` inside the `_mpk.tar.gz`.)
   - Output volume is a red herring: seg moves 1.79 M floats through its tail in 9.5 ms; pose moves
     only 0.47 M in 1822 ms.
   - The **post-MLA tail** is the cost. Dumping the MPK contract
     (`ModelOptions.verbose.planner = True`) shows pose routes **every one of its 9 outputs through
     its own `slice_MLA_0/tuple_get_item_N_slice_transform` stage**, whereas `yolo11s-seg` emits most
     outputs from a **single fused `MLA_0_ofm_unpack_transform`**. Nine serial per-output slice
     transforms is the 1.8 s.
   - `ModelOptions.processcvu.post_run_target = "EV74"` does **not** help — it is the stage
     count/shape, not the execution target.

   **Fix: recompile `yolo26s-pose` so its post-MLA tail fuses into one `ofm_unpack`.** Not yet done.
   Meanwhile run a pose-free set: `--tasks detection,segmentation,yolox`.

2. **Teardown heap abort.** The process intermittently dies at exit with
   `malloc(): mismatching next->prev_size` / `double free` (exit 134/139). It fires **after** the
   profile prints, so the numbers are valid, but it is a real defect in the teardown path.

3. Set `QSQM_DEBUG=1` to print the raw tensor shapes each model delivers.
