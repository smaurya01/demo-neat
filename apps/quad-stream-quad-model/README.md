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

## Measured behaviour (DevKit 192.168.135.203, 20 frames/stream)

Single-process, single-thread round-robin. Per-stream numbers are *service-time*
FPS (push+pull+decode+annotate+encode for that stream alone):

| stream | task | service FPS |
| --- | --- | --- |
| 0 | detection (on-device decode) | ~15.7 |
| 1 | segmentation (host decode + mask assembly) | ~3.6 |
| 2 | pose (host decode) | ~0.5 |
| 3 | yolox (host decode) | ~12.7 |

**Aggregate ≈ 1.7 FPS across all four** because the four streams are serviced
serially in one Python thread — aggregate ≈ 1 / Σ(per-frame times), so the
slowest stream (pose host-decode) gates the whole loop. This is a genuine,
measured limit, not a target: to sustain four realtime streams you would run one
worker thread per stream (the graphs are already independent) and lighten the
A65 host-decode cost. See `TEACHING.md` → "Measuring & the host-decode
bottleneck". A 2-stream configuration (`--num-streams 2`: detection + seg) runs
comfortably at ~8 FPS aggregate.

## Known limitation

The **YOLOX host decoder runs end to end** (valid annotated output, correct
banner) but currently emits **0 boxes** — the exposed decoupled-head channel
order needs one more on-board verification (`0:4` reg / `4` obj / `5:85` cls per
`model-compilation/work/yolox_s/reports/SURGERY.md`). Detection, segmentation and
pose all decode correctly and validate the raw-head → host-decode design. Set
`QSQM_DEBUG=1` to print the raw tensor shapes each model delivers.
