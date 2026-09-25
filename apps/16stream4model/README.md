# 16stream4model: 16 × 720p30 RTSP streams, 4 models, one process

Sixteen 1280×720 @ 30 fps H.264 RTSP cameras run through four models on one Modalix DevKit.
Each group of four streams shares one model:

- three object detectors: YOLO26n, YOLO11n, and YOLOv8n
- one pose estimator: YOLO26n-pose

Video and results go to Neat Insight. The encoded video is forwarded untouched, and Insight
draws boxes and skeletons from JSON metadata. The host never decodes a frame to draw on it or
re-encodes one.

**Result:** the output rate equals the input rate, 480 / 480 fps, with every stream at 30 fps.
See [Measured results](#measured-results).

## Contents

- [Topology](#topology)
- [RTSP stream mapping](#rtsp-stream-mapping)
- [Models](#models)
- [Build](#build-sdk-host)
- [Run the test](#run-the-test)
- [Measured results](#measured-results)
- [Configuration](#configuration)
- [Command-line flags](#command-line-flags)
- [Layout](#layout)

## Topology

```
                       ┌─ model 0: YOLO26n det   ◄── streams 0–3
16 RTSP streams ──────►├─ model 1: YOLO11n det   ◄── streams 4–7
(720p30 H.264)         ├─ model 2: YOLOv8n det   ◄── streams 8–11
                       └─ model 3: YOLO26n pose  ◄── streams 12–15
```

Each model has its own `Graph`, its own `Run`, and its own puller thread, so a stall in one group
cannot back-pressure another group's decoders. Per stream, Core fuses two branches out of **one**
RTSP session:

```
RTSP H.264
  |- encoded edge (untouched) --> VideoSender ------> Insight video    channel i  (UDP 9000+i)
  \- HW decode -> model of its group -------------> MetadataSender -> Insight metadata channel i (UDP 9100+i)
```

Frames enter each model through a per-stream "latest frame" mux (`RealtimeLatestByStream`). A
slow model drops that stream's stale frames instead of stalling its decoder.

## RTSP stream mapping

Stream `i` reads the RTSP URL of Insight source `src<i+1>` (`<rtsp-url-i+1>` in the config) and
publishes to Insight channel `i`. This holds
under `--only-model` and `--streams` too: a stream always keeps its channel, ports and stream id.
`scripts/start_sources.sh` assigns exactly this mapping. The clips come from the Insight catalog,
and every clip is 1280×720, 30/1 fps, H.264 Constrained Baseline (no B-frames).

| stream | Insight source | config value (`stream<i>_rtsp`) | clip | length | model | video UDP | metadata UDP |
|---:|---|---|---|---:|---|---:|---:|
| 0  | src1  | `<rtsp-url-1>`                    | `parking_garage_cars_720p30_30fps_h264.mp4`        | 35.7 s | YOLO26n det  | 9000 | 9100 |
| 1  | src2  | `<rtsp-url-2>`                    | `sj_almaden_street_720p30_30fps_h264.mp4`          | 37.8 s | YOLO26n det  | 9001 | 9101 |
| 2  | src3  | `<rtsp-url-3>`                    | `sj_highway_720p30_30fps_h264.mp4`                 | 31.2 s | YOLO26n det  | 9002 | 9102 |
| 3  | src4  | `<rtsp-url-4>`                    | `sj_intersection_san_carlos_720p30_30fps_h264.mp4` | 31.1 s | YOLO26n det  | 9003 | 9103 |
| 4  | src5  | `<rtsp-url-5>`                    | `sj_highway2_720p30_30fps_h264.mp4`                | 31.5 s | YOLO11n det  | 9004 | 9104 |
| 5  | src6  | `<rtsp-url-6>`                    | `parking_garage_cars_720p30_30fps_h264.mp4`        | 35.7 s | YOLO11n det  | 9005 | 9105 |
| 6  | src7  | `<rtsp-url-7>`                    | `sj_highway3_720p30_30fps_h264.mp4`                | 31.0 s | YOLO11n det  | 9006 | 9106 |
| 7  | src8  | `<rtsp-url-8>`                    | `sj_highway4_720p30_30fps_h264.mp4`                | 36.9 s | YOLO11n det  | 9007 | 9107 |
| 8  | src9  | `<rtsp-url-9>`                    | `sj_park_720p30_30fps_h264.mp4`                    | 30.8 s | YOLOv8n det  | 9008 | 9108 |
| 9  | src10 | `<rtsp-url-10>`                   | `sj_intersection_san_carlos_720p30_30fps_h264.mp4` | 31.1 s | YOLOv8n det  | 9009 | 9109 |
| 10 | src11 | `<rtsp-url-11>`                   | `sj_walking_street_720p30_30fps_h264.mp4`          | 31.3 s | YOLOv8n det  | 9010 | 9110 |
| 11 | src12 | `<rtsp-url-12>`                   | `sj_almaden_street_720p30_30fps_h264.mp4`          | 37.8 s | YOLOv8n det  | 9011 | 9111 |
| 12 | src13 | `<rtsp-url-13>`                   | `person_gym_bike_720p30_30fps_h264.mp4`            | 31.9 s | YOLO26n pose | 9012 | 9112 |
| 13 | src14 | `<rtsp-url-14>`                   | `person_gym_jumping_720p30_30fps_h264.mp4`         | 31.1 s | YOLO26n pose | 9013 | 9113 |
| 14 | src15 | `<rtsp-url-15>`                   | `person_gym_threadmill_720p30_30fps_h264.mp4`      | 35.2 s | YOLO26n pose | 9014 | 9114 |
| 15 | src16 | `<rtsp-url-16>`                   | `person_gym_workout_720p30_30fps_h264.mp4`         | 33.2 s | YOLO26n pose | 9015 | 9115 |

- Clip paths in Insight are `catalog/<name>/<name>_720p30_30fps_h264.mp4`.
- The pose group gets the four person/gym clips. The detector groups get traffic, street, and
  parking scenes.
- Some clips appear twice, on different models: `parking_garage_cars` (src1, src6),
  `sj_almaden_street` (src2, src12), and `sj_intersection_san_carlos` (src4, src10). This is a
  handy way to compare YOLO26n, YOLO11n, and YOLOv8n side by side on the same scene.
- The clips loop inside the RTSP session, with no gap at the loop point.
- Verify the sources in Insight. Its media sources page must show src1–src16 playing the clips
  above, and the viewer must show all 16 channels live once the app runs (see
  [Watch in Insight](#4-watch-in-insight)).

## Models

| model | streams | archive (`assets/models/`) | source | decode | classes |
|---|---|---|---|---|---:|
| 0 `yolo26n_det`  | 0–3   | `yolo26n-det-int8-b1.tar.gz` | SDK 2.1.3 direct artifact | `yolo26` | 80 (COCO) |
| 1 `yolo11n_det`  | 4–7   | `yolo_11n_mpk.tar.gz`        | model zoo `yolo_11n`      | `yolov8` | 80 (COCO) |
| 2 `yolov8n_det`  | 8–11  | `yolo_v8n_mpk.tar.gz`        | model zoo `yolo_v8n`      | `yolov8` | 80 (COCO) |
| 3 `yolo26n_pose` | 12–15 | `yolo_26n_pose_mpk.tar.gz`   | model zoo `yolo_26n_pose` | `YoloV26Pose`, `score_is_prob=true` | 1 (person), 17 keypoints |

- The zoo YOLO11n and YOLOv8n keep the raw DFL head, so they use the `yolov8` decode family.
- The zoo `yolo_26n_pose` score head is already sigmoided, so it needs `score_is_prob=true`.
  Without it, every background anchor scores about 0.5 and floods the output with false poses.

Download commands, run from `assets/models/`:

```bash
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK2.1.3/models/modalix/yolo26-detection/yolo26n-det-int8-b1.tar.gz"
sima-cli modelzoo --boardtype modalix get yolo_11n
sima-cli modelzoo --boardtype modalix get yolo_v8n
sima-cli modelzoo --boardtype modalix get yolo_26n_pose
```

Model archives are gitignored, so download them after cloning.

Metadata sent to Insight:

- **Detection streams:** `type: "object-detection"`, `data.objects[]` (label, confidence,
  bbox).
- **Pose streams:** `type: "pose-estimation"`, `data.poses[]` (bbox, confidence, and 17 named
  COCO keypoints with x, y, and confidence).
- Both carry `rtp_timestamp`, which Insight uses to match overlays to video frames.

## Build (SDK host)

```bash
cd /workspace/demo-neat/apps/16stream4model
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr \
      -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++-12
cmake --build build -- -j1
```

The result is `build/16stream4model`, an aarch64 binary that runs on the board.

## Run the test

### 1. Start the 16 sources (SDK host)

```bash
./scripts/start_sources.sh          # stop all, assign the table above, start, 20 s delivery check
./scripts/start_sources.sh --check  # verify only; changes nothing
./scripts/start_sources.sh --stop   # stop all sources
```

The script exits non-zero with a message when Insight is unreachable, a clip is missing, or an
assign or start call fails. The expected result is that every row reads about 30 fps and says `OK`, followed by
`all 16 sources delivering -- safe to launch the app`. Do not launch until then.

Restart the sources after any aborted app run.

Before the first run, fill in the placeholders in `config/default.conf`:

- `stream<i>_rtsp=<rtsp-url-N>`: the RTSP URL of Insight source `srcN`, copied from Insight's
  media sources page. The DevKit connects from outside the SDK container, so use the host
  address the DevKit can reach, not `127.0.0.1`.
- `insight_host=<insight-host-ip>`: the Insight host's IP as seen from the DevKit. `curl -sk
  https://127.0.0.1:9900/api/server-ip` on the SDK host reports it.

The app refuses to start, and names the key, while any `<...>` placeholder is left. To keep a
filled-in copy out of git, save it outside the repo, or under `logs/`, and pass it with
`--config <path>`.

### 2. Run the app on the DevKit

**From the SDK host with `dk`.** This is the usual way. Run it from the app directory:

```bash
cd /workspace/demo-neat/apps/16stream4model
dk ./build/16stream4model --duration 60 --report-interval 10
```

- `dk` runs the binary on the DevKit over SSH, in the same directory on the shared
  `/workspace`, so relative paths such as `./config/default.conf` and `./assets/models/` resolve
  as they do locally.
- It streams the app's output back, with each line prefixed `[DEVKIT][STDOUT]`.
- It returns the app's exit code when the app finishes on its own (see
  [Read the output](#3-read-the-output)).
- Ctrl-C stops the remote app and leaves no process behind on the board. `dk` then exits 0, not
  the app's 130.
- Any flag works the same way:

```bash
dk ./build/16stream4model                                  # until Ctrl-C
dk ./build/16stream4model --only-model 3 --duration 30     # one model's streams only
dk ./build/16stream4model --config ./config/default.conf --duration 300 --report-interval 30
```

- `dk status` checks the DevKit connection.
- `dk shell` opens a shell on the board.
- `scripts/start_sources.sh` is not run through `dk`. It talks to Insight on the SDK host, so
  run it there directly.

**Directly on the board.** Use this for long runs that must outlive the host session:

```bash
ssh sima@<devkit-ip>
cd /workspace/demo-neat/apps/16stream4model
./build/16stream4model --duration 60 --report-interval 10
```

For a long detached soak, run this on the board:

```bash
nohup ./build/16stream4model --duration 1800 --report-interval 60 > logs/soak.log 2>&1 &
```

When SIGHUP is already ignored (as it is under `nohup`), the app leaves it that way, so the soak
survives an SSH drop. SIGINT and SIGTERM stop it cleanly. A second Ctrl-C exits immediately.

### 3. Read the output

The live line shows total fps against the input, then per-stream fps in four groups, one per
model:

```
[t=90  s] 480.0  fps (in 480)  | 30.0 30.0 30.0 30.0  | 30.0 30.0 30.0 30.0  | 30.0 30.0 30.0 30.0  | 30.0 30.0 30.0 30.0
             total    input      model 0 (26n det)       model 1 (11n det)       model 2 (v8n det)       model 3 (26n pose)
```

- The **first window reads above 30 fps**. That is the pipeline catching up on frames the RTSP
  sources buffered while the four graphs were built. It is not a problem.
- The **final summary** lists each stream's frames, fps, ratio to source, objects per frame,
  metadata sent/failed, and host-side parse and metadata time. It ends with:
  ```
  aggregate: 479.5 fps of 480.0 input (99.9%) across 16 streams, 145156 frames
  slowest stream: 99.1% of source rate  ->  PASS (target >= 97.0% per stream)
  ```
- **Exit code:**
  - 0 on PASS.
  - 3 on FAIL (a stream below `target_ratio`).
  - 1 on an error: a bad config value or flag, a model or graph build failure, or a pipeline
    that failed at runtime.
  - 130 when stopped by a signal. A shortened window is not a verdict.
- `[stall]` / `[resume]` lines report a stream that produced no frame for `stall_warn_s` (3 s),
  and when it came back. A stream that never delivers its first frame is reported too.
- **A model's pipeline fails:** for example, a source whose rate does not match the pinned fps.
  Its puller logs the error and the app stops with exit 1 after 10 s without a frame. It stops
  at once if the endpoint closes.

### 4. Watch in Insight

Open the viewer on the SDK host. Get the URL with:

```bash
curl -sk "https://127.0.0.1:9900/api/viewer-url?src=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
```

Keep one viewer open while checking overlays. Insight renders metadata reliably for a single
viewer only.

## Measured results

Current config: YOLO26n det / YOLO11n det / YOLOv8n det / YOLO26n pose, 4 streams each, all
sources 1280×720 @ 30 fps. Measured on the DevKit on NEAT 0.4.0 / SDK 2.1.3 with a 300 s run.

| model | streams | input fps | output fps | per stream | objects / frame |
|---|---|---:|---:|---|---|
| YOLO26n det  | 0–3   | 120 | 120.0 | 30.00 | 4.6–7.3 |
| YOLO11n det  | 4–7   | 120 | 119.8 | 29.85–30.00 | 3.3–6.6 |
| YOLOv8n det  | 8–11  | 120 | 119.7 | 29.73–30.00 | 2.2–6.6 |
| YOLO26n pose | 12–15 | 120 | 120.0 | 29.97–30.00 | 1.0–1.5 persons |
| **total** | 16 | **480** | **479.5 (99.9%)** | every 30 s window from t=60 s: 30.0 on all 16 | |

- **Result:** PASS, with no stalls and no metadata send failures.
- **Host work per frame:** parse takes about 0.05 ms for detection and 0.13 ms for pose.
  Building and sending the metadata JSON takes 0.1–0.2 ms for detection and 0.3–0.45 ms for
  pose.
- **Board load:** the app used about 3.6 A65 cores, and the board stayed about 60% idle.
- **Headroom:** the four models together keep the MLA close to full. A heavier model or a 60 fps
  source in any group pulls every group below the source rate.

## Configuration

Everything is in `config/default.conf`, in `key=value` form. Numeric values are parsed strictly:
`fps=29.97` is an error, not 29. An unknown key prints a warning.

**Per model** (`m` = 0–3):

| key | example | meaning |
|---|---|---|
| `model<m>_name` | `yolo26n_pose` | Label used in logs and reports. |
| `model<m>_archive` | `./assets/models/yolo_26n_pose_mpk.tar.gz` | Compiled model archive. |
| `model<m>_labels` | `./assets/labels/person.txt` | One label per line. |
| `model<m>_classes` | `1` | Real class count (required). |
| `model<m>_task` | `detection` \| `pose` | `pose` selects `YoloV26Pose` and pose metadata. |
| `model<m>_decode` | `yolo26` \| `yolov8` | Detection decode family (ignored for pose). |
| `model<m>_score_is_prob` | `false` | `true` for archives whose score head is already sigmoided. |

**Per stream** (`i` = 0–15):

| key | default | meaning |
|---|---|---|
| `stream<i>_rtsp` | none (required) | Source URL (`<rtsp-url-i+1>` in the config). |
| `stream<i>_model` | `i / 4` | Which model this stream feeds. |
| `stream<i>_fps` | global `fps` | This stream's integer source rate (pinned). |

**Global:**

| key | default | meaning |
|---|---|---|
| `visible_streams`, `video_enabled` | -1, `true` | Publish only channels below this number (-1 = all); `false` sends metadata only. |
| `rtsp_transport`, `latency_ms`, `drop_on_latency` | `tcp`, 100, `true` | RTSP transport and jitter buffer. |
| `decoder_tuning` | `throughput-low-latency` | Hardware decoder profile. |
| `verbose`, `print_backend` | `false`, `false` | Same as `--verbose` and `--print-backend`. |
| `insight_host`, `video_port_base`, `metadata_port_base` | `127.0.0.1`, 9000, 9100 | Insight destination: the Insight host's IP as seen from the DevKit (`<insight-host-ip>` in the config). Channel `i` uses base + `i`. |
| `width`, `height`, `fps` | 1280, 720, 30 | Source contract. |
| `skip_rtsp_probe` | `true` | Trust the contract instead of probing each URL. This is also what lets Core admit decoders with zero-copy output. |
| `decoder_buffers`, `decoder_input_buffers` | 8, 2 | Core admission raises output buffers to 18 when every decoder has known caps. |
| `max_inflight_per_stream`, `max_inflight_total` | 2, 8 | Frames admitted into each model, per stream and per model. |
| `queue_depth`, `internal_queue_depth`, `inference_async` | 16, 1, `true` | Detections queue depth and graph execution options. |
| `detections_drop` | `false` | With 4 streams per model, back-pressure beats dropping. |
| `min_score`, `nms_iou`, `max_detections` | 0.30, 0.60, 50 | Decode thresholds, applied to every model. |
| `warmup_frames` | 30 | Frames per stream excluded from the fps figure. |
| `target_ratio` | 0.97 | PASS when every stream reaches this fraction of its source rate. |
| `report_interval`, `duration`, `stall_warn_s` | 5, 0, 3 | Reporting cadence, run length (0 = until Ctrl-C), and stall warning threshold. |

## Command-line flags

| flag | effect |
|---|---|
| `--config <path>` | Config file (default `./config/default.conf`). |
| `--duration <s>` | Stop after this many seconds; 0 = until Ctrl-C. |
| `--report-interval <s>` | Live per-stream fps every N seconds; 0 = summary only. |
| `--streams <n>` | Use only the first n streams. |
| `--only-model <m>` | Run only model m's streams. This measures one model's own ceiling. |
| `--visible <n>` | Publish only the first n streams to Insight. |
| `--no-video` | Metadata only, no video passthrough. |
| `--source-only` | Receive and decode every stream, run no model. This measures the source path ceiling. |
| `--inflight <per>,<total>` | Override `max_inflight_per_stream` and `max_inflight_total`. |
| `--internal-queue <n>` | Override `internal_queue_depth` (-1 = framework default). |
| `--measure [m]` | Neat in-graph timing, with the plugin trace on model m. |
| `--verbose` | Unmute the Neat model planner. |
| `--print-backend` | Dump each model's generated GStreamer pipeline. |
| `--help` | Print usage. |

## Layout

```
main.cpp                    the app
CMakeLists.txt              build (links SimaNeat::sima_neat)
support/detection_egress.h  Insight object-detection JSON envelope (vendored)
config/default.conf         16 streams, 4 models (the configuration above)
scripts/start_sources.sh    assign, start, and verify the 16 Insight sources (the mapping above)
assets/models/              model archives (gitignored; see Models for download commands)
assets/labels/              coco_label.txt (80 classes), person.txt (pose)
logs/                       run logs (gitignored)
```
