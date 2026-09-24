# Chapter 18 — A C++ video application, end to end

*RTSP in, YOLO26 on the MLA, annotated video out to Insight — built in the SDK, running on the board.*

---

This is the chapter that ties the rest together. Everything you have installed gets used at once:
the SDK to build, Insight to supply a stream and render the result, `dk` to run on the board, and
the MLA to do the work.

```text
  Insight (host)                    DevKit (board)                    Insight (host)
  ──────────────                    ──────────────                    ──────────────
  video file                                                          Video Viewer
      │                                                                     ▲
      └── RTSP ──▶  decode ──┬──▶ H.264 encode ── UDP 9000 ────────────────┘
       src1                  │                                              ▲
                             └──▶ YOLO26 (MLA) ── detections ── UDP 9100 ───┘
```

One graph, two branches. Both carry timestamps from the same decoded frame, which is what lets
Insight draw each box on the frame it came from.

---

## Before you start

| Requirement | Chapter |
|---|---|
| Board on the network, software 2.1.3 | [6](06-connect-to-router.md) · [9](09-check-and-update-image.md) |
| NVMe mounted at `/media/nvme` | [8](08-mount-nvme.md) |
| Neat SDK installed and paired | [15](15-install-neat-sdk.md) |
| `dk status` names your DevKit | [16](16-devkit-tool-dk.md) |
| Insight reachable in a browser | [17](17-neat-insight.md) |
| A YOLO26 model on the board | [12](12-object-detection.md) |
| A **720p** video file to stream | Any `mp4` with people, cars or traffic in it |

---

## Step 1 — Start the SDK and attach VS Code

Start the SDK container on the host, then attach the editor to it:

1. Open VS Code on the host.
2. Command Palette (`Ctrl+Shift+P`) → **Dev Containers: Attach to Running Container…**
3. Pick the `sima-neat/sdk` container.
4. In the attached window, open `/workspace`.

You are now editing files that the host, the SDK and the DevKit all see. Confirm the board is
there before going further:

```bash
sima-user@sdk:/workspace$ dk status
```

<!-- screenshot: VS Code attached to the sima-neat/sdk container with /workspace open -->

---

## Step 2 — Open Insight

Ask the SDK for the URL rather than assuming the port:

```bash
sima-user@sdk:/workspace$ neat
```

Take the **Insight Web UI** address from the `Exposed Ports` block and open it in a browser. Note
two more rows from that same table — you will need them in the config:

| Row | Used for |
|---|---|
| `videoUDP` (from 9000) | Where the application sends H.264 video |
| `metadataUDP` (from 9100) | Where it sends detections |

Accept the self-signed certificate warning. Check the **devkit-ip** appears in the top-right corner
of the UI — if it does not, the SDK is not paired and the video will have nowhere to come from.

<!-- screenshot: Insight landing page with devkit-ip visible top right -->

---

## Step 3 — Create the RTSP stream

Insight turns a video file into a live RTSP source, so you do not need a camera on your desk.

**3a. Get the video in.** Go to **Media Library** → **Import Media**. The **Catalog** tab is the
path of least resistance here, because it lets you *choose* the encoding rather than discover it
afterwards ([Chapter 17](17-neat-insight.md#getting-test-media-in)):

1. Pick a scene with plenty to detect — `Parking Garage Cars`, `San Jose Highway 2` and
   `San Jose San Carlos Intersection` are all good for a COCO detector.
2. Set **Resolution & FPS** to **`720p30`** and **Codec** to **H.264**. The variant list should
   then read **`720p30 / 30 fps / H.264 / 1280×720`** — that exact one.
3. Check the size, then press **Import**.

**Use 1280×720 at 30 fps.** Every other number in this chapter follows from it: the `stream_*`
values in `config.yaml`, the frame size the encoder is built for, and the input bounds the model
is given. A 120 fps or 1080p variant will run, but you would have to change all three together,
and a mismatch between the stream and the config is the most tedious failure to debug here.

Uploading your own clip through **Local files** works equally well — as long as it is **H.264,
1280×720, 30 fps**. Select it after import and confirm those numbers on the metadata panel.

<!-- screenshot: Media Library with the 720p H.264 catalog variant imported -->

**3b. Assign it to a source slot.** Go to the **RTSP Source** tab. Insight has three slots —
`src1`, `src2`, `src3` — all served on port `8554`. Assign your file to **`src1`**, either by
picking it in the slot's dropdown or with **auto-assign**.

**3c. Start the stream.** Press start on `src1`. The slot should switch to a running state and
show its URL:

```text
rtsp://<insight-host>:8554/src1
```

Copy that URL exactly — it goes straight into the config. Leave the stream running.

<!-- screenshot: RTSP Source tab, src1 assigned and started, URL visible -->

> **Copy that URL.** It is `<rtsp-url>` in the config in Step 4.

---

## Step 4 — The project

Create it in the shared workspace, so the SDK builds it and the board can see it:

```bash
sima-user@sdk:/workspace$ mkdir -p /workspace/rtsp-detector && cd /workspace/rtsp-detector
```

```text
/workspace/rtsp-detector/
├── CMakeLists.txt
├── config.yaml
├── coco_labels.txt      ← copy from tutorial/assets/coco_labels.txt
├── main.cpp
└── build/               ← created by cmake
```

### `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.16)
project(rtsp_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(rtsp_detector main.cpp)
target_link_libraries(rtsp_detector PRIVATE SimaNeat::sima_neat)
```

### `config.yaml`

A flat mapping — one `key: value` per line — so the application can read it without pulling in a
YAML library:

```yaml
# Model
model_path: /media/nvme/example/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
labels: /workspace/rtsp-detector/coco_labels.txt

# Source
rtsp_url: <rtsp-url>                        # the URL Insight shows for src1
stream_width: 1280
stream_height: 720
stream_fps: 30
latency_ms: 100

# Where Insight is listening
insight_host: <host-ip>                     # must be reachable from the board
video_port: 9000
metadata_port: 9100

# Detection
min_score: 0.30
nms_iou: 0.60
max_detections: 50
```

Fill in the two placeholders:

| Placeholder | Value |
|---|---|
| `<rtsp-url>` | The URL Insight showed for `src1` in Step 3, copied exactly |
| `<host-ip>` | The machine running Insight — the IP from the **Insight Web UI** row of `neat` |

Both must be reachable **from the board**, which is where the application runs.

The `stream_*` values are already correct for the `720p30` variant from Step 3.

### `main.cpp`

```cpp
// rtsp-detector — one RTSP stream in, YOLO26 on the MLA,
// H.264 video and detection metadata out to Insight.

#include "neat.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace neat = simaai::neat;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r");
  const auto e = s.find_last_not_of(" \t\r");
  return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

// The config is a flat "key: value" mapping, so splitting on the first colon
// is enough. Values such as rtsp:// URLs keep their own colons.
std::map<std::string, std::string> load_config(const std::string& path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open config: " + path);

  std::map<std::string, std::string> cfg;
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos)
      line = line.substr(0, hash);
    const auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    const std::string key = trim(line.substr(0, colon));
    const std::string value = trim(line.substr(colon + 1));
    if (!key.empty() && !value.empty())
      cfg[key] = value;
  }
  return cfg;
}

std::string require(const std::map<std::string, std::string>& cfg, const std::string& key) {
  const auto it = cfg.find(key);
  if (it == cfg.end())
    throw std::runtime_error("config is missing '" + key + "'");
  return it->second;
}

int as_int(const std::map<std::string, std::string>& cfg, const std::string& key, int fallback) {
  const auto it = cfg.find(key);
  return it == cfg.end() ? fallback : std::stoi(it->second);
}

double as_double(const std::map<std::string, std::string>& cfg, const std::string& key,
                 double fallback) {
  const auto it = cfg.find(key);
  return it == cfg.end() ? fallback : std::stod(it->second);
}

std::vector<std::string> load_labels(const std::string& path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open labels: " + path);
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    const std::string name = trim(line);
    if (!name.empty())
      labels.push_back(name);
  }
  if (labels.empty())
    throw std::runtime_error("labels file is empty: " + path);
  return labels;
}

// Insight expects {"objects":[{"id","label","confidence","bbox":[x,y,w,h]}]}
// with bbox in frame pixels.
std::string detections_json(const neat::Tensor& boxes, const std::vector<std::string>& labels,
                            int frame_w, int frame_h) {
  const auto rows = boxes.shape.empty() ? 0 : boxes.shape[0];
  auto mapping = boxes.map_read();
  const float* data = static_cast<const float*>(mapping.data);

  std::ostringstream out;
  out << "{\"objects\":[";
  for (int64_t i = 0; i < rows; ++i) {
    const float* row = data + i * 6;
    const int x1 = std::max(0, static_cast<int>(row[0]));
    const int y1 = std::max(0, static_cast<int>(row[1]));
    int w = std::max(0, static_cast<int>(row[2] - row[0]));
    int h = std::max(0, static_cast<int>(row[3] - row[1]));
    if (x1 + w > frame_w)
      w = frame_w - x1;
    if (y1 + h > frame_h)
      h = frame_h - y1;

    const int class_id = static_cast<int>(row[5]);
    const std::string label =
        (class_id >= 0 && class_id < static_cast<int>(labels.size())) ? labels[class_id]
                                                                     : "unknown";
    if (i > 0)
      out << ',';
    out << "{\"id\":\"obj_" << (i + 1) << "\",\"label\":\"" << label << "\",\"confidence\":"
        << row[4] << ",\"bbox\":[" << x1 << ',' << y1 << ',' << w << ',' << h << "]}";
  }
  out << "]}";
  return out.str();
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string config_path = "config.yaml";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config" && i + 1 < argc)
        config_path = argv[++i];
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const auto cfg = load_config(config_path);
    const std::string model_path = require(cfg, "model_path");
    const std::string rtsp_url = require(cfg, "rtsp_url");
    const std::string insight_host = require(cfg, "insight_host");
    const auto labels = load_labels(require(cfg, "labels"));

    const int frame_w = as_int(cfg, "stream_width", 1280);
    const int frame_h = as_int(cfg, "stream_height", 720);
    const int fps = as_int(cfg, "stream_fps", 30);
    const int max_detections = as_int(cfg, "max_detections", 50);

    // ── The source: RTSP in, decoded to NV12 frames on the board ────────────
    neat::nodes::groups::RtspDecodedInputOptions source_opt;
    source_opt.url = rtsp_url;
    source_opt.codec = neat::nodes::groups::RtspCodec::H264;
    source_opt.payload_type = 96;
    source_opt.tcp = true;
    source_opt.latency_ms = as_int(cfg, "latency_ms", 100);
    source_opt.insert_queue = true;
    source_opt.out_format = "NV12";
    source_opt.decoder_name = "decoder";
    source_opt.decoder_raw_output = true;
    source_opt.auto_caps_from_stream = true;
    source_opt.source_fps = fps;
    source_opt.fallback_h264_width = frame_w;
    source_opt.fallback_h264_height = frame_h;
    source_opt.output_caps.enable = true;
    source_opt.output_caps.format = "NV12";
    source_opt.output_caps.width = frame_w;
    source_opt.output_caps.height = frame_h;
    source_opt.output_caps.fps = fps;
    source_opt.output_caps.memory = neat::CapsMemory::Any;

    // ── The model: decoded frame in, decoded boxes out ──────────────────────
    neat::Model::Options model_opt;
    model_opt.preprocess.kind = neat::InputKind::Image;
    model_opt.preprocess.enable = neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
    model_opt.preprocess.input_max_width = frame_w;
    model_opt.preprocess.input_max_height = frame_h;
    model_opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
    model_opt.decode_type = neat::BoxDecodeType::YoloV26;
    model_opt.score_threshold = static_cast<float>(as_double(cfg, "min_score", 0.30));
    model_opt.nms_iou_threshold = static_cast<float>(as_double(cfg, "nms_iou", 0.60));
    model_opt.top_k = max_detections;
    neat::Model model(model_path, model_opt);

    // ── The video branch: same frames, H.264-encoded, RTP over UDP ──────────
    auto video_opt = neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(frame_w, frame_h,
                                                                               fps);
    video_opt.host = insight_host;
    video_opt.channel = 0;
    video_opt.video_port_base = as_int(cfg, "video_port", 9000);
    video_opt.encoder.bitrate_kbps = 1000;

    // ── One graph. The decoded frame branches to the encoder and the model,
    //    so both carry timestamps from the same frame and Insight can line
    //    the boxes up with the picture. ──────────────────────────────────────
    neat::Graph graph;
    auto source = neat::nodes::groups::RtspDecodedInput(source_opt);
    auto branch = neat::graphs::Branch("source", {"video", "model"});

    neat::Graph video_graph("video");
    video_graph.connect(neat::nodes::Input("video"),
                        neat::nodes::groups::VideoSender(video_opt));

    neat::Graph model_graph("model");
    model_graph.connect(neat::nodes::Input("model"), model);

    neat::Graph detections_graph("detections");
    detections_graph.add(
        neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));

    graph.connect(source, branch);
    graph.connect(branch, video_graph);
    graph.connect(branch, model_graph);
    graph.connect(model_graph, detections_graph);

    neat::RunOptions run_opt;
    run_opt.preset = neat::RunPreset::Realtime;
    run_opt.queue_depth = 3;
    run_opt.overflow_policy = neat::OverflowPolicy::KeepLatest;
    run_opt.output_memory = neat::OutputMemory::ZeroCopy;
    neat::Run run = graph.build(run_opt);

    // ── Detections leave the graph here and go to Insight as JSON ───────────
    neat::MetadataSenderOptions metadata_opt;
    metadata_opt.host = insight_host;
    metadata_opt.channel = 0;
    metadata_opt.metadata_port_base = as_int(cfg, "metadata_port", 9100);
    std::string metadata_err;
    neat::MetadataSender metadata_sender(metadata_opt, &metadata_err);
    if (!metadata_sender.ok())
      throw std::runtime_error("metadata sender: " + metadata_err);

    std::cout << "source=" << rtsp_url << " stream=" << frame_w << "x" << frame_h << "@" << fps
              << " insight=" << insight_host << " video=" << video_opt.video_port()
              << " metadata=" << metadata_sender.metadata_port() << "\n"
              << "Ctrl-C to stop.\n";

    int processed = 0;
    while (!g_stop) {
      neat::Sample sample;
      neat::PullError pull_error;
      const auto status = run.pull("detections", 20000, sample, &pull_error);

      if (status == neat::PullStatus::Timeout) {
        std::cerr << "[warn] no detections for 20s — is the RTSP source still running?\n";
        continue;
      }
      if (status == neat::PullStatus::Closed)
        break;
      if (status != neat::PullStatus::Ok)
        throw std::runtime_error("pull failed: " + pull_error.message);

      // BBOX payload in, [N, 6] rows of x1, y1, x2, y2, score, class_id out.
      neat::TensorList bbox = sample.tensor ? neat::TensorList{*sample.tensor} : sample.tensors;
      const auto decoded = neat::decode_bbox(bbox, frame_w, frame_h, max_detections, false);
      if (decoded.empty())
        continue;

      const std::string data_json = detections_json(decoded[0], labels, frame_w, frame_h);
      const int64_t ts_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1000000 : -1;
      const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";

      std::string send_err;
      if (!metadata_sender.send_metadata("object-detection", data_json, ts_ms, frame_id,
                                         &send_err))
        std::cerr << "[warn] metadata send failed: " << send_err << "\n";

      if (++processed % 50 == 0)
        std::cout << "frames=" << processed << "\n";
    }

    std::cout << "stopping after " << processed << " frames\n";
    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
```

---

## Step 5 — Build

From the SDK shell, in the app folder:

```bash
sima-user@sdk:/workspace/rtsp-detector$ cmake -S . \
    -B ./build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
sima-user@sdk:/workspace/rtsp-detector$ cmake --build ./build --parallel
```

`CMAKE_PREFIX_PATH` points CMake at the SDK sysroot, where the aarch64 Neat package lives. Without
it, `find_package(SimaNeat)` fails.

<!-- screenshot: successful cmake build in the VS Code terminal -->

---

## Step 6 — Run it on the board

Do not run the binary in the container — it was cross-compiled for the board. Hand it to `dk`:

```bash
sima-user@sdk:/workspace/rtsp-detector$ dk ./build/rtsp_detector --config ./config.yaml
```

The binary executes on the DevKit and its output comes back to your SDK terminal:

```text
source=rtsp://<host-ip>:8554/src1 stream=1280x720@30 insight=<host-ip> video=9000 metadata=9100
Ctrl-C to stop.
frames=50
frames=100
```

`Ctrl-C` stops it cleanly — the signal handler closes the `Run` so the encoder, decoder and model
release properly.

---

## Step 7 — Watch it in Insight

Back in the browser, open the **Video Viewer** tab. Channel 0 should show your video with detection
boxes drawn on it.

What you are confirming, in order:
1. **Video arrives** — the picture moves. If not, the video branch or UDP 9000 is the problem.
2. **Boxes appear** — metadata is arriving on UDP 9100.
3. **Boxes sit on the objects** — the two branches are timestamp-aligned, which is the whole point
   of keeping them in one graph.

<!-- screenshot: Insight Video Viewer, channel 0, boxes on the 720p stream -->

---

## What you have now

A complete Neat development loop: edit in VS Code attached to the SDK, build with CMake against
`SimaNeat::sima_neat`, run on the board with `dk`, and see the result in Insight — without copying
a file by hand at any point.

From here, the packaged examples are the natural next read. The application above is a trimmed
version of `single-stream-object-detector`, which adds source probing, MJPEG and H.265 support,
annotated-frame saving and profiling:

```bash
sima@modalix:~$ ls ~/prebuilt-apps/examples/object-detection/
```

| Next | Where |
|---|---|
| Multiple streams at once | `multi-stream-object-detector` in the apps bundle |
| Tracking, pose, segmentation | The other categories under `prebuilt-apps/examples/` |
| The C++ API reference | [developer.sima.ai](https://developer.sima.ai/software/develop-apps/) |

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 17 · Neat Insight](17-neat-insight.md) | [All chapters](../README.md) | [Troubleshooting](troubleshooting.md) |
