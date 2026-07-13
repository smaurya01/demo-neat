// quad_stream_quad_model — 4 RTSP streams -> 4 DIFFERENT models -> 4 UDP sinks (C++).
//
// This is the C++ port of main.py. Same four models, same config file, same output.
// The reason it exists: the Python version is gated by the GIL. Four streams means four
// overlay threads, and NumPy's fancy-index blend holds the interpreter lock, so the
// per-stream overlay stage measured ~60-140 ms for work that costs under 6 ms in
// isolation. Here each stream owns a real OS thread and the overlay is a plain memory
// write, so the four run genuinely concurrently.
//
// Task routing (config/default.conf, per stream slot 0..3). EVERY stream decodes
// on-device with Neat's fused BoxDecode — nothing is decoded on the host:
//
//   task          model            source          Neat decode family     normalization
//   detection     yolo_11s         MODEL ZOO       BoxDecodeType::YoloV8     COCO_YOLO
//   segmentation  yolo_11s_seg     MODEL ZOO       BoxDecodeType::YoloV8Seg  COCO_YOLO
//   pose          yolo26s-pose     self-compiled   BoxDecodeType::YoloV26Pose COCO_YOLO
//   yolox         yolox_s          self-compiled   BoxDecodeType::YoloX      None (raw 0-255)
//
// Two things that are easy to get wrong and fail SILENTLY (see LEARNING.md):
//   * The decode family follows the shape of the archive's HEAD, not the model's version
//     number. There is no "YoloV11" — zoo YOLO11 keeps raw 64-channel DFL heads and so
//     decodes as YoloV8. A self-compiled YOLO11 folds the DFL away and decodes as YoloV26.
//   * Megvii YOLOX is trained on RAW 0-255 pixels. Hand it the COCO_YOLO preset (x/255)
//     that its three neighbours want and it detects NOTHING, at full speed, with no error.
//
// Design provenance (https://github.com/sima-neat/core and sibling apps):
//   * three-graph shuttle (source / model / video) — apps/single-stream-yolo-yolo11/main.cpp
//   * per-task ModelSpec + NormalizePreset::None — apps/multi-model-load-probe/main.cpp
//   * NV12 mask blend + letterbox-inverse — apps/single-stream-yolov8n-seg/main.cpp
//   * decode_bbox_tensor / decode_pose_tensor / decode_segmentation_tensor
//     — core/include/pipeline/DetectionTypes.h

#include <neat.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

using Clock = std::chrono::steady_clock;

namespace {

std::atomic<bool> g_stop{false};
// Set once every source+worker thread has finished, so the reporter thread stops too.
// Without it a --frames run (which sets neither g_stop nor a deadline) would leave the
// reporter spinning forever and hang on join().
std::atomic<bool> g_done{false};

void handle_signal(int) {
  g_stop.store(true);
}

double ms_since(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// ── tasks ────────────────────────────────────────────────────────────────────
enum class Task { Detection, Segmentation, Pose, Yolox };

std::string task_name(Task t) {
  switch (t) {
  case Task::Detection: return "detection";
  case Task::Segmentation: return "segmentation";
  case Task::Pose: return "pose";
  case Task::Yolox: return "yolox";
  }
  return "?";
}

std::optional<Task> task_from_string(const std::string& s) {
  if (s == "detection") return Task::Detection;
  if (s == "segmentation") return Task::Segmentation;
  if (s == "pose") return Task::Pose;
  if (s == "yolox") return Task::Yolox;
  return std::nullopt;
}

// The decode family is keyed on the HEAD SHAPE of the archive, not on the model name.
neat::BoxDecodeType decode_family(Task t) {
  switch (t) {
  case Task::Detection:    return neat::BoxDecodeType::YoloV8;      // zoo: 64-ch DFL + 80-ch class
  case Task::Segmentation: return neat::BoxDecodeType::YoloV8Seg;   // + 32-ch coeffs + 32x160 proto
  case Task::Pose:         return neat::BoxDecodeType::YoloV26Pose; // 4-ch ltrb + 1-ch + 51-ch kpt
  case Task::Yolox:        return neat::BoxDecodeType::YoloX;       // (4, 1, 80) per scale
  }
  throw std::runtime_error("unknown task");
}

// Pose is single-class ("person"); the rest are COCO-80.
int num_classes_for(Task t) {
  return t == Task::Pose ? 1 : 80;
}

// Ultralytics models want x/255. Megvii YOLOX wants the raw 0-255 pixel.
bool uses_coco_yolo_normalize(Task t) {
  return t != Task::Yolox;
}


// ── config (same keys as config/default.conf, shared with main.py) ───────────
struct StreamSpec {
  int id = 0;
  Task task = Task::Detection;
  std::string rtsp_url;
  std::string model_path;
  int port = 0;
};

struct Config {
  std::string rtsp_default = "rtsp://<rtsp-server-ip>:8555/stream";
  std::array<std::string, 4> stream_rtsp{};
  std::array<std::string, 4> stream_model{};
  std::array<std::string, 4> stream_task{};
  bool tcp = true;
  int num_streams = 4;
  std::string udp_host = "127.0.0.1";
  int udp_port_base = 5206;
  int udp_port_stride = 2;
  int model_width = 640;
  int model_height = 640;
  int fallback_width = 1280;
  int fallback_height = 720;
  int fallback_fps = 30;
  int latency_ms = 200;
  float score_threshold = 0.25f;
  float nms_iou = 0.50f;
  int top_k = 100;
  int queue_depth = 3;
  int bitrate_kbps = 4000;
  int frames = 0;
  double duration_s = 0.0;
  int warmup_frames = 20;
  // How often the live time profile is printed to the terminal while running.
  // 0 disables it and you only get the summary at exit.
  double profile_interval_s = 5.0;

  // Frames the model graph may hold in flight at once. The pusher will not hand it a
  // frame beyond this; excess frames are dropped at the source mailbox instead.
  int model_queue_depth = 4;
  // Frames parked between the model side and the output thread.
  int output_queue_depth = 2;
  bool print_backend = false;
  bool no_overlay = false;
  std::string cvu_pre_target = "EV74";
  std::string cvu_post_target = "EV74";
};

std::string trim(std::string v) {
  const auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
  v.erase(v.begin(), std::find_if(v.begin(), v.end(), not_space));
  v.erase(std::find_if(v.rbegin(), v.rend(), not_space).base(), v.end());
  return v;
}

bool to_bool(const std::string& v) {
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

void set_config_value(Config& cfg, const std::string& key, const std::string& value) {
  auto slot_key = [&](const char* prefix, const char* suffix) -> int {
    // stream<i>_rtsp / stream<i>_model / stream<i>_task
    const std::string p(prefix);
    const std::string s(suffix);
    if (key.size() < p.size() + s.size() + 1) return -1;
    if (key.compare(0, p.size(), p) != 0) return -1;
    if (key.compare(key.size() - s.size(), s.size(), s) != 0) return -1;
    const char digit = key[p.size()];
    if (digit < '0' || digit > '3') return -1;
    return digit - '0';
  };

  if (int i = slot_key("stream", "_rtsp"); i >= 0) { cfg.stream_rtsp[i] = value; return; }
  if (int i = slot_key("stream", "_model"); i >= 0) { cfg.stream_model[i] = value; return; }
  if (int i = slot_key("stream", "_task"); i >= 0) { cfg.stream_task[i] = value; return; }

  if (key == "rtsp_default") cfg.rtsp_default = value;
  else if (key == "rtsp_transport") cfg.tcp = (value == "tcp");
  else if (key == "num_streams") cfg.num_streams = std::stoi(value);
  else if (key == "udp_host") cfg.udp_host = value;
  else if (key == "udp_port_base") cfg.udp_port_base = std::stoi(value);
  else if (key == "udp_port_stride") cfg.udp_port_stride = std::stoi(value);
  else if (key == "model_width") cfg.model_width = std::stoi(value);
  else if (key == "model_height") cfg.model_height = std::stoi(value);
  else if (key == "fallback_width") cfg.fallback_width = std::stoi(value);
  else if (key == "fallback_height") cfg.fallback_height = std::stoi(value);
  else if (key == "fallback_fps") cfg.fallback_fps = std::stoi(value);
  else if (key == "latency_ms") cfg.latency_ms = std::stoi(value);
  else if (key == "score_threshold") cfg.score_threshold = std::stof(value);
  else if (key == "nms_iou") cfg.nms_iou = std::stof(value);
  else if (key == "top_k") cfg.top_k = std::stoi(value);
  else if (key == "queue_depth") cfg.queue_depth = std::stoi(value);
  else if (key == "bitrate_kbps") cfg.bitrate_kbps = std::stoi(value);
  else if (key == "frames") cfg.frames = std::stoi(value);
  else if (key == "warmup_frames") cfg.warmup_frames = std::stoi(value);
  else if (key == "profile_interval") cfg.profile_interval_s = std::stod(value);
  else if (key == "model_queue_depth") cfg.model_queue_depth = std::stoi(value);
  else if (key == "output_queue_depth") cfg.output_queue_depth = std::stoi(value);
  else if (key == "print_backend") cfg.print_backend = to_bool(value);
  else if (key == "cvu_pre_target") cfg.cvu_pre_target = value;
  else if (key == "cvu_post_target") cfg.cvu_post_target = value;
  // Unknown keys are ignored on purpose: config/default.conf is shared with main.py,
  // which understands a few extra Python-only knobs (pipeline_depth, serial, ...).
}

Config read_config(const std::string& path) {
  Config cfg;
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open config: " + path);
  }
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    set_config_value(cfg, trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
  }
  return cfg;
}

std::vector<StreamSpec> build_specs(const Config& cfg) {
  static const std::array<const char*, 4> kDefaultTask{"detection", "segmentation", "pose", "yolox"};
  std::vector<StreamSpec> specs;
  const int n = std::clamp(cfg.num_streams, 1, 4);
  for (int i = 0; i < n; ++i) {
    StreamSpec s;
    s.id = i;
    const std::string tname =
        cfg.stream_task[i].empty() ? std::string(kDefaultTask[i]) : cfg.stream_task[i];
    const auto t = task_from_string(tname);
    if (!t.has_value()) {
      throw std::runtime_error("stream" + std::to_string(i) + "_task: unknown task '" + tname + "'");
    }
    s.task = *t;
    s.rtsp_url = cfg.stream_rtsp[i].empty() ? cfg.rtsp_default : cfg.stream_rtsp[i];
    if (cfg.stream_model[i].empty()) {
      throw std::runtime_error("stream" + std::to_string(i) + "_model is not set in the config");
    }
    s.model_path = cfg.stream_model[i];
    s.port = cfg.udp_port_base + i * cfg.udp_port_stride;
    specs.push_back(std::move(s));
  }
  return specs;
}

// ── model + graphs ───────────────────────────────────────────────────────────
std::unique_ptr<neat::Model> make_model(const Config& cfg, const StreamSpec& spec) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.input_max_width = cfg.fallback_width;
  opt.preprocess.input_max_height = cfg.fallback_height;
  opt.preprocess.input_max_depth = 1;
  opt.preprocess.resize.enable = neat::AutoFlag::On;
  opt.preprocess.resize.width = cfg.model_width;
  opt.preprocess.resize.height = cfg.model_height;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.resize.pad_value = 114;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;

  if (uses_coco_yolo_normalize(spec.task)) {
    opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;   // x/255
  } else {
    // YOLOX: raw 0-255, i.e. normalization OFF.
    //
    // Set the preset and NOTHING else. Do NOT also set explicit normalize stats of
    // mean=0/stddev=1: those are interpreted in [0,1] space, so stddev=1 re-applies the
    // very x/255 we are trying to avoid, and YOLOX goes back to detecting nothing.
    // (Measured: with explicit stats, objs=0 on every frame; with the bare preset,
    // objs=2-7 on the same stream.)
    //
    // This pairs with the archive, compiled with std=1/255 (models.yaml) so its input
    // quantization expects 0-255 (q_scale ~ 1.0). Both halves are required.
    opt.preprocess.preset = neat::NormalizePreset::None;
  }

  // Pin the CVU stages. AUTO does not always pick the accelerator; measured ~12% slower
  // aggregate, and historically it once put pose's post stage on the A65 at ~1.8 s/frame.
  opt.processcvu.pre_run_target = cfg.cvu_pre_target;
  opt.processcvu.post_run_target = cfg.cvu_post_target;

  // Every task decodes ON-DEVICE. Leaving decode_type unset would publish raw heads and
  // force a host decode — which is exactly the cost this app exists to avoid.
  opt.decode_type = decode_family(spec.task);
  opt.score_threshold = cfg.score_threshold;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.top_k;
  opt.num_classes = num_classes_for(spec.task);

  // decode_type_option is deliberately LEFT AT Auto. These are MPK-backed models, so the
  // archive's own contract already pins tensor order, layout, dtype and score domain, and
  // SimaBoxDecode.h says application code should set only the decode FAMILY and the
  // thresholds. Forcing an explicit sub-variant makes the model-managed route fail to
  // compile, and it surfaces as "Missing prepared runtime stage for graph-owned processcvu"
  // on the PREPROC element — a thoroughly confusing place to learn you broke the decode.
  // NOT setting boxdecode_original_width/height — deprecated in core/include/model/Model.h.

  return std::make_unique<neat::Model>(spec.model_path, opt);
}

groups::RtspDecodedInputOptions make_rtsp_options(const Config& cfg, const StreamSpec& spec) {
  groups::RtspDecodedInputOptions opt;
  opt.url = spec.rtsp_url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = neat::FormatTag::NV12;
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.fallback_h264_width = cfg.fallback_width;
  opt.fallback_h264_height = cfg.fallback_height;
  opt.fallback_h264_fps = cfg.fallback_fps;
  opt.output_caps.enable = true;
  opt.output_caps.format = neat::FormatTag::NV12;
  opt.output_caps.width = cfg.fallback_width;
  opt.output_caps.height = cfg.fallback_height;
  opt.output_caps.fps = cfg.fallback_fps;
  opt.output_caps.memory = neat::CapsMemory::Any;
  return opt;
}

neat::InputOptions make_nv12_input_options(const Config& cfg) {
  neat::InputOptions opt;
  opt.payload_type = neat::PayloadType::Image;
  opt.format = neat::FormatTag::NV12;
  opt.width = cfg.fallback_width;
  opt.height = cfg.fallback_height;
  opt.depth = 1;
  opt.max_width = cfg.fallback_width;
  opt.max_height = cfg.fallback_height;
  opt.max_depth = 1;
  opt.fps_n = std::max(1, cfg.fallback_fps);
  opt.fps_d = 1;
  return opt;
}

neat::Graph make_source_graph(const Config& cfg, const StreamSpec& spec) {
  neat::Graph g("qsqm_source_" + std::to_string(spec.id));
  g.add(groups::RtspDecodedInput(make_rtsp_options(cfg, spec)));
  g.add(neat::nodes::Output("frame", neat::OutputOptions::Latest()));
  return g;
}

neat::Graph make_model_graph(const Config& cfg, const StreamSpec& spec, const neat::Model& model) {
  neat::Graph g("qsqm_model_" + task_name(spec.task) + "_" + std::to_string(spec.id));
  g.add(neat::nodes::Input("image", make_nv12_input_options(cfg)));
  g.add(model);
  // One endpoint for all four tasks: each model graph ends in an on-device BoxDecode
  // stage, so what comes out is always a decoded payload, never raw heads.
  //
  // EveryFrame(max_buffers) = how many FINISHED results the output sink may park for the
  // puller. This must be at least model_queue_depth, or the two settings contradict each
  // other: --mode pipelined deliberately keeps N frames in flight, and if the sink can only
  // hold 1 finished result, results back up inside the graph and throttle the very pipelining
  // we are paying for. (apps/multi-model-load-probe uses 4; the library default is 30.)
  const int out_buffers = std::max(1, cfg.model_queue_depth);
  g.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(out_buffers)));
  return g;
}

neat::Graph make_video_graph(const Config& cfg, const StreamSpec& spec) {
  auto so = groups::VideoSenderOptions::H264RtpUdpFromRaw(
      cfg.fallback_width, cfg.fallback_height, std::max(1, cfg.fallback_fps));
  so.host = cfg.udp_host;
  so.channel = 0;
  so.video_port_base = spec.port;
  so.encoder.bitrate_kbps = cfg.bitrate_kbps;

  neat::Graph g("qsqm_video_" + std::to_string(spec.id));
  g.add(neat::nodes::Input("video", make_nv12_input_options(cfg)));
  g.add(groups::VideoSender(so));
  return g;
}

// ── NV12 plumbing (verbatim contract from single-stream-yolo-yolo11) ──────────
bool infer_dims(const neat::Tensor& t, int& width, int& height) {
  width = t.width();
  height = t.height();
  if ((width <= 0 || height <= 0) && t.shape.size() >= 2) {
    height = static_cast<int>(t.shape[0]);
    width = static_cast<int>(t.shape[1]);
  }
  return width > 0 && height > 0;
}

bool init_nv12_tensor_meta(neat::Tensor& out, int width, int height, std::string& err) {
  if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0) {
    err = "NV12 requires positive even width/height";
    return false;
  }
  out.dtype = neat::TensorDType::UInt8;
  out.layout = neat::TensorLayout::HW;
  out.shape = {height, width};
  out.strides_bytes = {width, 1};
  out.byte_offset = 0;
  out.device = {neat::DeviceType::CPU, 0};
  out.read_only = false;

  neat::ImageSpec image;
  image.format = neat::ImageSpec::PixelFormat::NV12;
  out.semantic.image = image;

  neat::Plane y;
  y.role = neat::PlaneRole::Y;
  y.shape = {height, width};
  y.strides_bytes = {width, 1};
  y.byte_offset = 0;

  neat::Plane uv;
  uv.role = neat::PlaneRole::UV;
  uv.shape = {height / 2, width};
  uv.strides_bytes = {width, 1};
  uv.byte_offset = static_cast<std::int64_t>(width) * static_cast<std::int64_t>(height);

  out.planes.clear();
  out.planes.push_back(std::move(y));
  out.planes.push_back(std::move(uv));
  return true;
}

neat::Tensor make_blank_nv12_tensor(int width, int height) {
  const std::size_t bytes =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U / 2U;
  neat::Tensor out;
  out.storage = neat::make_cpu_owned_storage(bytes);
  std::memset(out.storage->data, 16, static_cast<std::size_t>(width) * height);
  std::memset(static_cast<std::uint8_t*>(out.storage->data) +
                  static_cast<std::size_t>(width) * height,
              128, bytes - static_cast<std::size_t>(width) * height);
  std::string err;
  if (!init_nv12_tensor_meta(out, width, height, err)) {
    throw std::runtime_error(err);
  }
  return out;
}

neat::Tensor copy_nv12_to_cpu_tensor(const neat::Tensor& input) {
  int width = 0;
  int height = 0;
  if (!input.is_nv12()) {
    throw std::runtime_error("expected NV12 tensor");
  }
  if (!infer_dims(input, width, height)) {
    throw std::runtime_error("invalid NV12 tensor dimensions");
  }
  const auto bytes = input.copy_nv12_contiguous();
  if (bytes.empty()) {
    throw std::runtime_error("NV12 copy failed");
  }
  neat::Tensor out;
  out.storage = neat::make_cpu_owned_storage(bytes.size());
  std::memcpy(out.storage->data, bytes.data(), bytes.size());
  std::string err;
  if (!init_nv12_tensor_meta(out, width, height, err)) {
    throw std::runtime_error(err);
  }
  return out;
}

std::uint8_t* nv12_y_plane(neat::Tensor& t) {
  return static_cast<std::uint8_t*>(t.storage->data);
}

std::uint8_t* nv12_uv_plane(neat::Tensor& t, int width, int height) {
  return static_cast<std::uint8_t*>(t.storage->data) +
         static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

// ── overlay primitives (NV12 Y + UV, no OpenCV) ──────────────────────────────
struct Nv12Color {
  std::uint8_t y = 235;
  std::uint8_t u = 128;
  std::uint8_t v = 128;
};

void fill_nv12_rect(std::uint8_t* y, std::uint8_t* uv, int width, int height, int x1, int y1,
                    int x2, int y2, const Nv12Color& c) {
  x1 = std::max(0, std::min(width, x1));
  x2 = std::max(0, std::min(width, x2));
  y1 = std::max(0, std::min(height, y1));
  y2 = std::max(0, std::min(height, y2));
  if (x2 <= x1 || y2 <= y1) return;

  for (int row = y1; row < y2; ++row) {
    std::memset(y + static_cast<std::size_t>(row) * width + x1, c.y,
                static_cast<std::size_t>(x2 - x1));
  }
  const int uv_y1 = y1 / 2;
  const int uv_y2 = (y2 + 1) / 2;
  const int uv_x1 = x1 & ~1;
  const int uv_x2 = (x2 + 1) & ~1;
  for (int row = std::max(0, uv_y1); row < std::min(height / 2, uv_y2); ++row) {
    auto* uv_row = uv + static_cast<std::size_t>(row) * width;
    for (int col = std::max(0, uv_x1); col + 1 < std::min(width, uv_x2); col += 2) {
      uv_row[col] = c.u;
      uv_row[col + 1] = c.v;
    }
  }
}

void draw_box(std::uint8_t* y, std::uint8_t* uv, int w, int h, int x1, int y1, int x2, int y2,
              const Nv12Color& c, int th = 3) {
  fill_nv12_rect(y, uv, w, h, x1, y1, x2, y1 + th, c);
  fill_nv12_rect(y, uv, w, h, x1, y2 - th, x2, y2, c);
  fill_nv12_rect(y, uv, w, h, x1, y1, x1 + th, y2, c);
  fill_nv12_rect(y, uv, w, h, x2 - th, y1, x2, y2, c);
}

void draw_line(std::uint8_t* y, std::uint8_t* uv, int w, int h, int x0, int y0, int x1, int y1,
               const Nv12Color& c, int th = 2) {
  // Integer Bresenham; the skeleton is the only thing that needs arbitrary lines.
  int dx = std::abs(x1 - x0);
  int dy = -std::abs(y1 - y0);
  int sx = x0 < x1 ? 1 : -1;
  int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  for (int guard = 0; guard < 4096; ++guard) {
    fill_nv12_rect(y, uv, w, h, x0 - th / 2, y0 - th / 2, x0 + (th + 1) / 2, y0 + (th + 1) / 2, c);
    if (x0 == x1 && y0 == y1) break;
    const int e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

std::array<std::string_view, 7> glyph_for(char c) {
  switch (c) {
  case 'A': return {"01110", "10001", "10001", "11111", "10001", "10001", "10001"};
  case 'B': return {"11110", "10001", "10001", "11110", "10001", "10001", "11110"};
  case 'C': return {"01111", "10000", "10000", "10000", "10000", "10000", "01111"};
  case 'D': return {"11110", "10001", "10001", "10001", "10001", "10001", "11110"};
  case 'E': return {"11111", "10000", "10000", "11110", "10000", "10000", "11111"};
  case 'F': return {"11111", "10000", "10000", "11110", "10000", "10000", "10000"};
  case 'G': return {"01111", "10000", "10000", "10011", "10001", "10001", "01110"};
  case 'H': return {"10001", "10001", "10001", "11111", "10001", "10001", "10001"};
  case 'I': return {"11111", "00100", "00100", "00100", "00100", "00100", "11111"};
  case 'J': return {"00111", "00010", "00010", "00010", "10010", "10010", "01100"};
  case 'K': return {"10001", "10010", "10100", "11000", "10100", "10010", "10001"};
  case 'L': return {"10000", "10000", "10000", "10000", "10000", "10000", "11111"};
  case 'M': return {"10001", "11011", "10101", "10101", "10001", "10001", "10001"};
  case 'N': return {"10001", "11001", "10101", "10011", "10001", "10001", "10001"};
  case 'O': return {"01110", "10001", "10001", "10001", "10001", "10001", "01110"};
  case 'P': return {"11110", "10001", "10001", "11110", "10000", "10000", "10000"};
  case 'Q': return {"01110", "10001", "10001", "10001", "10101", "10010", "01101"};
  case 'R': return {"11110", "10001", "10001", "11110", "10100", "10010", "10001"};
  case 'S': return {"01111", "10000", "10000", "01110", "00001", "00001", "11110"};
  case 'T': return {"11111", "00100", "00100", "00100", "00100", "00100", "00100"};
  case 'U': return {"10001", "10001", "10001", "10001", "10001", "10001", "01110"};
  case 'V': return {"10001", "10001", "10001", "10001", "10001", "01010", "00100"};
  case 'W': return {"10001", "10001", "10001", "10101", "10101", "10101", "01010"};
  case 'X': return {"10001", "10001", "01010", "00100", "01010", "10001", "10001"};
  case 'Y': return {"10001", "10001", "01010", "00100", "00100", "00100", "00100"};
  case 'Z': return {"11111", "00001", "00010", "00100", "01000", "10000", "11111"};
  case '0': return {"01110", "10001", "10011", "10101", "11001", "10001", "01110"};
  case '1': return {"00100", "01100", "00100", "00100", "00100", "00100", "01110"};
  case '2': return {"01110", "10001", "00001", "00010", "00100", "01000", "11111"};
  case '3': return {"11110", "00001", "00001", "01110", "00001", "00001", "11110"};
  case '4': return {"10010", "10010", "10010", "11111", "00010", "00010", "00010"};
  case '5': return {"11111", "10000", "10000", "11110", "00001", "00001", "11110"};
  case '6': return {"01111", "10000", "10000", "11110", "10001", "10001", "01110"};
  case '7': return {"11111", "00001", "00010", "00100", "01000", "01000", "01000"};
  case '8': return {"01110", "10001", "10001", "01110", "10001", "10001", "01110"};
  case '9': return {"01110", "10001", "10001", "01111", "00001", "00001", "11110"};
  case ':': return {"00000", "01100", "01100", "00000", "01100", "01100", "00000"};
  case '-': return {"00000", "00000", "00000", "11111", "00000", "00000", "00000"};
  case '.': return {"00000", "00000", "00000", "00000", "00000", "01100", "01100"};
  case ' ': return {"00000", "00000", "00000", "00000", "00000", "00000", "00000"};
  default:  return {"11111", "00001", "00010", "00100", "00100", "00000", "00100"};
  }
}

char overlay_char(char c) {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
}

void draw_text(std::uint8_t* y, std::uint8_t* uv, int width, int height, int x, int y0,
               const std::string& text, const Nv12Color& c, int scale = 2) {
  int cursor = x;
  for (char raw : text) {
    const auto glyph = glyph_for(overlay_char(raw));
    for (int row = 0; row < static_cast<int>(glyph.size()); ++row) {
      for (int col = 0; col < static_cast<int>(glyph[row].size()); ++col) {
        if (glyph[row][col] == '1') {
          fill_nv12_rect(y, uv, width, height, cursor + col * scale, y0 + row * scale,
                         cursor + (col + 1) * scale, y0 + (row + 1) * scale, c);
        }
      }
    }
    cursor += 6 * scale;
    if (cursor >= width - 6 * scale) break;
  }
}

const std::vector<std::string>& coco_labels() {
  static const std::vector<std::string> labels = {
      "PERSON", "BICYCLE", "CAR", "MOTORCYCLE", "AIRPLANE", "BUS", "TRAIN", "TRUCK",
      "BOAT", "TRAFFIC LIGHT", "FIRE HYDRANT", "STOP SIGN", "PARKING METER", "BENCH",
      "BIRD", "CAT", "DOG", "HORSE", "SHEEP", "COW", "ELEPHANT", "BEAR", "ZEBRA",
      "GIRAFFE", "BACKPACK", "UMBRELLA", "HANDBAG", "TIE", "SUITCASE", "FRISBEE",
      "SKIS", "SNOWBOARD", "SPORTS BALL", "KITE", "BASEBALL BAT", "BASEBALL GLOVE",
      "SKATEBOARD", "SURFBOARD", "TENNIS RACKET", "BOTTLE", "WINE GLASS", "CUP",
      "FORK", "KNIFE", "SPOON", "BOWL", "BANANA", "APPLE", "SANDWICH", "ORANGE",
      "BROCCOLI", "CARROT", "HOT DOG", "PIZZA", "DONUT", "CAKE", "CHAIR", "COUCH",
      "POTTED PLANT", "BED", "DINING TABLE", "TOILET", "TV", "LAPTOP", "MOUSE",
      "REMOTE", "KEYBOARD", "CELL PHONE", "MICROWAVE", "OVEN", "TOASTER", "SINK",
      "REFRIGERATOR", "BOOK", "CLOCK", "VASE", "SCISSORS", "TEDDY BEAR", "HAIR DRIER",
      "TOOTHBRUSH"};
  return labels;
}

std::string class_label(int id) {
  const auto& l = coco_labels();
  if (id >= 0 && id < static_cast<int>(l.size())) return l[static_cast<std::size_t>(id)];
  return "CLASS " + std::to_string(id);
}

// COCO-17 skeleton (Ultralytics pose order).
constexpr std::array<std::pair<int, int>, 19> kSkeleton{{
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
    {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2},
    {1, 3}, {2, 4}, {3, 5}, {4, 6}}};

// ── segmentation mask projection (from single-stream-yolov8n-seg) ────────────
struct MaskRect {
  int x = 0, y = 0, width = 0, height = 0;
};

MaskRect mask_rect_for_frame_box(int x1, int y1, int x2, int y2, int frame_w, int frame_h,
                                 int mask_w, int mask_h) {
  const int model_w = mask_w * 4;   // 160 * 4 = 640
  const int model_h = mask_h * 4;
  const double scale = std::min(static_cast<double>(model_w) / frame_w,
                                static_cast<double>(model_h) / frame_h);
  const double pad_x = (model_w - frame_w * scale) / 2.0;
  const double pad_y = (model_h - frame_h * scale) / 2.0;
  const auto to_mask_x = [&](double fx) { return (fx * scale + pad_x) * mask_w / model_w; };
  const auto to_mask_y = [&](double fy) { return (fy * scale + pad_y) * mask_h / model_h; };

  MaskRect r;
  r.x = std::clamp(static_cast<int>(std::floor(to_mask_x(x1))), 0, std::max(0, mask_w - 1));
  r.y = std::clamp(static_cast<int>(std::floor(to_mask_y(y1))), 0, std::max(0, mask_h - 1));
  const int rx2 = std::clamp(static_cast<int>(std::ceil(to_mask_x(x2))), r.x + 1, mask_w);
  const int ry2 = std::clamp(static_cast<int>(std::ceil(to_mask_y(y2))), r.y + 1, mask_h);
  r.width = rx2 - r.x;
  r.height = ry2 - r.y;
  return r;
}

std::uint8_t blend_u8(std::uint8_t base, std::uint8_t over, float alpha) {
  const float v = (1.0f - alpha) * base + alpha * over;
  return static_cast<std::uint8_t>(std::clamp(v, 0.0f, 255.0f));
}

void blend_mask_pixel(std::uint8_t* yp, std::uint8_t* uvp, int w, int h, int x, int y,
                      const Nv12Color& c, float alpha) {
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  const std::size_t yo = static_cast<std::size_t>(y) * w + x;
  yp[yo] = blend_u8(yp[yo], c.y, alpha);
  const int ux = x & ~1;
  const int uy = y / 2;
  if (ux + 1 < w && uy < h / 2) {
    const std::size_t uo = static_cast<std::size_t>(uy) * w + ux;
    uvp[uo] = blend_u8(uvp[uo], c.u, alpha);
    uvp[uo + 1] = blend_u8(uvp[uo + 1], c.v, alpha);
  }
}

// ── decoded-result containers ────────────────────────────────────────────────
struct Instance {
  neat::Box box;
  const std::uint8_t* mask = nullptr;      // [160*160], segmentation only
  const float* keypoints = nullptr;        // [17*3] (x, y, visibility), pose only
};

std::vector<neat::Box> boxes_from_tensor(const neat::Tensor& t) {
  std::vector<neat::Box> out;
  if (!t.storage || t.shape.size() < 2) return out;
  const auto n = static_cast<std::size_t>(t.shape[0]);
  const auto* p = static_cast<const float*>(t.storage->data);
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float* r = p + i * 6;
    neat::Box b;
    b.x1 = r[0]; b.y1 = r[1]; b.x2 = r[2]; b.y2 = r[3];
    b.score = r[4];
    b.class_id = static_cast<int>(r[5]);
    out.push_back(b);
  }
  return out;
}

// ── per-stream runtime ───────────────────────────────────────────────────────
//
// The pipeline spans two threads, so the profile does too:
//
//   source thread : decode                                  (RTSP/H.264 -> NV12)
//   worker thread : infer -> postproc -> overlay -> encode
//
// Two of these stages are PIPELINED, and their host-side timing is not the device's
// internal cost. Reporting them as if it were is the easiest way to publish a confident,
// wrong number, so each is named for what it actually measures:
//
//   dec_wait   blocking pull on the source graph. ARRIVAL-GATED: a healthy 60 fps stream
//              hands back a frame every ~16.7 ms however fast the decoder is, so this is
//              the frame INTERVAL, not the decode cost. It only rises above the interval
//              once the decoder has genuinely fallen behind -> read it as a health check,
//              and read `decode fps` for whether the decoder is keeping up.
//   decode     memcpy of the decoded NV12 frame out of the zero-copy pool (~1.4 MB). This
//              is the real, host-side cost of getting a decoded frame. The H.264 decode
//              proper runs on the hardware decoder and is not observable from the host.
//   qwait      worker blocked waiting for the source thread. Idle, not work.
//   infer      model push + pull: on-device preprocess + MLA + fused BoxDecode. Real cost.
//   postproc   host-side read of the decoded payload + instance build. Real host CPU.
//   overlay    host-side NV12 draw. Real host CPU (0 under --no-overlay).
//   encode     video push. This ENQUEUES to the encoder and returns — it is NOT the
//              encoder's latency. It sits near zero until the encoder falls behind, at
//              which point its backpressure surfaces here. Read it as encoder HEADROOM.
struct SourceProfile {
  std::vector<double> wait;   // one sample per DECODED frame (including ones we then drop)
  std::vector<double> copy;   // one sample per frame actually handed to the worker
  int decoded = 0;
  Clock::time_point steady_start{};
  double steady_seconds = 0.0;
};

struct StageProfile {
  std::vector<double> qwait, infer, postproc, overlay, encode, latency;

  void add(double q, double i, double p, double o, double e, double l) {
    qwait.push_back(q); infer.push_back(i); postproc.push_back(p);
    overlay.push_back(o); encode.push_back(e); latency.push_back(l);
  }
};

double mean_of(const std::vector<double>& v, std::size_t skip) {
  if (v.size() <= skip) return 0.0;
  double s = 0.0;
  for (std::size_t i = skip; i < v.size(); ++i) s += v[i];
  return s / static_cast<double>(v.size() - skip);
}

double pct_of(std::vector<double> v, std::size_t skip, double p) {
  if (v.size() <= skip) return 0.0;
  std::vector<double> w(v.begin() + static_cast<long>(skip), v.end());
  std::sort(w.begin(), w.end());
  const auto idx = static_cast<std::size_t>(p / 100.0 * static_cast<double>(w.size() - 1));
  return w[std::min(idx, w.size() - 1)];
}

// Single-slot drop-oldest mailbox between a stream's source thread and its worker.
//
// This is what keeps the RTSP source DRAINED. Without it the source graph's internal edge
// queue (256 frames) fills — a live 60 fps camera does not wait for a ~45 fps consumer —
// and every stream then dies with "sink backpressure timeout" after exactly 256 frames.
// A live source must be drained continuously and the stale frames thrown away; the drop
// belongs HERE, at the source, not inside the model. Same intent as OverflowPolicy::KeepLatest.
struct FrameMailbox {
  std::mutex m;
  std::condition_variable cv;
  neat::Tensor frame;
  int width = 0;
  int height = 0;
  bool has = false;
  bool closed = false;
  long long dropped = 0;

  // True when the worker has taken the last frame and is ready for another.
  //
  // The source thread checks this BEFORE copying. That ordering matters: the RTSP graph
  // must still be pulled at the full 60 fps to keep its edge queue drained, but copying
  // every one of those frames costs 1.4 MB per stream per frame — ~336 MB/s of memcpy
  // across four streams — which starves the workers of the very CPU they need. So we pull
  // always, and copy only when someone is actually waiting for the frame. Drop before the
  // copy, not after it.
  bool wants_frame() {
    std::lock_guard<std::mutex> lk(m);
    return !has;
  }

  void put(neat::Tensor t, int w, int h) {
    {
      std::lock_guard<std::mutex> lk(m);
      if (has) ++dropped;   // overwrite the stale frame: drop-oldest
      frame = std::move(t);
      width = w;
      height = h;
      has = true;
    }
    cv.notify_one();
  }

  bool take(neat::Tensor& out, int& w, int& h) {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return has || closed; });
    if (!has) return false;
    out = std::move(frame);
    w = width;
    h = height;
    has = false;
    return true;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lk(m);
      closed = true;
    }
    cv.notify_all();
  }
};

// One frame after the model has run, on its way to postproc + overlay + encode.
//
// It carries the model's output Sample. That Sample is ZeroCopy, so holding it keeps one of the
// model graph's output buffers checked out — which is exactly why out_q is small (bounded by
// --output-queue-depth, default 2). Make it large and you starve the model of output buffers.
struct InferItem {
  neat::Tensor frame;          // CPU NV12, annotated in place
  int fw = 0;
  int fh = 0;
  neat::Sample det;            // model output (ZeroCopy)
  Clock::time_point t_in{};
  double qwait_ms = 0.0;
  double infer_ms = 0.0;
};

// A frame that has been pushed into the model graph and is awaiting its result (pipelined mode).
struct PendingFrame {
  neat::Tensor frame;
  int fw = 0;
  int fh = 0;
  Clock::time_point t_in{};
  Clock::time_point push_begin{};
  double qwait_ms = 0.0;
};

// Bounded blocking queue. push() blocks when full — that backpressure is deliberate: if the
// output thread falls behind, the model side must wait rather than pile up ZeroCopy samples.
template <typename T>
class BoundedQueue {
public:
  void set_capacity(std::size_t c) { cap_ = std::max<std::size_t>(1, c); }

  bool push(T v) {
    std::unique_lock<std::mutex> lk(m_);
    cv_space_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return false;
    q_.push_back(std::move(v));
    lk.unlock();
    cv_item_.notify_one();
    return true;
  }

  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    cv_item_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;   // closed and drained
    out = std::move(q_.front());
    q_.pop_front();
    lk.unlock();
    cv_space_.notify_one();
    return true;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    cv_space_.notify_all();
    cv_item_.notify_all();
  }

private:
  std::mutex m_;
  std::condition_variable cv_space_, cv_item_;
  std::deque<T> q_;
  std::size_t cap_ = 2;
  bool closed_ = false;
};

struct StreamRuntime {
  StreamSpec spec;
  std::unique_ptr<neat::Model> model;
  neat::Graph source_graph, model_graph, video_graph;
  neat::Run source_run, model_run, video_run;
  FrameMailbox mailbox;

  // FIX 1: model side -> output side.
  BoundedQueue<InferItem> out_q;
  // FIX 2: frames pushed into the model graph, awaiting their result. OverflowPolicy::Block
  // keeps push/pull strictly FIFO-paired, so a plain deque is enough to match each result to
  // the frame it came from — no tagging required.
  std::mutex pending_mu;
  std::condition_variable pending_cv;   // signalled by the puller when a slot frees
  std::deque<PendingFrame> pending;
  std::atomic<bool> push_done{false};   // pusher has finished; puller may drain and exit
  // Frames handed to the model graph but not yet pulled back. This is the number the
  // backpressure gate bounds. It is reported as `inflt` so you can SEE whether the bound
  // holds instead of inferring it from latency: if this reads > --model-queue-depth, the
  // gate is not working. (Serial/split push and pull on one thread, so it is 0 or 1 there.)
  std::atomic<int> in_flight{0};

  // Guards source_profile, profile, `processed` and `last_objs` — the live reporter thread
  // reads all four WHILE the source and worker threads are still writing them. Without this
  // a push_back that reallocates under the reporter's feet is a use-after-free, not a
  // wrong number. Taken ~60x/sec per stream; the contention is nil.
  std::mutex prof_mu;
  SourceProfile source_profile;   // written by the source thread
  StageProfile profile;           // written by the worker thread
  // Atomic, not plain int: in split/pipelined the OUTPUT thread increments `processed` while
  // the model/pusher thread reads it as its `--frames` stop condition. A plain int there is a
  // data race, not just a stale read.
  std::atomic<int> processed{0};
  // Frames the MODEL completed (pull returned). In serial/split this equals `processed`; in
  // pipelined mode it is the model's true throughput, which `1000/infer` no longer reports —
  // there, `infer` is a frame's in-graph LATENCY with several frames in flight, not its period.
  std::atomic<int> pulled{0};
  int dropped = 0;
  int last_objs = 0;
  int pull_timeouts = 0;
  Clock::time_point steady_start{};
  double steady_seconds = 0.0;
};

// Source thread: pull the RTSP graph as fast as it produces, copy each frame out of the
// zero-copy pool, and park only the newest one for the worker.
//
// Copying out and releasing the Sample immediately is what lets the source recycle its
// buffers. Holding a ZeroCopy Sample across the model + overlay + encode stalls the source.
void run_source(const Config& cfg, StreamRuntime& rt, const Clock::time_point& deadline) {
  while (!g_stop.load()) {
    if (cfg.duration_s > 0.0 && Clock::now() >= deadline) break;

    neat::Sample frame_sample;
    neat::PullError err;
    const auto wait_begin = Clock::now();
    auto st = rt.source_run.pull("frame", 20000, frame_sample, &err);
    const auto wait_end = Clock::now();
    if (st == neat::PullStatus::Timeout) { ++rt.pull_timeouts; continue; }
    if (st == neat::PullStatus::Closed) break;
    if (st != neat::PullStatus::Ok) {
      std::cerr << "[warn] stream " << rt.spec.id << ": source pull: " << err.message << "\n";
      continue;
    }

    // A frame came off the decoder. Count it BEFORE the drop test below: the decoder did
    // the work whether or not the worker had room for it, and `decode fps` is a statement
    // about the decoder, not about what the worker managed to consume.
    auto& sp = rt.source_profile;
    {
      std::lock_guard<std::mutex> lk(rt.prof_mu);
      ++sp.decoded;
      if (sp.decoded == cfg.warmup_frames + 1) sp.steady_start = Clock::now();
      sp.wait.push_back(ms_since(wait_begin, wait_end));
    }

    // The pull above already drained one frame from the source's edge queue, which is the
    // job. If the worker is still busy with the previous frame, drop this one HERE —
    // before paying for the copy.
    if (!rt.mailbox.wants_frame()) {
      ++rt.mailbox.dropped;
      continue;
    }
    const auto copy_begin = Clock::now();
    const auto tensors = neat::tensors_from_sample(frame_sample, true);
    if (tensors.empty()) continue;
    int w = 0, h = 0;
    if (!infer_dims(tensors.front(), w, h)) continue;
    auto frame_cpu = copy_nv12_to_cpu_tensor(tensors.front());
    const auto copy_end = Clock::now();
    {
      std::lock_guard<std::mutex> lk(rt.prof_mu);
      sp.copy.push_back(ms_since(copy_begin, copy_end));
    }

    rt.mailbox.put(std::move(frame_cpu), w, h);
  }
  if (rt.source_profile.steady_start.time_since_epoch().count() != 0) {
    rt.source_profile.steady_seconds =
        std::chrono::duration<double>(Clock::now() - rt.source_profile.steady_start).count();
  }
  rt.mailbox.close();
}

// ── the per-stream worker ─────────────────────────────────────────────────────
//
// Three topologies, chosen with --mode. They share ONE implementation of
// postproc/overlay/encode (finish_frame), so switching topology cannot change what is drawn
// or published — only who does the work and when.
//
//   serial     (default, the original)
//       source -> [ take | infer | postproc | overlay | encode ]            1 worker thread
//       Nothing overlaps: a stream's frame period is the SUM of every stage, which is why
//       `latency == infer + postproc + overlay + encode` holds to 0.01 ms.
//
//   split      (FIX 1)
//       source -> [ take | infer ] -> out_q -> [ postproc | overlay | encode ]   2 threads
//       Frame period becomes max(infer, postproc+overlay+encode) instead of the sum. This is
//       what segmentation's 11.4 ms overlay was costing: it was charged against the frame rate.
//
//   pipelined  (FIX 2 = FIX 1 + decoupled push/pull)
//       source -> [ take | push ] -> model graph -> [ pull ] -> out_q -> [ output ]  3 threads
//       In serial/split the same thread pushes a frame and then blocks on its pull, so exactly
//       ONE frame is ever inside the model graph and queue_depth is inert. Splitting push from
//       pull lets the graph hold --model-queue-depth frames at once, so frame i+1's EV74
//       preprocess overlaps frame i's MLA and decode. Requires OverflowPolicy::Block: it keeps
//       push/pull FIFO-paired (so `pending` matches results to frames) AND applies backpressure
//       instead of silently dropping. Taken from the C++ 4-stream demo and multi-stream-yolo.

// Read the decoded payload the on-device BoxDecode stage produced.
//
// `pose_out` / `seg_out` are OUT params on purpose: `Instance::keypoints` and `Instance::mask`
// point INTO their storage, so they must outlive the returned instances. Keeping them in the
// caller's frame is what makes that lifetime obvious rather than accidental.
std::vector<Instance> decode_payload(const Config& cfg, StreamRuntime& rt,
                                     const neat::Sample& det_sample, int fw, int fh,
                                     neat::PoseDecodeTensors& pose_out,
                                     neat::SegmentationDecodeTensors& seg_out) {
  std::vector<Instance> instances;
  const auto out_tensors = neat::tensors_from_sample(det_sample, true);
  if (out_tensors.empty()) return instances;

  try {
    switch (rt.spec.task) {
    case Task::Pose: {
      pose_out = neat::decode_pose_tensor(out_tensors.front(), fw, fh, cfg.top_k, false);
      const auto boxes = boxes_from_tensor(pose_out.boxes);
      const auto* kp = pose_out.keypoints.storage
                           ? static_cast<const float*>(pose_out.keypoints.storage->data)
                           : nullptr;
      for (std::size_t i = 0; i < boxes.size(); ++i) {
        if (boxes[i].score < cfg.score_threshold) continue;
        Instance in;
        in.box = boxes[i];
        if (kp != nullptr) in.keypoints = kp + i * 17 * 3;
        instances.push_back(in);
      }
      break;
    }
    case Task::Segmentation: {
      seg_out = neat::decode_segmentation_tensor(out_tensors.front(), fw, fh, cfg.top_k, false);
      const auto boxes = boxes_from_tensor(seg_out.boxes);
      const auto* mk = seg_out.masks.storage
                           ? static_cast<const std::uint8_t*>(seg_out.masks.storage->data)
                           : nullptr;
      for (std::size_t i = 0; i < boxes.size(); ++i) {
        if (boxes[i].score < cfg.score_threshold) continue;
        Instance in;
        in.box = boxes[i];
        if (mk != nullptr) in.mask = mk + i * 160 * 160;
        instances.push_back(in);
      }
      break;
    }
    default: {  // detection + yolox: plain boxes
      const auto decoded = neat::decode_bbox_tensor(out_tensors.front(), fw, fh, cfg.top_k, false);
      for (const auto& b : decoded.boxes) {
        if (b.score < cfg.score_threshold) continue;
        Instance in;
        in.box = b;
        instances.push_back(in);
      }
      break;
    }
    }
  } catch (const std::exception& e) {
    std::cerr << "[warn] stream " << rt.spec.id << ": decode: " << e.what() << "\n";
  }
  return instances;
}

// Burn the annotation into the NV12 frame, in place.
void draw_overlay(const StreamRuntime& rt, neat::Tensor& frame, int fw, int fh,
                  const std::vector<Instance>& instances, const std::string& banner) {
  const Nv12Color kBox{235, 128, 128};
  const Nv12Color kMask{210, 90, 200};
  const Nv12Color kKpt{255, 128, 128};

  auto* yp = nv12_y_plane(frame);
  auto* uvp = nv12_uv_plane(frame, fw, fh);
  for (const auto& in : instances) {
    const int x1 = static_cast<int>(in.box.x1);
    const int y1 = static_cast<int>(in.box.y1);
    const int x2 = static_cast<int>(in.box.x2);
    const int y2 = static_cast<int>(in.box.y2);
    if (x2 <= x1 || y2 <= y1) continue;

    if (in.mask != nullptr) {
      const auto rect = mask_rect_for_frame_box(x1, y1, x2, y2, fw, fh, 160, 160);
      const int bw = x2 - x1;
      const int bh = y2 - y1;
      for (int by = 0; by < bh; ++by) {
        const int my = rect.y + by * rect.height / std::max(1, bh);
        if (my < 0 || my >= 160) continue;
        for (int bx = 0; bx < bw; ++bx) {
          const int mx = rect.x + bx * rect.width / std::max(1, bw);
          if (mx < 0 || mx >= 160) continue;
          if (in.mask[static_cast<std::size_t>(my) * 160 + mx] != 0) {
            blend_mask_pixel(yp, uvp, fw, fh, x1 + bx, y1 + by, kMask, 0.45f);
          }
        }
      }
    }

    draw_box(yp, uvp, fw, fh, x1, y1, x2, y2, kBox);

    const std::string label =
        (rt.spec.task == Task::Pose) ? "PERSON" : class_label(in.box.class_id);
    const int ly = (y1 >= 20) ? y1 - 18 : std::min(fh - 16, y1 + 4);
    draw_text(yp, uvp, fw, fh, x1, ly, label, kBox, 2);

    if (in.keypoints != nullptr) {
      for (int k = 0; k < 17; ++k) {
        if (in.keypoints[k * 3 + 2] < 0.3f) continue;
        const int kx = static_cast<int>(in.keypoints[k * 3 + 0]);
        const int ky = static_cast<int>(in.keypoints[k * 3 + 1]);
        fill_nv12_rect(yp, uvp, fw, fh, kx - 2, ky - 2, kx + 3, ky + 3, kKpt);
      }
      for (const auto& [a, b] : kSkeleton) {
        if (in.keypoints[a * 3 + 2] < 0.3f || in.keypoints[b * 3 + 2] < 0.3f) continue;
        draw_line(yp, uvp, fw, fh, static_cast<int>(in.keypoints[a * 3 + 0]),
                  static_cast<int>(in.keypoints[a * 3 + 1]),
                  static_cast<int>(in.keypoints[b * 3 + 0]),
                  static_cast<int>(in.keypoints[b * 3 + 1]), kKpt, 2);
      }
    }
  }
  draw_text(yp, uvp, fw, fh, 8, 8, banner, kBox, 3);
}

// postproc -> overlay -> encode -> record the profile. The ONE place these happen, in all
// three topologies. In `serial` the model thread calls it; otherwise the output thread does.
void finish_frame(const Config& cfg, StreamRuntime& rt, InferItem& item,
                  const std::string& banner) {
  const auto post_begin = Clock::now();
  neat::PoseDecodeTensors pose_out;
  neat::SegmentationDecodeTensors seg_out;
  const auto instances = decode_payload(cfg, rt, item.det, item.fw, item.fh, pose_out, seg_out);
  const auto post_end = Clock::now();

  const auto overlay_begin = Clock::now();
  if (!cfg.no_overlay) {
    draw_overlay(rt, item.frame, item.fw, item.fh, instances, banner);
  }
  const auto overlay_end = Clock::now();

  const auto enc_begin = Clock::now();
  if (!rt.video_run.push("video", neat::TensorList{item.frame})) {
    std::cerr << "[warn] stream " << rt.spec.id << ": video push failed\n";
  }
  const auto enc_end = Clock::now();

  std::lock_guard<std::mutex> lk(rt.prof_mu);
  const int n = rt.processed.fetch_add(1) + 1;
  rt.last_objs = static_cast<int>(instances.size());
  if (n == cfg.warmup_frames + 1) rt.steady_start = Clock::now();
  rt.profile.add(item.qwait_ms, item.infer_ms, ms_since(post_begin, post_end),
                 ms_since(overlay_begin, overlay_end), ms_since(enc_begin, enc_end),
                 ms_since(item.t_in, enc_end));
}

std::string stream_banner(const StreamRuntime& rt) {
  return "S" + std::to_string(rt.spec.id) + " " + task_name(rt.spec.task) + " :" +
         std::to_string(rt.spec.port);
}

// Take the newest frame the source parked for us. Returns false when the stream is finished.
bool take_frame(const Config& cfg, StreamRuntime& rt, const Clock::time_point& deadline,
                neat::Tensor& frame, int& fw, int& fh, double& qwait_ms,
                Clock::time_point& t_in) {
  if (g_stop.load()) return false;
  if (cfg.duration_s > 0.0 && Clock::now() >= deadline) return false;

  const auto qwait_begin = Clock::now();
  if (!rt.mailbox.take(frame, fw, fh)) return false;
  const auto qwait_end = Clock::now();

  qwait_ms = ms_since(qwait_begin, qwait_end);
  // Latency starts when we HAVE a frame, not when we started waiting for one — otherwise every
  // stream's "latency" would just absorb its own idle time and read ~16.7 ms regardless.
  t_in = qwait_end;
  return true;
}

void mark_steady_done(StreamRuntime& rt) {
  if (rt.steady_start.time_since_epoch().count() != 0) {
    rt.steady_seconds = std::chrono::duration<double>(Clock::now() - rt.steady_start).count();
  }
}



// ── pusher: mailbox -> model graph (bounded by model_queue_depth) ────────────
void run_pusher(const Config& cfg, StreamRuntime& rt, const Clock::time_point& deadline) {
  while (!g_stop.load()) {
    if (cfg.frames > 0 && rt.processed.load() >= cfg.frames) break;

    PendingFrame pf;
    if (!take_frame(cfg, rt, deadline, pf.frame, pf.fw, pf.fh, pf.qwait_ms, pf.t_in)) break;

    pf.push_begin = Clock::now();

    // Tensor is a refcounted handle, so this is a pointer copy, not a frame copy. We need our
    // own reference because `pf` is about to be moved into `pending`.
    neat::Tensor to_push = pf.frame;

    // RESERVE A SLOT BEFORE PUSHING. This is the backpressure — NOT OverflowPolicy::Block.
    //
    // Block does not bound how many frames sit in the graph: a Neat graph has a large internal
    // edge queue (the same 256-frame queue the source-graph comment above warns about), and
    // push() returns as soon as the frame lands in it. So for any model slower than the source
    // — segmentation here — input outruns the model, the edge queue fills toward its physical
    // limit, and per-frame latency climbs into the SECONDS and stays there. Measured: seg went
    // from 30 ms to 7319 ms, and then "delivered" 85 fps from a 60 fps camera while it drained
    // the backlog. That is not throughput, it is a queue emptying.
    //
    // Gating on pending.size() bounds frames-in-flight exactly, whatever the graph does
    // internally. Excess frames are then dropped at the mailbox (drop-oldest, where a live
    // source SHOULD shed load) instead of silently queueing.
    {
      const auto depth = static_cast<std::size_t>(std::max(1, cfg.model_queue_depth));
      std::unique_lock<std::mutex> lk(rt.pending_mu);
      // wait_for, not wait: a lost wakeup here would hang shutdown forever, and Ctrl-C has no
      // way to signal this condition variable. Re-checking on a timer makes that impossible.
      while (rt.pending.size() >= depth && !g_stop.load()) {
        rt.pending_cv.wait_for(lk, std::chrono::milliseconds(50));
      }
      if (g_stop.load()) break;
      // Record BEFORE the push: under Block the result can come back the instant push() returns,
      // and a puller that found `pending` empty would have nothing to match it against.
      rt.pending.push_back(std::move(pf));
    }
    // The lock is NOT held across push(): draining is the puller's job and it needs this same
    // lock, so holding it here would deadlock the pair the moment the queue filled.
    rt.in_flight.fetch_add(1);
    if (!rt.model_run.push("image", neat::TensorList{to_push})) {
      // A failed push means the Run is closing. Stop rather than race the puller to un-record
      // the pending entry — the stale entry is harmless: the puller times out, sees push_done,
      // and drains.
      rt.in_flight.fetch_sub(1);
      ++rt.dropped;
      break;
    }
  }
  rt.push_done.store(true);
  rt.pending_cv.notify_all();
}

// ── puller: model graph -> output queue ──────────────────────────────────────
void run_puller(StreamRuntime& rt) {
  neat::PullError err;

  while (true) {
    if (g_stop.load()) break;
    // Exit only once the pusher has stopped AND every in-flight frame has come back.
    if (rt.push_done.load()) {
      std::lock_guard<std::mutex> lk(rt.pending_mu);
      if (rt.pending.empty()) break;
    }

    InferItem item;
    // A short timeout (not 20 s) so shutdown is prompt: this loop is also the exit check.
    auto st = rt.model_run.pull("detections", 1000, item.det, &err);
    if (st == neat::PullStatus::Timeout) {
      // Pusher is finished and nothing more is coming back — so anything still in `pending`
      // never made it into the graph. Without this the loop would spin on it forever.
      if (rt.push_done.load()) break;
      continue;
    }
    if (st == neat::PullStatus::Closed) break;
    if (st != neat::PullStatus::Ok) {
      std::cerr << "[warn] stream " << rt.spec.id << ": model pull: " << err.message << "\n";
      continue;
    }

    // Block keeps push/pull FIFO-paired, so the oldest pending frame is this result's frame.
    PendingFrame pf;
    {
      std::lock_guard<std::mutex> lk(rt.pending_mu);
      if (rt.pending.empty()) {
        std::cerr << "[warn] stream " << rt.spec.id << ": result with no pending frame\n";
        continue;
      }
      pf = std::move(rt.pending.front());
      rt.pending.pop_front();
    }
    rt.in_flight.fetch_sub(1);
    rt.pending_cv.notify_one();   // a slot freed: let the pusher hand the graph another frame

    item.frame = std::move(pf.frame);
    item.fw = pf.fw;
    item.fh = pf.fh;
    item.t_in = pf.t_in;
    item.qwait_ms = pf.qwait_ms;
    // With N frames in flight this is the frame's IN-GRAPH LATENCY, not its period. Model
    // throughput is `pull fps` in the report, not 1000/infer.
    item.infer_ms = ms_since(pf.push_begin, Clock::now());

    rt.pulled.fetch_add(1);
    if (!rt.out_q.push(std::move(item))) break;
  }
  rt.out_q.close();
}

// ── output thread: postproc -> overlay -> encode ─────────────────────────────
void run_output_thread(const Config& cfg, StreamRuntime& rt) {
  const std::string banner = stream_banner(rt);
  InferItem item;
  while (rt.out_q.pop(item)) {
    finish_frame(cfg, rt, item, banner);
    item = InferItem{};   // release the ZeroCopy sample promptly
  }
  mark_steady_done(rt);
}

// ── live time profile ─────────────────────────────────────────────────────────
//
// ONE reporter thread prints the whole table every `profile_interval` seconds, so the
// terminal shows stage timings as the run happens rather than only at exit. Every number
// is the mean over THAT WINDOW (since the previous print), not a cumulative average — a
// cumulative mean hides a stream that degrades halfway through the run.
double mean_slice(const std::vector<double>& v, std::size_t from) {
  if (v.size() <= from) return 0.0;
  double s = 0.0;
  for (std::size_t i = from; i < v.size(); ++i) s += v[i];
  return s / static_cast<double>(v.size() - from);
}

// Where the previous window ended, per stream.
struct LiveCursor {
  std::size_t copy_n = 0;
  std::size_t work_n = 0;
  int decoded = 0;
  int processed = 0;
  int pulled = 0;
  Clock::time_point t{};
};

void print_live(std::vector<std::unique_ptr<StreamRuntime>>& rts, std::vector<LiveCursor>& cur,
                Clock::time_point start) {
  const auto now = Clock::now();

  // Build the whole table into one string and write it with a single << . Four worker
  // threads no longer print at all, but std::cout is still shared with the warning path,
  // and a table assembled across many << calls can be sliced in half by a stray warning.
  std::ostringstream os;
  os << "\n── t=" << std::fixed << std::setprecision(1)
     << std::chrono::duration<double>(now - start).count()
     << "s ─── ms/frame, mean over this window ───\n";
  os << std::left << std::setw(7) << "stream" << std::setw(14) << "task" << std::right
     << std::setw(8) << "decode" << std::setw(8) << "infer" << std::setw(10) << "postproc"
     << std::setw(9) << "overlay" << std::setw(8) << "encode" << std::setw(9) << "latency"
     << std::setw(10) << "dec fps" << std::setw(10) << "pull fps" << std::setw(11) << "deliv fps"
     << std::setw(7) << "inflt" << std::setw(7) << "objs" << "\n";

  double aggregate = 0.0;
  for (std::size_t i = 0; i < rts.size(); ++i) {
    auto& rt = *rts[i];
    auto& c = cur[i];

    double dec, inf, post, ovl, enc, lat;
    int decoded, processed, pulled, objs, inflt;
    {
      std::lock_guard<std::mutex> lk(rt.prof_mu);
      dec  = mean_slice(rt.source_profile.copy, c.copy_n);
      inf  = mean_slice(rt.profile.infer,       c.work_n);
      post = mean_slice(rt.profile.postproc,    c.work_n);
      ovl  = mean_slice(rt.profile.overlay,     c.work_n);
      enc  = mean_slice(rt.profile.encode,      c.work_n);
      lat  = mean_slice(rt.profile.latency,     c.work_n);
      c.copy_n  = rt.source_profile.copy.size();
      c.work_n  = rt.profile.infer.size();
      decoded   = rt.source_profile.decoded;
      processed = rt.processed.load();
      pulled    = rt.pulled.load();
      inflt     = rt.in_flight.load();
      objs      = rt.last_objs;
    }

    const double dt = std::chrono::duration<double>(now - c.t).count();
    const double dec_fps = dt > 0.0 ? (decoded - c.decoded) / dt : 0.0;
    const double dlv_fps = dt > 0.0 ? (processed - c.processed) / dt : 0.0;
    // The model's TRUE throughput: frames it actually completed per second. In pipelined mode
    // `infer` is an in-flight latency, so 1000/infer understates the model badly — this does not.
    const double pull_fps = dt > 0.0 ? (pulled - c.pulled) / dt : 0.0;
    c.decoded = decoded;
    c.processed = processed;
    c.pulled = pulled;
    c.t = now;
    aggregate += dlv_fps;

    os << std::left << std::setw(7) << rt.spec.id << std::setw(14) << task_name(rt.spec.task)
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(8) << dec << std::setw(8) << inf << std::setw(10) << post
       << std::setw(9) << ovl << std::setw(8) << enc << std::setw(9) << lat
       << std::setprecision(1)
       << std::setw(10) << dec_fps << std::setw(10) << pull_fps << std::setw(11) << dlv_fps
       << std::setw(7) << inflt << std::setw(7) << objs << "\n";
  }
  os << std::string(52, ' ') << "aggregate delivered " << std::fixed << std::setprecision(1)
     << aggregate << " fps\n";

  std::cout << os.str() << std::flush;
}

void run_reporter(const Config& cfg, std::vector<std::unique_ptr<StreamRuntime>>& rts,
                  Clock::time_point start, Clock::time_point deadline) {
  if (cfg.profile_interval_s <= 0.0) return;

  std::vector<LiveCursor> cur(rts.size());
  for (auto& c : cur) c.t = start;

  const auto finished = [&] {
    return g_stop.load() || g_done.load() ||
           (cfg.duration_s > 0.0 && Clock::now() >= deadline);
  };

  while (!finished()) {
    // Wake often so Ctrl-C, --frames and --duration are all honoured promptly rather than
    // after a full interval of sleeping.
    const auto next = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                         std::chrono::duration<double>(cfg.profile_interval_s));
    while (!finished() && Clock::now() < next) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (finished()) return;
    print_live(rts, cur, start);
  }
}

void print_report(const Config& cfg, std::vector<std::unique_ptr<StreamRuntime>>& rts,
                  double wall_s) {
  const auto skip = static_cast<std::size_t>(std::max(0, cfg.warmup_frames));

  auto cell = [&](const std::vector<double>& v) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << mean_of(v, skip) << "|" << pct_of(v, skip, 95);
    return os.str();
  };

  // ── 1. the four pipeline components, per model-stream ──────────────────────
  std::cout << "\n=== time profile (C++, " << (cfg.no_overlay ? "no-overlay" : "with-overlay")
            << ") — ms/frame, mean | p95 ===\n";
  std::cout << std::left << std::setw(7) << "stream" << std::setw(14) << "task"
            << std::right << std::setw(7) << "frames"
            << std::setw(14) << "decode" << std::setw(14) << "infer"
            << std::setw(14) << "postproc" << std::setw(14) << "overlay"
            << std::setw(14) << "encode" << std::setw(16) << "latency" << "\n";
  for (auto& rt : rts) {
    const auto& p = rt->profile;
    std::cout << std::left << std::setw(7) << rt->spec.id << std::setw(14)
              << task_name(rt->spec.task) << std::right << std::setw(7) << rt->processed.load()
              << std::setw(14) << cell(rt->source_profile.copy)
              << std::setw(14) << cell(p.infer)
              << std::setw(14) << cell(p.postproc) << std::setw(14) << cell(p.overlay)
              << std::setw(14) << cell(p.encode) << std::setw(16) << cell(p.latency) << "\n";
  }

  // Say what each column IS. Two of them do not mean what their name suggests, and a
  // number that is quietly measuring the wrong thing is worse than no number at all.
  std::cout
      << "\n  decode   memcpy of the decoded NV12 frame out of the zero-copy pool (host CPU).\n"
      << "           The H.264 decode itself runs on the hardware decoder and is NOT visible\n"
      << "           from the host — see `decode fps` below for whether it is keeping up.\n"
      << "  infer    model push+pull: on-device preprocess + MLA + fused BoxDecode.\n"
      << "  postproc host-side read of the decoded payload + instance build.\n"
      << "  overlay  host-side NV12 draw (0.00 under --no-overlay).\n"
      << "  encode   video push. This ENQUEUES to the encoder and returns, so it is encoder\n"
      << "           HEADROOM, not encode latency: it sits near 0 until the encoder falls\n"
      << "           behind, and only then does its backpressure show up here.\n"
      << "  latency  frame in hand -> frame handed to the encoder (worker thread).\n";

  // ── 2. throughput + the waits, which tell you WHERE the bottleneck is ──────
  std::cout << "\n=== per model-stream throughput (window " << std::fixed << std::setprecision(1)
            << wall_s << "s) ===\n";
  std::cout << std::left << std::setw(7) << "stream" << std::setw(14) << "task"
            << std::right << std::setw(12) << "decode fps"
            << std::setw(11) << "pull fps" << std::setw(15) << "delivered fps"
            << std::setw(12) << "src wait" << std::setw(10) << "qwait"
            << std::setw(10) << "dropped" << std::setw(10) << "pull t/o" << "\n";
  double aggregate = 0.0;
  for (auto& rt : rts) {
    const auto& sp = rt->source_profile;

    const int steady_decoded = std::max(0, sp.decoded - cfg.warmup_frames);
    const double dec_secs = sp.steady_seconds > 0.0 ? sp.steady_seconds : wall_s;
    const double decode_fps = dec_secs > 0.0 ? steady_decoded / dec_secs : 0.0;

    const int steady_frames = std::max(0, rt->processed.load() - cfg.warmup_frames);
    const double secs = rt->steady_seconds > 0.0 ? rt->steady_seconds : wall_s;
    const double delivered = secs > 0.0 ? steady_frames / secs : 0.0;
    const int steady_pulled = std::max(0, rt->pulled.load() - cfg.warmup_frames);
    const double pull_fps = secs > 0.0 ? steady_pulled / secs : 0.0;
    aggregate += delivered;

    std::cout << std::left << std::setw(7) << rt->spec.id << std::setw(14)
              << task_name(rt->spec.task) << std::right << std::fixed
              << std::setprecision(1) << std::setw(12) << decode_fps
              << std::setprecision(2) << std::setw(11) << pull_fps
              << std::setw(15) << delivered
              << std::setw(12) << mean_of(sp.wait, skip)
              << std::setw(10) << mean_of(rt->profile.qwait, skip)
              << std::setw(10) << rt->mailbox.dropped
              << std::setw(10) << rt->pull_timeouts << "\n";
  }
  std::cout
      << "\n  decode fps    frames the decoder produced (INCLUDING ones the worker was too\n"
      << "                busy to take — `dropped`). This is the source's true rate.\n"
      << "  pull fps      frames the model actually completed per second — its true throughput.\n"
      << "                Do NOT read 1000/infer as a rate: with several frames in flight, `infer`\n"
      << "                is a frame's in-graph LATENCY, not its period.\n"
      << "  delivered fps frames that actually reached the encoder. The number that matters.\n"
      << "  src wait      source thread blocked on pull. ~1000/stream_fps (e.g. 16.7 ms at\n"
      << "                60 fps) means the decoder is keeping up; well above it means it is not.\n"
      << "  qwait         worker idle waiting for a frame. High qwait + low delivered fps means\n"
      << "                the SOURCE is the bottleneck, not the model.\n";

  std::cout << "\naggregate delivered: " << std::fixed << std::setprecision(2) << aggregate
            << " fps across " << rts.size() << " stream-model pairs\n";
}

void print_usage() {
  std::cout
      << "quad_stream_quad_model (C++)\n"
      << "  --config <path>       config file (default ./config/default.conf)\n"
      << "  --duration <seconds>  measure over a wall-clock window (preferred on a shared MLA)\n"
      << "  --frames <n>          frames PER stream; 0 = until interrupted\n"
      << "  --num-streams <1..4>  how many stream slots to run\n"
      << "  --no-overlay          skip the annotation (isolates the model rate)\n"
      << "  --profile-interval <s> how often to print the live time profile (0 = off, default 5)\n"
      << "  --model-queue-depth <n>  frames the model graph may hold in flight (default 4)\n"
      << "  --output-queue-depth <n> frames parked before the output thread (default 2)\n"
      << "  --pre-target <t>      AUTO | EV74 | A65\n"
      << "  --post-target <t>     AUTO | EV74 | A65\n"
      << "  --print-backend       dump the generated GStreamer backends\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  std::string config_path = "./config/default.conf";
  std::vector<std::string> args(argv + 1, argv + argc);
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--help" || args[i] == "-h") { print_usage(); return 0; }
    if (args[i] == "--config" && i + 1 < args.size()) { config_path = args[++i]; }
  }

  // We hand the model graph a CPU-backed NV12 tensor, and its preprocess route is on the
  // EV74. Allow the runtime's compatibility copy rather than failing the push.
  ::setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);

  try {
    Config cfg = read_config(config_path);

    // CLI overrides (applied after the file, so they win).
    for (std::size_t i = 0; i < args.size(); ++i) {
      const auto& a = args[i];
      auto next = [&]() -> std::string { return (i + 1 < args.size()) ? args[++i] : std::string(); };
      if (a == "--duration") cfg.duration_s = std::stod(next());
      else if (a == "--frames") cfg.frames = std::stoi(next());
      else if (a == "--num-streams") cfg.num_streams = std::stoi(next());
      else if (a == "--warmup-frames") cfg.warmup_frames = std::stoi(next());
      else if (a == "--profile-interval") cfg.profile_interval_s = std::stod(next());
      else if (a == "--model-queue-depth") cfg.model_queue_depth = std::stoi(next());
      else if (a == "--output-queue-depth") cfg.output_queue_depth = std::stoi(next());
      else if (a == "--no-overlay") cfg.no_overlay = true;
      else if (a == "--print-backend") cfg.print_backend = true;
      else if (a == "--pre-target") cfg.cvu_pre_target = next();
      else if (a == "--post-target") cfg.cvu_post_target = next();
    }

    const auto specs = build_specs(cfg);

    neat::RunOptions run_options;
    run_options.preset = neat::RunPreset::Realtime;
    run_options.queue_depth = cfg.queue_depth;
    run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
    run_options.output_memory = neat::OutputMemory::ZeroCopy;

    // The SOURCE run stays Realtime/KeepLatest: a live camera must be drained continuously and
    // its stale frames thrown away. The MODEL run is different — Block, not KeepLatest:
    //   * Block keeps push/pull strictly FIFO-paired, so `pending` can match each result back
    //     to the frame it came from. KeepLatest silently drops results and breaks that pairing.
    //   * It does NOT bound how many frames sit in the graph — that is the pusher's in-flight
    //     gate (see run_pusher). Relying on Block alone let segmentation accumulate a 600-frame,
    //     11-second backlog.
    neat::RunOptions model_run_options = run_options;
    model_run_options.preset = neat::RunPreset::Reliable;
    model_run_options.queue_depth = cfg.model_queue_depth;
    model_run_options.overflow_policy = neat::OverflowPolicy::Block;

    neat::RunOptions video_run_options = run_options;
    video_run_options.output_memory = neat::OutputMemory::Owned;

    // StreamRuntime owns a mutex/condition_variable, so it is neither copyable nor
    // movable — hold them by pointer so the vector never has to relocate one.
    std::vector<std::unique_ptr<StreamRuntime>> rts;
    rts.reserve(specs.size());

    for (const auto& spec : specs) {
      auto rt = std::make_unique<StreamRuntime>();
      rt->spec = spec;
      rt->model = make_model(cfg, spec);
      rt->source_graph = make_source_graph(cfg, spec);
      rt->model_graph = make_model_graph(cfg, spec, *rt->model);
      rt->video_graph = make_video_graph(cfg, spec);

      if (cfg.print_backend) {
        std::cout << "--- stream " << spec.id << " (" << task_name(spec.task) << ")\n"
                  << rt->model_graph.describe_backend() << "\n";
      }

      // NOTE: the source run is deliberately NOT built here — see below.
      rt->model_run = rt->model_graph.build(model_run_options);
      rt->out_q.set_capacity(static_cast<std::size_t>(std::max(1, cfg.output_queue_depth)));
      neat::Tensor seed = make_blank_nv12_tensor(cfg.fallback_width, cfg.fallback_height);
      rt->video_run = rt->video_graph.build(neat::TensorList{seed}, video_run_options);

      std::cout << "Stream " << spec.id << ": " << std::left << std::setw(13)
                << task_name(spec.task) << spec.model_path << "\n"
                << "  RTSP " << spec.rtsp_url << " -> udp://" << cfg.udp_host << ":" << spec.port
                << "\n";
      rts.push_back(std::move(rt));
    }

    // Start the RTSP sources LAST, after every model is loaded, and immediately before the
    // threads that drain them.
    //
    // Graph::build() starts the pipeline running. Building each stream's source inside the
    // loop above meant stream 0's camera started streaming while streams 1-3 were still
    // loading their models (seconds each, an MLA archive at a time). Nobody was pulling
    // yet, so its 256-frame edge queue filled and it was already dead on arrival with a
    // "sink backpressure timeout". The symptom was diagnostic: stream 3 — built last, and
    // so idle for the shortest time — was the only one that survived.
    for (auto& rt : rts) {
      rt->source_run = rt->source_graph.build(run_options);
    }

    std::cout << "build " << __DATE__ << " " << __TIME__ << "\n"
              << "Running " << rts.size() << " streams x 4 threads"
              << " (model_queue_depth=" << cfg.model_queue_depth
              << ", output_queue_depth=" << cfg.output_queue_depth
              << ", overlay=" << (cfg.no_overlay ? "off" : "on") << "). Ctrl-C to stop.\n"
              << std::flush;

    const auto start = Clock::now();
    const auto deadline =
        start + std::chrono::milliseconds(static_cast<long>(cfg.duration_s * 1000.0));

    // Four threads per stream:
    //   source  RTSP -> drop-oldest mailbox
    //   pusher  mailbox -> model graph        (bounded by model_queue_depth)
    //   puller  model graph -> output queue
    //   output  postproc -> overlay -> encode
    std::vector<std::thread> sources, pushers, pullers, outputs;
    for (auto* v : {&sources, &pushers, &pullers, &outputs}) v->reserve(rts.size());

    for (auto& rt : rts) {
      StreamRuntime* p = rt.get();
      sources.emplace_back([&cfg, p, deadline]() { run_source(cfg, *p, deadline); });
      pushers.emplace_back([&cfg, p, deadline]() { run_pusher(cfg, *p, deadline); });
      pullers.emplace_back([p]() { run_puller(*p); });
      outputs.emplace_back([&cfg, p]() { run_output_thread(cfg, *p); });
    }

    // One reporter thread prints the live time profile for ALL streams on an interval.
    std::thread reporter([&cfg, &rts, start, deadline]() { run_reporter(cfg, rts, start, deadline); });

    // Shut down in pipeline order, so no stage is ever waiting on a stage that already exited.
    for (auto& s : sources) s.join();
    for (auto& rt : rts) rt->mailbox.close();   // releases a worker/pusher blocked in take()
    for (auto& rt : rts) rt->pending_cv.notify_all();   // and one blocked waiting for a slot
    for (auto& u : pushers) u.join();           // sets push_done
    for (auto& q : pullers) q.join();           // drains the in-flight frames, then closes out_q
    for (auto& rt : rts) rt->out_q.close();     // idempotent
    for (auto& o : outputs) o.join();

    // The reporter reads the profiles these threads write, so it MUST be joined before the
    // final report reads them unlocked.
    g_done.store(true);
    reporter.join();

    const double wall_s = std::chrono::duration<double>(Clock::now() - start).count();
    print_report(cfg, rts, wall_s);

    for (auto& rt : rts) {
      rt->video_run.close();
      rt->model_run.close();
      rt->source_run.close();
    }
    return g_stop.load() ? 130 : 0;
  } catch (const neat::NeatError& e) {
    std::cerr << "NEAT error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
