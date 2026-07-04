#include <neat.h>

#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {


constexpr const char* kConfigPath = "./config/default.conf";

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

struct Config {
  std::string rtsp_url;
  std::string model_path;
  int fallback_width = 1280;
  int fallback_height = 720;
  int fallback_fps = 25;
  int model_width = 640;
  int model_height = 640;
  int latency_ms = 200;
  float score_threshold = 0.25f;
  float nms_iou = 0.50f;
  int top_k = 100;
  int num_classes = 80;
  int frames = 0;
  std::string udp_host;
  int udp_port_base = 0;
  int bitrate_kbps = 4000;
  bool tcp = true;
  bool print_backend = false;
};

using Clock = std::chrono::steady_clock;

struct StageTimes {
  double decoder_ms = 0.0;
  double inference_ms = 0.0;
  double overlay_ms = 0.0;
  double encoder_ms = 0.0;

  double total_ms() const {
    return decoder_ms + inference_ms + overlay_ms + encoder_ms;
  }
};

struct ProfileTotals {
  StageTimes sum;

  void add(const StageTimes& frame) {
    sum.decoder_ms += frame.decoder_ms;
    sum.inference_ms += frame.inference_ms;
    sum.overlay_ms += frame.overlay_ms;
    sum.encoder_ms += frame.encoder_ms;
  }

  StageTimes average(int frames) const {
    if (frames <= 0) {
      return {};
    }
    const double n = static_cast<double>(frames);
    return {sum.decoder_ms / n, sum.inference_ms / n, sum.overlay_ms / n,
            sum.encoder_ms / n};
  }
};

double ms_since(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool should_log_frame(int processed, int target_frames) {
  return processed == 1 || (processed % 30) == 0 ||
         (target_frames > 0 && processed == target_frames);
}

int to_int(const std::string& value) { return std::stoi(value); }
float to_float(const std::string& value) { return std::stof(value); }
bool to_bool(const std::string& value) {
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string trim(std::string value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

void set_config_value(Config& cfg, const std::string& key, const std::string& value) {
  if (key == "rtsp_url") {
    cfg.rtsp_url = value;
  } else if (key == "model_path") {
    cfg.model_path = value;
  } else if (key == "fallback_width") {
    cfg.fallback_width = to_int(value);
  } else if (key == "fallback_height") {
    cfg.fallback_height = to_int(value);
  } else if (key == "fallback_fps") {
    cfg.fallback_fps = to_int(value);
  } else if (key == "model_width") {
    cfg.model_width = to_int(value);
  } else if (key == "model_height") {
    cfg.model_height = to_int(value);
  } else if (key == "latency_ms") {
    cfg.latency_ms = to_int(value);
  } else if (key == "score_threshold") {
    cfg.score_threshold = to_float(value);
  } else if (key == "nms_iou") {
    cfg.nms_iou = to_float(value);
  } else if (key == "top_k") {
    cfg.top_k = to_int(value);
  } else if (key == "num_classes") {
    cfg.num_classes = to_int(value);
  } else if (key == "frames") {
    cfg.frames = to_int(value);
  } else if (key == "udp_host") {
    cfg.udp_host = value;
  } else if (key == "udp_port_base") {
    cfg.udp_port_base = to_int(value);
  } else if (key == "bitrate_kbps") {
    cfg.bitrate_kbps = to_int(value);
  } else if (key == "rtsp_transport") {
    cfg.tcp = value != "udp";
  } else if (key == "print_backend") {
    cfg.print_backend = to_bool(value);
  } else {
    throw std::runtime_error("unknown config key: " + key);
  }
}

Config read_config() {
  Config cfg;
  std::ifstream file(kConfigPath);
  if (!file) {
    throw std::runtime_error(std::string("config file not found: ") + kConfigPath);
  }
  std::string line;
  while (std::getline(file, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line.erase(comment);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    const auto equal = line.find('=');
    set_config_value(cfg, trim(line.substr(0, equal)), trim(line.substr(equal + 1)));
  }
  return cfg;
}

struct ModelSpec {
  std::string name;
  std::string path;
  std::optional<neat::BoxDecodeType> decode_type;
  int model_width = 640;
  int model_height = 640;
  bool coco_yolo_preprocess = true;
};

struct ModelRuntime {
  ModelSpec spec;
  std::unique_ptr<neat::Model> model;
  neat::Graph model_graph;
  neat::Run model_run;
  neat::Graph udp_graph;
  neat::Run udp_run;
  int udp_port = 0;
  int processed = 0;
  int last_visible_boxes = -1;
  Clock::time_point inference_begin;
  StageTimes current_profile;
  ProfileTotals profile;
};

struct Nv12Color {
  std::uint8_t y = 235;
  std::uint8_t u = 128;
  std::uint8_t v = 128;
};

Nv12Color class_color(int class_id) {
  static constexpr std::array<Nv12Color, 12> palette = {
      Nv12Color{82, 90, 240},   Nv12Color{170, 60, 180}, Nv12Color{210, 40, 40},
      Nv12Color{180, 40, 150},  Nv12Color{200, 100, 30}, Nv12Color{145, 54, 34},
      Nv12Color{160, 210, 190}, Nv12Color{120, 220, 90}, Nv12Color{225, 110, 170},
      Nv12Color{190, 80, 210},  Nv12Color{235, 128, 128}, Nv12Color{95, 200, 220},
  };
  return palette[static_cast<std::size_t>(std::max(0, class_id)) % palette.size()];
}

struct SegmentationOverlay {
  int instances = 0;
  int visible_instances = 0;
};

const std::vector<std::string>& coco_labels() {
  static const std::vector<std::string> labels = {
      "PERSON",        "BICYCLE",      "CAR",          "MOTORCYCLE", "AIRPLANE",
      "BUS",           "TRAIN",        "TRUCK",        "BOAT",       "TRAFFIC LIGHT",
      "FIRE HYDRANT",  "STOP SIGN",    "PARKING METER", "BENCH",      "BIRD",
      "CAT",           "DOG",          "HORSE",        "SHEEP",      "COW",
      "ELEPHANT",      "BEAR",         "ZEBRA",        "GIRAFFE",    "BACKPACK",
      "UMBRELLA",      "HANDBAG",      "TIE",          "SUITCASE",   "FRISBEE",
      "SKIS",          "SNOWBOARD",    "SPORTS BALL",  "KITE",       "BASEBALL BAT",
      "BASEBALL GLOVE", "SKATEBOARD",  "SURFBOARD",    "TENNIS RACKET", "BOTTLE",
      "WINE GLASS",    "CUP",          "FORK",         "KNIFE",      "SPOON",
      "BOWL",          "BANANA",       "APPLE",        "SANDWICH",   "ORANGE",
      "BROCCOLI",      "CARROT",       "HOT DOG",      "PIZZA",      "DONUT",
      "CAKE",          "CHAIR",        "COUCH",        "POTTED PLANT", "BED",
      "DINING TABLE",  "TOILET",       "TV",           "LAPTOP",     "MOUSE",
      "REMOTE",        "KEYBOARD",     "CELL PHONE",   "MICROWAVE",  "OVEN",
      "TOASTER",       "SINK",         "REFRIGERATOR", "BOOK",       "CLOCK",
      "VASE",          "SCISSORS",     "TEDDY BEAR",   "HAIR DRIER", "TOOTHBRUSH"};
  return labels;
}

std::string class_label(int class_id) {
  const auto& labels = coco_labels();
  if (class_id >= 0 && class_id < static_cast<int>(labels.size())) {
    return labels[static_cast<std::size_t>(class_id)];
  }
  return "CLASS " + std::to_string(class_id);
}

std::vector<ModelSpec> make_specs(const Config& cfg) {
  return {{"yolov8n-seg", cfg.model_path, neat::BoxDecodeType::YoloV8Seg,
           cfg.model_width, cfg.model_height, true}};
}

std::unique_ptr<neat::Model> make_model(const Config& cfg, const ModelSpec& spec) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.input_max_width = cfg.fallback_width;
  opt.preprocess.input_max_height = cfg.fallback_height;
  opt.preprocess.input_max_depth = 1;
  opt.preprocess.resize.width = spec.model_width;
  opt.preprocess.resize.height = spec.model_height;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  if (spec.coco_yolo_preprocess) {
    opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  } else {
    opt.preprocess.preset = neat::NormalizePreset::None;
    opt.preprocess.resize.enable = neat::AutoFlag::On;
    opt.preprocess.normalize.enable = neat::AutoFlag::On;
    opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
    opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
    opt.preprocess.normalize.has_explicit_stats = true;
  }
  if (spec.decode_type.has_value()) {
    opt.decode_type = *spec.decode_type;
    opt.score_threshold = cfg.score_threshold;
    opt.nms_iou_threshold = cfg.nms_iou;
    opt.top_k = cfg.top_k;
    opt.num_classes = cfg.num_classes;
  }
  return std::make_unique<neat::Model>(spec.path, opt);
}

groups::RtspDecodedInputOptions make_rtsp_options(const Config& cfg) {
  groups::RtspDecodedInputOptions opt;
  opt.url = cfg.rtsp_url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = neat::FormatTag::NV12;
  opt.decoder_name = "decoder";
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

neat::Graph make_source_pipeline(const Config& cfg) {
  auto source = groups::RtspDecodedInput(make_rtsp_options(cfg));

  neat::Graph app("multi_model_load_probe_source");
  app.add(source);
  app.add(neat::nodes::Output("frame", neat::OutputOptions::Latest()));
  return app;
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
  opt.fps_n = cfg.fallback_fps;
  opt.fps_d = 1;
  opt.caps_override = "video/x-raw,format=NV12,width=" + std::to_string(cfg.fallback_width) +
                      ",height=" + std::to_string(cfg.fallback_height) + ",framerate=" +
                      std::to_string(cfg.fallback_fps) + "/1";
  opt.use_simaai_pool = false;
  return opt;
}

neat::Graph make_model_pipeline(const Config& cfg, const neat::Model& model,
                                const std::string& name) {
  neat::Graph app("multi_model_load_probe_" + name + "_model");
  app.add(neat::nodes::Input("image", make_nv12_input_options(cfg)));
  app.add(model);
  app.add(neat::nodes::Output("result", neat::OutputOptions::EveryFrame(4)));
  return app;
}

neat::Graph make_udp_pipeline(const Config& cfg, int udp_port, const std::string& name) {
  auto video_options = groups::VideoSenderOptions::H264RtpUdpFromRaw(
      cfg.fallback_width, cfg.fallback_height, cfg.fallback_fps);
  video_options.host = cfg.udp_host;
  video_options.channel = 0;
  video_options.video_port_base = udp_port;
  video_options.encoder.bitrate_kbps = cfg.bitrate_kbps;

  neat::Graph app("multi_model_load_probe_" + name + "_udp");
  app.add(neat::nodes::Input("video", make_nv12_input_options(cfg)));
  app.add(groups::VideoSender(video_options));
  return app;
}

bool infer_dims(const neat::Tensor& tensor, int& width, int& height) {
  width = tensor.width();
  height = tensor.height();
  if ((width <= 0 || height <= 0) && tensor.shape.size() >= 2) {
    height = static_cast<int>(tensor.shape[0]);
    width = static_cast<int>(tensor.shape[1]);
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
  std::string err;
  const std::size_t bytes =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U / 2U;
  neat::Tensor out;
  out.storage = neat::make_cpu_owned_storage(bytes);
  std::memset(out.storage->data, 0, bytes);
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

std::string sample_summary(const neat::Sample& sample);

std::uint8_t* nv12_y_plane(neat::Tensor& tensor, int width, int height) {
  (void)height;
  if (!tensor.storage || !tensor.storage->data) {
    throw std::runtime_error("NV12 tensor has no CPU storage");
  }
  return static_cast<std::uint8_t*>(tensor.storage->data) + tensor.byte_offset;
}

std::uint8_t* nv12_uv_plane(neat::Tensor& tensor, int width, int height) {
  return nv12_y_plane(tensor, width, height) +
         static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

void draw_nv12_point(neat::Tensor& frame, int width, int height, int x, int y,
                     const Nv12Color& color) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }
  auto* y_plane = nv12_y_plane(frame, width, height);
  auto* uv_plane = nv12_uv_plane(frame, width, height);
  y_plane[static_cast<std::size_t>(y) * width + x] = color.y;

  const int uv_x = x & ~1;
  const int uv_y = y / 2;
  if (uv_x + 1 < width && uv_y >= 0 && uv_y < height / 2) {
    const std::size_t offset = static_cast<std::size_t>(uv_y) * width + uv_x;
    uv_plane[offset] = color.u;
    uv_plane[offset + 1] = color.v;
  }
}

void fill_nv12_rect(neat::Tensor& frame, int width, int height, int x1, int y1, int x2, int y2,
                    const Nv12Color& color) {
  x1 = std::max(0, std::min(x1, width));
  x2 = std::max(0, std::min(x2, width));
  y1 = std::max(0, std::min(y1, height));
  y2 = std::max(0, std::min(y2, height));
  if (x2 <= x1 || y2 <= y1) {
    return;
  }

  auto* y_plane = nv12_y_plane(frame, width, height);
  auto* uv_plane = nv12_uv_plane(frame, width, height);
  for (int y = y1; y < y2; ++y) {
    std::memset(y_plane + static_cast<std::size_t>(y) * width + x1, color.y,
                static_cast<std::size_t>(x2 - x1));
  }

  const int uv_x1 = x1 & ~1;
  const int uv_x2 = (x2 + 1) & ~1;
  const int uv_y1 = y1 / 2;
  const int uv_y2 = (y2 + 1) / 2;
  for (int row = std::max(0, uv_y1); row < std::min(height / 2, uv_y2); ++row) {
    auto* row_ptr = uv_plane + static_cast<std::size_t>(row) * width;
    for (int col = std::max(0, uv_x1); col + 1 < std::min(width, uv_x2); col += 2) {
      row_ptr[col] = color.u;
      row_ptr[col + 1] = color.v;
    }
  }
}

void draw_nv12_line_h(neat::Tensor& frame, int width, int height, int x1, int x2, int y,
                      int thickness, const Nv12Color& color) {
  fill_nv12_rect(frame, width, height, x1, y, x2, y + thickness, color);
}

void draw_nv12_line_v(neat::Tensor& frame, int width, int height, int x, int y1, int y2,
                      int thickness, const Nv12Color& color) {
  fill_nv12_rect(frame, width, height, x, y1, x + thickness, y2, color);
}

void draw_nv12_line(neat::Tensor& frame, int width, int height, int x1, int y1, int x2, int y2,
                    int thickness, const Nv12Color& color) {
  const int dx = std::abs(x2 - x1);
  const int sx = x1 < x2 ? 1 : -1;
  const int dy = -std::abs(y2 - y1);
  const int sy = y1 < y2 ? 1 : -1;
  int err = dx + dy;
  while (true) {
    fill_nv12_rect(frame, width, height, x1 - thickness / 2, y1 - thickness / 2,
                   x1 + thickness / 2 + 1, y1 + thickness / 2 + 1, color);
    if (x1 == x2 && y1 == y2) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x1 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y1 += sy;
    }
  }
}

void draw_nv12_circle(neat::Tensor& frame, int width, int height, int cx, int cy, int radius,
                      const Nv12Color& color) {
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      if (x * x + y * y <= radius * radius) {
        draw_nv12_point(frame, width, height, cx + x, cy + y, color);
      }
    }
  }
}

void draw_nv12_box(neat::Tensor& frame, int width, int height, const neat::Box& box,
                   const Nv12Color& color) {
  const int x1 = std::max(0, std::min(static_cast<int>(box.x1), width - 1));
  const int y1 = std::max(0, std::min(static_cast<int>(box.y1), height - 1));
  const int x2 = std::max(0, std::min(static_cast<int>(box.x2), width - 1));
  const int y2 = std::max(0, std::min(static_cast<int>(box.y2), height - 1));
  if (x2 <= x1 || y2 <= y1) {
    return;
  }
  constexpr int kThickness = 3;
  draw_nv12_line_h(frame, width, height, x1, x2, y1, kThickness, color);
  draw_nv12_line_h(frame, width, height, x1, x2, y2 - kThickness + 1, kThickness, color);
  draw_nv12_line_v(frame, width, height, x1, y1, y2, kThickness, color);
  draw_nv12_line_v(frame, width, height, x2 - kThickness + 1, y1, y2, kThickness, color);
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
  case '-': return {"00000", "00000", "00000", "11111", "00000", "00000", "00000"};
  case '_': return {"00000", "00000", "00000", "00000", "00000", "00000", "11111"};
  case ':': return {"00000", "00100", "00100", "00000", "00100", "00100", "00000"};
  case '.': return {"00000", "00000", "00000", "00000", "00000", "01100", "01100"};
  case ' ': return {"00000", "00000", "00000", "00000", "00000", "00000", "00000"};
  default: return {"11111", "00001", "00010", "00100", "00100", "00000", "00100"};
  }
}

char overlay_char(char c) {
  if (c >= 'a' && c <= 'z') {
    return static_cast<char>(c - 'a' + 'A');
  }
  return c;
}

void draw_nv12_text(neat::Tensor& frame, int width, int height, int x, int y,
                    const std::string& text, const Nv12Color& color, int scale = 2) {
  int cursor_x = x;
  for (char raw : text) {
    const auto glyph = glyph_for(overlay_char(raw));
    for (int row = 0; row < static_cast<int>(glyph.size()); ++row) {
      for (int col = 0; col < static_cast<int>(glyph[row].size()); ++col) {
        if (glyph[row][col] != '1') {
          continue;
        }
        fill_nv12_rect(frame, width, height, cursor_x + col * scale, y + row * scale,
                       cursor_x + (col + 1) * scale, y + (row + 1) * scale, color);
      }
    }
    cursor_x += 6 * scale;
    if (cursor_x >= width - 6 * scale) {
      break;
    }
  }
}

void draw_nv12_label(neat::Tensor& frame, int width, int height, int x, int y,
                     const std::string& text, const Nv12Color& fg) {
  constexpr int kScale = 2;
  const int text_w = static_cast<int>(std::min<std::size_t>(text.size(), 22)) * 6 * kScale;
  const int text_h = 7 * kScale;
  const int bx = std::max(0, std::min(x, width - 1));
  const int by = std::max(0, std::min(y, height - 1));
  const Nv12Color black{16, 128, 128};
  fill_nv12_rect(frame, width, height, bx, by, std::min(width, bx + text_w + 6),
                 std::min(height, by + text_h + 4), black);
  draw_nv12_text(frame, width, height, bx + 3, by + 2, text.substr(0, 22), fg, kScale);
}

std::vector<float> tensor_to_float_vector(const neat::Tensor& tensor) {
  if (tensor.dtype != neat::TensorDType::Float32) {
    return {};
  }
  const auto bytes = tensor.copy_dense_bytes_tight();
  if (bytes.empty() || (bytes.size() % sizeof(float)) != 0) {
    return {};
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
  return values;
}

std::vector<neat::Box> boxes_from_decoded_tensor(const neat::Tensor& tensor) {
  std::vector<neat::Box> boxes;
  const auto values = tensor_to_float_vector(tensor);
  if (values.empty() || (values.size() % static_cast<std::size_t>(neat::kDecodedBoxColumns)) != 0) {
    return boxes;
  }
  const std::size_t count = values.size() / static_cast<std::size_t>(neat::kDecodedBoxColumns);
  boxes.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const float* row = values.data() + i * static_cast<std::size_t>(neat::kDecodedBoxColumns);
    boxes.push_back(neat::Box{row[0], row[1], row[2], row[3], row[4], static_cast<int>(row[5])});
  }
  return boxes;
}

std::vector<neat::Box> decode_boxes_from_sample(const neat::Sample& sample, int width, int height,
                                                int top_k) {
  const auto tensors = neat::tensors_from_sample(sample, true);
  if (tensors.empty()) {
    return {};
  }
  const auto decoded = neat::decode_bbox_tensor(tensors.front(), width, height, top_k, false);
  return decoded.boxes;
}

std::uint8_t blend_u8(std::uint8_t base, std::uint8_t overlay, float alpha) {
  const float v = (1.0f - alpha) * static_cast<float>(base) + alpha * static_cast<float>(overlay);
  return static_cast<std::uint8_t>(std::max(0.0f, std::min(255.0f, v)));
}

void blend_nv12_mask_pixel(neat::Tensor& frame, int width, int height, int x, int y,
                           const Nv12Color& color, float alpha) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }
  auto* y_plane = nv12_y_plane(frame, width, height);
  auto* uv_plane = nv12_uv_plane(frame, width, height);
  const std::size_t y_offset = static_cast<std::size_t>(y) * width + x;
  y_plane[y_offset] = blend_u8(y_plane[y_offset], color.y, alpha);

  const int uv_x = x & ~1;
  const int uv_y = y / 2;
  if (uv_x + 1 < width && uv_y >= 0 && uv_y < height / 2) {
    const std::size_t uv_offset = static_cast<std::size_t>(uv_y) * width + uv_x;
    uv_plane[uv_offset] = blend_u8(uv_plane[uv_offset], color.u, alpha);
    uv_plane[uv_offset + 1] = blend_u8(uv_plane[uv_offset + 1], color.v, alpha);
  }
}

struct MaskRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

MaskRect mask_rect_for_frame_box(int x1, int y1, int x2, int y2, int frame_w, int frame_h,
                                 int mask_w, int mask_h) {
  const int model_w = mask_w * 4;
  const int model_h = mask_h * 4;
  const double scale =
      std::min(static_cast<double>(model_w) / std::max(1, frame_w),
               static_cast<double>(model_h) / std::max(1, frame_h));
  const double pad_x = (static_cast<double>(model_w) - frame_w * scale) * 0.5;
  const double pad_y = (static_cast<double>(model_h) - frame_h * scale) * 0.5;
  const auto to_mask_x = [&](double frame_x) {
    return (frame_x * scale + pad_x) * static_cast<double>(mask_w) / model_w;
  };
  const auto to_mask_y = [&](double frame_y) {
    return (frame_y * scale + pad_y) * static_cast<double>(mask_h) / model_h;
  };

  MaskRect rect;
  rect.x = std::clamp(static_cast<int>(std::floor(to_mask_x(x1))), 0, std::max(0, mask_w - 1));
  rect.y = std::clamp(static_cast<int>(std::floor(to_mask_y(y1))), 0, std::max(0, mask_h - 1));
  const int right =
      std::clamp(static_cast<int>(std::ceil(to_mask_x(x2))), rect.x + 1, mask_w);
  const int bottom =
      std::clamp(static_cast<int>(std::ceil(to_mask_y(y2))), rect.y + 1, mask_h);
  rect.width = right - rect.x;
  rect.height = bottom - rect.y;
  return rect;
}

std::uint8_t projected_mask_value(const std::uint8_t* mask, int mask_w, int mask_h,
                                  const MaskRect& mask_rect, int box_w, int box_h, int box_x,
                                  int box_y) {
  const double mx =
      mask_rect.x + (static_cast<double>(box_x) + 0.5) * mask_rect.width / std::max(1, box_w);
  const double my =
      mask_rect.y + (static_cast<double>(box_y) + 0.5) * mask_rect.height / std::max(1, box_h);
  const int x0 = std::clamp(static_cast<int>(std::floor(mx)), 0, mask_w - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(my)), 0, mask_h - 1);
  const int x1 = std::min(mask_w - 1, x0 + 1);
  const int y1 = std::min(mask_h - 1, y0 + 1);
  const double fx = mx - x0;
  const double fy = my - y0;
  const auto at = [&](int x, int y) -> double {
    return static_cast<double>(mask[static_cast<std::size_t>(y) * mask_w + x]);
  };
  const double top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
  const double bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
  return static_cast<std::uint8_t>(std::clamp(top * (1.0 - fy) + bottom * fy, 0.0, 255.0));
}

SegmentationOverlay draw_segmentation_overlay(neat::Tensor& frame, const neat::Sample& sample,
                                              const Config& cfg, const Nv12Color& color) {
  SegmentationOverlay stats;
  int width = 0;
  int height = 0;
  if (!infer_dims(frame, width, height)) {
    return stats;
  }

  const auto tensors = neat::tensors_from_sample(sample, true);
  if (tensors.empty()) {
    return stats;
  }

  neat::SegmentationDecodeTensors decoded;
  try {
    decoded = neat::decode_segmentation_tensor(tensors.front(), width, height, cfg.top_k, false);
  } catch (const std::exception& e) {
    std::cerr << "[warn] failed to decode segmentation tensor: " << e.what() << "\n";
    return stats;
  }

  const auto boxes = boxes_from_decoded_tensor(decoded.boxes);
  const auto masks = decoded.masks.copy_dense_bytes_tight();
  constexpr int kMaskW = static_cast<int>(neat::kDecodedMaskWidth);
  constexpr int kMaskH = static_cast<int>(neat::kDecodedMaskHeight);
  const std::size_t per_mask = static_cast<std::size_t>(kMaskW) * static_cast<std::size_t>(kMaskH);
  if (masks.empty() || per_mask == 0) {
    return stats;
  }

  stats.instances = static_cast<int>(std::min<std::size_t>(boxes.size(), masks.size() / per_mask));
  for (int i = 0; i < stats.instances; ++i) {
    const auto& box = boxes[static_cast<std::size_t>(i)];
    if (box.score < cfg.score_threshold) {
      continue;
    }
    ++stats.visible_instances;
    const Nv12Color instance_color = class_color(box.class_id);
    draw_nv12_box(frame, width, height, box, instance_color);
    draw_nv12_label(frame, width, height, static_cast<int>(box.x1),
                    std::max(36, static_cast<int>(box.y1) - 18), class_label(box.class_id),
                    instance_color);

    const int x1 = std::max(0, std::min(static_cast<int>(box.x1), width - 1));
    const int y1 = std::max(0, std::min(static_cast<int>(box.y1), height - 1));
    const int x2 = std::max(0, std::min(static_cast<int>(box.x2), width - 1));
    const int y2 = std::max(0, std::min(static_cast<int>(box.y2), height - 1));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const std::uint8_t* mask = masks.data() + static_cast<std::size_t>(i) * per_mask;
    const int box_w = x2 - x1 + 1;
    const int box_h = y2 - y1 + 1;
    const MaskRect mask_rect =
        mask_rect_for_frame_box(x1, y1, x2 + 1, y2 + 1, width, height, kMaskW, kMaskH);
    for (int y = y1; y <= y2; y += 2) {
      for (int x = x1; x <= x2; x += 2) {
        if (projected_mask_value(mask, kMaskW, kMaskH, mask_rect, box_w, box_h, x - x1,
                                 y - y1) <= 127) {
          continue;
        }
        blend_nv12_mask_pixel(frame, width, height, x, y, instance_color, 0.55f);
        blend_nv12_mask_pixel(frame, width, height, x + 1, y, instance_color, 0.55f);
        blend_nv12_mask_pixel(frame, width, height, x, y + 1, instance_color, 0.55f);
        blend_nv12_mask_pixel(frame, width, height, x + 1, y + 1, instance_color, 0.55f);
      }
    }
  }
  return stats;
}

void draw_model_overlay(neat::Tensor& frame, ModelRuntime& runtime, const neat::Sample& model_sample,
                        const Config& cfg) {
  int width = 0;
  int height = 0;
  if (!infer_dims(frame, width, height)) {
    throw std::runtime_error("invalid overlay frame dimensions");
  }

  const Nv12Color white{235, 128, 128};
  const Nv12Color black{16, 128, 128};
  const Nv12Color green{210, 40, 40};
  const Nv12Color blue{90, 240, 110};
  const Nv12Color orange{190, 80, 210};
  const Nv12Color magenta{160, 210, 190};
  const Nv12Color model_color = runtime.spec.name == "yolov8n"       ? green
                                : runtime.spec.name == "yolo26n"     ? orange
                                : runtime.spec.name == "yolov8n-seg" ? blue
                                                                       : magenta;

  fill_nv12_rect(frame, width, height, 0, 0, width, 34, black);
  draw_nv12_text(frame, width, height, 10, 8, runtime.spec.name, model_color, 2);
  draw_nv12_text(frame, width, height, 220, 8, sample_summary(model_sample), white, 2);

  const auto seg = draw_segmentation_overlay(frame, model_sample, cfg, model_color);
  runtime.last_visible_boxes = seg.visible_instances;
  draw_nv12_text(frame, width, height, 520, 8, "MASKS:" + std::to_string(seg.visible_instances),
                 white, 2);

  draw_nv12_line_h(frame, width, height, 0, width, 34, 3, model_color);
}

void print_report(const neat::GraphReport& report) {
  if (!report.error_code.empty()) {
    std::cerr << "error_code: " << report.error_code << "\n";
  }
  if (!report.repro_note.empty()) {
    std::cerr << "repro_note: " << report.repro_note << "\n";
  }
  for (const auto& msg : report.bus) {
    if (msg.type == "ERROR" || msg.type == "WARNING") {
      std::cerr << "bus " << msg.type << " [" << msg.src << "]: " << msg.detail << "\n";
    }
  }
  if (!report.repro_gst_launch.empty()) {
    std::cerr << "repro_gst_launch:\n" << report.repro_gst_launch << "\n";
  }
}

std::string sample_summary(const neat::Sample& sample) {
  const auto tensors = neat::tensors_from_sample(sample, false);
  return std::to_string(tensors.size()) + " tensors";
}


} // namespace

int main() {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  try {
    const Config cfg = read_config();
    const auto specs = make_specs(cfg);

    neat::RunOptions run_options;
    run_options.preset = neat::RunPreset::Realtime;
    run_options.queue_depth = 3;
    run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
    run_options.output_memory = neat::OutputMemory::ZeroCopy;

    std::vector<ModelRuntime> runtimes;
    runtimes.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
      ModelRuntime rt;
      rt.spec = specs[i];
      rt.udp_port = cfg.udp_port_base + static_cast<int>(i);
      rt.model = make_model(cfg, rt.spec);
      rt.model_graph = make_model_pipeline(cfg, *rt.model, rt.spec.name);
      rt.udp_graph = make_udp_pipeline(cfg, rt.udp_port, rt.spec.name);

      if (cfg.print_backend) {
        std::cout << rt.spec.name << " model backend:\n"
                  << rt.model_graph.describe_backend() << "\n";
        std::cout << rt.spec.name << " UDP backend:\n" << rt.udp_graph.describe_backend()
                  << "\n";
      }

      rt.model_run = rt.model_graph.build(run_options);

      neat::RunOptions udp_run_options = run_options;
      udp_run_options.output_memory = neat::OutputMemory::Owned;
      neat::Tensor udp_seed = make_blank_nv12_tensor(cfg.fallback_width, cfg.fallback_height);
      rt.udp_run = rt.udp_graph.build(neat::TensorList{udp_seed}, udp_run_options);
      runtimes.push_back(std::move(rt));
    }

    neat::Graph source_graph = make_source_pipeline(cfg);
    if (cfg.print_backend) {
      std::cout << "Source backend:\n" << source_graph.describe_backend() << "\n";
    }
    neat::Run source_run = source_graph.build(run_options);

    std::cout << "RTSP input: " << cfg.rtsp_url << "\n";
    std::cout << "Loaded models:\n";
    for (const auto& rt : runtimes) {
      std::cout << "  " << rt.spec.name << " -> " << rt.spec.path << " -> udp://"
                << cfg.udp_host << ":" << rt.udp_port << "\n";
    }
    std::cout << "Running. Press Ctrl-C to stop.\n";

    int processed = 0;
    const auto start = Clock::now();
    Clock::time_point steady_start = start;
    while (!g_stop.load() && (cfg.frames == 0 || processed < cfg.frames)) {
      neat::Sample frame_sample;
      neat::PullError frame_error;
      const auto decoder_begin = Clock::now();
      auto status = source_run.pull("frame", 20000, frame_sample, &frame_error);
      const auto decoder_end = Clock::now();
      if (status == neat::PullStatus::Timeout) {
        std::cerr << "[warn] timed out waiting for RTSP frame\n";
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        std::cerr << "RTSP source closed\n";
        break;
      }
      if (status != neat::PullStatus::Ok) {
        throw std::runtime_error("failed to pull frame: " + frame_error.message);
      }

      const auto frame_tensors = neat::tensors_from_sample(frame_sample, true);
      if (frame_tensors.empty()) {
        std::cerr << "[warn] frame sample has no tensors\n";
        continue;
      }

      for (auto& rt : runtimes) {
        rt.current_profile = {};
        rt.current_profile.decoder_ms = ms_since(decoder_begin, decoder_end);
        rt.inference_begin = Clock::now();
        if (!rt.model_run.push("image", neat::TensorList{frame_tensors.front()})) {
          std::cerr << "[warn] failed to push frame to " << rt.spec.name << "\n";
          continue;
        }
      }

      for (auto& rt : runtimes) {
        neat::Sample model_sample;
        neat::PullError pull_error;
        status = rt.model_run.pull("result", 20000, model_sample, &pull_error);
        if (status == neat::PullStatus::Timeout) {
          std::cerr << "[warn] timed out waiting for " << rt.spec.name << "\n";
          continue;
        }
        if (status == neat::PullStatus::Closed) {
          std::cerr << rt.spec.name << " model pipeline closed\n";
          continue;
        }
        if (status != neat::PullStatus::Ok) {
          throw std::runtime_error("failed to pull " + rt.spec.name + ": " + pull_error.message);
        }
        rt.current_profile.inference_ms = ms_since(rt.inference_begin, Clock::now());

        const auto overlay_begin = Clock::now();
        neat::Tensor output_frame = copy_nv12_to_cpu_tensor(frame_tensors.front());
        draw_model_overlay(output_frame, rt, model_sample, cfg);
        const auto overlay_end = Clock::now();
        rt.current_profile.overlay_ms = ms_since(overlay_begin, overlay_end);

        const auto encoder_begin = Clock::now();
        if (!rt.udp_run.push("video", neat::TensorList{output_frame})) {
          std::cerr << "[warn] failed to push " << rt.spec.name << " frame to UDP encoder\n";
        }
        const auto encoder_end = Clock::now();
        rt.current_profile.encoder_ms = ms_since(encoder_begin, encoder_end);
        ++rt.processed;
        rt.profile.add(rt.current_profile);

        if (should_log_frame(processed + 1, cfg.frames)) {
          const auto now = Clock::now();
          const double elapsed = std::chrono::duration<double>(now - start).count();
          const double fps = elapsed > 0.0 ? static_cast<double>(processed + 1) / elapsed : 0.0;
          const double steady_elapsed =
              std::chrono::duration<double>(now - steady_start).count();
          const double steady_fps =
              processed > 0 && steady_elapsed > 0.0
                  ? static_cast<double>(processed) / steady_elapsed
                  : fps;
          const StageTimes avg = rt.profile.average(rt.processed);
          std::cout << "[" << rt.spec.name << "] frame=" << (processed + 1)
                    << " result=" << sample_summary(model_sample)
                    << " sample_frame_id=" << model_sample.frame_id;
          if (rt.last_visible_boxes >= 0) {
            std::cout << " masks=" << rt.last_visible_boxes;
          }
          std::cout << " fps=" << std::fixed << std::setprecision(2) << fps
                    << " steady_fps=" << steady_fps
                    << " ms(decoder=" << rt.current_profile.decoder_ms
                    << ", inference=" << rt.current_profile.inference_ms
                    << ", overlay=" << rt.current_profile.overlay_ms
                    << ", encoder=" << rt.current_profile.encoder_ms
                    << ", total=" << rt.current_profile.total_ms() << ")"
                    << " avg_ms(decoder=" << avg.decoder_ms
                    << ", inference=" << avg.inference_ms
                    << ", overlay=" << avg.overlay_ms
                    << ", encoder=" << avg.encoder_ms
                    << ", total=" << avg.total_ms() << ")\n";
        }
      }

      ++processed;
      if (processed == 1) {
        steady_start = Clock::now();
      }
      if (should_log_frame(processed, cfg.frames)) {
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - start).count();
        const double fps = elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0;
        const double steady_elapsed = std::chrono::duration<double>(now - steady_start).count();
        const double steady_fps =
            processed > 1 && steady_elapsed > 0.0
                ? static_cast<double>(processed - 1) / steady_elapsed
                : fps;
        std::cout << "[probe] source_frames=" << processed << " active_models="
                  << runtimes.size() << " fps=" << std::fixed << std::setprecision(2) << fps
                  << " steady_fps=" << steady_fps << "\n";
      }
    }

    for (auto& rt : runtimes) {
      rt.udp_run.close();
      rt.model_run.close();
    }
    source_run.close();
    return g_stop.load() ? 130 : 0;
  } catch (const neat::NeatError& e) {
    std::cerr << "NEAT error: " << e.what() << "\n";
    print_report(e.report());
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
