#include <neat.h>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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
  int udp_port = 0;
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

int to_int(const std::string& value) {
  return std::stoi(value);
}

float to_float(const std::string& value) {
  return std::stof(value);
}

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
  } else if (key == "udp_port") {
    cfg.udp_port = to_int(value);
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

std::unique_ptr<neat::Model> make_model(const Config& cfg) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.input_max_width = cfg.fallback_width;
  opt.preprocess.input_max_height = cfg.fallback_height;
  opt.preprocess.input_max_depth = 1;
  opt.preprocess.resize.width = cfg.model_width;
  opt.preprocess.resize.height = cfg.model_height;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = neat::BoxDecodeType::YoloV8;
  opt.score_threshold = cfg.score_threshold;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.top_k;
  opt.num_classes = cfg.num_classes;
  return std::make_unique<neat::Model>(cfg.model_path, opt);
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

  neat::Graph app("single_stream_yolo_yolov8n_source");
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

neat::Graph make_model_pipeline(const Config& cfg, const neat::Model& model) {
  neat::Graph app("single_stream_yolo_yolov8n_model");
  app.add(neat::nodes::Input("image", make_nv12_input_options(cfg)));
  app.add(model);
  app.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));
  return app;
}

neat::Graph make_udp_pipeline(const Config& cfg) {
  auto video_options = groups::VideoSenderOptions::H264RtpUdpFromRaw(
      cfg.fallback_width, cfg.fallback_height, cfg.fallback_fps);
  video_options.host = cfg.udp_host;
  video_options.channel = 0;
  video_options.video_port_base = cfg.udp_port;
  video_options.encoder.bitrate_kbps = cfg.bitrate_kbps;

  neat::Graph app("single_stream_yolo_yolov8n_udp");
  app.add(neat::nodes::Input("video", make_nv12_input_options(cfg)));
  app.add(groups::VideoSender(video_options));
  return app;
}

std::vector<neat::Box> decode_boxes(const neat::Sample& sample, const Config& cfg) {
  const auto tensors = neat::tensors_from_sample(sample, true);
  if (tensors.empty()) {
    return {};
  }
  const auto decoded =
      neat::decode_bbox_tensor(tensors.front(), cfg.fallback_width, cfg.fallback_height,
                               cfg.top_k, false);
  return decoded.boxes;
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
  const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U / 2U;
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

void draw_y_line(std::uint8_t* y, int width, int height, int x1, int x2, int yv, std::uint8_t value,
                 int thickness) {
  if (yv < 0 || yv >= height) {
    return;
  }
  x1 = std::max(0, std::min(width - 1, x1));
  x2 = std::max(0, std::min(width - 1, x2));
  if (x2 < x1) {
    std::swap(x1, x2);
  }
  for (int t = -thickness / 2; t <= thickness / 2; ++t) {
    const int yy = yv + t;
    if (yy < 0 || yy >= height) {
      continue;
    }
    std::memset(y + static_cast<std::size_t>(yy) * width + x1, value,
                static_cast<std::size_t>(x2 - x1 + 1));
  }
}

void draw_y_col(std::uint8_t* y, int width, int height, int xv, int y1, int y2, std::uint8_t value,
                int thickness) {
  if (xv < 0 || xv >= width) {
    return;
  }
  y1 = std::max(0, std::min(height - 1, y1));
  y2 = std::max(0, std::min(height - 1, y2));
  if (y2 < y1) {
    std::swap(y1, y2);
  }
  for (int yy = y1; yy <= y2; ++yy) {
    for (int t = -thickness / 2; t <= thickness / 2; ++t) {
      const int xx = xv + t;
      if (xx >= 0 && xx < width) {
        y[static_cast<std::size_t>(yy) * width + xx] = value;
      }
    }
  }
}

void draw_uv_rect(std::uint8_t* uv, int width, int height, int x1, int y1, int x2, int y2,
                  std::uint8_t u_value, std::uint8_t v_value) {
  const int uv_width = width / 2;
  const int uv_height = height / 2;
  x1 = std::max(0, std::min(uv_width - 1, x1 / 2));
  x2 = std::max(0, std::min(uv_width - 1, x2 / 2));
  y1 = std::max(0, std::min(uv_height - 1, y1 / 2));
  y2 = std::max(0, std::min(uv_height - 1, y2 / 2));
  if (x2 < x1) {
    std::swap(x1, x2);
  }
  if (y2 < y1) {
    std::swap(y1, y2);
  }
  const auto set_uv = [&](int x, int y) {
    const std::size_t offset = (static_cast<std::size_t>(y) * uv_width + x) * 2U;
    uv[offset] = u_value;
    uv[offset + 1U] = v_value;
  };
  for (int x = x1; x <= x2; ++x) {
    set_uv(x, y1);
    set_uv(x, y2);
  }
  for (int y = y1; y <= y2; ++y) {
    set_uv(x1, y);
    set_uv(x2, y);
  }
}

void fill_nv12_rect(std::uint8_t* y, std::uint8_t* uv, int width, int height, int x1, int y1,
                    int x2, int y2, std::uint8_t y_value, std::uint8_t u_value,
                    std::uint8_t v_value) {
  x1 = std::max(0, std::min(width, x1));
  x2 = std::max(0, std::min(width, x2));
  y1 = std::max(0, std::min(height, y1));
  y2 = std::max(0, std::min(height, y2));
  if (x2 <= x1 || y2 <= y1) {
    return;
  }

  for (int row = y1; row < y2; ++row) {
    std::memset(y + static_cast<std::size_t>(row) * width + x1, y_value,
                static_cast<std::size_t>(x2 - x1));
  }

  const int uv_y1 = y1 / 2;
  const int uv_y2 = (y2 + 1) / 2;
  const int uv_x1 = x1 & ~1;
  const int uv_x2 = (x2 + 1) & ~1;
  for (int row = std::max(0, uv_y1); row < std::min(height / 2, uv_y2); ++row) {
    auto* uv_row = uv + static_cast<std::size_t>(row) * width;
    for (int col = std::max(0, uv_x1); col + 1 < std::min(width, uv_x2); col += 2) {
      uv_row[col] = u_value;
      uv_row[col + 1] = v_value;
    }
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
  case '-': return {"00000", "00000", "00000", "11111", "00000", "00000", "00000"};
  case '_': return {"00000", "00000", "00000", "00000", "00000", "00000", "11111"};
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

void draw_nv12_text(std::uint8_t* y, std::uint8_t* uv, int width, int height, int x, int y0,
                    const std::string& text, std::uint8_t y_value, std::uint8_t u_value,
                    std::uint8_t v_value, int scale = 2) {
  int cursor_x = x;
  for (char raw : text) {
    const auto glyph = glyph_for(overlay_char(raw));
    for (int row = 0; row < static_cast<int>(glyph.size()); ++row) {
      for (int col = 0; col < static_cast<int>(glyph[row].size()); ++col) {
        if (glyph[row][col] == '1') {
          fill_nv12_rect(y, uv, width, height, cursor_x + col * scale, y0 + row * scale,
                         cursor_x + (col + 1) * scale, y0 + (row + 1) * scale, y_value,
                         u_value, v_value);
        }
      }
    }
    cursor_x += 6 * scale;
    if (cursor_x >= width - 6 * scale) {
      break;
    }
  }
}

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
  return "CLASS";
}

void draw_box_label(std::uint8_t* y, std::uint8_t* uv, int width, int height, int x, int y0,
                    const std::string& label) {
  constexpr int kScale = 2;
  constexpr int kMaxChars = 22;
  const std::string text = label.substr(0, kMaxChars);
  const int bx = std::max(0, std::min(x, width - 1));
  const int by = std::max(0, std::min(y0, height - 1));
  draw_nv12_text(y, uv, width, height, bx, by, text, 235, 128, 128, kScale);
}

void draw_boxes_on_nv12(neat::Tensor& frame, const std::vector<neat::Box>& boxes,
                        float min_score) {
  int width = 0;
  int height = 0;
  if (!infer_dims(frame, width, height) || !frame.storage || !frame.storage->data) {
    return;
  }
  auto* base = static_cast<std::uint8_t*>(frame.storage->data);
  auto* y = base;
  auto* uv = base + static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

  constexpr std::uint8_t kY = 76;
  constexpr std::uint8_t kU = 84;
  constexpr std::uint8_t kV = 255;
  constexpr int kThickness = 3;

  for (const auto& box : boxes) {
    if (box.score < min_score) {
      continue;
    }
    const int x1 = std::max(0, static_cast<int>(box.x1));
    const int y1 = std::max(0, static_cast<int>(box.y1));
    const int x2 = std::min(width - 1, static_cast<int>(box.x2));
    const int y2 = std::min(height - 1, static_cast<int>(box.y2));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    fill_nv12_rect(y, uv, width, height, x1, y1, x2 + 1, y1 + kThickness, kY, kU, kV);
    fill_nv12_rect(y, uv, width, height, x1, y2 - kThickness + 1, x2 + 1, y2 + 1, kY,
                   kU, kV);
    fill_nv12_rect(y, uv, width, height, x1, y1, x1 + kThickness, y2 + 1, kY, kU, kV);
    fill_nv12_rect(y, uv, width, height, x2 - kThickness + 1, y1, x2 + 1, y2 + 1, kY,
                   kU, kV);
    const int label_y = y1 >= 22 ? y1 - 20 : std::min(height - 18, y1 + 6);
    draw_box_label(y, uv, width, height, x1, label_y, class_label(box.class_id));
  }
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

} // namespace

int main() {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  try {
    const Config cfg = read_config();
    auto model = make_model(cfg);
    neat::Graph source_graph = make_source_pipeline(cfg);
    neat::Graph model_graph = make_model_pipeline(cfg, *model);
    neat::Graph udp_graph = make_udp_pipeline(cfg);

    if (cfg.print_backend) {
      std::cout << "Source backend:\n" << source_graph.describe_backend() << "\n";
      std::cout << "Model backend:\n" << model_graph.describe_backend() << "\n";
      std::cout << "UDP backend:\n" << udp_graph.describe_backend() << "\n";
    }

    neat::RunOptions run_options;
    run_options.preset = neat::RunPreset::Realtime;
    run_options.queue_depth = 3;
    run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
    run_options.output_memory = neat::OutputMemory::ZeroCopy;

    neat::Run source_run = source_graph.build(run_options);
    neat::Run model_run = model_graph.build(run_options);

    neat::RunOptions udp_run_options = run_options;
    udp_run_options.output_memory = neat::OutputMemory::Owned;
    neat::Tensor udp_seed = make_blank_nv12_tensor(cfg.fallback_width, cfg.fallback_height);
    neat::Run udp_run = udp_graph.build(neat::TensorList{udp_seed}, udp_run_options);

    std::cout << "RTSP input: " << cfg.rtsp_url << "\n";
    std::cout << "Model:      " << cfg.model_path << "\n";
    std::cout << "UDP output: udp://" << cfg.udp_host << ":" << cfg.udp_port
              << " H264/RTP payload=96 " << cfg.fallback_width << "x" << cfg.fallback_height
              << "@" << cfg.fallback_fps << "\n";
    std::cout << "Receiver example:\n"
              << "  gst-launch-1.0 -v udpsrc port=" << cfg.udp_port
              << " caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" "
                 "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink\n";
    std::cout << "Running. Press Ctrl-C to stop.\n";

    int processed = 0;
    ProfileTotals profile;
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

      const auto inference_begin = Clock::now();
      if (!model_run.push("image", neat::TensorList{frame_tensors.front()})) {
        std::cerr << "[warn] failed to push frame to model\n";
        continue;
      }

      neat::Sample detection_sample;
      neat::PullError pull_error;
      status = model_run.pull("detections", 20000, detection_sample, &pull_error);
      if (status == neat::PullStatus::Timeout) {
        std::cerr << "[warn] timed out waiting for detections\n";
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        std::cerr << "Pipeline closed\n";
        break;
      }
      if (status != neat::PullStatus::Ok) {
        throw std::runtime_error("failed to pull detections: " + pull_error.message);
      }
      const auto inference_end = Clock::now();

      const auto overlay_begin = Clock::now();
      auto boxes = decode_boxes(detection_sample, cfg);
      neat::Tensor annotated = copy_nv12_to_cpu_tensor(frame_tensors.front());
      draw_boxes_on_nv12(annotated, boxes, cfg.score_threshold);
      const auto overlay_end = Clock::now();

      const auto encoder_begin = Clock::now();
      if (!udp_run.push("video", neat::TensorList{annotated})) {
        std::cerr << "[warn] failed to push annotated frame to UDP encoder\n";
      }
      const auto encoder_end = Clock::now();

      ++processed;
      const StageTimes frame_profile{ms_since(decoder_begin, decoder_end),
                                     ms_since(inference_begin, inference_end),
                                     ms_since(overlay_begin, overlay_end),
                                     ms_since(encoder_begin, encoder_end)};
      profile.add(frame_profile);
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
        const StageTimes avg = profile.average(processed);
        std::cout << "[visualize] frame=" << processed << " boxes=" << boxes.size()
                  << " sample_frame_id=" << detection_sample.frame_id << " fps=" << std::fixed
                  << std::setprecision(2) << fps << " steady_fps=" << steady_fps
                  << " ms(decoder=" << frame_profile.decoder_ms
                  << ", inference=" << frame_profile.inference_ms
                  << ", overlay=" << frame_profile.overlay_ms
                  << ", encoder=" << frame_profile.encoder_ms
                  << ", total=" << frame_profile.total_ms() << ")"
                  << " avg_ms(decoder=" << avg.decoder_ms
                  << ", inference=" << avg.inference_ms
                  << ", overlay=" << avg.overlay_ms
                  << ", encoder=" << avg.encoder_ms
                  << ", total=" << avg.total_ms() << ")\n";
      }
    }
    udp_run.close();
    model_run.close();
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
