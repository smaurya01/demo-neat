#include <neat.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr const char* kDefaultConfigPath =
    "/workspace/demo-neat/single-stream-yolo-yolov8n/config/default.conf";

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

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --config <path>           Key=value config file (default: " << kDefaultConfigPath
      << ")\n"
      << "  --rtsp <url>              Input RTSP URL\n"
      << "  --model <path>            Compiled model archive (.tar.gz)\n"
      << "  --width <px>              Fallback stream width if RTSP caps are incomplete\n"
      << "  --height <px>             Fallback stream height if RTSP caps are incomplete\n"
      << "  --fps <fps>               Fallback stream FPS\n"
      << "  --model-width <px>        Model input width\n"
      << "  --model-height <px>       Model input height\n"
      << "  --score <0..1>            Detection score threshold\n"
      << "  --nms <0..1>              NMS IoU threshold\n"
      << "  --top-k <n>               Max detections per frame\n"
      << "  --classes <n>             Number of detection classes\n"
      << "  --frames <n>              Stop after n detection outputs (0 = run until interrupted)\n"
      << "  --udp-host <host>         UDP receiver host/IP\n"
      << "  --udp-port <port>         UDP receiver port\n"
      << "  --bitrate <kbps>          H.264 encoder bitrate\n"
      << "  --rtsp-udp                Use UDP transport for RTSP\n"
      << "  --print-backend           Print generated backend pipeline\n"
      << "  -h, --help                Show this help\n";
}

int parse_int_arg(const char* name, const char* value) {
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string(name) + " requires an integer");
  }
}

float parse_float_arg(const char* name, const char* value) {
  try {
    return std::stof(value);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string(name) + " requires a number");
  }
}

std::string trim(std::string value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

bool parse_bool_value(const std::string& value) {
  if (value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" || value == "no" || value == "off") {
    return false;
  }
  throw std::runtime_error("boolean value must be true/false, yes/no, on/off, or 1/0");
}

void apply_config_value(Config& cfg, const std::string& key, const std::string& value) {
  if (key == "rtsp_url") {
    cfg.rtsp_url = value;
  } else if (key == "model_path") {
    cfg.model_path = value;
  } else if (key == "fallback_width") {
    cfg.fallback_width = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "fallback_height") {
    cfg.fallback_height = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "fallback_fps") {
    cfg.fallback_fps = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "model_width") {
    cfg.model_width = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "model_height") {
    cfg.model_height = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "latency_ms") {
    cfg.latency_ms = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "score_threshold") {
    cfg.score_threshold = parse_float_arg(key.c_str(), value.c_str());
  } else if (key == "nms_iou") {
    cfg.nms_iou = parse_float_arg(key.c_str(), value.c_str());
  } else if (key == "top_k") {
    cfg.top_k = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "num_classes") {
    cfg.num_classes = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "frames") {
    cfg.frames = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "udp_host") {
    cfg.udp_host = value;
  } else if (key == "udp_port") {
    cfg.udp_port = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "bitrate_kbps") {
    cfg.bitrate_kbps = parse_int_arg(key.c_str(), value.c_str());
  } else if (key == "rtsp_transport") {
    if (value == "tcp") {
      cfg.tcp = true;
    } else if (value == "udp") {
      cfg.tcp = false;
    } else {
      throw std::runtime_error("rtsp_transport must be tcp or udp");
    }
  } else if (key == "print_backend") {
    cfg.print_backend = parse_bool_value(value);
  } else {
    throw std::runtime_error("unknown config key: " + key);
  }
}

void load_config_file(Config& cfg, const std::string& path, bool required) {
  std::ifstream file(path);
  if (!file) {
    if (required) {
      throw std::runtime_error("config file not found: " + path);
    }
    return;
  }

  std::string line;
  int line_no = 0;
  while (std::getline(file, line)) {
    ++line_no;
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line.erase(comment);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    const auto equal = line.find('=');
    if (equal == std::string::npos) {
      throw std::runtime_error(path + ":" + std::to_string(line_no) + ": expected key=value");
    }
    const std::string key = trim(line.substr(0, equal));
    const std::string value = trim(line.substr(equal + 1));
    if (key.empty()) {
      throw std::runtime_error(path + ":" + std::to_string(line_no) + ": empty key");
    }
    apply_config_value(cfg, key, value);
  }
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  std::string config_path = kDefaultConfigPath;
  bool explicit_config = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a value");
      }
      config_path = argv[++i];
      explicit_config = true;
    }
  }
  load_config_file(cfg, config_path, explicit_config);

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[++i];
    };

    if (arg == "--config") {
      (void)need_value("--config");
    } else if (arg == "--rtsp") {
      cfg.rtsp_url = need_value("--rtsp");
    } else if (arg == "--model") {
      cfg.model_path = need_value("--model");
    } else if (arg == "--width") {
      cfg.fallback_width = parse_int_arg("--width", need_value("--width"));
    } else if (arg == "--height") {
      cfg.fallback_height = parse_int_arg("--height", need_value("--height"));
    } else if (arg == "--fps") {
      cfg.fallback_fps = parse_int_arg("--fps", need_value("--fps"));
    } else if (arg == "--model-width") {
      cfg.model_width = parse_int_arg("--model-width", need_value("--model-width"));
    } else if (arg == "--model-height") {
      cfg.model_height = parse_int_arg("--model-height", need_value("--model-height"));
    } else if (arg == "--score") {
      cfg.score_threshold = parse_float_arg("--score", need_value("--score"));
    } else if (arg == "--nms") {
      cfg.nms_iou = parse_float_arg("--nms", need_value("--nms"));
    } else if (arg == "--top-k") {
      cfg.top_k = parse_int_arg("--top-k", need_value("--top-k"));
    } else if (arg == "--classes") {
      cfg.num_classes = parse_int_arg("--classes", need_value("--classes"));
    } else if (arg == "--frames") {
      cfg.frames = parse_int_arg("--frames", need_value("--frames"));
    } else if (arg == "--udp-host") {
      cfg.udp_host = need_value("--udp-host");
    } else if (arg == "--udp-port") {
      cfg.udp_port = parse_int_arg("--udp-port", need_value("--udp-port"));
    } else if (arg == "--bitrate") {
      cfg.bitrate_kbps = parse_int_arg("--bitrate", need_value("--bitrate"));
    } else if (arg == "--rtsp-udp") {
      cfg.tcp = false;
    } else if (arg == "--print-backend") {
      cfg.print_backend = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (cfg.rtsp_url.empty()) {
    throw std::runtime_error("RTSP URL must not be empty");
  }
  if (cfg.model_path.empty()) {
    throw std::runtime_error("model path must not be empty");
  }
  if (!std::filesystem::exists(cfg.model_path)) {
    throw std::runtime_error("model file not found: " + cfg.model_path);
  }
  if (cfg.fallback_width <= 0 || cfg.fallback_height <= 0 || cfg.fallback_fps <= 0) {
    throw std::runtime_error("fallback width/height/fps must be positive");
  }
  if (cfg.model_width <= 0 || cfg.model_height <= 0) {
    throw std::runtime_error("model width/height must be positive");
  }
  if (cfg.score_threshold < 0.0f || cfg.score_threshold > 1.0f || cfg.nms_iou < 0.0f ||
      cfg.nms_iou > 1.0f) {
    throw std::runtime_error("score and nms must be in 0..1");
  }
  if (cfg.top_k <= 0 || cfg.num_classes <= 0) {
    throw std::runtime_error("top-k and classes must be positive");
  }
  if (cfg.frames < 0) {
    throw std::runtime_error("frames must be >= 0");
  }
  if (cfg.udp_host.empty()) {
    throw std::runtime_error("UDP host must not be empty");
  }
  if (cfg.udp_port <= 0 || cfg.udp_port > 65535) {
    throw std::runtime_error("UDP port must be in 1..65535");
  }
  if (cfg.bitrate_kbps <= 0) {
    throw std::runtime_error("bitrate must be positive");
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

  constexpr std::uint8_t kY = 210;
  constexpr std::uint8_t kU = 44;
  constexpr std::uint8_t kV = 21;
  constexpr int kThickness = 4;

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
    draw_y_line(y, width, height, x1, x2, y1, kY, kThickness);
    draw_y_line(y, width, height, x1, x2, y2, kY, kThickness);
    draw_y_col(y, width, height, x1, y1, y2, kY, kThickness);
    draw_y_col(y, width, height, x2, y1, y2, kY, kThickness);
    draw_uv_rect(uv, width, height, x1, y1, x2, y2, kU, kV);
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

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  try {
    const Config cfg = parse_args(argc, argv);
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
    const auto start = std::chrono::steady_clock::now();
    while (!g_stop.load() && (cfg.frames == 0 || processed < cfg.frames)) {
      neat::Sample frame_sample;
      neat::PullError frame_error;
      auto status = source_run.pull("frame", 20000, frame_sample, &frame_error);
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

      auto boxes = decode_boxes(detection_sample, cfg);
      neat::Tensor annotated = copy_nv12_to_cpu_tensor(frame_tensors.front());
      draw_boxes_on_nv12(annotated, boxes, cfg.score_threshold);
      if (!udp_run.push("video", neat::TensorList{annotated})) {
        std::cerr << "[warn] failed to push annotated frame to UDP encoder\n";
      }

      ++processed;
      if (processed == 1 || (processed % 30) == 0 || cfg.frames > 0) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start).count();
        const double fps = elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0;
        std::cout << "[visualize] frame=" << processed << " boxes=" << boxes.size()
                  << " sample_frame_id=" << detection_sample.frame_id << " fps=" << std::fixed
                  << std::setprecision(2) << fps << "\n";
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
