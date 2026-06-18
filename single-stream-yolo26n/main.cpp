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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr const char* kDefaultConfigPath =
    "/workspace/demo-neat/single-stream-yolo26n/config/default.conf";

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

struct Config {
  std::string rtsp_url;
  std::string models_dir;
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
  bool allow_missing = false;
  bool load_only = false;
  std::string only;
};

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

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --config <path>           Key=value config file (default: " << kDefaultConfigPath
      << ")\n"
      << "  --rtsp <url>              Input RTSP URL\n"
      << "  --models-dir <path>       Directory containing model archives\n"
      << "  --only <name>             Run one model: yolov8n, yolov8n-seg, yolo26n, open_pose\n"
      << "  --allow-missing           Skip model archives that are not present\n"
      << "  --load-only               Load/build selected model graphs, then exit before RTSP\n"
      << "  --width <px>              Fallback stream width if RTSP caps are incomplete\n"
      << "  --height <px>             Fallback stream height if RTSP caps are incomplete\n"
      << "  --fps <fps>               Fallback stream FPS\n"
      << "  --model-width <px>        Model input width\n"
      << "  --model-height <px>       Model input height\n"
      << "  --score <0..1>            Detection score threshold for boxdecode models\n"
      << "  --nms <0..1>              NMS IoU threshold for boxdecode models\n"
      << "  --top-k <n>               Max detections for boxdecode models\n"
      << "  --classes <n>             Number of detection classes\n"
      << "  --frames <n>              Stop after n source frames (0 = run until interrupted)\n"
      << "  --udp-host <host>         UDP receiver host/IP\n"
      << "  --udp-port-base <port>    First UDP receiver port\n"
      << "  --bitrate <kbps>          H.264 encoder bitrate per output stream\n"
      << "  --rtsp-udp                Use UDP transport for RTSP\n"
      << "  --print-backend           Print generated backend pipelines\n"
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
  } else if (key == "models_dir") {
    cfg.models_dir = value;
  } else if (key == "only") {
    cfg.only = value;
  } else if (key == "allow_missing") {
    cfg.allow_missing = parse_bool_value(value);
  } else if (key == "load_only") {
    cfg.load_only = parse_bool_value(value);
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
  } else if (key == "udp_port_base") {
    cfg.udp_port_base = parse_int_arg(key.c_str(), value.c_str());
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
    } else if (arg == "--models-dir") {
      cfg.models_dir = need_value("--models-dir");
    } else if (arg == "--only") {
      cfg.only = need_value("--only");
    } else if (arg == "--allow-missing") {
      cfg.allow_missing = true;
    } else if (arg == "--load-only") {
      cfg.load_only = true;
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
    } else if (arg == "--udp-port-base") {
      cfg.udp_port_base = parse_int_arg("--udp-port-base", need_value("--udp-port-base"));
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
  if (cfg.models_dir.empty()) {
    throw std::runtime_error("models directory must not be empty");
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
  if (cfg.udp_port_base <= 0 || cfg.udp_port_base > 65532) {
    throw std::runtime_error("UDP port base must allow up to four output ports");
  }
  if (cfg.bitrate_kbps <= 0) {
    throw std::runtime_error("bitrate must be positive");
  }
  return cfg;
}

std::string join_path(const std::string& dir, const std::string& file) {
  return (std::filesystem::path(dir) / file).string();
}

std::vector<ModelSpec> make_specs(const Config& cfg) {
  std::vector<ModelSpec> specs = {
      {"yolov8n", join_path(cfg.models_dir, "yolo_v8n_mpk.tar.gz"), neat::BoxDecodeType::YoloV8,
       cfg.model_width, cfg.model_height, true},
      {"yolov8n-seg", join_path(cfg.models_dir, "yolo_v8n_seg_mpk.tar.gz"),
       neat::BoxDecodeType::YoloV8Seg, cfg.model_width, cfg.model_height, true},
      {"yolo26n", join_path(cfg.models_dir, "yolo26n-det-bf16-mla_tess-b1.tar.gz"),
       neat::BoxDecodeType::YoloV26, cfg.model_width, cfg.model_height, true},
      {"open_pose", join_path(cfg.models_dir, "open_pose_mpk.tar.gz"), std::nullopt, 480, 480,
       false},
  };

  std::vector<ModelSpec> selected;
  for (const auto& spec : specs) {
    if (!cfg.only.empty() && spec.name != cfg.only) {
      continue;
    }
    if (!std::filesystem::exists(spec.path)) {
      if (cfg.allow_missing) {
        std::cerr << "[warn] skipping missing model " << spec.name << ": " << spec.path << "\n";
        continue;
      }
      throw std::runtime_error("model file not found for " + spec.name + ": " + spec.path);
    }
    selected.push_back(spec);
  }
  if (selected.empty()) {
    throw std::runtime_error("no models selected");
  }
  return selected;
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

struct PoseKeypoint {
  float x = 0.0f;
  float y = 0.0f;
  float score = 0.0f;
  int id = -1;
  int type = -1;
};

struct PosePerson {
  std::array<int, 18> keypoint_ids{};
  float score = 0.0f;
  int count = 0;

  PosePerson() {
    keypoint_ids.fill(-1);
  }
};

struct PoseOverlay {
  int persons = 0;
  int keypoints = 0;
};

struct OpenPoseTensorData {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<float> values;
};

bool openpose_tensor_geometry(const neat::Tensor& tensor, const std::vector<float>& values,
                              int& heat_w, int& heat_h, int& channels) {
  if (tensor.shape.size() >= 5) {
    heat_h = static_cast<int>(tensor.shape[tensor.shape.size() - 3]);
    heat_w = static_cast<int>(tensor.shape[tensor.shape.size() - 2]);
    channels = static_cast<int>(tensor.shape.back());
  } else if (tensor.shape.size() >= 3) {
    heat_h = static_cast<int>(tensor.shape[tensor.shape.size() - 3]);
    heat_w = static_cast<int>(tensor.shape[tensor.shape.size() - 2]);
    channels = static_cast<int>(tensor.shape.back());
  }
  return heat_w > 0 && heat_h > 0 && channels >= 18 &&
         values.size() >= static_cast<std::size_t>(heat_w) * heat_h * channels;
}

float tensor_channel_value(const OpenPoseTensorData& tensor, int x, int y, int channel) {
  x = std::clamp(x, 0, tensor.width - 1);
  y = std::clamp(y, 0, tensor.height - 1);
  channel = std::clamp(channel, 0, tensor.channels - 1);
  const std::size_t offset =
      (static_cast<std::size_t>(y) * tensor.width + x) * tensor.channels + channel;
  return offset < tensor.values.size() ? tensor.values[offset] : 0.0f;
}

std::pair<int, int> model_to_frame(float model_x, float model_y, int frame_w, int frame_h,
                                   int model_w, int model_h) {
  const double scale =
      std::min(static_cast<double>(model_w) / std::max(1, frame_w),
               static_cast<double>(model_h) / std::max(1, frame_h));
  const double scaled_w = frame_w * scale;
  const double scaled_h = frame_h * scale;
  const double pad_x = (model_w - scaled_w) * 0.5;
  const double pad_y = (model_h - scaled_h) * 0.5;
  const int frame_x =
      static_cast<int>(std::lround((model_x - pad_x) / std::max(1e-9, scale)));
  const int frame_y =
      static_cast<int>(std::lround((model_y - pad_y) / std::max(1e-9, scale)));
  return {std::clamp(frame_x, 0, frame_w - 1), std::clamp(frame_y, 0, frame_h - 1)};
}

std::vector<std::vector<PoseKeypoint>> find_openpose_peaks(const OpenPoseTensorData& heatmap) {
  constexpr float kPeakThreshold = 0.15f;
  constexpr float kSuppressRadius = 6.0f;
  constexpr int kMaxPeaksPerType = 16;
  std::vector<std::vector<PoseKeypoint>> by_type(18);
  int next_id = 0;

  for (int type = 0; type < 18; ++type) {
    std::vector<PoseKeypoint> candidates;
    for (int y = 1; y + 1 < heatmap.height; ++y) {
      for (int x = 1; x + 1 < heatmap.width; ++x) {
        const float score = tensor_channel_value(heatmap, x, y, type);
        if (score < kPeakThreshold || score <= tensor_channel_value(heatmap, x - 1, y, type) ||
            score <= tensor_channel_value(heatmap, x + 1, y, type) ||
            score <= tensor_channel_value(heatmap, x, y - 1, type) ||
            score <= tensor_channel_value(heatmap, x, y + 1, type)) {
          continue;
        }
        candidates.push_back(PoseKeypoint{static_cast<float>(x), static_cast<float>(y), score, -1,
                                          type});
      }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const PoseKeypoint& a, const PoseKeypoint& b) { return a.score > b.score; });
    for (const auto& candidate : candidates) {
      bool suppressed = false;
      for (const auto& kept : by_type[static_cast<std::size_t>(type)]) {
        const float dx = candidate.x - kept.x;
        const float dy = candidate.y - kept.y;
        if (std::sqrt(dx * dx + dy * dy) < kSuppressRadius) {
          suppressed = true;
          break;
        }
      }
      if (suppressed) {
        continue;
      }
      PoseKeypoint kept = candidate;
      kept.id = next_id++;
      by_type[static_cast<std::size_t>(type)].push_back(kept);
      if (static_cast<int>(by_type[static_cast<std::size_t>(type)].size()) >= kMaxPeaksPerType) {
        break;
      }
    }
  }
  return by_type;
}

struct PoseConnection {
  int a_id = -1;
  int b_id = -1;
  float score = 0.0f;
  int a_local = -1;
  int b_local = -1;
};

std::optional<float> paf_connection_score(const OpenPoseTensorData& paf, const PoseKeypoint& a,
                                          const PoseKeypoint& b, int paf_x_channel,
                                          int paf_y_channel) {
  constexpr int kSamples = 10;
  constexpr float kMinPafScore = 0.08f;
  constexpr float kMinSuccessRatio = 0.80f;
  const float dx = b.x - a.x;
  const float dy = b.y - a.y;
  const float norm = std::sqrt(dx * dx + dy * dy);
  if (norm < 1.0f) {
    return std::nullopt;
  }
  const float ux = dx / norm;
  const float uy = dy / norm;
  float score_sum = 0.0f;
  int valid = 0;
  for (int i = 0; i < kSamples; ++i) {
    const float t = kSamples == 1 ? 0.0f : static_cast<float>(i) / (kSamples - 1);
    const int x = static_cast<int>(std::lround(a.x + dx * t));
    const int y = static_cast<int>(std::lround(a.y + dy * t));
    const float px = tensor_channel_value(paf, x, y, paf_x_channel);
    const float py = tensor_channel_value(paf, x, y, paf_y_channel);
    const float dot = px * ux + py * uy;
    if (dot > kMinPafScore) {
      score_sum += dot;
      ++valid;
    }
  }
  const float success_ratio = static_cast<float>(valid) / kSamples;
  if (success_ratio < kMinSuccessRatio || valid == 0) {
    return std::nullopt;
  }
  return score_sum / valid;
}

std::vector<PoseConnection>
match_limb_connections(const std::vector<PoseKeypoint>& a_candidates,
                       const std::vector<PoseKeypoint>& b_candidates,
                       const OpenPoseTensorData& paf, int paf_x_channel, int paf_y_channel) {
  std::vector<PoseConnection> candidates;
  for (std::size_t ai = 0; ai < a_candidates.size(); ++ai) {
    for (std::size_t bi = 0; bi < b_candidates.size(); ++bi) {
      const auto score =
          paf_connection_score(paf, a_candidates[ai], b_candidates[bi], paf_x_channel,
                               paf_y_channel);
      if (score.has_value()) {
        candidates.push_back(PoseConnection{a_candidates[ai].id, b_candidates[bi].id, *score,
                                            static_cast<int>(ai), static_cast<int>(bi)});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const PoseConnection& a, const PoseConnection& b) { return a.score > b.score; });

  std::vector<PoseConnection> out;
  std::vector<bool> used_a(a_candidates.size(), false);
  std::vector<bool> used_b(b_candidates.size(), false);
  for (const auto& candidate : candidates) {
    if (used_a[static_cast<std::size_t>(candidate.a_local)] ||
        used_b[static_cast<std::size_t>(candidate.b_local)]) {
      continue;
    }
    used_a[static_cast<std::size_t>(candidate.a_local)] = true;
    used_b[static_cast<std::size_t>(candidate.b_local)] = true;
    out.push_back(candidate);
  }
  return out;
}

void add_pose_connection(PosePerson& person, int ka, int kb, const PoseConnection& connection,
                         const std::vector<PoseKeypoint>& all_keypoints) {
  if (person.keypoint_ids[static_cast<std::size_t>(ka)] < 0) {
    person.keypoint_ids[static_cast<std::size_t>(ka)] = connection.a_id;
    person.count += 1;
    person.score += all_keypoints[static_cast<std::size_t>(connection.a_id)].score;
  }
  if (person.keypoint_ids[static_cast<std::size_t>(kb)] < 0) {
    person.keypoint_ids[static_cast<std::size_t>(kb)] = connection.b_id;
    person.count += 1;
    person.score += all_keypoints[static_cast<std::size_t>(connection.b_id)].score;
  }
  person.score += connection.score;
}

void merge_pose_people(PosePerson& target, const PosePerson& source) {
  for (std::size_t i = 0; i < target.keypoint_ids.size(); ++i) {
    if (target.keypoint_ids[i] < 0 && source.keypoint_ids[i] >= 0) {
      target.keypoint_ids[i] = source.keypoint_ids[i];
    }
  }
  target.score += source.score;
  target.count = 0;
  for (int id : target.keypoint_ids) {
    if (id >= 0) {
      ++target.count;
    }
  }
}

std::vector<PosePerson>
assemble_openpose_people(const std::vector<std::vector<PoseKeypoint>>& by_type,
                         const std::vector<PoseKeypoint>& all_keypoints,
                         const OpenPoseTensorData& paf) {
  static constexpr std::array<std::pair<int, int>, 19> kBodyParts = {
      std::pair{1, 2},  {1, 5},  {2, 3},   {3, 4},   {5, 6},   {6, 7},  {1, 8},
      {8, 9},          {9, 10}, {1, 11},  {11, 12}, {12, 13}, {1, 0},  {0, 14},
      {14, 16},        {0, 15}, {15, 17}, {2, 16},  {5, 17},
  };
  static constexpr std::array<std::pair<int, int>, 19> kPafParts = {
      std::pair{12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1},
      {2, 3},           {4, 5},   {6, 7},   {8, 9},   {10, 11}, {28, 29}, {30, 31},
      {34, 35},         {32, 33}, {36, 37}, {18, 19}, {26, 27},
  };

  std::vector<PosePerson> people;
  for (std::size_t part_id = 0; part_id < kBodyParts.size(); ++part_id) {
    const auto [ka, kb] = kBodyParts[part_id];
    const auto& a_candidates = by_type[static_cast<std::size_t>(ka)];
    const auto& b_candidates = by_type[static_cast<std::size_t>(kb)];
    if (a_candidates.empty() || b_candidates.empty()) {
      continue;
    }
    const auto [paf_x, paf_y] = kPafParts[part_id];
    const auto connections = match_limb_connections(a_candidates, b_candidates, paf, paf_x, paf_y);
    if (connections.empty()) {
      continue;
    }

    if (people.empty()) {
      for (const auto& connection : connections) {
        PosePerson person;
        person.keypoint_ids[static_cast<std::size_t>(ka)] = connection.a_id;
        person.keypoint_ids[static_cast<std::size_t>(kb)] = connection.b_id;
        person.count = 2;
        person.score = all_keypoints[static_cast<std::size_t>(connection.a_id)].score +
                       all_keypoints[static_cast<std::size_t>(connection.b_id)].score +
                       connection.score;
        people.push_back(person);
      }
      continue;
    }

    for (const auto& connection : connections) {
      std::vector<int> matched_indices;
      for (std::size_t i = 0; i < people.size(); ++i) {
        if (people[i].keypoint_ids[static_cast<std::size_t>(ka)] == connection.a_id ||
            people[i].keypoint_ids[static_cast<std::size_t>(kb)] == connection.b_id) {
          matched_indices.push_back(static_cast<int>(i));
        }
      }
      if (!matched_indices.empty()) {
        auto& person = people[static_cast<std::size_t>(matched_indices.front())];
        add_pose_connection(person, ka, kb, connection, all_keypoints);
        if (matched_indices.size() > 1) {
          for (std::size_t idx = matched_indices.size() - 1; idx > 0; --idx) {
            const int source_index = matched_indices[idx];
            merge_pose_people(person, people[static_cast<std::size_t>(source_index)]);
            people.erase(people.begin() + source_index);
          }
        }
        continue;
      }

      if (part_id == 17 || part_id == 18) {
        for (auto& person : people) {
          if (person.keypoint_ids[static_cast<std::size_t>(ka)] == connection.a_id &&
              person.keypoint_ids[static_cast<std::size_t>(kb)] < 0) {
            person.keypoint_ids[static_cast<std::size_t>(kb)] = connection.b_id;
            break;
          }
          if (person.keypoint_ids[static_cast<std::size_t>(kb)] == connection.b_id &&
              person.keypoint_ids[static_cast<std::size_t>(ka)] < 0) {
            person.keypoint_ids[static_cast<std::size_t>(ka)] = connection.a_id;
            break;
          }
        }
        continue;
      }

      PosePerson person;
      person.keypoint_ids[static_cast<std::size_t>(ka)] = connection.a_id;
      person.keypoint_ids[static_cast<std::size_t>(kb)] = connection.b_id;
      person.count = 2;
      person.score = all_keypoints[static_cast<std::size_t>(connection.a_id)].score +
                     all_keypoints[static_cast<std::size_t>(connection.b_id)].score +
                     connection.score;
      people.push_back(person);
    }
  }

  std::vector<PosePerson> filtered;
  for (const auto& person : people) {
    if (person.count >= 4 && person.score / std::max(1, person.count) >= 0.35f) {
      filtered.push_back(person);
    }
  }
  return filtered;
}

PoseOverlay draw_openpose_overlay(neat::Tensor& frame, const neat::Sample& sample,
                                  const ModelSpec& spec, const Nv12Color& color) {
  PoseOverlay stats;
  int width = 0;
  int height = 0;
  if (!infer_dims(frame, width, height)) {
    return stats;
  }

  const auto tensors = neat::tensors_from_sample(sample, false);
  if (tensors.empty()) {
    return stats;
  }

  OpenPoseTensorData heatmap;
  OpenPoseTensorData paf;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    auto values = tensor_to_float_vector(tensors[i]);
    int candidate_w = 0;
    int candidate_h = 0;
    int candidate_c = 0;
    if (!openpose_tensor_geometry(tensors[i], values, candidate_w, candidate_h, candidate_c)) {
      continue;
    }
    if (candidate_c == 19) {
      heatmap.width = candidate_w;
      heatmap.height = candidate_h;
      heatmap.channels = candidate_c;
      heatmap.values = std::move(values);
    } else if (candidate_c == 38) {
      paf.width = candidate_w;
      paf.height = candidate_h;
      paf.channels = candidate_c;
      paf.values = std::move(values);
    }
  }
  if (heatmap.values.empty() || paf.values.empty() || heatmap.width != paf.width ||
      heatmap.height != paf.height) {
    return stats;
  }

  const auto by_type = find_openpose_peaks(heatmap);
  std::vector<PoseKeypoint> all_keypoints;
  for (const auto& candidates : by_type) {
    for (const auto& candidate : candidates) {
      if (candidate.id >= static_cast<int>(all_keypoints.size())) {
        all_keypoints.resize(static_cast<std::size_t>(candidate.id) + 1);
      }
      all_keypoints[static_cast<std::size_t>(candidate.id)] = candidate;
    }
  }

  const auto people = assemble_openpose_people(by_type, all_keypoints, paf);
  stats.persons = static_cast<int>(people.size());

  static constexpr std::array<std::pair<int, int>, 17> kDrawSkeleton = {
      std::pair{1, 2},  {1, 5},  {2, 3},   {3, 4},   {5, 6},   {6, 7},
      {1, 8},          {8, 9},  {9, 10},  {1, 11},  {11, 12}, {12, 13},
      {1, 0},          {0, 14}, {14, 16}, {0, 15},  {15, 17},
  };
  for (std::size_t person_index = 0; person_index < people.size(); ++person_index) {
    const Nv12Color person_color = class_color(static_cast<int>(person_index) + 40);
    const auto& person = people[person_index];
    for (const auto& [a, b] : kDrawSkeleton) {
      const int id_a = person.keypoint_ids[static_cast<std::size_t>(a)];
      const int id_b = person.keypoint_ids[static_cast<std::size_t>(b)];
      if (id_a < 0 || id_b < 0 || id_a >= static_cast<int>(all_keypoints.size()) ||
          id_b >= static_cast<int>(all_keypoints.size())) {
        continue;
      }
      const auto& ka = all_keypoints[static_cast<std::size_t>(id_a)];
      const auto& kb = all_keypoints[static_cast<std::size_t>(id_b)];
      const float model_ax = (ka.x + 0.5f) * spec.model_width / heatmap.width;
      const float model_ay = (ka.y + 0.5f) * spec.model_height / heatmap.height;
      const float model_bx = (kb.x + 0.5f) * spec.model_width / heatmap.width;
      const float model_by = (kb.y + 0.5f) * spec.model_height / heatmap.height;
      const auto [fax, fay] =
          model_to_frame(model_ax, model_ay, width, height, spec.model_width, spec.model_height);
      const auto [fbx, fby] =
          model_to_frame(model_bx, model_by, width, height, spec.model_width, spec.model_height);
      draw_nv12_line(frame, width, height, fax, fay, fbx, fby, 4, person_color);
    }
    for (int id : person.keypoint_ids) {
      if (id < 0 || id >= static_cast<int>(all_keypoints.size())) {
        continue;
      }
      const auto& kp = all_keypoints[static_cast<std::size_t>(id)];
      const float model_x = (kp.x + 0.5f) * spec.model_width / heatmap.width;
      const float model_y = (kp.y + 0.5f) * spec.model_height / heatmap.height;
      const auto [fx, fy] =
          model_to_frame(model_x, model_y, width, height, spec.model_width, spec.model_height);
      draw_nv12_circle(frame, width, height, fx, fy, 5, Nv12Color{235, 128, 128});
    }
    stats.keypoints += person.count;
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

  if (runtime.spec.decode_type == neat::BoxDecodeType::YoloV8Seg) {
    const auto seg = draw_segmentation_overlay(frame, model_sample, cfg, model_color);
    runtime.last_visible_boxes = seg.visible_instances;
    draw_nv12_text(frame, width, height, 520, 8, "MASKS:" + std::to_string(seg.visible_instances),
                   white, 2);
  } else if (runtime.spec.name == "open_pose") {
    const auto pose = draw_openpose_overlay(frame, model_sample, runtime.spec, model_color);
    runtime.last_visible_boxes = pose.persons;
    draw_nv12_text(frame, width, height, 520, 8,
                   "PERSONS:" + std::to_string(pose.persons) + " KPTS:" +
                       std::to_string(pose.keypoints),
                   white, 2);
  } else {
    const auto boxes = runtime.spec.decode_type.has_value()
                           ? decode_boxes_from_sample(model_sample, width, height, cfg.top_k)
                           : std::vector<neat::Box>{};
    int visible = 0;
    for (const auto& box : boxes) {
      if (box.score < cfg.score_threshold) {
        continue;
      }
      draw_nv12_box(frame, width, height, box, model_color);
      draw_nv12_label(frame, width, height, static_cast<int>(box.x1),
                      std::max(36, static_cast<int>(box.y1) - 18), class_label(box.class_id),
                      model_color);
      ++visible;
    }
    runtime.last_visible_boxes = runtime.spec.decode_type.has_value() ? visible : -1;

    if (!boxes.empty()) {
      draw_nv12_text(frame, width, height, 520, 8, "BOXES:" + std::to_string(visible), white, 2);
    } else {
      const std::string label = runtime.spec.decode_type.has_value() ? "BOXES:0" : "TENSORS";
      draw_nv12_text(frame, width, height, 520, 8, label, white, 2);
    }
  }

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

std::string tensor_shape_summary(const neat::Sample& sample) {
  const auto tensors = neat::tensors_from_sample(sample, false);
  std::string out;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (!out.empty()) {
      out += " ";
    }
    out += "t" + std::to_string(i) + "=[";
    for (std::size_t j = 0; j < tensors[i].shape.size(); ++j) {
      if (j > 0) {
        out += "x";
      }
      out += std::to_string(tensors[i].shape[j]);
    }
    out += "]";
  }
  return out;
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

    if (cfg.load_only) {
      std::cout << "Loaded model graphs successfully:\n";
      for (const auto& rt : runtimes) {
        std::cout << "  " << rt.spec.name << " -> " << rt.spec.path << "\n";
      }
      for (auto& rt : runtimes) {
        rt.udp_run.close();
        rt.model_run.close();
      }
      return 0;
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

      for (auto& rt : runtimes) {
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

        neat::Tensor output_frame = copy_nv12_to_cpu_tensor(frame_tensors.front());
        draw_model_overlay(output_frame, rt, model_sample, cfg);
        if (!rt.udp_run.push("video", neat::TensorList{output_frame})) {
          std::cerr << "[warn] failed to push " << rt.spec.name << " frame to UDP encoder\n";
        }
        ++rt.processed;

        if (processed == 0 || ((processed + 1) % 30) == 0 || cfg.frames > 0) {
          std::cout << "[" << rt.spec.name << "] frame=" << (processed + 1)
                    << " result=" << sample_summary(model_sample)
                    << " sample_frame_id=" << model_sample.frame_id;
          if (rt.last_visible_boxes >= 0) {
            if (rt.spec.decode_type == neat::BoxDecodeType::YoloV8Seg) {
              std::cout << " masks=" << rt.last_visible_boxes;
            } else if (rt.spec.name == "open_pose") {
              std::cout << " persons=" << rt.last_visible_boxes;
            } else {
              std::cout << " boxes=" << rt.last_visible_boxes;
            }
          }
          if (rt.spec.name == "open_pose") {
            std::cout << " " << tensor_shape_summary(model_sample);
          }
          std::cout << "\n";
        }
      }

      ++processed;
      if (processed == 1 || (processed % 30) == 0 || cfg.frames > 0) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start).count();
        const double fps = elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0;
        std::cout << "[probe] source_frames=" << processed << " active_models="
                  << runtimes.size() << " fps=" << std::fixed << std::setprecision(2) << fps
                  << "\n";
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
