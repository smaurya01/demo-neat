// RTSP single-stream Neat demo publishing clean video via VideoSender and
// detections as JSON via MetadataSender — no boxes are drawn on the frames.
// Neat Insight (or any metadata-aware viewer) overlays the boxes client-side.
#include <neat.h>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr const char* kConfigPath = "./config/default.conf";

constexpr std::array<const char*, 80> kCocoLabels = {
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

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

struct Config {
  std::string rtsp_url;
  std::string model_path;
  std::string insight_host;
  int channel = 0;
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  int fallback_width = 1280;
  int fallback_height = 720;
  int fallback_fps = 60;
  int model_width = 640;
  int model_height = 640;
  int latency_ms = 200;
  float score_threshold = 0.25f;
  float nms_iou = 0.50f;
  int top_k = 100;
  int num_classes = 80;
  int frames = 0;
  int bitrate_kbps = 4000;
  bool tcp = true;
  bool print_backend = false;
};

using Clock = std::chrono::steady_clock;

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
  } else if (key == "insight_host") {
    cfg.insight_host = value;
  } else if (key == "channel") {
    cfg.channel = to_int(value);
  } else if (key == "video_port_base") {
    cfg.video_port_base = to_int(value);
  } else if (key == "metadata_port_base") {
    cfg.metadata_port_base = to_int(value);
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

void validate_config(const Config& cfg) {
  if (cfg.rtsp_url.empty()) {
    throw std::runtime_error("rtsp_url must not be empty");
  }
  if (cfg.insight_host.empty() || cfg.insight_host.front() == '<') {
    throw std::runtime_error("insight_host must be set to the host running Neat Insight");
  }
  if (cfg.channel < 0) {
    throw std::runtime_error("channel must be >= 0");
  }
  const int video_port = cfg.video_port_base + cfg.channel;
  const int metadata_port = cfg.metadata_port_base + cfg.channel;
  if (video_port <= 0 || video_port > 65535 || metadata_port <= 0 || metadata_port > 65535) {
    throw std::runtime_error("derived UDP ports must be in 1..65535");
  }
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
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  // Zoo yolo_11n exposes raw 64-channel DFL heads -> YoloV8 decode family.
  // A self-compiled YOLO11 archive needs BoxDecodeType::YoloV26 instead.
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

// One in-graph pipeline: the decoded stream branches to the VideoSender (clean
// frames, never touched by the CPU) and to the model, whose decoded detections
// surface at the named "detections" output.
neat::Graph make_pipeline(const Config& cfg, const neat::Model& model, int& video_port_out) {
  auto source = groups::RtspDecodedInput(make_rtsp_options(cfg));
  auto branch = neat::graphs::Branch("source", {"video", "model"});

  auto video_options = groups::VideoSenderOptions::H264RtpUdpFromRaw(
      cfg.fallback_width, cfg.fallback_height, cfg.fallback_fps);
  video_options.host = cfg.insight_host;
  video_options.channel = cfg.channel;
  video_options.video_port_base = cfg.video_port_base;
  video_options.encoder.bitrate_kbps = cfg.bitrate_kbps;
  video_port_out = video_options.video_port();

  neat::Graph video_graph("video");
  video_graph.connect(neat::nodes::Input("video"), groups::VideoSender(video_options));

  neat::Graph model_graph("model");
  model_graph.connect(neat::nodes::Input("model"), model);

  neat::Graph detections_graph("detections");
  detections_graph.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));

  neat::Graph graph("single_stream_yolo_insight");
  neat::GraphLinkOptions live_link;
  live_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  graph.connect(source, branch);
  graph.connect(branch, video_graph, live_link);
  graph.connect(branch, model_graph, live_link);
  graph.connect(model_graph, detections_graph);
  return graph;
}

std::vector<neat::Box> decode_boxes(const neat::Sample& sample, const Config& cfg) {
  const auto tensors = neat::tensors_from_sample(sample, true);
  if (tensors.empty()) {
    return {};
  }
  const auto decoded = neat::decode_bbox_tensor(tensors.front(), cfg.fallback_width,
                                                cfg.fallback_height, cfg.top_k, false);
  return decoded.boxes;
}

std::string class_label(int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(kCocoLabels.size())) {
    return kCocoLabels[static_cast<std::size_t>(class_id)];
  }
  return "CLASS " + std::to_string(class_id);
}

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

// Insight expects object-detection bbox as [x, y, w, h] top-left in
// encoded-frame pixels; decode_bbox_tensor returns corners, so convert here.
std::string build_objects_json(const std::vector<neat::Box>& boxes, const Config& cfg,
                               int& published_out) {
  std::ostringstream json;
  json << "{\"objects\":[";
  int published = 0;
  for (const auto& box : boxes) {
    if (box.score < cfg.score_threshold) {
      continue;
    }
    const int x = std::max(0, static_cast<int>(box.x1));
    const int y = std::max(0, static_cast<int>(box.y1));
    const int w = std::min(cfg.fallback_width - x, static_cast<int>(box.x2 - box.x1));
    const int h = std::min(cfg.fallback_height - y, static_cast<int>(box.y2 - box.y1));
    if (w <= 0 || h <= 0) {
      continue;
    }
    if (published > 0) {
      json << ",";
    }
    ++published;
    json << "{\"id\":\"obj_" << published << "\""
         << ",\"label\":\"" << json_escape(class_label(box.class_id)) << "\""
         << ",\"confidence\":" << box.score
         << ",\"bbox\":[" << x << "," << y << "," << w << "," << h << "]}";
  }
  json << "]}";
  published_out = published;
  return json.str();
}

int64_t now_epoch_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

int run_app() {
  const Config cfg = read_config();
  validate_config(cfg);

  const auto model = make_model(cfg);
  int video_port = 0;
  auto graph = make_pipeline(cfg, *model, video_port);
  if (cfg.print_backend) {
    std::cout << "Backend:\n" << graph.describe_backend() << "\n";
  }

  neat::RunOptions run_options;
  run_options.preset = neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = neat::OutputMemory::ZeroCopy;
  auto run = graph.build(run_options);

  neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = cfg.channel;
  metadata_options.metadata_port_base = cfg.metadata_port_base;
  std::string metadata_err;
  neat::MetadataSender metadata_sender(metadata_options, &metadata_err);
  if (!metadata_sender.ok()) {
    throw std::runtime_error("metadata sender init failed: " + metadata_err);
  }

  std::cout << "RTSP input:   " << cfg.rtsp_url << "\n"
            << "Model:        " << cfg.model_path << "\n"
            << "Video out:    udp://" << cfg.insight_host << ":" << video_port
            << " (clean frames, no overlay)\n"
            << "Metadata out: udp://" << cfg.insight_host << ":"
            << metadata_sender.metadata_port() << " (object-detection JSON, channel="
            << cfg.channel << ")\n"
            << "Viewer:       Neat Insight Video Viewer channel " << cfg.channel
            << " draws boxes from metadata\n";

  int processed = 0;
  double pull_ms_sum = 0.0;
  double send_ms_sum = 0.0;
  const auto run_start = Clock::now();
  while (!g_stop.load() && (cfg.frames <= 0 || processed < cfg.frames)) {
    neat::Sample sample;
    neat::PullError pull_error;
    const auto pull_start = Clock::now();
    const auto status = run.pull("detections", 20000, sample, &pull_error);
    const auto pull_end = Clock::now();
    if (status == neat::PullStatus::Timeout) {
      std::cerr << "[warn] timed out waiting for detections\n";
      continue;
    }
    if (status == neat::PullStatus::Closed) {
      break;
    }
    if (status != neat::PullStatus::Ok) {
      throw std::runtime_error("failed to pull detections: " + pull_error.message);
    }

    const auto boxes = decode_boxes(sample, cfg);
    int published = 0;
    const std::string data_json = build_objects_json(boxes, cfg, published);

    const int64_t frame_id = sample.frame_id >= 0 ? sample.frame_id : processed;
    const auto send_start = Clock::now();
    // Send every frame, including an empty list, so stale boxes never linger
    // in the viewer. UDP is fire-and-forget; success only proves the datagram
    // left this host.
    std::string send_err;
    if (!metadata_sender.send_metadata("object-detection", data_json, now_epoch_ms(),
                                       std::to_string(frame_id), &send_err)) {
      std::cerr << "[warn] metadata send failed: " << send_err << "\n";
    }
    const auto send_end = Clock::now();

    ++processed;
    pull_ms_sum += ms_since(pull_start, pull_end);
    send_ms_sum += ms_since(send_start, send_end);
    if (should_log_frame(processed, cfg.frames)) {
      const double elapsed_s =
          std::chrono::duration<double>(Clock::now() - run_start).count();
      const double fps_now = elapsed_s > 0.0 ? processed / elapsed_s : 0.0;
      std::cout << "frame=" << processed << " detections=" << boxes.size()
                << " published=" << published << " fps=" << fps_now
                << " avg_ms(pull=" << pull_ms_sum / processed
                << ", metadata_send=" << send_ms_sum / processed << ")\n"
                << std::flush;
    }
  }

  run.close();
  std::cout << "processed=" << processed << " video=" << cfg.insight_host << ":" << video_port
            << " metadata=" << cfg.insight_host << ":" << metadata_sender.metadata_port()
            << "\n";
  return 0;
}

} // namespace

int main() {
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  try {
    return run_app();
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
