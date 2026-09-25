// 16stream4model — 16 x 1280x720@30 RTSP streams across 4 models, one process.
//
// Topology (4 streams per model, one Graph + Run + puller thread per model). The default config:
//
//   stream 0..3   -> model 0  YOLO26n detection  (YoloV26 decode)
//   stream 4..7   -> model 1  YOLO11n detection  (YoloV8 decode: zoo keeps the raw DFL head)
//   stream 8..11  -> model 2  YOLOv8n detection  (YoloV8 decode)
//   stream 12..15 -> model 3  YOLO26n pose       (YoloV26Pose decode, 1 class, 17 keypoints;
//                                                 zoo score head is pre-sigmoided: score_is_prob)
//
// Per stream, Core fuses two branches out of ONE RTSP session:
//
//   RTSP H.264
//     |- latest encoded edge --> VideoSender ------> Insight video    channel i (9000+i)
//     \- decode -> detector for its model --------> MetadataSender -> Insight metadata channel i
//                                                   (9100+i)
//
// The encoded access unit goes to Insight untouched and Insight draws the overlay from JSON
// metadata, so nothing is decoded to the host, annotated or re-encoded.
//
// Derived from Worker_Safety/app16 with the VLM, the per-stream local frame (snapshot) branch
// and the violation/event path removed; the pose path now sends Insight pose-estimation
// metadata. Graph execution options follow the
// high-density-multi-stream-object-detector example (internal_queue_depth=1, async MLA).
//
// Things that silently go wrong if changed:
//  1. Graphs are wired BY ENDPOINT NAME. Each model has its own "detector_frame_<m>" /
//     "detections_<m>" pair; two models sharing a name cross streams between them.
//  2. VideoSenderOptions::async = false. The sender is fused into the same live pipeline as the
//     decoder fan-in; an async UDP sink can hold the whole pipeline in PAUSED waiting for preroll.
//  3. skip_rtsp_probe pins an INTEGER frame rate. A 29.97 source fails negotiation outright.
//  4. Start each puller the moment its own Run is built: build() starts streaming, and an
//     undrained non-dropping detections sink back-pressures the decoders.

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include "support/detection_egress.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kMaxStreams = 16;
constexpr int kNumModels = 4;

std::atomic<int> g_stop{0};        // stop requested: 1 = signal, 2 = fatal pipeline error
std::atomic<bool> g_pull_done{false};
constexpr int kStopSignal = 1;
constexpr int kStopFatal = 2;
// A second Ctrl-C exits immediately, in case run.close() hangs on the way out.
void handle_signal(int) {
  int expected = 0;
  if (!g_stop.compare_exchange_strong(expected, kStopSignal)) std::_Exit(130);
}
void request_fatal_stop() {
  int expected = 0;
  g_stop.compare_exchange_strong(expected, kStopFatal);
}

// Per-stream heartbeat, written by the pullers and read by the main thread's reporter.
std::array<std::atomic<std::uint64_t>, kMaxStreams> g_frames{};

// ── config ──────────────────────────────────────────────────────────────────
enum class Task { Detection, Pose };

// COCO keypoint names in the order the BoxDecode pose payload emits them. Insight joins
// skeleton edges by name, so these strings are part of the metadata contract.
constexpr int kPoseKeypoints = 17;
constexpr std::array<const char*, kPoseKeypoints> kCocoKeypointNames = {
    "nose",           "left_eye",   "right_eye",   "left_ear",   "right_ear",   "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
    "right_hip",      "left_knee",  "right_knee",  "left_ankle", "right_ankle"};

struct ModelCfg {
  std::string name;
  std::string archive;
  std::string labels_path;
  // Must be the model's real class count: archives bake a generic 80-class BoxDecode stub,
  // and the pose archives still declare 80 although their score head is single-class.
  int num_classes = 80;
  Task task = Task::Detection;
  // Follows the archive's HEAD SHAPE, not the model's name: zoo YOLO11/YOLOv8 keep the raw
  // 64-channel DFL head (YoloV8); YOLO26 exports 4-channel l/t/r/b (YoloV26).
  neat::BoxDecodeType decode = neat::BoxDecodeType::YoloV26;
  // The archive's score head already ends in a sigmoid (the zoo yolo_26n_pose does: its score
  // tensors dequantize to [0, x] with zp=-128). BoxDecode applies a second sigmoid, which puts
  // every background anchor at ~0.5, so a 0.30 threshold passes everything. When set, the
  // threshold is mapped through sigmoid() and decoded scores back through logit().
  bool score_is_prob = false;
  std::vector<std::string> labels;
};

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float logitf(float p) {
  p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
  return std::log(p / (1.0f - p));
}

struct StreamCfg {
  std::string url;
  int group = -1;   // -1 = derive from the index: stream i feeds model i / 4
  int fps = 0;      // 0 = the global fps. Must be the source's real integer rate: it is pinned.
  int slot = -1;    // config index i; stays the Insight channel and stream id under filtering
};

struct Config {
  std::array<ModelCfg, kNumModels> models{};
  std::vector<StreamCfg> streams;

  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  int visible_streams = -1;   // -1 = all
  bool video_enabled = true;

  // Source contract. skip_rtsp_probe=true trusts these instead of probing every URL.
  int width = 1280;
  int height = 720;
  int fps = 30;
  int latency_ms = 100;
  bool tcp = true;
  bool drop_on_latency = true;
  bool skip_rtsp_probe = true;
  int decoder_buffers = 8;
  int decoder_input_buffers = 2;
  std::string decoder_tuning = "throughput-low-latency";

  // Inference backpressure (per model graph).
  int queue_depth = 16;
  int internal_queue_depth = 1;    // GraphOptions advanced_execution; -1 = framework default
  bool inference_async = true;
  int max_inflight_per_stream = 2;
  int max_inflight_total = 8;
  bool detections_drop = false;    // 4 streams per model: back-pressure beats dropping
  float min_score = 0.30f;
  float nms_iou = 0.60f;
  int max_detections = 50;

  int warmup_frames = 30;
  double duration_s = 0.0;
  double report_interval_s = 5.0;
  double target_ratio = 0.97;      // pass when every stream delivers >= ratio * fps
  double stall_warn_s = 3.0;
  bool verbose = false;
  bool print_backend = false;
  bool source_only = false;
  bool measure = false;
  int measure_model = 0;
};

std::string trim(std::string v) {
  const auto ws = " \t\r\n";
  const auto b = v.find_first_not_of(ws);
  if (b == std::string::npos) return {};
  const auto e = v.find_last_not_of(ws);
  return v.substr(b, e - b + 1);
}

bool to_bool(const std::string& v) {
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

// std::stoi/stof accept trailing junk ("29.97" -> 29) and report errors without the key, so
// every numeric value goes through these.
int parse_int(const std::string& key, const std::string& value) {
  std::size_t pos = 0;
  int v = 0;
  try {
    v = std::stoi(value, &pos);
  } catch (const std::exception&) {
    pos = std::string::npos;
  }
  if (pos != value.size()) throw std::runtime_error(key + ": expected an integer, got '" + value + "'");
  return v;
}

double parse_double(const std::string& key, const std::string& value) {
  std::size_t pos = 0;
  double v = 0.0;
  try {
    v = std::stod(value, &pos);
  } catch (const std::exception&) {
    pos = std::string::npos;
  }
  if (pos != value.size()) throw std::runtime_error(key + ": expected a number, got '" + value + "'");
  return v;
}

// Matches "<prefix><digits><suffix>" and returns the number, or -1.
int indexed_slot(const std::string& key, const std::string& prefix, const std::string& suffix,
                 int limit) {
  if (key.size() <= prefix.size() + suffix.size()) return -1;
  if (key.compare(0, prefix.size(), prefix) != 0) return -1;
  if (key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) return -1;
  const std::string digits = key.substr(prefix.size(), key.size() - prefix.size() - suffix.size());
  if (digits.empty() || digits.size() > 3 ||
      !std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c); })) {
    return -1;
  }
  const int i = std::stoi(digits);
  return i < limit ? i : -1;
}

void set_value(Config& cfg, std::map<int, StreamCfg>& streams, const std::string& key,
               const std::string& value) {
  if (int i = indexed_slot(key, "stream", "_rtsp", kMaxStreams); i >= 0) {
    streams[i].url = value;
    return;
  }
  if (int i = indexed_slot(key, "stream", "_model", kMaxStreams); i >= 0) {
    streams[i].group = parse_int(key, value);
    if (streams[i].group < 0 || streams[i].group >= kNumModels) {
      throw std::runtime_error(key + " must be 0.." + std::to_string(kNumModels - 1));
    }
    return;
  }
  if (int i = indexed_slot(key, "stream", "_fps", kMaxStreams); i >= 0) {
    streams[i].fps = parse_int(key, value);
    return;
  }
  if (int m = indexed_slot(key, "model", "_name", kNumModels); m >= 0) {
    cfg.models[m].name = value;
    return;
  }
  if (int m = indexed_slot(key, "model", "_archive", kNumModels); m >= 0) {
    cfg.models[m].archive = value;
    return;
  }
  if (int m = indexed_slot(key, "model", "_labels", kNumModels); m >= 0) {
    cfg.models[m].labels_path = value;
    return;
  }
  if (int m = indexed_slot(key, "model", "_classes", kNumModels); m >= 0) {
    cfg.models[m].num_classes = parse_int(key, value);
    return;
  }
  if (int m = indexed_slot(key, "model", "_task", kNumModels); m >= 0) {
    if (value == "detection") cfg.models[m].task = Task::Detection;
    else if (value == "pose") cfg.models[m].task = Task::Pose;
    else throw std::runtime_error(key + " must be detection|pose");
    return;
  }
  if (int m = indexed_slot(key, "model", "_score_is_prob", kNumModels); m >= 0) {
    cfg.models[m].score_is_prob = to_bool(value);
    return;
  }
  if (int m = indexed_slot(key, "model", "_decode", kNumModels); m >= 0) {
    if (value == "yolo26") cfg.models[m].decode = neat::BoxDecodeType::YoloV26;
    else if (value == "yolov8") cfg.models[m].decode = neat::BoxDecodeType::YoloV8;
    else throw std::runtime_error(key + " must be yolo26|yolov8");
    return;
  }

  if (key == "insight_host") cfg.insight_host = value;
  else if (key == "video_port_base") cfg.video_port_base = parse_int(key, value);
  else if (key == "metadata_port_base") cfg.metadata_port_base = parse_int(key, value);
  else if (key == "visible_streams") cfg.visible_streams = parse_int(key, value);
  else if (key == "video_enabled") cfg.video_enabled = to_bool(value);
  else if (key == "width") cfg.width = parse_int(key, value);
  else if (key == "height") cfg.height = parse_int(key, value);
  else if (key == "fps") cfg.fps = parse_int(key, value);
  else if (key == "latency_ms") cfg.latency_ms = parse_int(key, value);
  else if (key == "rtsp_transport") cfg.tcp = (value == "tcp");
  else if (key == "drop_on_latency") cfg.drop_on_latency = to_bool(value);
  else if (key == "skip_rtsp_probe") cfg.skip_rtsp_probe = to_bool(value);
  else if (key == "decoder_buffers") cfg.decoder_buffers = parse_int(key, value);
  else if (key == "decoder_input_buffers") cfg.decoder_input_buffers = parse_int(key, value);
  else if (key == "decoder_tuning") cfg.decoder_tuning = value;
  else if (key == "queue_depth") cfg.queue_depth = parse_int(key, value);
  else if (key == "internal_queue_depth") cfg.internal_queue_depth = parse_int(key, value);
  else if (key == "inference_async") cfg.inference_async = to_bool(value);
  else if (key == "max_inflight_per_stream") cfg.max_inflight_per_stream = parse_int(key, value);
  else if (key == "max_inflight_total") cfg.max_inflight_total = parse_int(key, value);
  else if (key == "detections_drop") cfg.detections_drop = to_bool(value);
  else if (key == "min_score") cfg.min_score = static_cast<float>(parse_double(key, value));
  else if (key == "nms_iou") cfg.nms_iou = static_cast<float>(parse_double(key, value));
  else if (key == "max_detections") cfg.max_detections = parse_int(key, value);
  else if (key == "warmup_frames") cfg.warmup_frames = parse_int(key, value);
  else if (key == "duration") cfg.duration_s = parse_double(key, value);
  else if (key == "report_interval") cfg.report_interval_s = parse_double(key, value);
  else if (key == "target_ratio") cfg.target_ratio = parse_double(key, value);
  else if (key == "stall_warn_s") cfg.stall_warn_s = parse_double(key, value);
  else if (key == "verbose") cfg.verbose = to_bool(value);
  else if (key == "print_backend") cfg.print_backend = to_bool(value);
  else std::cerr << "[warn] unknown config key: " << key << "\n";
}

Config read_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);

  Config cfg;
  std::map<int, StreamCfg> streams;
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    line = trim(line);
    if (line.empty()) continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    set_value(cfg, streams, trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
  }
  for (const auto& [index, s] : streams) {
    if (s.url.empty()) continue;
    if (static_cast<int>(cfg.streams.size()) <= index) cfg.streams.resize(index + 1);
    cfg.streams[static_cast<std::size_t>(index)] = s;
  }
  return cfg;
}

std::vector<std::string> load_labels(const std::string& path, int num_classes,
                                     const std::string& id) {
  std::vector<std::string> labels;
  std::ifstream in(path);
  if (!in) {
    std::cerr << "[warn] " << id << ": labels file not found: " << path << "\n";
    return labels;
  }
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (!line.empty()) labels.push_back(line);
  }
  if (static_cast<int>(labels.size()) != num_classes) {
    std::cerr << "[warn] " << id << ": " << path << " has " << labels.size()
              << " labels but the model has " << num_classes << " classes\n";
  }
  return labels;
}

void validate(Config& cfg) {
  if (cfg.streams.empty()) throw std::runtime_error("no stream<i>_rtsp entries in the config");
  if (cfg.skip_rtsp_probe && (cfg.width <= 0 || cfg.height <= 0 || cfg.fps <= 0)) {
    throw std::runtime_error("skip_rtsp_probe=true needs integer width/height/fps");
  }
  // The shipped config carries <...> placeholders instead of lab addresses.
  const auto is_placeholder = [](const std::string& v) { return v.find('<') != std::string::npos; };
  if (is_placeholder(cfg.insight_host)) {
    throw std::runtime_error("insight_host is still the placeholder '" + cfg.insight_host +
                             "': set it to the Insight host's IP as seen from the DevKit");
  }
  for (std::size_t i = 0; i < cfg.streams.size(); ++i) {
    auto& s = cfg.streams[i];
    if (s.url.empty()) {
      throw std::runtime_error("stream" + std::to_string(i) +
                               "_rtsp is not set (slots must be contiguous from 0)");
    }
    if (is_placeholder(s.url)) {
      throw std::runtime_error("stream" + std::to_string(i) + "_rtsp is still the placeholder '" +
                               s.url + "': set it to the RTSP URL of Insight source src" +
                               std::to_string(i + 1));
    }
    s.slot = static_cast<int>(i);
    if (s.group < 0) s.group = static_cast<int>(i) / (kMaxStreams / kNumModels);
    if (s.fps <= 0) s.fps = cfg.fps;
  }
  for (int m = 0; m < kNumModels; ++m) {
    auto& g = cfg.models[static_cast<std::size_t>(m)];
    const bool used = std::any_of(cfg.streams.begin(), cfg.streams.end(),
                                  [m](const StreamCfg& s) { return s.group == m; });
    if (!used) continue;
    if (g.archive.empty()) {
      throw std::runtime_error("model" + std::to_string(m) +
                               "_archive is not set but streams are assigned to it");
    }
    if (!std::ifstream(g.archive)) {
      throw std::runtime_error("model" + std::to_string(m) + "_archive not found: " + g.archive);
    }
    if (g.num_classes <= 0) {
      throw std::runtime_error("model" + std::to_string(m) + "_classes must be > 0");
    }
    if (g.name.empty()) g.name = "model" + std::to_string(m);
    g.labels = load_labels(g.labels_path, g.num_classes, g.name);
  }
}

// ── naming ──────────────────────────────────────────────────────────────────
std::string frame_endpoint(int group) { return "detector_frame_" + std::to_string(group); }
std::string detections_endpoint(int group) { return "detections_" + std::to_string(group); }
std::string raw_frame_endpoint(int stream) { return "raw_frame_" + std::to_string(stream); }
// Global stream id, unique across all models, so a detection maps back to its camera.
std::string stream_id_for(int index) { return "stream" + std::to_string(index); }

// Maps a sample's "stream<slot>" id to its position in `sources`, or -1. A single-stream
// fan-in carries the producer's name instead of a stream id, hence the `sole_pos` fallback.
int source_pos_from(const neat::Sample& s, const std::array<int, kMaxStreams>& pos_of_slot,
                    int sole_pos) {
  const std::string prefix = "stream";
  if (s.stream_id.rfind(prefix, 0) != 0) return sole_pos;
  const std::string suffix = s.stream_id.substr(prefix.size());
  if (suffix.empty() || suffix.size() > 3 ||
      !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
    return -1;
  }
  const int slot = std::stoi(suffix);
  return slot < kMaxStreams ? pos_of_slot[static_cast<std::size_t>(slot)] : -1;
}

// ── per-stream runtime ──────────────────────────────────────────────────────
struct SourceRuntime {
  int index = 0;   // config slot: Insight channel, stream id, report label, g_frames index
  int group = 0;
  std::string url;
  int frame_w = 0;
  int frame_h = 0;
  int frame_fps = 0;
  int video_port = 0;
  std::unique_ptr<neat::MetadataSender> metadata_sender;
  groups::RtspDecodedInputOptions source_options;

  // Owned by this stream's model puller thread; read by the main thread only after join.
  // Live progress goes through g_frames.
  std::uint64_t processed = 0;
  std::uint64_t boxes_total = 0;
  std::uint64_t metadata_ok = 0;
  std::uint64_t metadata_fail = 0;
  double t_parse_ms = 0.0;
  double t_meta_ms = 0.0;
  std::uint64_t t_samples = 0;
  Clock::time_point steady_start{};
  bool steady_started = false;
};

struct ModelRuntime {
  std::unique_ptr<neat::Model> model;
  neat::Graph detector;   // Input -> model -> Output fragment
  neat::Graph graph;      // container: this model's sources + the fragment above
  neat::Run run;
  int stream_count = 0;
  int sole_pos = -1;      // position in `sources` when this model has exactly one stream
  std::thread puller;
  std::atomic<std::uint64_t> unattributed{0};
  std::atomic<std::uint64_t> pull_failures{0};
};

// ── model + graphs ──────────────────────────────────────────────────────────
std::unique_ptr<neat::Model> make_model(const Config& cfg, const ModelCfg& g) {
  neat::Model::Options opt;
  if (!cfg.verbose) opt.verbose = neat::VerboseOptions::quiet();
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  // Ultralytics wants plain x/255, which is exactly this preset.
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = g.task == Task::Pose ? neat::BoxDecodeType::YoloV26Pose : g.decode;
  opt.score_threshold = g.score_is_prob ? sigmoidf(cfg.min_score) : cfg.min_score;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.max_detections;
  opt.num_classes = g.num_classes;
  return std::make_unique<neat::Model>(g.archive, opt);
}

neat::Graph build_detector_graph(const Config& cfg, int group, std::unique_ptr<neat::Model>& model) {
  model = make_model(cfg, cfg.models[static_cast<std::size_t>(group)]);

  neat::Graph input_graph;
  auto in_opt = model->input_appsrc_options(false);
  const int buffers = std::max(1, in_opt.pool_max_buffers);
  in_opt.max_bytes = static_cast<std::uint64_t>(cfg.width) * static_cast<std::uint64_t>(cfg.height) *
                     3U / 2U * static_cast<std::uint64_t>(buffers);
  in_opt.pool_max_buffers = buffers;
  in_opt.block = true;
  input_graph.add(neat::nodes::Input(frame_endpoint(group), in_opt));

  neat::Graph model_graph = model->graph();

  neat::Graph detections_graph;
  auto det_out = neat::OutputOptions::EveryFrame(cfg.queue_depth);
  det_out.drop = cfg.detections_drop;
  detections_graph.add(neat::nodes::Output(detections_endpoint(group), det_out));

  neat::Graph graph;
  graph.connect(input_graph, model_graph);
  graph.connect(model_graph, detections_graph);
  return graph;
}

groups::RtspDecodedInputOptions make_source_options(const Config& cfg, const StreamCfg& sc) {
  groups::RtspDecodedInputOptions opt;
  opt.url = sc.url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.drop_on_latency = cfg.drop_on_latency;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  // Tells the decoder its consumer is the CVU preprocess stage, not an appsink.
  opt.decoder_next_element = "CVU";
  opt.decoder_input_buffers = cfg.decoder_input_buffers;
  opt.decoder_tuning = cfg.decoder_tuning;
  opt.decoder_memory_opt =
      cfg.decoder_tuning == "low-memory" || cfg.decoder_tuning == "throughput-low-latency";
  opt.auto_caps_from_stream = !cfg.skip_rtsp_probe;
  opt.num_buffers = cfg.decoder_buffers;
  opt.codec = groups::RtspCodec::H264;
  opt.dec_width = cfg.width;
  opt.dec_height = cfg.height;
  opt.fallback_h264_width = cfg.width;
  opt.fallback_h264_height = cfg.height;
  opt.output_caps.width = cfg.width;
  opt.output_caps.height = cfg.height;
  if (cfg.skip_rtsp_probe) {
    opt.source_fps = sc.fps;
    opt.output_caps.fps = sc.fps;
  } else {
    opt.source_fps = 0;
    opt.output_caps.fps = -1;
  }
  return opt;
}

neat::Graph make_rtsp_encoded_input(const groups::RtspDecodedInputOptions& opt) {
  groups::RtspEncodedInputOptions enc;
  enc.url = opt.url;
  enc.codec = opt.codec;
  enc.latency_ms = opt.latency_ms;
  enc.tcp = opt.tcp;
  enc.drop_on_latency = opt.drop_on_latency;
  enc.buffer_mode = opt.buffer_mode;
  enc.insert_queue = opt.insert_queue;
  enc.sync_mode = opt.sync_mode;
  enc.auto_caps_from_stream = opt.auto_caps_from_stream;
  enc.source_fps = opt.source_fps;
  enc.payload_type = opt.payload_type;
  enc.h264_parse_config_interval = opt.h264_parse_config_interval;
  enc.fallback_h264_width = opt.fallback_h264_width;
  enc.fallback_h264_height = opt.fallback_h264_height;
  return groups::RtspEncodedInput(enc);
}

neat::Graph make_decoder(const groups::RtspDecodedInputOptions& opt, int decoder_buffers,
                         const std::string& out_endpoint) {
  neat::Graph graph("decoder");
  neat::SimaDecodeOptions decode;
  decode.type = neat::SimaDecodeType::H264;
  decode.sima_allocator_type = opt.sima_allocator_type;
  decode.out_format = neat::FormatTag::NV12;
  decode.decoder_name = opt.decoder_name;
  decode.raw_output = opt.decoder_raw_output;
  decode.next_element = opt.decoder_next_element;
  decode.dec_width = opt.dec_width;
  decode.dec_height = opt.dec_height;
  decode.dec_fps = opt.source_fps;
  decode.num_buffers = decoder_buffers;
  decode.input_buffers = opt.decoder_input_buffers;
  decode.decoder_tuning = opt.decoder_tuning;
  decode.memory_opt = opt.decoder_memory_opt;
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  if (opt.output_caps.enable) {
    graph.add(neat::nodes::CapsRaw("NV12", opt.output_caps.width, opt.output_caps.height,
                                   opt.output_caps.fps, opt.output_caps.memory));
  }
  // In source-only mode the app drains this endpoint itself; a non-dropping sink would
  // back-pressure the hardware decoder the moment the consumer falls behind.
  auto frame_out = neat::OutputOptions::EveryFrame(4);
  frame_out.drop = true;
  graph.add(neat::nodes::Output(out_endpoint, frame_out));
  return graph;
}

void connect_source(neat::Graph& app_graph, const Config& cfg, SourceRuntime& src,
                    const neat::Graph& detector_graph) {
  auto rtsp = make_rtsp_encoded_input(src.source_options);
  const std::string out_ep =
      cfg.source_only ? raw_frame_endpoint(src.index) : frame_endpoint(src.group);
  auto decoder = make_decoder(src.source_options, cfg.decoder_buffers, out_ep);
  app_graph.connect(rtsp, decoder);

  if (!cfg.source_only) {
    // Per-stream latest-frame admission: a slow model sheds that stream's stale frames instead
    // of back-pressuring its decoder.
    neat::GraphLinkOptions detector_link;
    detector_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
    detector_link.queue_depth = cfg.queue_depth;
    detector_link.stream_id = stream_id_for(src.index);
    detector_link.max_inflight_per_stream = cfg.max_inflight_per_stream;
    detector_link.max_inflight_total = cfg.max_inflight_total;
    app_graph.connect(decoder, detector_graph, detector_link);
  }

  const bool visible = cfg.visible_streams < 0 || src.index < cfg.visible_streams;
  if (cfg.video_enabled && visible) {
    auto video_options = groups::VideoSenderOptions::Passthrough(groups::RtspCodec::H264);
    video_options.host = cfg.insight_host;
    video_options.channel = src.index;
    video_options.video_port_base = cfg.video_port_base;
    video_options.async = false;   // see header note 2
    src.video_port = video_options.video_port();

    auto video_sender = groups::VideoSender(video_options);
    video_sender.set_name("insight_video_" + std::to_string(src.index));
    neat::GraphLinkOptions video_link;
    video_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
    app_graph.connect(rtsp, video_sender, video_link);
  }
}

std::uint32_t rtp_timestamp_from_pts_ns(std::int64_t pts_ns) {
  // Insight matches metadata to video on the 90 kHz RTP clock within +/-1 ms, so scale in one
  // step; two truncating divisions drift by up to the whole match window.
  if (pts_ns < 0) return 0;
  const auto ticks = (static_cast<__int128>(pts_ns) * 90000) / 1000000000;
  return static_cast<std::uint32_t>(static_cast<std::uint64_t>(ticks));
}

void send_metadata(SourceRuntime& src, const ModelCfg& model, const neat::Sample& sample,
                   const std::vector<neat::Box>& boxes) {
  if (!src.metadata_sender) return;

  sima_examples::detection_egress::FrameMetadata md;
  md.stream_index = src.index;
  // Always the configured id: a single-stream fan-in carries the producer's name instead.
  const std::string stream_id = stream_id_for(src.index);
  md.stream_id = stream_id;
  md.frame_id = sample.frame_id;
  md.pts_ns = sample.pts_ns;
  md.dts_ns = sample.dts_ns;
  md.duration_ns = sample.duration_ns;
  md.input_seq = sample.input_seq;
  md.orig_input_seq = sample.orig_input_seq;
  if (sample.pts_ns >= 0) md.rtp_timestamp = rtp_timestamp_from_pts_ns(sample.pts_ns);

  std::string payload;
  try {
    payload = sima_examples::detection_egress::serialize(boxes, model.labels, src.frame_w,
                                                         src.frame_h, md);
  } catch (const std::exception& e) {
    std::cerr << "[warn] stream " << src.index << " metadata build failed: " << e.what() << "\n";
    return;
  }
  std::string err;
  if (src.metadata_sender->send_raw_json(payload, &err)) {
    ++src.metadata_ok;
  } else {
    const auto n = ++src.metadata_fail;
    if (n == 1 || n % 500 == 0) {
      std::cerr << "[warn] stream " << src.index << " metadata send failed (count=" << n
                << "): " << err << "\n";
    }
  }
}

// Same envelope as detection_egress (so Insight can match on rtp_timestamp), with the
// `pose-estimation` data shape used by the multi-stream-pose-estimator example.
// `kpts` is dense [N][17][3] (x, y, confidence), parallel to `boxes`.
void send_pose_metadata(SourceRuntime& src, const neat::Sample& sample,
                        const std::vector<neat::Box>& boxes, const std::vector<float>& kpts) {
  if (!src.metadata_sender) return;
  nlohmann::json payload;
  payload["type"] = "pose-estimation";
  payload["timestamp"] = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  payload["frame_id"] = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  payload["stream_id"] = stream_id_for(src.index);
  payload["stream_index"] = src.index;
  payload["pts_ns"] = sample.pts_ns;
  payload["dts_ns"] = sample.dts_ns;
  payload["duration_ns"] = sample.duration_ns;
  payload["input_seq"] = sample.input_seq;
  payload["orig_input_seq"] = sample.orig_input_seq;
  if (sample.pts_ns >= 0) payload["rtp_timestamp"] = rtp_timestamp_from_pts_ns(sample.pts_ns);

  auto& poses = payload["data"]["poses"];
  poses = nlohmann::json::array();
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const auto& b = boxes[i];
    nlohmann::json points = nlohmann::json::array();
    const float* k = kpts.data() + i * kPoseKeypoints * 3;
    for (int j = 0; j < kPoseKeypoints; ++j) {
      points.push_back({{"name", kCocoKeypointNames[static_cast<std::size_t>(j)]},
                        {"x", std::lround(k[j * 3])},
                        {"y", std::lround(k[j * 3 + 1])},
                        {"confidence", std::round(k[j * 3 + 2] * 1000.0f) / 1000.0f}});
    }
    poses.push_back({{"id", "pose_" + std::to_string(i + 1)},
                     {"label", "person"},
                     {"confidence", std::round(b.score * 1000.0f) / 1000.0f},
                     {"bbox",
                      {std::lround(b.x1), std::lround(b.y1), std::lround(std::max(0.0f, b.x2 - b.x1)),
                       std::lround(std::max(0.0f, b.y2 - b.y1))}},
                     {"keypoints", std::move(points)}});
  }
  std::string err;
  if (src.metadata_sender->send_raw_json(payload.dump(), &err)) {
    ++src.metadata_ok;
  } else {
    const auto n = ++src.metadata_fail;
    if (n == 1 || n % 500 == 0) {
      std::cerr << "[warn] stream " << src.index << " pose metadata send failed (count=" << n
                << "): " << err << "\n";
    }
  }
}

void mark_frame(const Config& cfg, SourceRuntime& src) {
  ++src.processed;
  g_frames[static_cast<std::size_t>(src.index)].fetch_add(1, std::memory_order_relaxed);
  if (src.processed == static_cast<std::uint64_t>(cfg.warmup_frames) + 1) {
    src.steady_start = Clock::now();
    src.steady_started = true;
  }
}

// ── pull loops (one thread per model; each touches only its own streams) ────
void source_only_loop(int group, const Config& cfg, ModelRuntime& mr,
                      std::vector<SourceRuntime>& sources) {
  std::vector<int> mine;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (sources[i].group == group) mine.push_back(static_cast<int>(i));
  }
  while (g_stop.load() == 0 && !g_pull_done.load()) {
    bool did_work = false;
    for (const int idx : mine) {
      auto& src = sources[static_cast<std::size_t>(idx)];
      const auto endpoint = raw_frame_endpoint(src.index);
      for (int drained = 0; drained < 32; ++drained) {
        neat::Sample sample;
        const auto status = mr.run.pull(endpoint, /*timeout_ms=*/0, sample);
        if (status != neat::PullStatus::Ok) break;
        did_work = true;
        mark_frame(cfg, src);
      }
    }
    if (!did_work) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// Consecutive seconds of pull errors, with no frame in between, after which a model's graph is
// treated as dead. Clip loops do not produce pull errors at all (none in 300 s soaks with
// 31-38 s clips); a persistent error is a real fault such as a caps mismatch.
constexpr double kFatalPullErrorS = 10.0;

void pull_loop(int group, const Config& cfg, ModelRuntime& mr, std::vector<SourceRuntime>& sources,
               const std::array<int, kMaxStreams>& pos_of_slot) {
  const auto endpoint = detections_endpoint(group);
  const auto& model = cfg.models[static_cast<std::size_t>(group)];
  const bool pose = model.task == Task::Pose;
  std::vector<neat::Box> boxes;
  std::vector<float> kpts;
  boxes.reserve(static_cast<std::size_t>(cfg.max_detections));
  std::optional<Clock::time_point> failing_since;

  while (g_stop.load() == 0 && !g_pull_done.load()) {
    neat::Sample sample;
    neat::PullError perr;
    const auto status = mr.run.pull(endpoint, /*timeout_ms=*/20, sample, &perr);
    if (status == neat::PullStatus::Timeout) continue;
    if (status == neat::PullStatus::Closed) {
      // End of stream: Run.h guarantees no more samples will come on this endpoint.
      if (g_stop.load() == 0) {
        std::cerr << "[error] " << endpoint << " closed: " << mr.run.last_error() << "\n";
        request_fatal_stop();
      }
      return;
    }
    if (status == neat::PullStatus::Error) {
      const auto now = Clock::now();
      if (!failing_since) failing_since = now;
      const auto n = mr.pull_failures.fetch_add(1) + 1;
      if (n == 1 || n % 1000 == 0) {
        std::cerr << "[warn] pull " << endpoint << " failed (" << n << "): " << perr.message << "\n";
      }
      if (std::chrono::duration<double>(now - *failing_since).count() > kFatalPullErrorS) {
        std::cerr << "[error] " << endpoint << " has failed for " << kFatalPullErrorS
                  << " s without a frame; stopping: " << perr.message << "\n";
        request_fatal_stop();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    if (status != neat::PullStatus::Ok) continue;
    failing_since.reset();

    const int index = source_pos_from(sample, pos_of_slot, mr.sole_pos);
    if (index < 0 || sources[static_cast<std::size_t>(index)].group != group) {
      if (++mr.unattributed % 500 == 1) {
        std::cerr << "[warn] detection with unusable stream id '" << sample.stream_id << "' on "
                  << endpoint << " (count=" << mr.unattributed.load() << ")\n";
      }
      continue;
    }
    auto& src = sources[static_cast<std::size_t>(index)];

    boxes.clear();
    kpts.clear();
    const auto t_parse0 = Clock::now();
    try {
      const auto tensors = neat::tensors_from_sample(sample, false);
      if (!tensors.empty() && pose) {
        // boxes [N, 6] (x1, y1, x2, y2, score, class), keypoints [N, 17, 3], positionally aligned.
        const auto decoded = neat::decode_pose_tensor(tensors.front(), src.frame_w, src.frame_h,
                                                      cfg.max_detections, false);
        const neat::Tensor& bt = decoded.boxes;
        const neat::Tensor& kt = decoded.keypoints;
        const std::int64_t n = bt.shape.empty() ? 0 : bt.shape[0];
        if (n > 0) {
          if (bt.shape.size() != 2 || bt.shape[1] != 6 || kt.shape.size() != 3 ||
              kt.shape[0] != n || kt.shape[1] != kPoseKeypoints || kt.shape[2] != 3) {
            throw std::runtime_error("unexpected pose tensor shapes");
          }
        }
        // Throws unless the tensor is CPU-resident, dense and contiguous; honours byte_offset.
        const float* bp = n > 0 ? bt.data_ptr<float>() : nullptr;
        const float* kp = n > 0 ? kt.data_ptr<float>() : nullptr;
        for (std::int64_t bi = 0; bi < n; ++bi) {
          const float* r = bp + bi * 6;
          const float score = model.score_is_prob ? logitf(r[4]) : r[4];
          if (score < cfg.min_score) continue;
          neat::Box b;
          b.x1 = r[0]; b.y1 = r[1]; b.x2 = r[2]; b.y2 = r[3];
          b.score = score;
          b.class_id = static_cast<int>(r[5]);
          boxes.push_back(b);
          const float* k = kp + bi * kPoseKeypoints * 3;
          kpts.insert(kpts.end(), k, k + kPoseKeypoints * 3);
        }
      } else if (!tensors.empty()) {
        const auto decoded = neat::decode_bbox_tensor(tensors.front(), src.frame_w, src.frame_h,
                                                      cfg.max_detections, false);
        for (auto b : decoded.boxes) {
          if (model.score_is_prob) b.score = logitf(b.score);
          if (b.score >= cfg.min_score) boxes.push_back(b);
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "[warn] stream " << src.index << " decode: " << e.what() << "\n";
      continue;
    }
    const double parse_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_parse0).count();

    mark_frame(cfg, src);
    src.boxes_total += boxes.size();
    if (src.processed > static_cast<std::uint64_t>(cfg.warmup_frames)) {
      src.t_parse_ms += parse_ms;   // same post-warmup window as t_meta_ms / t_samples
      const auto t_meta0 = Clock::now();
      if (pose) send_pose_metadata(src, sample, boxes, kpts);
      else send_metadata(src, model, sample, boxes);
      src.t_meta_ms += std::chrono::duration<double, std::milli>(Clock::now() - t_meta0).count();
      ++src.t_samples;
    }
  }
}

// ── reporting ───────────────────────────────────────────────────────────────
struct ReportState {
  std::array<std::uint64_t, kMaxStreams> last_frames{};
  Clock::time_point last = Clock::now();
};

// Live report: fps over the last interval, read from the atomic heartbeat only.
void print_window(const std::vector<SourceRuntime>& sources, ReportState& st, double elapsed_s) {
  const auto now = Clock::now();
  const double dt = std::chrono::duration<double>(now - st.last).count();
  st.last = now;
  double total = 0.0;
  int input_fps = 0;
  for (const auto& s : sources) input_fps += s.frame_fps;
  std::ostringstream row;
  row << std::fixed << std::setprecision(1);
  int prev_group = -1;
  for (const auto& s : sources) {
    const auto i = static_cast<std::size_t>(s.index);
    const auto n = g_frames[i].load(std::memory_order_relaxed);
    const double fps = dt > 0 ? static_cast<double>(n - st.last_frames[i]) / dt : 0.0;
    st.last_frames[i] = n;
    total += fps;
    row << (s.group != prev_group ? "  | " : " ") << std::setw(4) << fps;
    prev_group = s.group;
  }
  std::cout << "[t=" << std::fixed << std::setprecision(0) << std::setw(4) << elapsed_s << "s] "
            << std::setprecision(1) << std::setw(6) << total << " fps (in "
            << input_fps << ")" << row.str() << "\n"
            << std::flush;
}

bool print_summary(const Config& cfg, const std::vector<SourceRuntime>& sources, double wall_s) {
  std::cout << "\n=== per-stream summary (window " << std::fixed << std::setprecision(1) << wall_s
            << "s, fps measured after " << cfg.warmup_frames << " warmup frames) ===\n"
            << std::left << std::setw(7) << "stream" << std::setw(10) << "model" << std::right
            << std::setw(9) << "frames" << std::setw(8) << "fps" << std::setw(8) << "in fps"
            << std::setw(8) << "ratio" << std::setw(9) << "objs/f" << std::setw(10) << "meta ok"
            << std::setw(7) << "fail" << std::setw(10) << "parse ms" << std::setw(9) << "meta ms"
            << "\n";
  double total_fps = 0.0;
  double min_ratio = 1e9;
  std::uint64_t total_frames = 0;
  for (const auto& s : sources) {
    const double secs =
        s.steady_started ? std::chrono::duration<double>(Clock::now() - s.steady_start).count() : 0;
    const auto steady = s.processed > static_cast<std::uint64_t>(cfg.warmup_frames)
                            ? s.processed - static_cast<std::uint64_t>(cfg.warmup_frames)
                            : 0;
    const double fps = secs > 0.1 ? static_cast<double>(steady) / secs : 0.0;
    const double ratio = s.frame_fps > 0 ? fps / s.frame_fps : 0.0;
    min_ratio = std::min(min_ratio, ratio);
    total_fps += fps;
    total_frames += s.processed;
    const double n = s.t_samples ? static_cast<double>(s.t_samples) : 1.0;
    std::cout << std::left << std::setw(7) << s.index << std::setw(10)
              << cfg.models[static_cast<std::size_t>(s.group)].name << std::right << std::setw(9)
              << s.processed << std::setw(8) << std::setprecision(2) << fps << std::setw(8)
              << s.frame_fps << std::setw(7) << std::setprecision(1) << ratio * 100.0 << "%"
              << std::setw(9) << std::setprecision(2)
              << (s.processed ? static_cast<double>(s.boxes_total) / s.processed : 0.0)
              << std::setw(10) << s.metadata_ok << std::setw(7) << s.metadata_fail
              << std::setw(10) << std::setprecision(3) << s.t_parse_ms / n << std::setw(9)
              << s.t_meta_ms / n << "\n";
  }
  double input_fps = 0.0;
  for (const auto& s : sources) input_fps += s.frame_fps;
  const bool pass = min_ratio >= cfg.target_ratio;
  std::cout << "\naggregate: " << std::fixed << std::setprecision(1) << total_fps << " fps of "
            << input_fps << " input (" << (input_fps > 0 ? total_fps / input_fps * 100.0 : 0.0)
            << "%) across " << sources.size() << " streams, " << total_frames << " frames\n"
            << "slowest stream: " << min_ratio * 100.0 << "% of source rate  ->  "
            << (pass ? "PASS" : "FAIL") << " (target >= " << cfg.target_ratio * 100.0
            << "% per stream)\n"
            << std::flush;
  return pass;
}

void print_measure(int m, const std::string& name, const neat::MeasureReport& rep) {
  std::cout << "\n##### model " << m << " (" << name << ") in-graph profile, window "
            << std::fixed << std::setprecision(1) << rep.elapsed_s << "s, outputs=" << rep.outputs
            << "\n";
  if (!rep.plugin_latency.empty()) {
    struct Agg { std::uint64_t calls = 0; double total = 0, max = 0; std::string backend; };
    std::map<std::string, Agg> by_stage;
    for (const auto& p : rep.plugin_latency) {
      const std::string key = p.stage_name.empty() ? (p.name.empty() ? p.kernel_name : p.name)
                                                   : p.stage_name;
      auto& a = by_stage[key];
      a.calls += p.calls;
      a.total += p.total_ms;
      a.max = std::max(a.max, p.max_ms);
      if (a.backend.empty()) a.backend = p.backend;
    }
    std::cout << std::left << std::setw(34) << "stage" << std::setw(10) << "backend" << std::right
              << std::setw(10) << "calls" << std::setw(10) << "avg ms" << std::setw(10)
              << "max ms" << "\n";
    for (const auto& [k, a] : by_stage) {
      std::cout << std::left << std::setw(34) << k.substr(0, 33) << std::setw(10)
                << a.backend.substr(0, 9) << std::right << std::setw(10) << a.calls
                << std::setw(10) << std::setprecision(3)
                << (a.calls ? a.total / static_cast<double>(a.calls) : 0.0) << std::setw(10)
                << a.max << "\n";
    }
  }
  std::cout << "end-to-end avg " << std::setprecision(2) << rep.end_to_end.avg_ms << " ms  p50 "
            << rep.end_to_end.p50_ms << "  p95 " << rep.end_to_end.p95_ms << "  max "
            << rep.end_to_end.max_ms << " ms\n";
}

void print_usage() {
  std::cout << "16stream4model -- 16 x 720p30 RTSP streams across 4 models\n"
            << "  --config <path>          config file (default ./config/default.conf)\n"
            << "  --duration <s>           stop after this many seconds; 0 = until Ctrl-C\n"
            << "  --report-interval <s>    live per-stream fps every N seconds (0 = off)\n"
            << "  --streams <n>            use only the first n streams\n"
            << "  --only-model <m>         run only the streams of model m (0-3)\n"
            << "  --visible <n>            publish only the first n streams to Insight\n"
            << "  --no-video               detections only, no video passthrough\n"
            << "  --source-only            decode every stream, run no model\n"
            << "  --inflight <per>,<total> override max_inflight_per_stream/_total\n"
            << "  --internal-queue <n>     GraphOptions internal_queue_depth (-1 = default)\n"
            << "  --measure [m]            Neat in-graph timing; plugin trace on model m\n"
            << "  --verbose                unmute the Neat model planner\n"
            << "  --print-backend          dump the generated GStreamer backend\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  // Leave SIGHUP alone when it is already ignored, so `nohup` keeps a detached soak running
  // after the SSH session drops.
  struct sigaction old_hup {};
  if (sigaction(SIGHUP, nullptr, &old_hup) == 0 && old_hup.sa_handler != SIG_IGN) {
    std::signal(SIGHUP, handle_signal);
  }

  std::string config_path = "./config/default.conf";
  std::vector<std::string> args(argv + 1, argv + argc);
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--help" || args[i] == "-h") { print_usage(); return 0; }
    if (args[i] == "--config") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --config needs a value\n";
        return 1;
      }
      config_path = args[++i];
    }
  }

  try {
    Config cfg = read_config(config_path);
    int stream_limit = -1;
    int only_model = -1;
    for (std::size_t i = 0; i < args.size(); ++i) {
      const auto& a = args[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= args.size()) throw std::runtime_error(a + " needs a value");
        return args[++i];
      };
      if (a == "--config") ++i;
      else if (a == "--duration") cfg.duration_s = parse_double(a, next());
      else if (a == "--report-interval") cfg.report_interval_s = parse_double(a, next());
      else if (a == "--streams") stream_limit = parse_int(a, next());
      else if (a == "--only-model") only_model = parse_int(a, next());
      else if (a == "--visible") cfg.visible_streams = parse_int(a, next());
      else if (a == "--no-video") cfg.video_enabled = false;
      else if (a == "--source-only") cfg.source_only = true;
      else if (a == "--internal-queue") cfg.internal_queue_depth = parse_int(a, next());
      else if (a == "--inflight") {
        const auto v = next();
        const auto comma = v.find(',');
        cfg.max_inflight_per_stream = parse_int(a, v.substr(0, comma));
        if (comma != std::string::npos) cfg.max_inflight_total = parse_int(a, v.substr(comma + 1));
      }
      else if (a == "--measure") {
        cfg.measure = true;
        if (i + 1 < args.size() && std::isdigit(static_cast<unsigned char>(args[i + 1][0]))) {
          cfg.measure_model = parse_int(a, args[++i]);
        }
      }
      else if (a == "--verbose") cfg.verbose = true;
      else if (a == "--print-backend") cfg.print_backend = true;
      else throw std::runtime_error("unknown argument: " + a + " (see --help)");
    }
    if (stream_limit == 0 || stream_limit < -1) throw std::runtime_error("--streams must be >= 1");
    if (only_model < -1 || only_model >= kNumModels) {
      throw std::runtime_error("--only-model must be 0.." + std::to_string(kNumModels - 1));
    }
    if (cfg.measure_model < 0 || cfg.measure_model >= kNumModels) {
      throw std::runtime_error("--measure model must be 0.." + std::to_string(kNumModels - 1));
    }
    if (stream_limit > 0 && stream_limit < static_cast<int>(cfg.streams.size())) {
      cfg.streams.resize(static_cast<std::size_t>(stream_limit));
    }
    validate(cfg);
    // Filtering keeps each stream's config slot (StreamCfg::slot), so a stream keeps its Insight
    // channel, ports and stream id no matter which subset runs.
    if (only_model >= 0) {
      std::vector<StreamCfg> kept;
      for (const auto& sc : cfg.streams) {
        if (sc.group == only_model) kept.push_back(sc);
      }
      if (kept.empty()) throw std::runtime_error("--only-model matches no configured stream");
      cfg.streams = std::move(kept);
    }
    const int stream_count = static_cast<int>(cfg.streams.size());

    std::array<ModelRuntime, kNumModels> models;
    std::array<int, kMaxStreams> pos_of_slot{};
    pos_of_slot.fill(-1);
    for (int i = 0; i < stream_count; ++i) {
      const auto& sc = cfg.streams[static_cast<std::size_t>(i)];
      auto& mr = models[static_cast<std::size_t>(sc.group)];
      mr.sole_pos = (mr.stream_count == 0) ? i : -1;
      ++mr.stream_count;
      pos_of_slot[static_cast<std::size_t>(sc.slot)] = i;
    }

    neat::GraphOptions graph_options;
    if (cfg.internal_queue_depth >= 0) {
      graph_options.advanced_execution.internal_queue_depth = cfg.internal_queue_depth;
    }
    graph_options.advanced_execution.inference_async = cfg.inference_async;

    for (int m = 0; m < kNumModels; ++m) {
      auto& mr = models[static_cast<std::size_t>(m)];
      if (mr.stream_count == 0) continue;
      // GraphOptions belong on the final composition owner, not on the model fragment.
      mr.graph = neat::Graph(graph_options);
      if (!cfg.source_only) mr.detector = build_detector_graph(cfg, m, mr.model);
      const auto& g = cfg.models[static_cast<std::size_t>(m)];
      std::cout << "Model " << m << ": " << std::left << std::setw(14) << g.name
                << (g.task == Task::Pose ? "pose       " : "detection  ") << std::setw(4)
                << g.num_classes << std::right << "classes  " << mr.stream_count << " stream(s)  "
                << g.archive << "\n";
    }

    std::vector<SourceRuntime> sources;
    sources.reserve(cfg.streams.size());
    for (int i = 0; i < stream_count; ++i) {
      const auto& sc = cfg.streams[static_cast<std::size_t>(i)];
      SourceRuntime src;
      src.index = sc.slot;
      src.group = sc.group;
      src.url = sc.url;
      src.frame_w = cfg.width;
      src.frame_h = cfg.height;
      src.frame_fps = sc.fps;
      src.source_options = make_source_options(cfg, sc);
      const bool visible = cfg.visible_streams < 0 || sc.slot < cfg.visible_streams;
      if (visible && !cfg.source_only) {
        neat::MetadataSenderOptions mo;
        mo.host = cfg.insight_host;
        mo.channel = sc.slot;
        mo.metadata_port_base = cfg.metadata_port_base;
        neat::MetadataSenderSendOptions so;
        so.nonblocking = true;   // a congested Insight channel must never block inference
        std::string err;
        src.metadata_sender = std::make_unique<neat::MetadataSender>(mo, so, &err);
        if (!src.metadata_sender->ok()) throw std::runtime_error("metadata sender: " + err);
      }
      sources.push_back(std::move(src));
    }

    for (auto& src : sources) {
      auto& mr = models[static_cast<std::size_t>(src.group)];
      connect_source(mr.graph, cfg, src, mr.detector);
      std::cout << "[stream " << std::setw(2) << src.index << "] "
                << cfg.models[static_cast<std::size_t>(src.group)].name << "  " << src.url << "  "
                << src.frame_w << "x" << src.frame_h << "@" << src.frame_fps << "  video=";
      if (src.video_port > 0) std::cout << src.video_port; else std::cout << "off";
      std::cout << " metadata=";
      if (src.metadata_sender) std::cout << src.metadata_sender->metadata_port();
      else std::cout << "off";
      std::cout << "\n";
    }

    if (cfg.print_backend) {
      for (int m = 0; m < kNumModels; ++m) {
        if (models[static_cast<std::size_t>(m)].stream_count == 0) continue;
        std::cout << "--- model " << m << " graph\n"
                  << models[static_cast<std::size_t>(m)].graph.describe_backend() << "\n";
      }
    }

    neat::RunOptions run_options;
    run_options.preset = neat::RunPreset::Realtime;
    run_options.queue_depth = cfg.queue_depth;
    run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
    run_options.output_memory = neat::OutputMemory::ZeroCopy;

    std::cout << "config: decoder_buffers=" << cfg.decoder_buffers
              << " queue_depth=" << cfg.queue_depth
              << " internal_queue_depth=" << cfg.internal_queue_depth
              << " inference_async=" << cfg.inference_async
              << " inflight=" << cfg.max_inflight_per_stream << "/" << cfg.max_inflight_total
              << " detections_drop=" << cfg.detections_drop << "\n"
              << "Starting " << stream_count << " streams. Ctrl-C to stop.\n"
              << std::flush;

    // Joins the pullers on every exit path, including a later build() throwing. Declared after
    // everything a puller references (cfg, models, sources, pos_of_slot) so it is destroyed, and
    // joins, before any of them.
    struct PullerGuard {
      std::array<ModelRuntime, kNumModels>& models;
      ~PullerGuard() {
        g_pull_done.store(true);
        for (auto& mr : models) {
          if (mr.puller.joinable()) mr.puller.join();
        }
      }
    } puller_guard{models};

    // Start each puller the moment its own Run is built (header note 4).
    for (int m = 0; m < kNumModels; ++m) {
      auto& mr = models[static_cast<std::size_t>(m)];
      if (mr.stream_count == 0) continue;
      mr.run = mr.graph.build(run_options);
      mr.puller = std::thread([m, &cfg, &mr, &sources, &pos_of_slot]() {
        try {
          if (cfg.source_only) source_only_loop(m, cfg, mr, sources);
          else pull_loop(m, cfg, mr, sources, pos_of_slot);
        } catch (const std::exception& e) {
          std::cerr << "[error] model " << m << " puller: " << e.what() << "\n";
          request_fatal_stop();
        }
      });
    }

    const auto start = Clock::now();
    const auto deadline =
        cfg.duration_s > 0.0
            ? start + std::chrono::milliseconds(static_cast<long>(cfg.duration_s * 1000.0))
            : Clock::time_point::max();
    auto next_report =
        cfg.report_interval_s > 0.0
            ? start + std::chrono::milliseconds(static_cast<long>(cfg.report_interval_s * 1000.0))
            : Clock::time_point::max();

    // Only one LTTng collector may be active per process, so plugin tracing goes on one model;
    // the others report end-to-end only.
    std::array<std::optional<neat::MeasureScope>, kNumModels> scopes;
    if (cfg.measure && !cfg.source_only) {
      for (int m = 0; m < kNumModels; ++m) {
        auto& mr = models[static_cast<std::size_t>(m)];
        if (mr.stream_count == 0) continue;
        neat::MeasureOptions mo;
        mo.duration_ms = static_cast<int>(cfg.duration_s > 0 ? cfg.duration_s * 1000.0 : 30000.0);
        mo.warmup_ms = 3000;
        mo.include_plugin_latency = (m == cfg.measure_model);
        mo.include_edge_latency = false;
        mo.title = "16stream4model / " + cfg.models[static_cast<std::size_t>(m)].name;
        mo.input = std::to_string(mr.stream_count) + " RTSP streams";
        scopes[static_cast<std::size_t>(m)].emplace(mr.run.start_measurement(mo));
      }
    }

    ReportState report_state;
    std::array<std::uint64_t, kMaxStreams> seen{};
    std::array<Clock::time_point, kMaxStreams> last_change{};
    std::array<bool, kMaxStreams> stalled{};
    last_change.fill(start);

    while (g_stop.load() == 0 && Clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      const auto now = Clock::now();
      for (const auto& s : sources) {
        const auto i = static_cast<std::size_t>(s.index);
        const auto n = g_frames[i].load(std::memory_order_relaxed);
        if (n != seen[i]) {
          if (stalled[i]) {
            std::cout << "[resume] stream " << s.index << " after "
                      << std::chrono::duration<double>(now - last_change[i]).count() << " s\n";
          }
          seen[i] = n;
          last_change[i] = now;
          stalled[i] = false;
        } else if (!stalled[i] &&
                   std::chrono::duration<double>(now - last_change[i]).count() > cfg.stall_warn_s) {
          stalled[i] = true;
          std::cout << "[stall] stream " << s.index
                    << (n == 0 ? " has produced no frame yet after " : " no frame for ")
                    << cfg.stall_warn_s << " s\n" << std::flush;
        }
      }
      if (now >= next_report) {
        print_window(sources, report_state,
                     std::chrono::duration<double>(now - start).count());
        next_report = now + std::chrono::milliseconds(
                                static_cast<long>(cfg.report_interval_s * 1000.0));
      }
    }

    g_pull_done.store(true);
    for (auto& mr : models) {
      if (mr.puller.joinable()) mr.puller.join();
    }

    const bool pass =
        print_summary(cfg, sources, std::chrono::duration<double>(Clock::now() - start).count());
    for (int m = 0; m < kNumModels; ++m) {
      const auto& mr = models[static_cast<std::size_t>(m)];
      if (mr.pull_failures.load() || mr.unattributed.load()) {
        std::cout << "model " << m << ": pull errors=" << mr.pull_failures.load()
                  << " unattributed=" << mr.unattributed.load() << "\n";
      }
    }
    for (int m = 0; m < kNumModels; ++m) {
      auto& sc = scopes[static_cast<std::size_t>(m)];
      if (!sc.has_value()) continue;
      try {
        print_measure(m, cfg.models[static_cast<std::size_t>(m)].name, sc->stop());
      } catch (const std::exception& e) {
        std::cerr << "[warn] measurement stop failed for model " << m << ": " << e.what() << "\n";
      }
      sc.reset();
    }

    for (auto& mr : models) {
      if (mr.stream_count > 0) mr.run.close();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // 1 = the pipeline failed, 130 = stopped by a signal (a shortened window is not a verdict),
    // otherwise the fps verdict: 0 = PASS, 3 = FAIL.
    if (g_stop.load() == kStopFatal) return 1;
    if (g_stop.load() == kStopSignal) return 130;
    return pass ? 0 : 3;
  } catch (const neat::NeatError& e) {
    std::cerr << "NEAT error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
