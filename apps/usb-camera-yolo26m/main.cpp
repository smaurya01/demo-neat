// USB camera (UVC/MJPEG) -> YOLO26m object detection -> H.264 RTP/UDP + detection metadata
//
// Two pipeline modes, selected by `pipeline_mode` in the config.
//
// ── pipeline_mode=push (default, works on runtime 0.2.2) ──────────────────────
//
//   source graph:  Custom(v4l2src ! jpegparse ! jpegdec ! videoconvert ! NV12) -> Output("frame")
//   model  graph:  Input("image") -> Model(yolo26m)  -> Output("detections")
//   udp    graph:  Input("video") -> VideoSender (H264EncodeSima -> RTP -> UDP)
//
//   The app pulls each NV12 frame, pushes it into the model graph (appsrc), pulls the
//   boxes, draws them onto the frame, and pushes the annotated frame to the encoder.
//   Pushing through appsrc is what lands the frame in SiMa DMA memory, which the CVU
//   requires -- see SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY below.
//
// ── pipeline_mode=graph (zero-copy, needs a newer runtime) ────────────────────
//
//   Custom(camera) -> Branch -+-> VideoSender
//                             +-> Model -> Output("detections")
//
//   Strictly better on paper: the frame never touches the CPU. But on runtime 0.2.2 the
//   CVU silently reads system-memory buffers as black frames and yields zero detections,
//   because the private `neatcamerabridge` element that lands OS buffers into SiMa DMA
//   memory does not exist in libsima_neat.so.2.1.2. Set camera_bridge=true to append it
//   once you are on a runtime that ships it. See LEARNING.md.
//
// MJPEG is decoded on the CPU (jpegdec), NOT on the SiMa hardware decoder. That is
// deliberate and measured: neatdecoder in mjpeg mode runs at ~4 fps on this camera's
// JPEGs, while jpegdec sustains the camera's full rate. See LEARNING.md.

#include <neat.h>

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr const char* kConfigPath = "./config/default.conf";

std::atomic<bool> g_stop{false};
void handle_signal(int) { g_stop.store(true); }

struct Config {
  // Camera
  std::string camera_device = "/dev/video16";
  int width = 1920;
  int height = 1080;
  int fps = 30;

  // Model
  std::string model_path = "./assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz";
  int model_width = 640;
  int model_height = 640;
  float score_threshold = 0.30f;
  float nms_iou = 0.50f;
  int top_k = 100;
  int num_classes = 80;

  // Output
  std::string udp_host;
  int udp_port = 5205;
  int bitrate_kbps = 4000;
  std::string metadata_host;
  int metadata_port = 9100;

  // Runtime
  std::string pipeline_mode = "push";  // push | graph
  std::string flip = "none";           // none | rotate-180 | horizontal-flip | vertical-flip
  bool overlay = true;                 // draw boxes onto the streamed video (push mode only)
  int frames = 0;                      // 0 = run until Ctrl-C
  double profile_interval = 1.0;       // seconds between profile lines; 0 = off
  int queue_depth = 3;
  bool print_backend = false;
  bool verbose_planner = false;        // dump the MPK contract + route/fusion decisions
  bool camera_bridge = false;          // append neatcamerabridge (absent in runtime 0.2.2)

  // Diagnostic: replace the camera with a looping still image, so the model and the
  // preprocessor can be validated against a picture with a known answer. The fragment
  // must end producing NV12 at width x height. Empty = use the real camera.
  std::string source_override;
};

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// ── time profiling ────────────────────────────────────────────────────────────
//
// Two different means, on purpose:
//
//   * the periodic line reports the mean over THAT WINDOW only. A cumulative mean
//     silently hides a pipeline that degrades halfway through a run -- it just drifts
//     slowly, and by frame 6000 one bad minute is invisible.
//   * the exit summary reports mean and p95 over the whole run. p95 is what tells you
//     whether a stage is *occasionally* slow, which a mean never will.
//
// Memory is O(1): the mean comes from a running sum, and p95 from a bounded ring of the
// most recent samples. A `frames=0` run streams for hours; an unbounded sample vector
// would grow without limit.
class StageStat {
public:
  void add(double ms) {
    sum_ += ms;
    ++count_;
    win_sum_ += ms;
    ++win_count_;
    if (ring_.size() < kRing) {
      ring_.push_back(ms);
    } else {
      ring_[ring_pos_] = ms;
      ring_pos_ = (ring_pos_ + 1) % kRing;
    }
  }

  double window_mean() const {
    return win_count_ ? win_sum_ / static_cast<double>(win_count_) : 0.0;
  }
  double mean() const { return count_ ? sum_ / static_cast<double>(count_) : 0.0; }

  /// p95 over the most recent `kRing` samples (~11 min at 30 fps).
  double p95() const {
    if (ring_.empty()) return 0.0;
    std::vector<double> v = ring_;
    const auto k = static_cast<std::size_t>(0.95 * static_cast<double>(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    return v[k];
  }

  void reset_window() {
    win_sum_ = 0.0;
    win_count_ = 0;
  }

private:
  static constexpr std::size_t kRing = 20000;

  double sum_ = 0.0;
  std::uint64_t count_ = 0;
  double win_sum_ = 0.0;
  std::uint64_t win_count_ = 0;
  std::vector<double> ring_;
  std::size_t ring_pos_ = 0;
};

struct Profile {
  StageStat capture;  ///< blocking wait for the next camera frame
  StageStat infer;    ///< CVU preprocess + MLA + EV74 box decode
  StageStat overlay;  ///< NV12 copy + box drawing (CPU)
  StageStat encode;   ///< push into the H.264 encoder graph
  StageStat total;    ///< the whole per-frame loop

  void reset_windows() {
    capture.reset_window();
    infer.reset_window();
    overlay.reset_window();
    encode.reset_window();
    total.reset_window();
  }
};

void print_profile_summary(const Profile& p, int frames, double elapsed_s) {
  if (frames < 2 || elapsed_s <= 0.0) return;

  const double fps = static_cast<double>(frames - 1) / elapsed_s;

  std::cout << "\n── time profile ──────────────────────────────\n"
            << std::left << std::setw(10) << "stage" << std::right << std::setw(10)
            << "mean ms" << std::setw(10) << "p95 ms" << "\n";

  const auto row = [](const char* name, const StageStat& s) {
    std::cout << std::left << std::setw(10) << name << std::right << std::fixed
              << std::setprecision(2) << std::setw(10) << s.mean() << std::setw(10)
              << s.p95() << "\n";
  };
  row("capture", p.capture);
  row("infer", p.infer);
  row("overlay", p.overlay);
  row("encode", p.encode);
  row("total", p.total);

  std::cout << "\nframes " << frames << "   elapsed " << std::fixed << std::setprecision(1)
            << elapsed_s << "s   steady-state " << std::setprecision(2) << fps << " fps\n";

  // Say plainly what is holding the pipeline back. `infer` is the only stage that runs on
  // the accelerator, so its mean is the MLA's ceiling. If we are delivering well under
  // that, something upstream (the camera) is the constraint -- not the SoC.
  const double infer_ms = p.infer.mean();
  if (infer_ms > 0.0) {
    const double mla_ceiling = 1000.0 / infer_ms;
    if (mla_ceiling > fps * 1.05) {
      std::cout << "bottleneck: THE CAMERA. Inference takes " << std::setprecision(1)
                << infer_ms << " ms, so the MLA could sustain ~" << mla_ceiling
                << " fps; you are getting " << fps
                << ". A smaller/faster model will not help.\n";
    } else {
      std::cout << "bottleneck: INFERENCE. The MLA tops out near " << std::setprecision(1)
                << mla_ceiling << " fps and you are delivering " << fps
                << ". A smaller model would raise this.\n";
    }
  }
}

std::string trim(std::string v) {
  const auto b = v.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  const auto e = v.find_last_not_of(" \t\r\n");
  return v.substr(b, e - b + 1);
}

bool to_bool(const std::string& v) { return v == "true" || v == "1" || v == "yes"; }

void set_config_value(Config& cfg, const std::string& k, const std::string& v) {
  if (k == "camera_device")        cfg.camera_device = v;
  else if (k == "width")           cfg.width = std::stoi(v);
  else if (k == "height")          cfg.height = std::stoi(v);
  else if (k == "fps")             cfg.fps = std::stoi(v);
  else if (k == "model_path")      cfg.model_path = v;
  else if (k == "model_width")     cfg.model_width = std::stoi(v);
  else if (k == "model_height")    cfg.model_height = std::stoi(v);
  else if (k == "score_threshold") cfg.score_threshold = std::stof(v);
  else if (k == "nms_iou")         cfg.nms_iou = std::stof(v);
  else if (k == "top_k")           cfg.top_k = std::stoi(v);
  else if (k == "num_classes")     cfg.num_classes = std::stoi(v);
  else if (k == "udp_host")        cfg.udp_host = v;
  else if (k == "udp_port")        cfg.udp_port = std::stoi(v);
  else if (k == "bitrate_kbps")    cfg.bitrate_kbps = std::stoi(v);
  else if (k == "metadata_host")   cfg.metadata_host = v;
  else if (k == "metadata_port")   cfg.metadata_port = std::stoi(v);
  else if (k == "pipeline_mode")   cfg.pipeline_mode = v;
  else if (k == "flip")            cfg.flip = v;
  else if (k == "overlay")         cfg.overlay = to_bool(v);
  else if (k == "frames")          cfg.frames = std::stoi(v);
  else if (k == "profile_interval") cfg.profile_interval = std::stod(v);
  else if (k == "queue_depth")     cfg.queue_depth = std::stoi(v);
  else if (k == "print_backend")   cfg.print_backend = to_bool(v);
  else if (k == "verbose_planner") cfg.verbose_planner = to_bool(v);
  else if (k == "camera_bridge")   cfg.camera_bridge = to_bool(v);
  else if (k == "source_override") cfg.source_override = v;
  else throw std::runtime_error("unknown config key: " + k);
}

Config read_config(const std::string& path) {
  Config cfg;
  std::ifstream file(path);
  if (!file) throw std::runtime_error("config file not found: " + path);

  std::string line;
  while (std::getline(file, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    line = trim(line);
    if (line.empty()) continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    set_config_value(cfg, trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
  }
  if (cfg.udp_host.empty()) throw std::runtime_error("config missing: udp_host");
  if (cfg.pipeline_mode != "push" && cfg.pipeline_mode != "graph")
    throw std::runtime_error("pipeline_mode must be push or graph");
  return cfg;
}

// ── Camera source fragment ────────────────────────────────────────────────────
//
// Neat has no V4L2 source node, so this is emitted through the Custom() escape hatch.
//
//   io-mode=mmap      zero-copy DMA from the UVC driver (io-mode=rw memcpys every frame)
//   image/jpeg caps   pins MJPEG. Without it v4l2src negotiates YUYV, which the Brio 100
//                     only offers at 5 fps for 1080p (USB 2.0 bandwidth limit).
//   queue leaky       drop stale frames rather than stall the camera when the MLA is busy
//   jpegparse         frame the JPEG stream and fix up caps
//   jpegdec           CPU MJPEG decode -- faster than the SiMa HW decoder here
//   videoconvert      I420 (jpegdec's native output) -> NV12, consumed by CVU and encoder
//
// The fragment must not end on a bare caps string: gst_parse_launch parses a trailing
// "video/x-raw,..." as an element name and fails with `no element "video"`. Terminating
// on a real element keeps the caps a capsfilter.
std::string camera_fragment(const Config& cfg) {
  if (!cfg.source_override.empty()) return cfg.source_override;

  std::ostringstream ss;
  ss << "v4l2src device=" << cfg.camera_device << " io-mode=mmap"
     << " ! image/jpeg,width=" << cfg.width
     << ",height=" << cfg.height
     << ",framerate=" << cfg.fps << "/1"
     << " ! queue leaky=downstream max-size-buffers=2"
     << " ! jpegparse"
     << " ! jpegdec";
  // COCO models are trained on upright scenes and lose a lot of confidence on an inverted
  // one. If the camera is mounted upside-down, correcting it here is worth the CPU.
  if (cfg.flip != "none") {
    ss << " ! videoflip method=" << cfg.flip;
  }
  ss << " ! videoconvert n-threads=4"
     << " ! video/x-raw,format=NV12,width=" << cfg.width
     << ",height=" << cfg.height
     << ",framerate=" << cfg.fps << "/1";
  if (cfg.camera_bridge) {
    ss << " ! neatcamerabridge buffer-name=camera num-buffers=4 copy-allowed=true";
  }
  ss << " ! queue leaky=downstream max-size-buffers=2";
  return ss.str();
}

std::unique_ptr<neat::Model> make_model(const Config& cfg) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.input_max_width = cfg.width;
  opt.preprocess.input_max_height = cfg.height;
  opt.preprocess.input_max_depth = 1;
  opt.preprocess.resize.width = cfg.model_width;
  opt.preprocess.resize.height = cfg.model_height;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;

  // YOLO26 uses NMS-free raw l/t/r/b distance heads. Neat decodes these on the EV74.
  opt.decode_type = neat::BoxDecodeType::YoloV26;
  opt.score_threshold = cfg.score_threshold;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.top_k;
  opt.num_classes = cfg.num_classes;
  // Dumps the MPK contract and the planner's route decisions -- how it maps the packaged
  // stages onto CVU/MLA nodes, and what it fuses into boxdecode. This is what shows, e.g.,
  // `post_fusion=user_boxdecode(cast+detess+dequant)->boxdecode` for the int8 package.
  if (cfg.verbose_planner) {
    opt.verbose.level = neat::VerbosityLevel::Verbose;
    opt.verbose.planner = true;
  }
  return std::make_unique<neat::Model>(cfg.model_path, opt);
}

neat::InputOptions make_nv12_input_options(const Config& cfg) {
  neat::InputOptions opt;
  opt.payload_type = neat::PayloadType::Image;
  opt.format = neat::FormatTag::NV12;
  opt.width = cfg.width;
  opt.height = cfg.height;
  opt.depth = 1;
  opt.max_width = cfg.width;
  opt.max_height = cfg.height;
  opt.max_depth = 1;
  opt.fps_n = cfg.fps;
  opt.fps_d = 1;
  opt.caps_override = "video/x-raw,format=NV12,width=" + std::to_string(cfg.width) +
                      ",height=" + std::to_string(cfg.height) +
                      ",framerate=" + std::to_string(cfg.fps) + "/1";
  opt.use_simaai_pool = false;
  return opt;
}

groups::VideoSenderOptions make_video_options(const Config& cfg) {
  auto opt = groups::VideoSenderOptions::H264RtpUdpFromRaw(cfg.width, cfg.height, cfg.fps);
  opt.host = cfg.udp_host;
  opt.channel = 0;
  opt.video_port_base = cfg.udp_port;
  opt.encoder.bitrate_kbps = cfg.bitrate_kbps;
  return opt;
}

// ── COCO labels ───────────────────────────────────────────────────────────────

const std::vector<std::string>& coco_labels() {
  static const std::vector<std::string> labels = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
      "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
      "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
      "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
      "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
      "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
      "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
      "toothbrush"};
  return labels;
}

std::string class_label(int id) {
  const auto& l = coco_labels();
  if (id >= 0 && id < static_cast<int>(l.size())) return l[static_cast<std::size_t>(id)];
  return "class_" + std::to_string(id);
}

// ── NV12 helpers ──────────────────────────────────────────────────────────────

bool infer_dims(const neat::Tensor& t, int& w, int& h) {
  w = t.width();
  h = t.height();
  if ((w <= 0 || h <= 0) && t.shape.size() >= 2) {
    h = static_cast<int>(t.shape[0]);
    w = static_cast<int>(t.shape[1]);
  }
  return w > 0 && h > 0;
}

bool init_nv12_meta(neat::Tensor& out, int w, int h, std::string& err) {
  if (w <= 0 || h <= 0 || (w % 2) || (h % 2)) {
    err = "NV12 requires positive even width/height";
    return false;
  }
  out.dtype = neat::TensorDType::UInt8;
  out.layout = neat::TensorLayout::HW;
  out.shape = {h, w};
  out.strides_bytes = {w, 1};
  out.byte_offset = 0;
  out.device = {neat::DeviceType::CPU, 0};
  out.read_only = false;

  neat::ImageSpec image;
  image.format = neat::ImageSpec::PixelFormat::NV12;
  out.semantic.image = image;

  neat::Plane y;
  y.role = neat::PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  neat::Plane uv;
  uv.role = neat::PlaneRole::UV;
  uv.shape = {h / 2, w};
  uv.strides_bytes = {w, 1};
  uv.byte_offset = static_cast<std::int64_t>(w) * h;

  out.planes.clear();
  out.planes.push_back(std::move(y));
  out.planes.push_back(std::move(uv));
  return true;
}

neat::Tensor blank_nv12(int w, int h) {
  const std::size_t bytes = static_cast<std::size_t>(w) * h * 3U / 2U;
  neat::Tensor t;
  t.storage = neat::make_cpu_owned_storage(bytes);
  std::memset(t.storage->data, 0, bytes);
  std::string err;
  if (!init_nv12_meta(t, w, h, err)) throw std::runtime_error(err);
  return t;
}

neat::Tensor copy_nv12_to_cpu(const neat::Tensor& in) {
  int w = 0, h = 0;
  if (!in.is_nv12()) throw std::runtime_error("expected NV12 tensor");
  if (!infer_dims(in, w, h)) throw std::runtime_error("invalid NV12 dimensions");

  const auto bytes = in.copy_nv12_contiguous();
  if (bytes.empty()) throw std::runtime_error("NV12 copy failed");

  neat::Tensor out;
  out.storage = neat::make_cpu_owned_storage(bytes.size());
  std::memcpy(out.storage->data, bytes.data(), bytes.size());
  std::string err;
  if (!init_nv12_meta(out, w, h, err)) throw std::runtime_error(err);
  return out;
}

void fill_nv12_rect(std::uint8_t* y, std::uint8_t* uv, int w, int h, int x1, int y1, int x2,
                    int y2, std::uint8_t yv, std::uint8_t uvu, std::uint8_t uvv) {
  x1 = std::clamp(x1, 0, w);
  x2 = std::clamp(x2, 0, w);
  y1 = std::clamp(y1, 0, h);
  y2 = std::clamp(y2, 0, h);
  if (x2 <= x1 || y2 <= y1) return;

  for (int r = y1; r < y2; ++r)
    std::memset(y + static_cast<std::size_t>(r) * w + x1, yv,
                static_cast<std::size_t>(x2 - x1));

  for (int r = y1 / 2; r < (y2 + 1) / 2 && r < h / 2; ++r) {
    auto* row = uv + static_cast<std::size_t>(r) * w;
    for (int c = (x1 & ~1); c + 1 < std::min(w, (x2 + 1) & ~1); c += 2) {
      row[c] = uvu;
      row[c + 1] = uvv;
    }
  }
}

// ── label rendering ───────────────────────────────────────────────────────────
//
// A 5x7 bitmap font, drawn straight onto the NV12 planes. There is no OpenCV in this
// app (and no font on the board), so the glyphs are hand-coded -- the same approach the
// other single-stream apps use.

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
  case '.': return {"00000", "00000", "00000", "00000", "00000", "01100", "01100"};
  case '-': return {"00000", "00000", "00000", "11111", "00000", "00000", "00000"};
  case ' ': return {"00000", "00000", "00000", "00000", "00000", "00000", "00000"};
  default:  return {"11111", "00001", "00010", "00100", "00100", "00000", "00100"};  // '?'
  }
}

char overlay_char(char c) {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
}

void draw_nv12_text(std::uint8_t* y, std::uint8_t* uv, int w, int h, int x, int y0,
                    const std::string& text, std::uint8_t yv, std::uint8_t u,
                    std::uint8_t v, int scale) {
  int cx = x;
  for (const char raw : text) {
    const auto glyph = glyph_for(overlay_char(raw));
    for (int row = 0; row < 7; ++row) {
      for (int col = 0; col < 5; ++col) {
        if (glyph[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] == '1') {
          fill_nv12_rect(y, uv, w, h, cx + col * scale, y0 + row * scale,
                         cx + (col + 1) * scale, y0 + (row + 1) * scale, yv, u, v);
        }
      }
    }
    cx += 6 * scale;  // 5px glyph + 1px gap
    if (cx >= w - 6 * scale) break;
  }
}

void draw_boxes_on_nv12(neat::Tensor& frame, const std::vector<neat::Box>& boxes,
                        float min_score) {
  int w = 0, h = 0;
  if (!infer_dims(frame, w, h) || !frame.storage || !frame.storage->data) return;

  auto* base = static_cast<std::uint8_t*>(frame.storage->data);
  auto* y = base;
  auto* uv = base + static_cast<std::size_t>(w) * h;

  constexpr std::uint8_t kY = 76, kU = 84, kV = 255;      // red   — box + label bar
  constexpr std::uint8_t kTY = 235, kTU = 128, kTV = 128; // white — label text
  constexpr int kT = 3;

  // 5x7 glyphs are unreadable at 1080p unscaled. Scale with the frame: 3 at 1080p.
  const int scale = std::max(2, h / 360);
  const int glyph_h = 7 * scale;
  const int glyph_w = 6 * scale;
  const int pad = scale;

  for (const auto& b : boxes) {
    if (b.score < min_score) continue;
    const int x1 = std::max(0, static_cast<int>(b.x1));
    const int y1 = std::max(0, static_cast<int>(b.y1));
    const int x2 = std::min(w - 1, static_cast<int>(b.x2));
    const int y2 = std::min(h - 1, static_cast<int>(b.y2));
    if (x2 <= x1 || y2 <= y1) continue;

    fill_nv12_rect(y, uv, w, h, x1, y1, x2 + 1, y1 + kT, kY, kU, kV);
    fill_nv12_rect(y, uv, w, h, x1, y2 - kT + 1, x2 + 1, y2 + 1, kY, kU, kV);
    fill_nv12_rect(y, uv, w, h, x1, y1, x1 + kT, y2 + 1, kY, kU, kV);
    fill_nv12_rect(y, uv, w, h, x2 - kT + 1, y1, x2 + 1, y2 + 1, kY, kU, kV);

    // "PERSON 0.93"
    char score_buf[8];
    std::snprintf(score_buf, sizeof(score_buf), "%.2f", b.score);
    const std::string label = class_label(b.class_id) + " " + score_buf;

    const int bar_w = std::min(static_cast<int>(label.size()) * glyph_w + 2 * pad, w - x1);
    const int bar_h = glyph_h + 2 * pad;

    // Above the box by default; if there is no room up there, drop it just inside the top
    // edge so a detection touching the top of the frame still gets a readable label.
    const int bar_y = (y1 - bar_h >= 0) ? y1 - bar_h : y1 + kT;

    // Filled bar first: white glyphs on a bright background would be invisible.
    fill_nv12_rect(y, uv, w, h, x1, bar_y, x1 + bar_w, bar_y + bar_h, kY, kU, kV);
    draw_nv12_text(y, uv, w, h, x1 + pad, bar_y + pad, label, kTY, kTU, kTV, scale);
  }
}

std::vector<neat::Box> decode_boxes(const neat::Sample& sample, const Config& cfg) {
  const auto tensors = neat::tensors_from_sample(sample, true);
  if (tensors.empty()) return {};
  return neat::decode_bbox_tensor(tensors.front(), cfg.width, cfg.height, cfg.top_k, false)
      .boxes;
}

std::string boxes_to_json(const std::vector<neat::Box>& boxes, const Config& cfg) {
  std::ostringstream ss;
  ss << "{\"boxes\":[";
  bool first = true;
  for (const auto& b : boxes) {
    if (b.score < cfg.score_threshold) continue;
    if (!first) ss << ",";
    first = false;
    ss << "{\"x1\":" << static_cast<int>(b.x1) << ",\"y1\":" << static_cast<int>(b.y1)
       << ",\"x2\":" << static_cast<int>(b.x2) << ",\"y2\":" << static_cast<int>(b.y2)
       << ",\"score\":" << std::fixed << std::setprecision(3) << b.score
       << ",\"class_id\":" << b.class_id << ",\"label\":\"" << class_label(b.class_id)
       << "\"}";
  }
  ss << "]}";
  return ss.str();
}

std::string summarize(const std::vector<neat::Box>& boxes, float min_score) {
  std::ostringstream ss;
  int shown = 0;
  for (const auto& b : boxes) {
    if (b.score < min_score) continue;
    if (shown++ >= 5) { ss << " ..."; break; }
    ss << " " << class_label(b.class_id) << "(" << std::fixed << std::setprecision(2)
       << b.score << ")";
  }
  return ss.str();
}

void print_report(const neat::GraphReport& report) {
  if (!report.error_code.empty()) std::cerr << "error_code: " << report.error_code << "\n";
  for (const auto& m : report.bus) {
    if (m.type == "ERROR" || m.type == "WARNING")
      std::cerr << "bus " << m.type << " [" << m.src << "]: " << m.detail << "\n";
  }
  if (!report.repro_gst_launch.empty())
    std::cerr << "repro_gst_launch:\n" << report.repro_gst_launch << "\n";
}

void print_banner(const Config& cfg, const neat::MetadataSender* metadata) {
  std::cout << "Mode:    " << cfg.pipeline_mode
            << (cfg.pipeline_mode == "push" && cfg.overlay ? " (with overlay)" : "") << "\n";
  std::cout << "Camera:  " << cfg.camera_device << " MJPEG " << cfg.width << "x" << cfg.height
            << "@" << cfg.fps << "\n";
  std::cout << "Model:   " << cfg.model_path << " (YOLO26, " << cfg.model_width << "x"
            << cfg.model_height << ")\n";
  std::cout << "Video:   udp://" << cfg.udp_host << ":" << cfg.udp_port
            << " H264/RTP payload=96\n";
  if (metadata)
    std::cout << "Metadata: udp://" << cfg.metadata_host << ":" << metadata->metadata_port()
              << "\n";
  std::cout << "\nView the stream with:\n"
            << "  gst-launch-1.0 -v udpsrc port=" << cfg.udp_port
            << " caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\""
            << " ! rtpjitterbuffer ! rtph264depay ! h264parse ! avdec_h264"
            << " ! videoconvert ! autovideosink sync=false\n\n"
            << "Running. Press Ctrl-C to stop.\n";
}

// ── push mode ─────────────────────────────────────────────────────────────────

int run_push(const Config& cfg) {
  auto model = make_model(cfg);

  neat::Graph source_graph("usb_camera_source");
  source_graph.add(neat::nodes::Custom(camera_fragment(cfg), neat::InputRole::Source));
  source_graph.add(neat::nodes::Output("frame", neat::OutputOptions::Latest()));

  neat::Graph model_graph("usb_camera_model");
  model_graph.add(neat::nodes::Input("image", make_nv12_input_options(cfg)));
  model_graph.add(*model);
  model_graph.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));

  neat::Graph udp_graph("usb_camera_udp");
  udp_graph.add(neat::nodes::Input("video", make_nv12_input_options(cfg)));
  udp_graph.add(groups::VideoSender(make_video_options(cfg)));

  if (cfg.print_backend) {
    std::cout << "Source backend:\n" << source_graph.describe_backend() << "\n";
    std::cout << "Model backend:\n" << model_graph.describe_backend() << "\n";
    std::cout << "UDP backend:\n" << udp_graph.describe_backend() << "\n";
  }

  neat::RunOptions run_options;
  run_options.preset = neat::RunPreset::Realtime;
  run_options.queue_depth = cfg.queue_depth;
  run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = neat::OutputMemory::ZeroCopy;

  neat::Run source_run = source_graph.build(run_options);
  neat::Run model_run = model_graph.build(run_options);

  // The encoder graph is push-driven and terminal (VideoSender ends in udpsink), so it has
  // no Output node to pull from. Caps are pinned by InputOptions::caps_override.
  neat::RunOptions udp_options = run_options;
  udp_options.output_memory = neat::OutputMemory::Owned;
  neat::Run udp_run = udp_graph.build(udp_options);

  std::unique_ptr<neat::MetadataSender> metadata;
  if (!cfg.metadata_host.empty()) {
    neat::MetadataSenderOptions mopt;
    mopt.host = cfg.metadata_host;
    mopt.channel = 0;
    mopt.metadata_port_base = cfg.metadata_port;
    metadata = std::make_unique<neat::MetadataSender>(mopt);
    if (!metadata->ok()) {
      std::cerr << "[warn] metadata sender failed to init; continuing without it\n";
      metadata.reset();
    }
  }

  print_banner(cfg, metadata.get());

  int processed = 0;
  Profile prof;
  auto steady_start = Clock::now();
  auto last_log = steady_start;
  int last_log_frames = 0;

  while (!g_stop.load() && (cfg.frames == 0 || processed < cfg.frames)) {
    neat::Sample frame_sample;
    neat::PullError err;

    const auto t0 = Clock::now();
    auto status = source_run.pull("frame", 20000, frame_sample, &err);
    const auto t1 = Clock::now();

    if (status == neat::PullStatus::Timeout) {
      std::cerr << "[warn] timed out waiting for a camera frame\n";
      continue;
    }
    if (status == neat::PullStatus::Closed) {
      std::cout << "camera source closed\n";
      break;
    }
    if (status != neat::PullStatus::Ok)
      throw std::runtime_error("pull frame failed: " + err.message);

    const auto frame_tensors = neat::tensors_from_sample(frame_sample, true);
    if (frame_tensors.empty()) {
      std::cerr << "[warn] camera sample has no tensors\n";
      continue;
    }

    // Pushing through appsrc is what lands the frame in SiMa DMA memory for the CVU.
    const auto t2 = Clock::now();
    if (!model_run.push("image", neat::TensorList{frame_tensors.front()})) {
      std::cerr << "[warn] failed to push frame to model\n";
      continue;
    }

    neat::Sample det_sample;
    status = model_run.pull("detections", 20000, det_sample, &err);
    const auto t3 = Clock::now();

    if (status == neat::PullStatus::Timeout) {
      std::cerr << "[warn] timed out waiting for detections\n";
      continue;
    }
    if (status == neat::PullStatus::Closed) {
      std::cout << "model pipeline closed\n";
      break;
    }
    if (status != neat::PullStatus::Ok)
      throw std::runtime_error("pull detections failed: " + err.message);

    const auto boxes = decode_boxes(det_sample, cfg);

    const auto t4 = Clock::now();
    neat::Tensor out_frame = copy_nv12_to_cpu(frame_tensors.front());
    if (cfg.overlay) draw_boxes_on_nv12(out_frame, boxes, cfg.score_threshold);
    const auto t5 = Clock::now();

    if (!udp_run.push("video", neat::TensorList{out_frame}))
      std::cerr << "[warn] failed to push frame to encoder\n";
    const auto t6 = Clock::now();

    if (metadata) {
      const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
      metadata->send_metadata("detection", boxes_to_json(boxes, cfg), now_ms,
                              std::to_string(det_sample.frame_id));
    }

    ++processed;
    if (processed == 1) {
      steady_start = Clock::now();
      last_log = steady_start;
      last_log_frames = 1;
    }
    prof.capture.add(ms_since(t0, t1));
    prof.infer.add(ms_since(t2, t3));
    prof.overlay.add(ms_since(t4, t5));
    prof.encode.add(ms_since(t5, t6));
    prof.total.add(ms_since(t0, t6));

    const auto now = Clock::now();
    const double since_log = std::chrono::duration<double>(now - last_log).count();
    if (cfg.profile_interval > 0.0 && since_log >= cfg.profile_interval) {
      // fps over THIS window, not since start -- so a stall shows up immediately instead
      // of being averaged away across thousands of earlier good frames.
      const double win_fps =
          since_log > 0.0 ? static_cast<double>(processed - last_log_frames) / since_log : 0.0;

      std::cout << "frame=" << processed << " fps=" << std::fixed << std::setprecision(1)
                << win_fps << " boxes=" << boxes.size() << " ms(capture=" << std::setprecision(1)
                << prof.capture.window_mean() << " infer=" << prof.infer.window_mean()
                << " overlay=" << prof.overlay.window_mean()
                << " encode=" << prof.encode.window_mean()
                << " total=" << prof.total.window_mean() << ")"
                << summarize(boxes, cfg.score_threshold) << "\n";

      prof.reset_windows();
      last_log = now;
      last_log_frames = processed;
    }
  }

  print_profile_summary(
      prof, processed,
      std::chrono::duration<double>(Clock::now() - steady_start).count());

  udp_run.close();
  model_run.close();
  source_run.close();
  return g_stop.load() ? 130 : 0;
}

// ── graph mode ────────────────────────────────────────────────────────────────

int run_graph(const Config& cfg) {
  auto source = neat::nodes::Custom(camera_fragment(cfg), neat::InputRole::Source);
  auto model = make_model(cfg);
  auto branch = neat::graphs::Branch("camera", {"video", "model"});

  neat::Graph video_graph("video");
  video_graph.connect(neat::nodes::Input("video"),
                      groups::VideoSender(make_video_options(cfg)));

  neat::Graph model_graph("model");
  model_graph.connect(neat::nodes::Input("model"), *model);

  neat::Graph detections_graph("detections");
  detections_graph.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));

  // RealtimeLatestByStream: if one branch falls behind, drop its stale frames rather than
  // back-pressuring the camera. The video branch must never stall the MLA.
  neat::GraphLinkOptions live;
  live.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

  // connect() registers the source; calling add(source) too would emit the camera
  // fragment twice and start two v4l2src elements.
  neat::Graph graph("usb_camera_yolo26m");
  graph.connect(source, branch);
  graph.connect(branch, video_graph, live);
  graph.connect(branch, model_graph, live);
  graph.connect(model_graph, detections_graph);

  if (cfg.print_backend) std::cout << "Backend:\n" << graph.describe_backend() << "\n";

  neat::RunOptions run_options;
  run_options.preset = neat::RunPreset::Realtime;
  run_options.queue_depth = cfg.queue_depth;
  run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = neat::OutputMemory::ZeroCopy;

  neat::Run run = graph.build(run_options);

  std::unique_ptr<neat::MetadataSender> metadata;
  if (!cfg.metadata_host.empty()) {
    neat::MetadataSenderOptions mopt;
    mopt.host = cfg.metadata_host;
    mopt.channel = 0;
    mopt.metadata_port_base = cfg.metadata_port;
    metadata = std::make_unique<neat::MetadataSender>(mopt);
    if (!metadata->ok()) metadata.reset();
  }

  print_banner(cfg, metadata.get());

  int processed = 0;
  Profile prof;
  auto steady_start = Clock::now();
  auto last_log = steady_start;
  int last_log_frames = 0;

  while (!g_stop.load() && (cfg.frames == 0 || processed < cfg.frames)) {
    neat::Sample sample;
    neat::PullError err;

    // In graph mode the whole camera -> CVU -> MLA -> boxdecode chain runs inside the
    // pipeline, so this single pull IS the pipeline. There is no separate capture or
    // encode stage to time on this side: `infer` is the wait for the next result, and
    // `overlay` is the CPU box decode.
    const auto t0 = Clock::now();
    const auto status = run.pull("detections", 20000, sample, &err);
    const auto t1 = Clock::now();

    if (status == neat::PullStatus::Timeout) {
      std::cerr << "[warn] timed out waiting for detections\n";
      continue;
    }
    if (status == neat::PullStatus::Closed) {
      std::cout << "pipeline closed\n";
      break;
    }
    if (status != neat::PullStatus::Ok)
      throw std::runtime_error("pull failed: " + err.message);

    const auto boxes = decode_boxes(sample, cfg);
    const auto t2 = Clock::now();

    if (metadata) {
      const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
      metadata->send_metadata("detection", boxes_to_json(boxes, cfg), now_ms,
                              std::to_string(sample.frame_id));
    }

    ++processed;
    if (processed == 1) {
      steady_start = Clock::now();
      last_log = steady_start;
      last_log_frames = 1;
    }
    prof.infer.add(ms_since(t0, t1));
    prof.overlay.add(ms_since(t1, t2));
    prof.total.add(ms_since(t0, t2));

    const auto now = Clock::now();
    const double since_log = std::chrono::duration<double>(now - last_log).count();
    if (cfg.profile_interval > 0.0 && since_log >= cfg.profile_interval) {
      const double win_fps =
          since_log > 0.0 ? static_cast<double>(processed - last_log_frames) / since_log : 0.0;

      std::cout << "frame=" << processed << " fps=" << std::fixed << std::setprecision(1)
                << win_fps << " boxes=" << boxes.size()
                << " ms(pipeline=" << std::setprecision(1) << prof.infer.window_mean()
                << " decode=" << prof.overlay.window_mean()
                << " total=" << prof.total.window_mean() << ")"
                << summarize(boxes, cfg.score_threshold) << "\n";

      prof.reset_windows();
      last_log = now;
      last_log_frames = processed;
    }
  }

  print_profile_summary(
      prof, processed,
      std::chrono::duration<double>(Clock::now() - steady_start).count());

  run.close();
  return g_stop.load() ? 130 : 0;
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  // Lets Neat copy a CPU-resident appsrc buffer into EV74/SiMa memory for the CVU.
  // Without this, push mode fails to hand frames to the preprocessor.
  setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 0);

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  const std::string config_path = (argc > 1) ? argv[1] : kConfigPath;

  try {
    const Config cfg = read_config(config_path);
    return (cfg.pipeline_mode == "graph") ? run_graph(cfg) : run_push(cfg);
  } catch (const neat::NeatError& e) {
    std::cerr << "NEAT error: " << e.what() << "\n";
    print_report(e.report());
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
