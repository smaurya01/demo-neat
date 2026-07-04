#include "utils.h"

#include <neat.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr const char* kConfigPath = "./config/default.conf";

std::atomic<bool> g_stop{false};

struct Config {
  std::string rtsp_url;
  std::string model_path;
  std::string udp_host;
  int udp_port_base = 5204;
  int fallback_width = 1280;
  int fallback_height = 720;
  int fallback_fps = 25;
  int latency_ms = 200;
  int frames = 0;
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

void handle_signal(int) {
  g_stop.store(true);
}

std::string trim(std::string value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

int to_int(const std::string& value) {
  return std::stoi(value);
}

bool to_bool(const std::string& value) {
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

void set_config_value(Config& cfg, const std::string& key, const std::string& value) {
  if (key == "rtsp_url") {
    cfg.rtsp_url = value;
  } else if (key == "model_path") {
    cfg.model_path = value;
  } else if (key == "rtsp_transport") {
    cfg.tcp = value != "udp";
  } else if (key == "udp_host") {
    cfg.udp_host = value;
  } else if (key == "udp_port_base") {
    cfg.udp_port_base = to_int(value);
  } else if (key == "fallback_width") {
    cfg.fallback_width = to_int(value);
  } else if (key == "fallback_height") {
    cfg.fallback_height = to_int(value);
  } else if (key == "fallback_fps") {
    cfg.fallback_fps = to_int(value);
  } else if (key == "latency_ms") {
    cfg.latency_ms = to_int(value);
  } else if (key == "frames") {
    cfg.frames = to_int(value);
  } else if (key == "bitrate_kbps") {
    cfg.bitrate_kbps = to_int(value);
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

std::unique_ptr<neat::Model> make_open_pose_model(const Config& cfg) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.input_max_width = cfg.fallback_width;
  opt.preprocess.input_max_height = cfg.fallback_height;
  opt.preprocess.input_max_depth = 1;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.resize.enable = neat::AutoFlag::On;
  opt.preprocess.resize.width = 480;
  opt.preprocess.resize.height = 480;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.preset = neat::NormalizePreset::None;
  opt.preprocess.normalize.enable = neat::AutoFlag::On;
  opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
  opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
  opt.preprocess.normalize.has_explicit_stats = true;
  return std::make_unique<neat::Model>(cfg.model_path, opt);
}

neat::InputOptions nv12_input_options(const Config& cfg) {
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

groups::RtspDecodedInputOptions rtsp_options(const Config& cfg) {
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

neat::Graph make_source_graph(const Config& cfg) {
  neat::Graph graph("single_stream_open_pose_source");
  graph.add(groups::RtspDecodedInput(rtsp_options(cfg)));
  graph.add(neat::nodes::Output("frame", neat::OutputOptions::Latest()));
  return graph;
}

neat::Graph make_model_graph(const Config& cfg, const neat::Model& model) {
  neat::Graph graph("single_stream_open_pose_model");
  graph.add(neat::nodes::Input("image", nv12_input_options(cfg)));
  graph.add(model);
  graph.add(neat::nodes::Output("result", neat::OutputOptions::EveryFrame(4)));
  return graph;
}

neat::Graph make_udp_graph(const Config& cfg) {
  auto opt = groups::VideoSenderOptions::H264RtpUdpFromRaw(
      cfg.fallback_width, cfg.fallback_height, cfg.fallback_fps);
  opt.host = cfg.udp_host;
  opt.channel = 0;
  opt.video_port_base = cfg.udp_port_base;
  opt.encoder.bitrate_kbps = cfg.bitrate_kbps;

  neat::Graph graph("single_stream_open_pose_udp");
  graph.add(neat::nodes::Input("video", nv12_input_options(cfg)));
  graph.add(groups::VideoSender(opt));
  return graph;
}

neat::RunOptions realtime_options() {
  neat::RunOptions opt;
  opt.preset = neat::RunPreset::Realtime;
  opt.queue_depth = 3;
  opt.overflow_policy = neat::OverflowPolicy::KeepLatest;
  opt.output_memory = neat::OutputMemory::ZeroCopy;
  return opt;
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
    auto model = make_open_pose_model(cfg);
    auto source_graph = make_source_graph(cfg);
    auto model_graph = make_model_graph(cfg, *model);
    auto udp_graph = make_udp_graph(cfg);

    if (cfg.print_backend) {
      std::cout << "Source backend:\n" << source_graph.describe_backend() << "\n";
      std::cout << "OpenPose backend:\n" << model_graph.describe_backend() << "\n";
      std::cout << "UDP backend:\n" << udp_graph.describe_backend() << "\n";
    }

    auto run_options = realtime_options();
    auto source_run = source_graph.build(run_options);
    auto model_run = model_graph.build(run_options);

    auto udp_options = realtime_options();
    udp_options.output_memory = neat::OutputMemory::Owned;
    auto udp_seed = make_blank_nv12_tensor(cfg.fallback_width, cfg.fallback_height);
    auto udp_run = udp_graph.build(neat::TensorList{udp_seed}, udp_options);

    std::cout << "RTSP input: " << cfg.rtsp_url << "\n";
    std::cout << "OpenPose model: " << cfg.model_path << "\n";
    std::cout << "UDP output: udp://" << cfg.udp_host << ":" << cfg.udp_port_base << "\n";
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
        std::cerr << "[warn] failed to push frame to OpenPose\n";
        continue;
      }

      neat::Sample model_sample;
      neat::PullError model_error;
      status = model_run.pull("result", 20000, model_sample, &model_error);
      if (status == neat::PullStatus::Timeout) {
        std::cerr << "[warn] timed out waiting for OpenPose\n";
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        std::cerr << "OpenPose model pipeline closed\n";
        break;
      }
      if (status != neat::PullStatus::Ok) {
        throw std::runtime_error("failed to pull OpenPose result: " + model_error.message);
      }
      const auto inference_end = Clock::now();

      const auto overlay_begin = Clock::now();
      auto output_frame = copy_nv12_to_cpu_tensor(frame_tensors.front());
      const auto overlay = draw_openpose_overlay(output_frame, model_sample);
      const auto overlay_end = Clock::now();

      const auto encoder_begin = Clock::now();
      if (!udp_run.push("video", neat::TensorList{output_frame})) {
        std::cerr << "[warn] failed to push frame to UDP encoder\n";
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
        const double seconds = std::chrono::duration<double>(now - start).count();
        const double fps = seconds > 0.0 ? static_cast<double>(processed) / seconds : 0.0;
        const double steady_seconds = std::chrono::duration<double>(now - steady_start).count();
        const double steady_fps =
            processed > 1 && steady_seconds > 0.0
                ? static_cast<double>(processed - 1) / steady_seconds
                : fps;
        const StageTimes avg = profile.average(processed);
        std::cout << "[open_pose] frame=" << processed << " result="
                  << sample_summary(model_sample) << " persons=" << overlay.persons
                  << " keypoints=" << overlay.keypoints << " "
                  << tensor_shape_summary(model_sample) << " fps=" << std::fixed
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
