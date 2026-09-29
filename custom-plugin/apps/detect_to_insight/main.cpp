// detect_to_insight: object detection with the custom plugin on the display branch only.
//
//   RTSP (from Insight) -> decode -> branch -+-> framemarker -> H.264 encode -> Insight video
//                                            +-> YOLO26 (preprocess, MLA, box decode)
//                                                  -> "detections" -> MetadataSender -> Insight
//
// The model sees the clean decoded frame; only what the viewer sees is marked. Put the
// plugin before the branch instead if the model should see the modified frame too.
//
// framemarker changes pixels but not timestamps, so Insight still pairs each detection with
// the video frame it came from and draws the boxes in the right place.
//
// Usage:
//   detect_to_insight --url rtsp://<insight-host>:8554/src1 --insight-host <insight-host>
//                     --model models/yolo26m-det-bf16-mla_tess-b1.tar.gz
//                     [--labels coco_labels.txt] [--channel 0]
//                     [--video-port-base 9000] [--metadata-port-base 9100]
//                     [--mode none|grayscale|invert] [--border 12] [--color 0xFF0000]
//                     [--no-bar] [--min-score 0.35] [--frames N] [--print-graph]

#include "../common/app_common.h"
#include "../common/framemarker_node.h"

#include <neat.h>
#include <nodes/groups/RtspDecodedInput.h>
#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;
using namespace custom_plugin;

namespace {

constexpr int kTopK = 100;

groups::RtspDecodedInputOptions make_source_options(const std::string& url,
                                                    const StreamGeometry& g) {
  groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.tcp = true;
  opt.latency_ms = 200;
  opt.codec = groups::RtspCodec::H264;
  opt.payload_type = 96;
  opt.source_fps = g.fps;
  opt.fallback_h264_width = g.width;
  opt.fallback_h264_height = g.height;
  opt.out_format = "NV12";
  opt.output_caps.enable = true;
  opt.output_caps.format = "NV12";
  opt.output_caps.width = g.width;
  opt.output_caps.height = g.height;
  opt.output_caps.fps = g.fps;
  opt.output_caps.memory = neat::CapsMemory::Any;
  return opt;
}

std::vector<std::string> load_labels(const std::string& path) {
  std::vector<std::string> labels;
  std::ifstream in(path);
  for (std::string line; std::getline(in, line);)
    if (!line.empty())
      labels.push_back(line);
  return labels;
}

// Insight object-detection payload: {"objects":[{"id","label","confidence","bbox":[x,y,w,h]}]}
std::string boxes_to_json(const std::vector<neat::Box>& boxes,
                          const std::vector<std::string>& labels, int frame_w, int frame_h) {
  nlohmann::json data;
  data["objects"] = nlohmann::json::array();
  int index = 1;
  for (const auto& b : boxes) {
    const float x1 = std::clamp(b.x1, 0.0f, static_cast<float>(frame_w));
    const float y1 = std::clamp(b.y1, 0.0f, static_cast<float>(frame_h));
    const float x2 = std::clamp(b.x2, 0.0f, static_cast<float>(frame_w));
    const float y2 = std::clamp(b.y2, 0.0f, static_cast<float>(frame_h));
    const bool known = b.class_id >= 0 && b.class_id < static_cast<int>(labels.size());
    data["objects"].push_back({
        {"id", "obj_" + std::to_string(index++)},
        {"label", known ? labels[b.class_id] : std::string("unknown")},
        {"confidence", b.score},
        {"bbox", {x1, y1, x2 - x1, y2 - y1}},
    });
  }
  return data.dump();
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf); // progress lines appear at once, even when logged to a file
  try {
    const Args args(argc, argv);
    const std::string url = args.get("--url");
    const std::string model_path = args.get("--model");
    if (url.empty() || model_path.empty() || args.has("--help")) {
      std::cerr << "usage: detect_to_insight --url rtsp://<host>:8554/srcN "
                   "--insight-host <host> --model <yolo26 .tar.gz> [--labels coco_labels.txt] "
                   "[--channel 0] [--mode none|grayscale|invert] [--border N] "
                   "[--color 0xRRGGBB] [--no-bar] [--min-score 0.35] [--frames N] "
                   "[--print-graph]\n";
      return (url.empty() || model_path.empty()) ? 1 : 0;
    }
    const std::string insight_host = args.get("--insight-host", "127.0.0.1");
    const int channel = args.get_int("--channel", 0);
    const int frames_limit = args.get_int("--frames", 0);
    const float min_score = std::stof(args.get("--min-score", "0.35"));
    const auto labels = load_labels(args.get("--labels", "coco_labels.txt"));
    if (labels.empty())
      std::cerr << "[warn] no labels loaded; boxes will be labelled 'unknown'\n";

    StreamGeometry geometry{args.get_int("--width", 0), args.get_int("--height", 0),
                            args.get_int("--fps", 0)};
    if (geometry.width <= 0 || geometry.height <= 0 || geometry.fps <= 0)
      geometry = probe_stream(url);

    // ---- Model -------------------------------------------------------------------------------
    // The same options the Apps single-stream detector uses for YOLO26 on NV12 input.
    neat::Model::Options model_opt;
    model_opt.preprocess.kind = neat::InputKind::Image;
    model_opt.preprocess.enable = neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
    model_opt.preprocess.input_max_width = geometry.width;
    model_opt.preprocess.input_max_height = geometry.height;
    model_opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
    model_opt.decode_type = neat::BoxDecodeType::YoloV26;
    model_opt.score_threshold = min_score;
    model_opt.nms_iou_threshold = 0.5f;
    model_opt.top_k = kTopK;
    neat::Model model(model_path, model_opt);

    // ---- Display branch: custom plugin, then the encoder -------------------------------------
    FrameMarkerOptions marker_opt;
    marker_opt.mode = parse_frame_marker_mode(args.get("--mode", "none"));
    marker_opt.border_width = args.get_uint("--border", 12);
    marker_opt.color = args.get_uint("--color", 0xFF0000);
    marker_opt.progress_bar = !args.has("--no-bar");
    marker_opt.label = "marker";

    auto sender_opt =
        groups::VideoSenderOptions::H264RtpUdpFromRaw(geometry.width, geometry.height, geometry.fps);
    sender_opt.host = insight_host;
    sender_opt.channel = channel;
    sender_opt.video_port_base = args.get_int("--video-port-base", 9000);
    sender_opt.encoder.bitrate_kbps = 2000;

    neat::Graph video("video");
    auto video_in = neat::nodes::Input("video");
    auto marker = FrameMarker(marker_opt);
    video.connect(video_in, marker);
    video.connect(marker, groups::VideoSender(sender_opt));

    // ---- Inference branch --------------------------------------------------------------------
    neat::Graph inference("model");
    inference.connect(neat::nodes::Input("model"), model);
    neat::Graph detections("detections");
    detections.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));

    // ---- Compose: both branches in one Run so they share one timeline -------------------------
    neat::Graph graph("detect_to_insight");
    auto source = groups::RtspDecodedInput(make_source_options(url, geometry));
    auto branch = neat::graphs::Branch("source", {"video", "model"});
    graph.connect(source, branch);
    graph.connect(branch, video);
    graph.connect(branch, inference);
    graph.connect(inference, detections);

    if (args.has("--print-graph")) {
      std::cout << graph.describe() << "\n";
      std::cout << "Backend:\n" << graph.describe_backend() << "\n";
    }

    neat::RunOptions run_opt;
    run_opt.preset = neat::RunPreset::Realtime;
    run_opt.queue_depth = 3;
    run_opt.overflow_policy = neat::OverflowPolicy::KeepLatest;
    run_opt.output_memory = neat::OutputMemory::ZeroCopy;
    neat::Run run = graph.build(run_opt);

    neat::MetadataSenderOptions meta_opt;
    meta_opt.host = insight_host;
    meta_opt.channel = channel; // must match the video channel
    meta_opt.metadata_port_base = args.get_int("--metadata-port-base", 9100);
    std::string meta_err;
    neat::MetadataSender metadata(meta_opt, &meta_err);
    if (!metadata.ok())
      throw std::runtime_error("metadata sender: " + meta_err);

    std::cout << "source=" << url << " " << geometry.width << "x" << geometry.height << "@"
              << geometry.fps << " model=" << model_path << " insight=" << insight_host
              << " video=" << sender_opt.video_port() << " metadata=" << metadata.metadata_port()
              << " channel=" << channel << "\nPress Ctrl-C to stop.\n";

    // ---- Pull detections, forward them to Insight ---------------------------------------------
    install_stop_handlers();
    using Clock = std::chrono::steady_clock;
    auto window_start = Clock::now();
    int total = 0, window = 0;
    std::size_t window_boxes = 0;
    while (!stop_requested() && (frames_limit <= 0 || total < frames_limit)) {
      neat::Sample sample;
      neat::PullError err;
      const auto status = run.pull("detections", 1000, sample, &err);
      if (status == neat::PullStatus::Timeout)
        continue;
      if (status == neat::PullStatus::Closed)
        break;
      if (status != neat::PullStatus::Ok)
        throw std::runtime_error("pull failed: " + err.message);

      const auto tensors = neat::tensors_from_sample(sample, false);
      std::vector<neat::Box> boxes;
      if (!tensors.empty()) {
        const auto decoded =
            neat::decode_bbox_tensor(tensors.front(), geometry.width, geometry.height, kTopK, false);
        for (const auto& b : decoded.boxes)
          if (b.score >= min_score)
            boxes.push_back(b);
      }

      // Insight matches metadata.timestamp (ms) against the video frame's timestamp.
      const int64_t ts_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
      const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
      std::string send_err;
      if (!metadata.send_metadata("object-detection",
                                  boxes_to_json(boxes, labels, geometry.width, geometry.height),
                                  ts_ms, frame_id, &send_err) &&
          !send_err.empty())
        std::cerr << "[warn] metadata send failed: " << send_err << "\n";

      ++total;
      ++window;
      window_boxes += boxes.size();
      const double secs = std::chrono::duration<double>(Clock::now() - window_start).count();
      if (secs >= 5.0) {
        std::cout << "frames=" << total << " fps=" << (window / secs)
                  << " avg_boxes=" << (window ? double(window_boxes) / window : 0.0) << "\n";
        window = 0;
        window_boxes = 0;
        window_start = Clock::now();
      }
    }

    run.close();
    std::cout << "stopped after " << total << " frames\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
