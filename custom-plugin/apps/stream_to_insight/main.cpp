// stream_to_insight: the smallest useful pipeline with a custom plugin in it.
//
//   RTSP (from Insight) -> H.264 decode -> framemarker -> branch -+-> H.264 encode -> Insight
//                                                                 +-> "stats" (pulled here)
//
// The "stats" output exists so the application has a loop to run in: it counts frames, prints
// the rate, and notices Ctrl-C. The video itself never leaves the pipeline.
//
// Usage:
//   stream_to_insight --url rtsp://<insight-host>:8554/src1 --insight-host <insight-host>
//                     [--channel 0] [--video-port-base 9000]
//                     [--mode none|grayscale|invert] [--border 12] [--color 0x00FF00]
//                     [--no-bar] [--custom] [--frames N] [--print-graph]
//                     [--width W --height H --fps F]

#include "../common/app_common.h"
#include "../common/framemarker_node.h"

#include <neat.h>
#include <nodes/groups/RtspDecodedInput.h>
#include <nodes/groups/VideoSender.h>

#include <chrono>
#include <exception>
#include <iostream>
#include <string>

namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;
using namespace custom_plugin;

namespace {

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
  // Pin the decoder output. This is also exactly the format framemarker's pad template
  // accepts, so caps negotiation between the two is trivial.
  opt.output_caps.enable = true;
  opt.output_caps.format = "NV12";
  opt.output_caps.width = g.width;
  opt.output_caps.height = g.height;
  opt.output_caps.fps = g.fps;
  opt.output_caps.memory = neat::CapsMemory::Any;
  return opt;
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf); // progress lines appear at once, even when logged to a file
  try {
    const Args args(argc, argv);
    const std::string url = args.get("--url");
    if (url.empty() || args.has("--help")) {
      std::cerr << "usage: stream_to_insight --url rtsp://<host>:8554/srcN "
                   "--insight-host <host> [--channel 0] [--mode none|grayscale|invert] "
                   "[--border N] [--color 0xRRGGBB] [--no-bar] [--custom] [--frames N] "
                   "[--print-graph]\n";
      return url.empty() ? 1 : 0;
    }
    const std::string insight_host = args.get("--insight-host", "127.0.0.1");
    const int channel = args.get_int("--channel", 0);
    const int frames_limit = args.get_int("--frames", 0);

    StreamGeometry geometry{args.get_int("--width", 0), args.get_int("--height", 0),
                            args.get_int("--fps", 0)};
    if (geometry.width <= 0 || geometry.height <= 0 || geometry.fps <= 0)
      geometry = probe_stream(url);

    // ---- The custom plugin, as a Neat Node ---------------------------------------------------
    FrameMarkerOptions marker_opt;
    marker_opt.mode = parse_frame_marker_mode(args.get("--mode", "grayscale"));
    marker_opt.border_width = args.get_uint("--border", 12);
    marker_opt.color = args.get_uint("--color", 0x00FF00);
    marker_opt.progress_bar = !args.has("--no-bar");
    marker_opt.label = "marker";

    // Two equivalent ways to put the element in the graph:
    //   --custom : nodes::Custom() with a raw launch fragment (public, stable API)
    //   default  : the typed FrameMarkerNode from framemarker_node.h
    std::shared_ptr<neat::Node> marker =
        args.has("--custom")
            ? neat::nodes::Custom("framemarker name=marker " + frame_marker_properties(marker_opt))
            : FrameMarker(marker_opt);

    // ---- Video out to Insight ---------------------------------------------------------------
    auto sender_opt =
        groups::VideoSenderOptions::H264RtpUdpFromRaw(geometry.width, geometry.height, geometry.fps);
    sender_opt.host = insight_host;
    sender_opt.channel = channel;
    sender_opt.video_port_base = args.get_int("--video-port-base", 9000);
    sender_opt.encoder.bitrate_kbps = 2000;

    neat::Graph video("video");
    video.connect(neat::nodes::Input("video"), groups::VideoSender(sender_opt));

    neat::Graph stats("stats");
    stats.add(neat::nodes::Output("stats", neat::OutputOptions::Latest()));

    // ---- Compose ------------------------------------------------------------------------------
    neat::Graph graph("stream_to_insight");
    auto source = groups::RtspDecodedInput(make_source_options(url, geometry));
    auto branch = neat::graphs::Branch("marked", {"video", "stats"});
    graph.connect(source, marker);
    graph.connect(marker, branch);
    graph.connect(branch, video);
    graph.connect(branch, stats);

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

    std::cout << "source=" << url << " " << geometry.width << "x" << geometry.height << "@"
              << geometry.fps << " mode=" << frame_marker_mode_name(marker_opt.mode)
              << " insight=" << insight_host << ":" << sender_opt.video_port()
              << " channel=" << channel << "\nPress Ctrl-C to stop.\n";

    // ---- Run until Ctrl-C or --frames ---------------------------------------------------------
    install_stop_handlers();
    using Clock = std::chrono::steady_clock;
    auto window_start = Clock::now();
    int total = 0, window = 0;
    while (!stop_requested() && (frames_limit <= 0 || total < frames_limit)) {
      neat::Sample sample;
      neat::PullError err;
      const auto status = run.pull("stats", 1000, sample, &err);
      if (status == neat::PullStatus::Timeout)
        continue;
      if (status == neat::PullStatus::Closed)
        break;
      if (status != neat::PullStatus::Ok)
        throw std::runtime_error("pull failed: " + err.message);
      ++total;
      ++window;
      const double secs = std::chrono::duration<double>(Clock::now() - window_start).count();
      if (secs >= 5.0) {
        std::cout << "frames=" << total << " pulled_fps=" << (window / secs)
                  << " last_frame_id=" << sample.frame_id << "\n";
        window = 0;
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
