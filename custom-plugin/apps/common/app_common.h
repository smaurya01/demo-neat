// Small helpers shared by the two sample apps: argument parsing, stream probing and
// Ctrl-C handling. Nothing here is specific to the custom plugin.
#pragma once

#include <opencv2/videoio.hpp>

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace custom_plugin {

class Args {
public:
  Args(int argc, char** argv) : argc_(argc), argv_(argv) {}

  bool has(const std::string& key) const {
    for (int i = 1; i < argc_; ++i)
      if (key == argv_[i])
        return true;
    return false;
  }

  std::string get(const std::string& key, const std::string& def = "") const {
    for (int i = 1; i + 1 < argc_; ++i)
      if (key == argv_[i])
        return argv_[i + 1];
    return def;
  }

  int get_int(const std::string& key, int def) const {
    const std::string v = get(key);
    return v.empty() ? def : std::stoi(v);
  }

  unsigned get_uint(const std::string& key, unsigned def) const {
    const std::string v = get(key);
    // Base 0 accepts decimal and 0xRRGGBB.
    return v.empty() ? def : static_cast<unsigned>(std::stoul(v, nullptr, 0));
  }

private:
  int argc_;
  char** argv_;
};

struct StreamGeometry {
  int width = 0;
  int height = 0;
  int fps = 0;
};

// Read width, height and frame rate from the RTSP stream. The VideoSender encoder needs all
// three up front, and pinning them on the decoder output lets the decoder use its fast path.
inline StreamGeometry probe_stream(const std::string& url) {
  cv::VideoCapture capture(url);
  if (!capture.isOpened())
    throw std::runtime_error("cannot open " + url + " to probe its size and frame rate");
  StreamGeometry g;
  g.width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  g.height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  g.fps = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FPS)));
  capture.release();
  if (g.width <= 0 || g.height <= 0 || g.fps <= 0)
    throw std::runtime_error("could not probe stream geometry; pass --width --height --fps");
  return g;
}

inline std::atomic<bool>& stop_requested() {
  static std::atomic<bool> flag{false};
  return flag;
}

// Ctrl-C, kill and a dropped SSH session all end the pull loop so that Run::close() runs and
// the decoder, encoder and MLA are released cleanly.
inline void install_stop_handlers() {
  auto handler = [](int) { stop_requested().store(true); };
  std::signal(SIGINT, handler);
  std::signal(SIGTERM, handler);
  std::signal(SIGHUP, handler);
}

} // namespace custom_plugin
