// A typed Neat Node that wraps the `framemarker` GStreamer element (../../plugin).
//
// A Node is the unit a Neat Graph is built from. It only has to say which GStreamer
// fragment it emits and which element names that fragment creates; Graph::build() stitches
// the fragments together, negotiates caps and runs the pipeline. Wrapping a plugin this way
// gives application code typed options instead of a hand-written launch string.
//
// The quick alternative, with no class at all, is:
//     graph.add(simaai::neat::nodes::Custom("framemarker mode=grayscale border-width=12"));
#pragma once

#include <neat.h>
#include <builder/Node.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace custom_plugin {

enum class FrameMarkerMode { None, Grayscale, Invert };

struct FrameMarkerOptions {
  FrameMarkerMode mode = FrameMarkerMode::None;
  unsigned border_width = 8;     ///< Pixels; 0 disables the border.
  unsigned color = 0x00FF00;     ///< Border colour as 0xRRGGBB.
  bool progress_bar = true;      ///< Bar that advances one step per frame.
  unsigned log_interval = 0;     ///< Print frames/fps every N frames; 0 = off.
  std::string label;             ///< Optional label shown in Graph::describe().
};

inline const char* frame_marker_mode_name(FrameMarkerMode mode) {
  switch (mode) {
  case FrameMarkerMode::None:
    return "none";
  case FrameMarkerMode::Grayscale:
    return "grayscale";
  case FrameMarkerMode::Invert:
    return "invert";
  }
  return "none";
}

inline FrameMarkerMode parse_frame_marker_mode(const std::string& name) {
  if (name == "none")
    return FrameMarkerMode::None;
  if (name == "grayscale")
    return FrameMarkerMode::Grayscale;
  if (name == "invert")
    return FrameMarkerMode::Invert;
  throw std::invalid_argument("mode must be none, grayscale or invert");
}

// The element properties exactly as gst-launch would take them. Shared by the typed Node
// and the nodes::Custom() path so both produce the same pipeline.
inline std::string frame_marker_properties(const FrameMarkerOptions& opt) {
  char color[16];
  std::snprintf(color, sizeof(color), "0x%06X", opt.color & 0xFFFFFFu);
  return std::string("mode=") + frame_marker_mode_name(opt.mode) +
         " border-width=" + std::to_string(opt.border_width) + " color=" + color +
         " progress-bar=" + (opt.progress_bar ? "true" : "false") +
         " log-interval=" + std::to_string(opt.log_interval);
}

class FrameMarkerNode : public simaai::neat::Node {
public:
  explicit FrameMarkerNode(FrameMarkerOptions opt) : opt_(std::move(opt)) {}

  // Stable type label used in reports and diagnostics.
  std::string kind() const override {
    return "FrameMarker";
  }

  std::string user_label() const override {
    return opt_.label;
  }

  // The element outputs whatever format/size it receives, so its caps follow upstream.
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }

  // The launch-string fragment. No leading "!"; the Graph adds the links. The element name
  // follows the framework's n<index>_<role> convention so it is deterministic.
  std::string backend_fragment(int node_index) const override {
    return "framemarker name=" + element_name(node_index) + " " + frame_marker_properties(opt_);
  }

  // Every name= the fragment creates.
  std::vector<std::string> element_names(int node_index) const override {
    return {element_name(node_index)};
  }

private:
  static std::string element_name(int node_index) {
    return "n" + std::to_string(node_index) + "_framemarker";
  }

  FrameMarkerOptions opt_;
};

// Factory in the same style as the built-in simaai::neat::nodes::* functions.
inline std::shared_ptr<simaai::neat::Node> FrameMarker(FrameMarkerOptions opt = {}) {
  return std::make_shared<FrameMarkerNode>(std::move(opt));
}

} // namespace custom_plugin
