# Writing a Custom Plugin for SiMa Neat

This tutorial shows how to add your own processing step to a SiMa Neat pipeline on a Modalix
DevKit. You write a GStreamer plugin, build it with the Neat SDK, check it on the board, wrap it
as a Neat `Node`, and use it in two real pipelines:

- **Example 1: RTSP camera streaming.** Video comes in from an Insight RTSP source, passes
  through the plugin, and goes back out to the Insight video viewer.
- **Example 2: object detection.** YOLO26 runs on the clean frames. The plugin marks only the
  video the viewer sees, and Insight draws the detection boxes on top.

Everything is in this folder and builds as-is. For just the build and run commands, see
[README.md](README.md). The sample plugin is called **`framemarker`**.
It can turn frames grayscale or invert them, and it draws a coloured border plus a progress bar
that moves one step per frame. The result is easy to see in Insight, so you can tell at a glance
whether your code runs on every frame.

---

## Contents

1. [How plugins fit into Neat](#1-how-plugins-fit-into-neat)
2. [Do you need a plugin?](#2-do-you-need-a-plugin)
3. [What is in this folder](#3-what-is-in-this-folder)
4. [Prerequisites](#4-prerequisites)
5. [Step 1: Know what Neat gives your element](#step-1-know-what-neat-gives-your-element)
6. [Step 2: Write the element](#step-2-write-the-element)
7. [Step 3: Build it with the SDK](#step-3-build-it-with-the-sdk)
8. [Step 4: Install it on the DevKit and check that it loads](#step-4-install-it-on-the-devkit-and-check-that-it-loads)
9. [Step 5: Use it from a Neat graph](#step-5-use-it-from-a-neat-graph)
10. [Step 6: Example 1: stream an RTSP camera through the plugin](#step-6-example-1-stream-an-rtsp-camera-through-the-plugin)
11. [Step 7: Example 2: object detection with the plugin](#step-7-example-2-object-detection-with-the-plugin)
12. [Step 8: Debugging](#step-8-debugging)
13. [Rules for a well-behaved plugin](#rules-for-a-well-behaved-plugin)
14. [Deploying beyond this folder](#deploying-beyond-this-folder)
15. [Going further](#going-further)

---

## 1. How plugins fit into Neat

Your application builds a **`Graph`** out of **`Node`s**: RTSP input, decode, preprocess,
inference, box decode, encode, output. Neat compiles that graph into a **GStreamer** pipeline
and runs it. Every Node produces a small piece of a GStreamer launch string, for example
`neatdecoder name=n5_decoder ...`. `Graph::build()` joins those pieces, negotiates formats
between them, and starts the pipeline.

```
 your app ──► Graph ──► Nodes ──► GStreamer launch fragments ──► GStreamer elements (plugins)
               │                                                        ▲
               └── Neat prints the final launch string as "Pipeline:" ──┘
```

So in Neat, a **plugin is a GStreamer element**, and it joins your application through a
**Node**. That gives you three levels of customisation:

| Level | What you write | When to use it |
|---|---|---|
| 0 | Ordinary C++/Python code between `run.pull()` and your next step | Business logic on results: filtering detections, rules, alerts, logging |
| 1 | `nodes::Custom("element prop=value ...")` | An element that **already exists** (stock GStreamer, or one you built earlier) has to run inside the pipeline |
| 2 | A **new GStreamer element** (this tutorial), used through `nodes::Custom` or a typed `Node` subclass | Per-frame processing that must happen **inside** the pipeline, on every frame, before encode or inference |

## 2. Do you need a plugin?

A plugin is extra code to build, deploy and maintain, so check the cheaper options first.

- **Working on detections, poses or masks?** Pull them with `run.pull("detections")` and handle
  them in your app. That is level 0 and needs no plugin. The Apps examples all work this way.
- **Drawing boxes for display?** Send detections to Insight with `MetadataSender`. The viewer
  draws them, so you don't have to change pixels.
- **Resize, colour conversion or normalisation before a model?** `Model::Options.preprocess`
  already does this on the CVU hardware. Don't do it on the CPU in a plugin.
- **A stock GStreamer element already does it** (`videoflip`, `videocrop`, `videobalance`,
  `textoverlay`, ...)? Use level 1: `nodes::Custom("videoflip method=horizontal-flip")`.

Write your own element when you need pixel-level or buffer-level work **inside** the pipeline
and nothing above covers it. Typical cases:

- privacy masking before video leaves the device
- burning a watermark, logo or timestamp into the video
- custom frame validation, dropping or tagging
- bridging to a proprietary library or device
- a CPU algorithm that must run on every frame at stream rate

## 3. What is in this folder

```
custom-plugin/
├── README.md                         ← quick start: build and run commands
├── CUSTOM_PLUGIN_TUTORIAL.md         ← this tutorial
├── plugin/
│   ├── gstframemarker.c              ← the GStreamer element (C, GstVideoFilter)
│   └── CMakeLists.txt                ← builds libgstframemarker.so
├── apps/
│   ├── CMakeLists.txt                ← builds both C++ apps against the Neat library
│   ├── common/
│   │   ├── framemarker_node.h        ← typed Neat Node wrapping the element
│   │   └── app_common.h              ← arg parsing, stream probe, Ctrl-C handling
│   ├── stream_to_insight/
│   │   ├── main.cpp                  ← Example 1 (C++)
│   │   └── main.py                   ← Example 1 (Python)
│   └── detect_to_insight/
│       ├── main.cpp                  ← Example 2 (C++, YOLO26)
│       └── coco_labels.txt
├── scripts/
│   ├── env.sh                        ← sets the plugin path variables for this folder
│   └── test_plugin.sh                ← plain GStreamer smoke test on the DevKit
└── install/                          ← created by the build: lib/gstreamer-1.0/libgstframemarker.so
```

## 4. Prerequisites

| Item | Used for | Check it with |
|---|---|---|
| **Neat Development Environment (SDK)** v2.1.3 or newer | Cross-compiling the plugin and the apps | `echo $SDK_IMAGE_TAG`, `echo $SYSROOT` (should print `/opt/toolchain/aarch64/modalix`) |
| **Modalix DevKit** reachable over SSH | Running everything | `ssh sima@<devkit-ip> uname -m` prints `aarch64` |
| **Neat Insight** (bundled in the SDK) | The RTSP test source and the video viewer | `insight-admin status`, `neat --json` |
| **pyneat** on the DevKit (Python example only) | Python bindings | `source ~/pyneat/bin/activate && python3 -c "import pyneat"` |
| **sima-cli** on the DevKit (Example 2 only) | Downloading the YOLO26 model | `sima-cli --help` |

Throughout this tutorial:

- **`<sdk-host-ip>`** is the IP address of the machine running the SDK container, which is also
  where Insight runs. `neat --json` prints it as `insight.webUiUrl`.
- **`<devkit-ip>`** is your DevKit's IP address.
- **Build** commands run in the SDK. **Run** commands run on the DevKit.
- **`/path/to/demo-neat`** is where you cloned the demo-neat repository. This tutorial lives in
  its `custom-plugin/` folder.
- If your SDK shares its workspace with the DevKit (the SDK's DevKit sync sets this up over NFS),
  the DevKit already sees the build output at the same path. If not, copy the folder over after
  building:
  ```bash
  ssh sima@<devkit-ip> mkdir -p /path/to/demo-neat
  rsync -a /path/to/demo-neat/custom-plugin/ sima@<devkit-ip>:/path/to/demo-neat/custom-plugin/
  ```

---

## Step 1: Know what Neat gives your element

Before writing any code, find out what data will reach your element and what the next element
expects from it. In GStreamer terms these are the **caps**: media type, pixel format, size and
frame rate.

Useful facts for a Modalix pipeline:

| Where your element sits | What it receives | Notes |
|---|---|---|
| After `RtspDecodedInput` / the Neat hardware decoder (`neatdecoder`) | `video/x-raw, format=NV12` (or `I420`), CPU-mappable memory | This tutorial's case. Check with `gst-inspect-1.0 neatdecoder` on the board. |
| Before `VideoSender` (H.264 encode for Insight) | Must output raw NV12 at the size and rate you gave `H264RtpUdpFromRaw(w, h, fps)` | Don't change the resolution unless you also change the sender options. |
| After a `Model` | Tensors / BBOX byte payloads, **not** video | Usually better handled in the app (level 0). |
| Inside a `Model` (preprocess, MLA, box decode) | SiMa-internal formats | Don't insert elements here. Use `Model::Options`. |

**NV12 memory layout** (what `framemarker` works on):

```
plane 0 (Y, luma):      height rows × width bytes     one byte per pixel
plane 1 (UV, chroma):   height/2 rows × width bytes   U,V,U,V,... one pair per 2×2 pixel block
Each row may be padded: always step rows with the plane's *stride*, never with width.
```

I420 is the same except that U and V are in separate planes (1 and 2), each `width/2` bytes wide.

**Timestamps matter.** Each buffer carries a PTS (presentation timestamp). Insight pairs a
detection with a video frame by comparing the metadata timestamp with the video timestamp, with
a tolerance of about 1 ms. Your element must pass PTS through unchanged. The base class used
below does that automatically.

---

## Step 2: Write the element

Source: [`plugin/gstframemarker.c`](plugin/gstframemarker.c). The walkthrough below covers each part.

### 2.1 Pick a base class

GStreamer gives you base classes that handle most of the pad, caps and buffer plumbing. Pick the
most specific one that fits:

| Base class | Use when | Main method you implement |
|---|---|---|
| **`GstVideoFilter`** | Raw video in, raw video out, same size (filters, overlays, masks) | `transform_frame_ip` (in place) or `transform_frame` (copy) |
| `GstBaseTransform` | One buffer in, one buffer out, any media type (tensors, bytes) | `transform_ip` / `transform` |
| `GstAggregator` / `GstVideoAggregator` | Several inputs merged into one output | `aggregate` |
| `GstBaseSrc` / `GstPushSrc` | Your element produces data (custom camera or device) | `create` / `fill` |
| `GstBaseSink` | Your element consumes data (custom output) | `render` |
| `GstElement` | Anything else: variable output count, custom pads | `chain` function on the sink pad |

`framemarker` uses **`GstVideoFilter`**. It parses the caps into a `GstVideoInfo` and maps each
buffer as a `GstVideoFrame`, so you get plane pointers and strides. It also makes the buffer
writable before you modify it, and it keeps timestamps and metadata.

### 2.2 Declare the type

```c
#define GST_TYPE_FRAME_MARKER (gst_frame_marker_get_type())
G_DECLARE_FINAL_TYPE(GstFrameMarker, gst_frame_marker, GST, FRAME_MARKER, GstVideoFilter)

struct _GstFrameMarker {
  GstVideoFilter parent;     /* must be first */
  /* properties */
  GstFrameMarkerMode mode;
  guint border_width;
  guint color;
  gboolean progress_bar;
  guint log_interval;
  /* per-stream state */
  guint64 frames;
  ...
};
G_DEFINE_TYPE(GstFrameMarker, gst_frame_marker, GST_TYPE_VIDEO_FILTER)
```

This is standard GObject boilerplate. Rename `FrameMarker` / `frame_marker` / `FRAME_MARKER`
consistently when you copy the file for your own element.

### 2.3 Declare the pad templates (the caps contract)

```c
#define FRAME_MARKER_CAPS GST_VIDEO_CAPS_MAKE("{ NV12, I420 }")
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK,
    GST_PAD_ALWAYS, GST_STATIC_CAPS(FRAME_MARKER_CAPS));
static GstStaticPadTemplate src_template  = GST_STATIC_PAD_TEMPLATE("src",  GST_PAD_SRC,
    GST_PAD_ALWAYS, GST_STATIC_CAPS(FRAME_MARKER_CAPS));
```

This is the most important decision in the file. It states exactly which formats the element
accepts and produces. If upstream offers something outside this list, `Graph::build()` fails with
a caps or `not-negotiated` error. **List only formats you have actually implemented.**

### 2.4 Properties

Properties are the knobs you set in the launch string (`framemarker mode=grayscale
border-width=12`). Each property needs:

1. an entry in the `enum { PROP_0, PROP_MODE, ... }`
2. a `g_object_class_install_property(... g_param_spec_xxx(...))` call in `class_init` with
   name, range and default
3. a `case` in `set_property` and `get_property`

`framemarker` has five:

| Property | Type | Default | Meaning |
|---|---|---|---|
| `mode` | enum `none` / `grayscale` / `invert` | `none` | Whole-frame pixel operation |
| `border-width` | uint 0–256 | `8` | Border thickness in pixels (0 = off) |
| `color` | uint `0xRRGGBB` | `0x00FF00` | Border colour |
| `progress-bar` | boolean | `true` | Bar that advances one step per frame |
| `log-interval` | uint | `0` | Print frames/fps every N frames (0 = off) |

Properties can be set from the application thread while frames run on the streaming thread, so
both `set_property` and the per-frame code take `GST_OBJECT_LOCK`. The per-frame code copies the
values into locals and releases the lock straight away.

### 2.5 The per-frame work: `transform_frame_ip`

```c
static GstFlowReturn gst_frame_marker_transform_frame_ip(GstVideoFilter *filter,
                                                         GstVideoFrame *frame) {
  /* 1. copy properties under the lock */
  /* 2. read geometry */
  const gint w = GST_VIDEO_FRAME_WIDTH(frame), h = GST_VIDEO_FRAME_HEIGHT(frame);
  /* 3. touch pixels via plane pointers + strides */
  guint8 *y = GST_VIDEO_FRAME_PLANE_DATA(frame, 0);
  gint   ys = GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0);
  ...
  return GST_FLOW_OK;
}
```

GStreamer calls this once per frame, with the frame already mapped read-write. `framemarker`
does three things here:

- **grayscale**: sets every chroma byte to 128 (neutral), which removes colour and leaves luma
  alone
- **invert**: `255 - value` on every luma and chroma byte
- **border and progress bar**: `fill_rect()` writes Y into plane 0 and U/V into the chroma
  plane(s) at half resolution. It handles NV12's interleaved UV and I420's separate U and V.

Return values: `GST_FLOW_OK` to pass the frame on, or `GST_BASE_TRANSFORM_FLOW_DROPPED` to drop
it. Return `GST_FLOW_ERROR` (after `GST_ELEMENT_ERROR(...)`) only for fatal problems, because it
stops the pipeline, and Neat reports it as a runtime error from `Run`.

### 2.6 Optional hooks

| Hook | `framemarker` uses it for |
|---|---|
| `GstBaseTransformClass::start` | Resetting counters when the pipeline starts |
| `GstVideoFilterClass::set_info` | Logging the negotiated format and size, and where you would allocate size-dependent buffers |
| `stop` (not used) | Freeing anything allocated in `start` or `set_info` |

### 2.7 Register the element and name the plugin

```c
static gboolean plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_frame_marker_debug, "framemarker", 0, "...");
  return gst_element_register(plugin, "framemarker", GST_RANK_NONE, GST_TYPE_FRAME_MARKER);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, framemarker, "description",
                  plugin_init, "1.0.0", "LGPL", PACKAGE, "https://...")
```

Three names are involved, and two of them must match:

| Name | Where | Rule |
|---|---|---|
| Plugin name | 3rd argument of `GST_PLUGIN_DEFINE` (`framemarker`) | **Must match the file name** `libgst<name>.so`, or GStreamer refuses to load it |
| Element name | `gst_element_register(..., "framemarker", ...)` | What you type in launch strings and in `nodes::Custom(...)`. A plugin can register several elements. |
| Debug category | `GST_DEBUG_CATEGORY_INIT(..., "framemarker", ...)` | What you pass to `GST_DEBUG=framemarker:5` |

Don't start your element names with `sima` or `neat`. Those prefixes belong to SiMa's own
plugins.

### 2.8 Start your own element from this one

1. Copy `plugin/` to a new folder.
2. Rename the file, the type macros and the three names from 2.7. For example, `privacymask` →
   `libgstprivacymask.so`.
3. Change the pad template formats to the ones you support.
4. Replace the body of `transform_frame_ip` with your algorithm.
5. Replace the properties with your own.

---

## Step 3: Build it with the SDK

The SDK shell already sets `CC`, `CFLAGS`, `LDFLAGS`, `SYSROOT` and the `PKG_CONFIG_*` variables
to point at the Modalix sysroot. A plain CMake build therefore **cross-compiles for the DevKit**
with no toolchain file. The sysroot includes GStreamer 1.22 headers and libraries
(`gstreamer-1.0`, `gstreamer-base-1.0`, `gstreamer-video-1.0`).

**In the SDK:**

```bash
cd /path/to/demo-neat/custom-plugin

# Optional: confirm the GStreamer dev packages are visible in the sysroot
pkg-config --modversion gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

# Configure, build, install into ./install
cmake -S plugin -B plugin/build -DCMAKE_BUILD_TYPE=Release
cmake --build plugin/build
cmake --install plugin/build --prefix $PWD/install

# Confirm it is an ARM64 shared object
file install/lib/gstreamer-1.0/libgstframemarker.so
# → ELF 64-bit LSB shared object, ARM aarch64 ...
```

What [`plugin/CMakeLists.txt`](plugin/CMakeLists.txt) does:

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
add_library(gstframemarker MODULE gstframemarker.c)       # MODULE = loaded with dlopen()
target_link_libraries(gstframemarker PRIVATE PkgConfig::GST)
install(TARGETS gstframemarker LIBRARY DESTINATION lib/gstreamer-1.0)
```

> **Building on the DevKit instead.** The same three `cmake` commands work natively on the
> board, provided the GStreamer development headers are installed there. The SDK route is
> faster and doesn't need anything installed on the board.

If you would rather not use CMake, this one line in the SDK does the same thing:

```bash
$CC $CFLAGS -shared -fPIC plugin/gstframemarker.c -DPACKAGE='"framemarker"' \
    -o libgstframemarker.so $LDFLAGS \
    $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
```

---

## Step 4: Install it on the DevKit and check that it loads

GStreamer looks for plugins in its system folder and in every folder listed in
**`GST_PLUGIN_PATH`**, or in **`GST_PLUGIN_PATH_1_0`**, which takes precedence when set. For
development, leave the `.so` in `install/` and point those variables at it.

> **Important: Neat apps ignore `GST_PLUGIN_PATH` by default.** When a Neat application starts,
> the Neat runtime replaces GStreamer's plugin path with its own plugin folder and unsets
> `GST_PLUGIN_PATH`. It does this because `SIMA_GST_NEAT_ONLY=1` is the default. As a result,
> `gst-inspect-1.0` and `gst-launch-1.0` find your plugin, but the Neat app fails with
> `build.plugin_missing ... Missing component: framemarker`. Inside a Neat app, the plugin
> loads only when both of these are set:
>
> ```bash
> export GST_PLUGIN_PATH_1_0=/path/to/demo-neat/custom-plugin/install/lib/gstreamer-1.0
> export SIMA_GST_NEAT_ONLY=0     # Neat then prepends its folder instead of replacing yours
> ```
>
> [`scripts/env.sh`](scripts/env.sh) sets `GST_PLUGIN_PATH`, `GST_PLUGIN_PATH_1_0` and
> `SIMA_GST_NEAT_ONLY=0` for the current shell, so everything in this tutorial works after
> `source scripts/env.sh`.

**On the DevKit:**

```bash
cd /path/to/demo-neat/custom-plugin
source scripts/env.sh

# 1. Does GStreamer see it?
gst-inspect-1.0 framemarker
```

You should see the element details, pad templates (`NV12`, `I420`) and the five properties. If
you see `No such element or plugin 'framemarker'`, check the following:

```bash
echo $GST_PLUGIN_PATH_1_0 $SIMA_GST_NEAT_ONLY       # install/lib/gstreamer-1.0 and 0?
ls -l install/lib/gstreamer-1.0/                    # file present? (NFS/rsync done?)
GST_DEBUG=GST_PLUGIN_LOADING:4 gst-inspect-1.0 framemarker 2>&1 | grep -i framemarker
rm -rf ~/.cache/gstreamer-1.0                       # clear a stale plugin registry cache
```

Next, run the element with plain GStreamer, with no Neat involved. This separates plugin bugs
from pipeline bugs:

```bash
# 2. Run 300 frames through it
gst-launch-1.0 videotestsrc num-buffers=300 \
  ! video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 \
  ! framemarker mode=grayscale border-width=16 color=0xFF0000 log-interval=100 \
  ! fakesink
```

Expected output includes lines like
`[framemarker framemarker0] frames=100 fps=... 1280x720 NV12`.

Or run the full smoke test. It covers every mode in NV12 and I420 and writes JPEG snapshots to
`out/` so you can see the effect:

```bash
./scripts/test_plugin.sh
```

---

## Step 5: Use it from a Neat graph

There are three ways, from quickest to most structured. All three produce the same GStreamer
pipeline.

### 5.1 `nodes::Custom`: public API, no extra code

```cpp
#include <neat.h>
namespace neat = simaai::neat;

auto marker = neat::nodes::Custom("framemarker name=marker mode=grayscale border-width=12");
graph.connect(source, marker);          // works in branching graphs
graph.connect(marker, next_stage);
```

```python
marker = pyneat.nodes.custom("framemarker name=marker mode=grayscale border-width=12")
graph.connect(source, marker)
graph.connect(marker, next_stage)
```

- The argument is a gst-launch fragment **without** a leading `!`.
- The optional second argument is the Node's `InputRole`. Leave it as `None` for a filter in
  the middle, use `Source` if your element generates data, and `Push` if it accepts
  `run.push()`.
- For a simple linear graph, `graph.custom("framemarker ...")` also works.
- `RtspDecodedInputOptions::extra_fragment = "framemarker ..."` appends the element at the end
  of the RTSP group, after the decoder and its output caps filter. The Apps examples use this
  hook.

### 5.2 A typed `Node` subclass: reusable, with typed options

[`apps/common/framemarker_node.h`](apps/common/framemarker_node.h) wraps the element so that
application code writes `FrameMarker({.mode = FrameMarkerMode::Grayscale})` instead of a string.
A Node only needs four methods:

```cpp
#include <neat.h>
#include <builder/Node.h>

class FrameMarkerNode : public simaai::neat::Node {
public:
  explicit FrameMarkerNode(FrameMarkerOptions opt) : opt_(std::move(opt)) {}

  // Stable type label shown in reports and errors.
  std::string kind() const override { return "FrameMarker"; }

  // Output caps follow the input (same format/size) → Dynamic.
  // Use Static only if your element always outputs fixed caps.
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }

  // The launch fragment. Name elements n<index>_<role> so names stay deterministic.
  std::string backend_fragment(int i) const override {
    return "framemarker name=n" + std::to_string(i) + "_framemarker " +
           frame_marker_properties(opt_);
  }

  // Every name= that backend_fragment() creates.
  std::vector<std::string> element_names(int i) const override {
    return {"n" + std::to_string(i) + "_framemarker"};
  }

private:
  FrameMarkerOptions opt_;
};

inline std::shared_ptr<simaai::neat::Node> FrameMarker(FrameMarkerOptions opt = {}) {
  return std::make_shared<FrameMarkerNode>(std::move(opt));
}
```

Optional overrides include `user_label()` (a label in `describe()` output), `input_role()` (for
source or push elements) and `memory_contract()`.

> **API status.** `include/builder/Node.h` calls the Node subclass "the framework's primary
> extension point". The Neat architecture document, however, lists the builder `Node` class as
> *internal*, so its signature may change between releases. `nodes::Custom()` is the fully
> public route. If you want to stay on stable API, keep the typed wrapper as a thin layer that
> builds a string and hands it to `nodes::Custom()`. The helper `frame_marker_properties()` in
> the header supports both routes.

### 5.3 Python

Python can't subclass a Neat Node, so use `pyneat.nodes.custom(...)` as in 5.1. See
[`apps/stream_to_insight/main.py`](apps/stream_to_insight/main.py).

### 5.4 Where to put it in the graph

Placement decides who sees the modified frames:

```
source ─► framemarker ─► branch ─┬─► VideoSender      (viewer AND model see marked frames)
                                 └─► Model

source ─► branch ─┬─► framemarker ─► VideoSender      (only the viewer sees marked frames)
                  └─► Model                           (model sees clean frames)
```

For display-only effects such as watermarks, borders and privacy masks for viewers, put the
plugin on the video branch as in Example 2. If the model **should** see the change, for example
a privacy mask that also hides people from analytics, put it before the branch.

---

## Step 6: Example 1: stream an RTSP camera through the plugin

```
RTSP (Insight src1) → RtspDecodedInput (HW H.264 decode, NV12) → framemarker
     → Branch ─┬─► VideoSender (HW H.264 encode → RTP/UDP) → Insight viewer, channel 0
               └─► Output("stats", Latest)  → the app counts frames, handles Ctrl-C
```

Source: [`apps/stream_to_insight/main.cpp`](apps/stream_to_insight/main.cpp) (C++) and
[`main.py`](apps/stream_to_insight/main.py) (Python).

### 6.1 Prepare an RTSP source in Insight

Insight can play a video file as an RTSP "camera" for the DevKit to read.

1. Open Insight at **`https://<sdk-host-ip>:9900/`**. It's HTTPS, so accept the certificate
   warning.
2. Go to the **Media Sources** tab and select a video from the **catalog** to load it.
3. Go to the **Streaming** tab, select the loaded video and press **Play**. This creates the
   RTSP stream.

The Streaming tab shows the stream's source number. From the DevKit, the stream is
**`rtsp://<sdk-host-ip>:8554/srcN`**, where N is that number. This tutorial uses **`src1`**;
replace it with yours.

### 6.2 Build the apps

**In the SDK** (after Step 3):

```bash
cd /path/to/demo-neat/custom-plugin
cmake -S apps -B apps/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
cmake --build apps/build -j2
ls apps/build/stream_to_insight apps/build/detect_to_insight
```

The apps link only against the Neat library (`SimaNeat::sima_neat`). The plugin is not linked
in: GStreamer loads it at run time from `GST_PLUGIN_PATH_1_0`, which `scripts/env.sh` sets together
with `SIMA_GST_NEAT_ONLY=0` (see Step 4).

### 6.3 Run it

**On the DevKit:**

```bash
cd /path/to/demo-neat/custom-plugin
source scripts/env.sh

./apps/build/stream_to_insight \
    --url rtsp://<sdk-host-ip>:8554/src1 \
    --insight-host <sdk-host-ip> --channel 0 \
    --mode grayscale --border 12 --color 0x00FF00
```

Python version:

```bash
source ~/pyneat/bin/activate
python3 apps/stream_to_insight/main.py \
    --url rtsp://<sdk-host-ip>:8554/src1 --insight-host <sdk-host-ip> --channel 0 \
    --mode invert --color 0xFF00FF
```

Options (both versions):

| Option | Default | Meaning |
|---|---|---|
| `--url` | required | RTSP H.264 URL |
| `--insight-host` | `127.0.0.1` | Where Insight runs |
| `--channel` | `0` | Insight viewer channel. Video goes to UDP `9000 + channel`. Must be a channel the SDK publishes (see `neat --json` → `videoUDP`, usually 0–15). |
| `--mode` | `grayscale` | `none`, `grayscale` or `invert` |
| `--border` / `--color` | `12` / `0x00FF00` | Border width and colour |
| `--no-bar` | off | Hide the progress bar |
| `--custom` | off | (C++) use `nodes::Custom` instead of the typed Node |
| `--frames N` | `0` (forever) | Stop after N frames |
| `--print-graph` | off | Print `Graph::describe()` (the node list) and `Graph::describe_backend()` (the backend plan) |
| `--width/--height/--fps` | probed | Skip the OpenCV probe of the stream |

Expected console output (numbers vary):

```
source=rtsp://<sdk-host-ip>:8554/src1 1280x720@30 mode=grayscale insight=<sdk-host-ip>:9000 channel=0
Press Ctrl-C to stop.
frames=250 pulled_fps=30.0 last_frame_id=249
...
```

### 6.4 Watch it

Open Insight at `https://<sdk-host-ip>:9900` → **Video Viewer** and choose channel 0. For a
direct link, run `curl -k "https://127.0.0.1:9900/api/viewer-url?src=0"` in the SDK. You should
see the video in grayscale with a green border and a white bar that fills along the bottom and
restarts every 120 frames. If the bar moves smoothly, the plugin is processing every frame.

---

## Step 7: Example 2: object detection with the plugin

```
RTSP → RtspDecodedInput → Branch ─┬─► framemarker ─► VideoSender ─────────► Insight video  (ch 0)
                                  └─► YOLO26 Model ─► Output("detections")
                                              └─► app ─► MetadataSender ─► Insight overlays (ch 0)
```

Source: [`apps/detect_to_insight/main.cpp`](apps/detect_to_insight/main.cpp).

The model gets clean frames. Only the viewer's copy is marked. Both branches run in **one**
`Run`, so they share one clock. `framemarker` doesn't change timestamps, so Insight can still
match each set of boxes to its frame.

### 7.1 Download the model

The example uses YOLO26 (COCO, 80 classes) from the SiMa Model Zoo. Download it **on the DevKit**
with `sima-cli`:

```bash
cd /path/to/demo-neat/custom-plugin
export MODELZOO_VERSION="2.1.3"
mkdir -p models && cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz"
cd ..
```

Other sizes from the same folder work too: `yolo26n-det-bf16-mla_tess-b1.tar.gz` (fastest),
`yolo26s-...`, `yolo26l-...` and `yolo26x-...`.

### 7.2 Run it

Build it as in 6.2 (one build produces both apps). Then **on the DevKit**:

```bash
cd /path/to/demo-neat/custom-plugin
source scripts/env.sh

./apps/build/detect_to_insight \
    --url rtsp://<sdk-host-ip>:8554/src1 \
    --insight-host <sdk-host-ip> --channel 0 \
    --model models/yolo26m-det-bf16-mla_tess-b1.tar.gz \
    --labels apps/build/coco_labels.txt \
    --border 12 --color 0xFF0000
```

Extra options: `--min-score 0.35` (detection threshold) and `--metadata-port-base 9100`. All
options from Example 1 also apply here, except that `--mode` defaults to `none` in this example.

Expected console output (numbers vary):

```
source=rtsp://<sdk-host-ip>:8554/src1 1280x720@30 model=models/yolo26m-det-bf16-mla_tess-b1.tar.gz insight=<sdk-host-ip> video=9000 metadata=9100 channel=0
frames=231 fps=30.07 avg_boxes=10.4
```

In the Insight video viewer, channel 0 shows the video with a **red border and progress bar
(from your plugin)** and **detection boxes with labels (drawn by Insight from the metadata)**.

To see why placement matters, try `--mode grayscale`. The viewer shows gray video, but the
detections don't change, because the model still receives the clean colour frames. The plugin
sits on the video branch only (see 5.4).

---

## Step 8: Debugging

### 8.1 See the real pipeline

```bash
./apps/build/stream_to_insight --url ... --insight-host ... --print-graph --frames 1
```

It prints three things:

- the node list from `Graph::describe()`: look for `n7: FrameMarker` (or `n7: CustomNode` with
  `--custom`)
- `Backend:`, the plan from `Graph::describe_backend()`. For branched graphs like these examples
  it's an `ExecutionGraphPlan` listing each pipeline segment, for example
  `segment 0 nodes=[..., SimaDecode, CapsRaw, FrameMarker]`. For a simple linear graph it's the
  launch string itself.
- `Pipeline:`, printed by Neat when each segment starts: the **exact** GStreamer launch string.
  Look for `framemarker name=n7_framemarker_<N> ...` in it. You can paste the relevant part
  into `gst-launch-1.0` to reproduce a problem outside Neat.

### 8.2 GStreamer logs for your element

```bash
GST_DEBUG=framemarker:5 ./apps/build/stream_to_insight ...        # your element only
GST_DEBUG=framemarker:5,neatdecoder:3 ...                         # plus the decoder
GST_DEBUG=3 ...                                                   # warnings from everything
```

At level 4 (INFO), `framemarker` prints the negotiated format from `set_info`. At level 6 (LOG),
it prints every frame with its PTS.

### 8.3 Common problems

| Symptom | Likely cause | Fix |
|---|---|---|
| `build.plugin_missing`, `Missing component: framemarker` in a Neat app, although `gst-inspect-1.0 framemarker` works | The Neat runtime replaced the plugin path (`SIMA_GST_NEAT_ONLY=1` default) | `source scripts/env.sh`: it sets `GST_PLUGIN_PATH_1_0` and `SIMA_GST_NEAT_ONLY=0` |
| `no element "framemarker"` in `gst-inspect-1.0` too | Plugin path not set in *this* shell, or file not copied to the board | `source scripts/env.sh`, check `ls install/lib/gstreamer-1.0/` |
| `gst-inspect` says the plugin is blacklisted | An older broken build was cached | `rm -rf ~/.cache/gstreamer-1.0` and retry |
| `wrong ELF class` / `cannot open shared object` | Built for x86 instead of aarch64, or run outside the SDK shell | Rebuild in the SDK shell; `file` must say `ARM aarch64` |
| `not-negotiated` / caps error at `build()` | Upstream format isn't in your pad template (for example RGB into NV12-only) | Add the format to the template (and implement it), or insert `nodes::CapsRaw("NV12")` / a converter before it |
| App runs at full fps but Insight shows nothing on the channel | The channel's UDP port isn't reachable. The SDK publishes only the channels in its port map (often 0–15, UDP 9000–9015 / 9100–9115), and some networks block UDP between the board and the SDK host | Use a channel inside `neat --json` → `videoUDP` range; check `curl -k https://127.0.0.1:9900/api/ingest/stats` for `packets_received` |
| Video in Insight but no change visible | `mode=none` and `border-width=0`, or you are watching another channel | Check the options and `--channel` |
| Overlays drift or vanish after adding your element | Your element changed PTS or re-timestamped buffers | Don't touch `GST_BUFFER_PTS/DTS`; with `GstVideoFilter` in-place this is automatic |
| Throughput drops | Per-frame CPU work too slow for the stream rate | Measure with `log-interval`; see the rules below |
| App hangs on exit | Your element blocks in `transform` (waiting on a lock or I/O) | Never block the streaming thread; use timeouts |

---

## Rules for a well-behaved plugin

1. **Keep per-frame work cheap and bounded.** `transform_frame_ip` runs on a streaming thread,
   and anything slow there stalls the whole pipeline. At 30 fps you have about 33 ms per frame
   **for the entire pipeline**, not just your element. Measure with `log-interval`, or with
   `SIMA_GST_ELEMENT_TIMINGS=1`, which the Apps examples enable in profile mode.
2. **Touch as few bytes as possible.** Changing only chroma (grayscale) or only a border is much
   cheaper than a full-frame pass. Whole-frame operations on 1080p NV12 touch about 3 MB per
   frame.
3. **Don't do on the CPU what the hardware does already.** Resize, colour conversion and
   normalisation for models belong in `Model::Options.preprocess` (CVU). Encode and decode
   belong to the Neat codec nodes.
4. **Preserve timestamps and metadata.** Don't rewrite PTS/DTS, and don't copy a buffer into a
   new one without copying its timestamps and metas (`gst_buffer_copy_into(..., GST_BUFFER_COPY_METADATA, ...)`).
5. **Use strides, never width, to step rows.** Hardware buffers are often padded.
6. **Protect properties with `GST_OBJECT_LOCK`.** The application may change them while frames
   are flowing.
7. **Advertise only formats you implement.** Wrong caps fail fast at `build()`. Wrong pixels
   fail silently.
8. **Report fatal errors with `GST_ELEMENT_ERROR` and return `GST_FLOW_ERROR`.** Neat turns
   that into a structured runtime error on `Run`. For recoverable issues, log a warning and
   continue.
9. **Free everything in `stop` / `finalize`.** Applications build and close runs repeatedly, for
   example to probe the stream first as these examples do.

---

## Deploying beyond this folder

| Option | How |
|---|---|
| Per-application (recommended) | Ship `libgstframemarker.so` next to your app and set `GST_PLUGIN_PATH_1_0` and `SIMA_GST_NEAT_ONLY=0` in its launcher script or systemd unit, the same way `scripts/env.sh` does |
| System-wide on the DevKit | `sudo cp install/lib/gstreamer-1.0/libgstframemarker.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0/`. The Neat runtime builds its plugin set from its own folder plus the `libgst*` files in this system folder, so no environment variables are needed. (Not verified on a board in this tutorial; use the per-application route if in doubt.) |
| From your app's code | `setenv("GST_PLUGIN_PATH_1_0", "/opt/myapp/gst-plugins", 1); setenv("SIMA_GST_NEAT_ONLY", "0", 1);` at the top of `main()`, **before** building any Graph |

Version your plugin with the 5th argument of `GST_PLUGIN_DEFINE` (`"1.0.0"`), and rebuild it
whenever the DevKit's GStreamer version changes. `gst-inspect-1.0 --version` shows the version.

---

## Going further

- **Attach data to frames instead of changing pixels.** Add a custom `GstMeta` or set
  `Sample::attributes` upstream (see the Neat docs,
  `advanced-concepts/data-model-contracts/frame_attributes.md`), then read it in the app after
  `pull()`.
- **Elements that take more than one input** (for example, merge two cameras): base the element
  on `GstVideoAggregator`. Wiring several graph branches into one custom element isn't covered
  by this sample.
- **A custom source** (a proprietary camera, for example): base the element on `GstPushSrc` and
  use `nodes::Custom("mysrc ...", simaai::neat::InputRole::Source)` as the first node. This
  sample doesn't demonstrate it.
- **Model-side customisation** (custom preprocessing, box decoding, CVU kernels) doesn't go
  through a plugin. Use `Model::Options` (`preprocess`, `decode_type`, thresholds) and the Model
  Compiler tools.

### Reference: files in the Neat SDK

| What | Where (SDK: `/neat-resources/core-src/`) |
|---|---|
| `Node` base class | `include/builder/Node.h` |
| `nodes::Custom`, `CapsRaw` | `include/nodes/common/Caps.h` |
| `Graph` (add, connect, custom, describe) | `include/pipeline/Graph.h` |
| RTSP input group | `include/nodes/groups/RtspDecodedInput.h` |
| Video sender to Insight | `include/nodes/groups/VideoSender.h` |
| Metadata sender to Insight | `include/nodes/io/MetadataSender.h` |
| How Neat sits on GStreamer | `docs/develop-apps/advanced-concepts/execution-model/gstreamer_layer.md` |
| Plugin search paths | `docs/develop-apps/contribute/start-here/architecture.md` |
| Error codes (`build.plugin_missing`, ...) | `docs/reference/error-codes.md`, `docs/reference/diagnostics.md` |
