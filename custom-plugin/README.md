# Custom GStreamer Plugins on SiMa Neat: a hands-on starter

Part of the [demo-neat](../README.md) repository. This folder is a complete, working example of
**adding your own processing step to a SiMa video/AI pipeline**. You'll build a small plugin
called **`framemarker`**, load it on a Modalix DevKit, put it inside a real Neat application
(with and without AI detection), and watch the result live in Neat Insight.

If you have never written a GStreamer plugin before, start here. Every step is explained, and
every command is one you can copy.

> Tested on: Modalix DevKit, eLxr 2.1.3, Neat runtime 0.4.0, GStreamer 1.22, Neat SDK v2.1.3.

---

## Contents

1. [What this project is about](#1-what-this-project-is-about)
2. [What you will see at the end](#2-what-you-will-see-at-the-end)
3. [Plugin basics in five minutes](#3-plugin-basics-in-five-minutes)
4. [What is in this folder](#4-what-is-in-this-folder)
5. [Prerequisites](#5-prerequisites)
6. [The whole journey at a glance](#6-the-whole-journey-at-a-glance)
7. [Step-by-step: build, run, verify](#7-step-by-step-build-run-verify)
8. [How to tell it is working](#8-how-to-tell-it-is-working)
9. [How the plugin works inside](#9-how-the-plugin-works-inside)
10. [Writing your own plugin from this one](#10-writing-your-own-plugin-from-this-one)
11. [Troubleshooting](#11-troubleshooting)
12. [FAQ](#12-faq)
13. [Glossary](#13-glossary)
14. [Where to go next](#14-where-to-go-next)

---

## 1. What this project is about

**The problem.** SiMa's Neat library gives you ready-made building blocks: RTSP input, hardware
decode, AI models, hardware encode, output to Insight. Sometimes you need a step that doesn't
exist yet. Examples:

- blur faces or number plates before video leaves the device
- burn a logo, timestamp or camera name into the video
- check or reject bad frames
- call a library of your own on every frame

**The answer: a custom plugin.** Neat pipelines are built on **GStreamer**, the open-source
media framework. Anything written as a GStreamer *element* can be dropped into a Neat pipeline
next to SiMa's own elements.

**What we do here:**

| Goal | How this folder does it |
|---|---|
| Learn what a plugin is and where it fits | Section 3 of this page, plus the tutorial |
| Write a real plugin | `plugin/gstframemarker.c`, about 400 lines of C with comments |
| Build it for the DevKit | CMake in the SDK, which cross-compiles to ARM64 |
| Prove it loads and works | `scripts/test_plugin.sh`, plain GStreamer with no Neat involved |
| Use it in a Neat app | `apps/stream_to_insight`: camera → plugin → Insight |
| Use it next to AI | `apps/detect_to_insight`: camera → plugin (video) + YOLO26 (boxes) → Insight |
| See it with your own eyes | The Neat Insight video viewer |

`framemarker` is deliberately simple and very visible, so you can tell at a glance that your
code is running. It's a **template**: once it works, you replace its per-frame code with your
own.

---

## 2. What you will see at the end

In the Insight video viewer:

- **Your plugin at work:** the video gets a **coloured border** and a **white progress bar**
  along the bottom. The bar grows by one step every frame and restarts every 120 frames. It can
  also turn the picture **grayscale** or **inverted**.
- **AI next to it (Example 2):** **boxes with labels** ("person", "car", ...) from YOLO26, drawn
  by Insight on top of the marked video.

```
 ┌──────────────────────────────────────────┐   ← border (your colour)
 │                                          │
 │        live camera video                 │
 │        (grayscale / inverted / normal)   │
 │            ┌────────┐                    │
 │            │ person │  ← box from YOLO26 │
 │            └────────┘   (Example 2)      │
 │ ██████████████░░░░░░░░░░░░░░░░░░░░░░░    │   ← progress bar, +1 step per frame
 └──────────────────────────────────────────┘
```

---

## 3. Plugin basics in five minutes

You only need these ideas to follow along. The [tutorial](CUSTOM_PLUGIN_TUTORIAL.md) goes
deeper.

**GStreamer** processes media as a chain of **elements** connected by `!`, like a production
line:

```
rtspsrc ! rtph264depay ! h264parse ! decoder ! MY_ELEMENT ! encoder ! udpsink
```

| Term | Plain meaning |
|---|---|
| **Element** | One step in the chain, for example a decoder, a filter or a sender. `framemarker` is an element. |
| **Plugin** | The `.so` file that contains one or more elements. Ours is `libgstframemarker.so`. |
| **Pad** | An element's input (*sink* pad) or output (*src* pad). |
| **Caps** | "Capabilities": the format flowing through a pad, such as `video/x-raw, format=NV12, width=1280, height=720`. Neighbouring elements must agree on caps. |
| **Buffer** | One chunk of data, here one video frame, with a **timestamp** (PTS). |
| **Property** | A setting of an element, written as `name=value`, for example `framemarker mode=grayscale`. |

**How Neat uses GStreamer.** Your Neat app builds a **Graph** of **Nodes**. Each Node produces a
piece of GStreamer launch string. `Graph::build()` joins the pieces, checks the formats and runs
the pipeline. A custom plugin joins the graph through a Node:

```
your app ─► Graph ─► Nodes ─► GStreamer launch string ─► GStreamer elements
                                                          (SiMa's + yours)
```

There are two ways to add a Node for your element:

- **Quick:** `nodes::Custom("framemarker mode=grayscale")` in C++, or
  `pyneat.nodes.custom(...)` in Python
- **Typed:** a small C++ class, as in [apps/common/framemarker_node.h](apps/common/framemarker_node.h)

**NV12, the frame format you'll work with.** SiMa's hardware decoder outputs NV12:

```
Y plane  (brightness): width × height bytes,          one per pixel
UV plane (colour):     width × height/2 bytes,        U,V pairs, one pair per 2×2 pixels
Rows can be padded, so always step rows by the plane's "stride", not by the width.
```

**Two rules that matter in a live pipeline:**

1. **Be fast.** Your code runs on every frame. At 30 fps, the *whole* pipeline has about 33 ms
   per frame.
2. **Don't touch timestamps.** Insight matches AI boxes to video frames by timestamp. If you
   change them, boxes drift away from objects.

**Do you even need a plugin?** Often you don't:

- Working with detection *results* (counting, alerts, rules)? Do it in your app after
  `run.pull()`.
- Drawing boxes for viewers? Send metadata to Insight and it draws them.
- Resize or normalise for a model? `Model::Options` does it on SiMa hardware.
- A standard GStreamer element already does it (`videoflip`, `videocrop`, ...)? Use it with
  `nodes::Custom("videoflip ...")`.

Write a plugin when you must change the frames themselves, inside the pipeline, on every frame.

---

## 4. What is in this folder

```
custom-plugin/
├── README.md                     ← you are here: overview + quick start
├── CUSTOM_PLUGIN_TUTORIAL.md     ← deep dive: how the plugin is written, line by line
│
├── plugin/                       ← THE PLUGIN
│   ├── gstframemarker.c          ← the element's source code (C)
│   └── CMakeLists.txt            ← builds libgstframemarker.so
│
├── apps/                         ← NEAT APPLICATIONS THAT USE THE PLUGIN
│   ├── CMakeLists.txt            ← builds both C++ apps
│   ├── common/
│   │   ├── framemarker_node.h    ← wraps the element as a Neat Node (typed options)
│   │   └── app_common.h          ← shared helpers: options, stream probe, Ctrl-C
│   ├── stream_to_insight/        ← EXAMPLE 1: camera → plugin → Insight (no AI)
│   │   ├── main.cpp              ←   C++ version
│   │   └── main.py               ←   Python version
│   └── detect_to_insight/        ← EXAMPLE 2: plugin + YOLO26 detection → Insight
│       ├── main.cpp
│       └── coco_labels.txt       ←   the 80 class names shown on boxes
│
├── scripts/
│   ├── env.sh                    ← sets the variables so GStreamer and Neat find the plugin
│   └── test_plugin.sh            ← self-test of the plugin on the DevKit (no Neat)
│
└── (created by the build)
    ├── install/lib/gstreamer-1.0/libgstframemarker.so   ← the built plugin
    ├── apps/build/stream_to_insight, detect_to_insight  ← the built apps
    └── out/                                             ← test snapshots
```

**The two sample apps:**

| App | Pipeline | Why it exists |
|---|---|---|
| `stream_to_insight` | RTSP → HW decode → **framemarker** → HW encode → Insight | The simplest proof that the plugin works inside a Neat graph |
| `detect_to_insight` | RTSP → HW decode → branch: (a) **framemarker** → encode → Insight video; (b) YOLO26 → boxes → Insight | Shows a plugin inside a real AI application. The model sees clean frames; only viewers see the marks. |

---

## 5. Prerequisites

### Hardware and software

| You need | Used for | Quick check |
|---|---|---|
| **Neat SDK** (Neat Development Environment) v2.1.3+ | Building the plugin and the apps | `echo $SDK_IMAGE_TAG` and `echo $SYSROOT` → `/opt/toolchain/aarch64/modalix` |
| **Modalix DevKit** reachable over SSH | Running everything | `ssh sima@<devkit-ip> uname -m` → `aarch64` |
| **Neat Insight** (bundled with the SDK) | The test video source (RTSP) and the viewer | `insight-admin status`; open `https://<sdk-host-ip>:9900` |
| **pyneat** on the DevKit (only for the Python example) | Python bindings | `source ~/pyneat/bin/activate && python3 -c "import pyneat"` |
| **sima-cli** on the DevKit (only for Example 2) | Downloading the YOLO26 model | `sima-cli --help` |

In the commands below:

- **`<sdk-host-ip>`** is the machine running the SDK and Insight. `neat --json` shows it as
  `insight.webUiUrl`.
- **`<devkit-ip>`** is your DevKit's address.
- **`/path/to/demo-neat`** is where you cloned this repository. Everything here lives in its
  `custom-plugin/` folder.
- The DevKit must be able to reach the SDK host on **TCP 8554** (RTSP in) and **UDP 9000–9015 /
  9100–9115** (video and boxes out).

### Helpful knowledge

| Topic | Needed? |
|---|---|
| Basic Linux shell, SSH, copying files | Yes |
| Reading C and C++ | Yes, to understand and change the code |
| GStreamer | No. Section 3 covers what you need. |
| Neat library | Helpful. The apps are small and commented. |
| Image formats (NV12) | Section 3 and [tutorial Step 1](CUSTOM_PLUGIN_TUTORIAL.md#step-1-know-what-neat-gives-your-element) |

---

## 6. The whole journey at a glance

```
  SDK (your PC)                                  DevKit (Modalix board)
 ───────────────                                ───────────────────────
 A. Start a test video in Insight (RTSP)
 B. Build the plugin  ──► libgstframemarker.so
 C. Build the apps    ──► stream_to_insight, detect_to_insight
                     D. Copy the folder to the DevKit ──►
                                                 E. source scripts/env.sh
                                                 F. Self-test the plugin (no Neat)
                                                 G. Example 1: camera → plugin → Insight
                                                 H. Example 2: + YOLO26 detection
 I. Watch & verify in the Insight viewer  ◄───────── video + boxes over UDP
```

---

## 7. Step-by-step: build, run, verify

### A. Start a test video in Insight (browser)

Insight can play a video file as an RTSP "camera" for the DevKit to read.

1. Open Insight at **`https://<sdk-host-ip>:9900/`**. It's HTTPS, so accept the certificate
   warning.
2. Go to the **Media Sources** tab and select a video from the **catalog** to load it.
3. Go to the **Streaming** tab, select the loaded video and press **Play**. This creates the
   RTSP stream.

The Streaming tab shows the stream's source number. From the DevKit, the stream is
**`rtsp://<sdk-host-ip>:8554/srcN`**, where N is that number. This guide uses **`src1`**;
replace it with yours.

### B. Build the plugin (SDK)

The SDK shell is already set up to cross-compile for the DevKit, so plain CMake is enough.

```bash
cd /path/to/demo-neat/custom-plugin
cmake -S plugin -B plugin/build -DCMAKE_BUILD_TYPE=Release && cmake --build plugin/build \
  && cmake --install plugin/build --prefix $PWD/install

file install/lib/gstreamer-1.0/libgstframemarker.so    # must say: ARM aarch64
```

### C. Build the apps (SDK)

```bash
cmake -S apps -B apps/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr \
  && cmake --build apps/build -j2
```

This produces `apps/build/stream_to_insight`, `apps/build/detect_to_insight` and
`apps/build/coco_labels.txt`. The apps are **not linked** to the plugin; GStreamer loads it at
run time.

### D. Put the folder on the DevKit

If your DevKit shares the SDK's workspace (the SDK's DevKit sync sets up NFS), there is nothing
to do: the board sees the same path. Otherwise copy the folder to the same path on the board:

```bash
ssh sima@<devkit-ip> mkdir -p /path/to/demo-neat
rsync -a /path/to/demo-neat/custom-plugin/ sima@<devkit-ip>:/path/to/demo-neat/custom-plugin/
```

### E. Set up the shell (DevKit; every new shell)

```bash
ssh sima@<devkit-ip>
cd /path/to/demo-neat/custom-plugin
source scripts/env.sh
```

> **Why this matters.** Neat apps ignore the usual `GST_PLUGIN_PATH`: the Neat runtime swaps in
> its own plugin folder at startup. `env.sh` sets **`GST_PLUGIN_PATH_1_0`** to this plugin's
> folder and **`SIMA_GST_NEAT_ONLY=0`**, so Neat keeps your folder. Without it, `gst-inspect-1.0`
> finds the plugin but Neat apps fail with `build.plugin_missing`.

### F. Self-test the plugin, without Neat (DevKit)

```bash
gst-inspect-1.0 framemarker      # should print "Frame marker", NV12/I420 pads, 5 properties
./scripts/test_plugin.sh         # runs every mode in NV12 and I420, then prints PASS
```

`test_plugin.sh` also saves `out/none_059.jpg`, `out/grayscale_059.jpg` and
`out/invert_059.jpg`. Open them to see the border, the half-filled bar and each mode on a
colour test pattern.

### G. Example 1: camera → plugin → Insight (DevKit)

```bash
./apps/build/stream_to_insight \
    --url rtsp://<sdk-host-ip>:8554/src1 \
    --insight-host <sdk-host-ip> --channel 0 \
    --mode grayscale --border 12 --color 0x00FF00
```

Expected console output (numbers vary):

```
source=rtsp://<sdk-host-ip>:8554/src1 1280x720@30 mode=grayscale insight=<sdk-host-ip>:9000 channel=0
Press Ctrl-C to stop.
frames=250 pulled_fps=30.0 last_frame_id=249
```

Open the viewer (section I). You should see grayscale video with a green border and the moving
bar. Stop with **Ctrl-C**.

Useful options: `--mode none|grayscale|invert`, `--border N`, `--color 0xRRGGBB`, `--no-bar`,
`--frames N` (stop after N frames), `--print-graph` (print the node list and the pipeline plan;
the exact launch string appears as `Pipeline:` at start-up), and `--custom` (use
`nodes::Custom` instead of the typed Node).

**Python version:**

```bash
source ~/pyneat/bin/activate
python3 apps/stream_to_insight/main.py \
    --url rtsp://<sdk-host-ip>:8554/src1 --insight-host <sdk-host-ip> --channel 0 --mode invert
```

### H. Example 2: plugin + YOLO26 detection (DevKit)

Download the model once:

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models && (cd models && sima-cli download \
  "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz")
```

Then run:

```bash
./apps/build/detect_to_insight \
    --url rtsp://<sdk-host-ip>:8554/src1 \
    --insight-host <sdk-host-ip> --channel 0 \
    --model models/yolo26m-det-bf16-mla_tess-b1.tar.gz \
    --labels apps/build/coco_labels.txt \
    --border 12 --color 0xFF0000
```

Expected console output (numbers vary):

```
source=rtsp://<sdk-host-ip>:8554/src1 1280x720@30 model=models/yolo26m-det-bf16-mla_tess-b1.tar.gz insight=<sdk-host-ip> video=9000 metadata=9100 channel=0
frames=231 fps=30.07 avg_boxes=10.4
```

The viewer shows colour video with a **red border and bar** (the plugin) and **labelled boxes**
(YOLO26). Try `--mode grayscale`: the picture turns gray but the boxes don't change, because the
model receives the clean frames on its own branch.

### I. Watch it in Insight (any browser)

Open **`https://<sdk-host-ip>:9900`** → **Video Viewer** → channel **0**. It's HTTPS, so accept
the certificate warning. For a direct link, run
`curl -k "https://127.0.0.1:9900/api/viewer-url?src=0"` in the SDK.

> **Channels:** for a single stream, use **channel 0**. It's the apps' default (`--channel` can
> be left out), and Insight's default single-stream view shows channel 0. Other channels are only
> needed when you run several streams at once. The SDK exposes channels **0–15** (UDP 9000–9015
> for video, 9100–9115 for boxes); a channel outside that range receives nothing. Check yours
> with `neat --json` → `videoUDP`.

---

## 8. How to tell it is working

Work down the list. Each level proves one more link in the chain.

| # | Question | How to check | Pass looks like |
|---|---|---|---|
| 1 | Is it built for the DevKit? | `file install/lib/gstreamer-1.0/libgstframemarker.so` (SDK) | `ELF 64-bit ... ARM aarch64` |
| 2 | Can GStreamer load it? | `gst-inspect-1.0 framemarker` (DevKit) | Element details, pads `NV12, I420`, 5 properties |
| 3 | Does it process frames correctly? | `./scripts/test_plugin.sh` | `PASS`, plus JPEGs in `out/` showing the border, bar and mode |
| 4 | Does it run inside Neat? | `stream_to_insight ... --print-graph` | The graph lists `n7: FrameMarker`, and the `Pipeline:` launch string contains `framemarker name=n7_framemarker_<N> ...`; steady `pulled_fps` ≈ source fps; no `build.plugin_missing` |
| 5 | Does the video reach Insight? | `curl -sk https://127.0.0.1:9900/api/ingest/stats` (SDK) | Your channel shows `active: true`, `codec: H264`, bitrate > 0 |
| 6 | Are timestamps preserved? | Same stats, `metadata` block, while Example 2 runs; take two readings a few seconds apart | `matched_*` counters rise, `expired_metadata` stays flat |
| 7 | Does it look right? | The Insight viewer | Border in your colour, bar moving **smoothly**, boxes sitting on the objects |

**Reading what you see:**

- **The bar moves smoothly:** your code runs on every frame.
- **The bar jumps:** frames are dropped *before* the plugin, for example because the network or
  the source is too slow.
- **Boxes lag or drift** after you change the plugin: something changed timestamps. Put the
  original code back and compare.
- **Nothing changes:** check `--mode`, `--border` and that you're viewing the right channel.

**Quick experiments to prove it's your code:** change one option at a time and watch the
viewer follow. For example `--color 0x0000FF --border 30` gives a thick blue border,
`--mode invert` gives a negative image, and `--no-bar` removes the bar.

---

## 9. How the plugin works inside

Full walkthrough: [tutorial Step 2](CUSTOM_PLUGIN_TUTORIAL.md#step-2-write-the-element). The
short version:

**It's built on `GstVideoFilter`,** a standard GStreamer base class for "video in → same
video out". The base class handles format negotiation, mapping frame memory and keeping
timestamps. Our code only has to:

| Part of `gstframemarker.c` | What it does |
|---|---|
| **Pad templates** `{ NV12, I420 }` | Declares which formats it accepts, matching what SiMa's decoder produces |
| **Properties** (`mode`, `border-width`, `color`, `progress-bar`, `log-interval`) | The settings you pass as `name=value` |
| **`transform_frame_ip()`** | Runs on **every frame**: applies the mode, draws the border and bar, counts frames. This is the function you replace for your own work. |
| **`set_info()`** | Called once when the format is known; logs width, height and format |
| **`plugin_init()` / `GST_PLUGIN_DEFINE`** | Registers the element name `framemarker` inside `libgstframemarker.so` |

**What each property does to the pixels:**

| Property | Values (default) | Effect |
|---|---|---|
| `mode` | `none` / `grayscale` / `invert` (`none`) | `grayscale` sets every colour (UV) byte to 128; `invert` sets every byte to 255 − value |
| `border-width` | 0–256 px (`8`) | Solid frame around the image; 0 turns it off |
| `color` | `0xRRGGBB` (`0x00FF00`) | Border colour, converted to Y/U/V (BT.601) |
| `progress-bar` | `true` / `false` (`true`) | White bar along the bottom, +1 step per frame, restarts every 120 frames |
| `log-interval` | N frames (`0` = off) | Prints `frames=... fps=...` every N frames |

**How the apps use it:**

- **Typed Node** ([framemarker_node.h](apps/common/framemarker_node.h)): C++ code writes
  `FrameMarker({.mode = FrameMarkerMode::Grayscale, .border_width = 12})`, and the Node emits
  `framemarker name=n7_framemarker mode=grayscale border-width=12 ...`.
- **Quick way:** `nodes::Custom("framemarker mode=grayscale border-width=12")` (C++) or
  `pyneat.nodes.custom(...)` (Python) produces the same pipeline with no class.

---

## 10. Writing your own plugin from this one

1. **Copy** `plugin/` to a new folder, for example `privacymask/`.
2. **Rename** consistently: the file, the type macros (`FrameMarker` → `PrivacyMask`), the
   element name in `gst_element_register(..., "privacymask", ...)`, and the plugin name in
   `GST_PLUGIN_DEFINE(..., privacymask, ...)`. The plugin name **must** match the file name
   `libgstprivacymask.so`. Don't start names with `sima` or `neat`; those belong to SiMa.
3. **Formats:** keep `{ NV12, I420 }` unless you really handle others.
4. **Replace the body of `transform_frame_ip()`** with your algorithm. Use the plane pointers and
   **strides** as the sample does.
5. **Replace the properties** with the settings your algorithm needs.
6. **Build, then test in this order:** `gst-inspect-1.0`, a `gst-launch-1.0` test like
   `scripts/test_plugin.sh`, then inside a Neat app with `nodes::Custom("privacymask ...")`.
7. **Measure.** Set a `log-interval`-style counter and make sure the app still holds the source
   frame rate.

Good habits: keep per-frame work short, never block, never change timestamps, and protect
properties with `GST_OBJECT_LOCK`. The tutorial's
[Rules for a well-behaved plugin](CUSTOM_PLUGIN_TUTORIAL.md#rules-for-a-well-behaved-plugin)
explains each one.

---

## 11. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `build.plugin_missing ... framemarker` in a Neat app, although `gst-inspect-1.0` finds it | The Neat runtime replaced the plugin path | `source scripts/env.sh` in **this** shell (sets `GST_PLUGIN_PATH_1_0` + `SIMA_GST_NEAT_ONLY=0`) |
| `No such element or plugin 'framemarker'` in `gst-inspect-1.0` | Path not set, or the file isn't on the board | `source scripts/env.sh`; `ls install/lib/gstreamer-1.0/` |
| Plugin reported as blacklisted | An earlier broken build was cached | `rm -rf ~/.cache/gstreamer-1.0` and retry |
| `wrong ELF class` / cannot open shared object | Built for the PC instead of the DevKit | Rebuild in the SDK shell; `file` must say `ARM aarch64` |
| `not-negotiated` / caps error at build | Incoming format isn't NV12/I420 | Feed decoded NV12 (as the apps do), or add the format to the plugin |
| App runs at full fps but Insight shows nothing | Channel outside 0–15, or UDP blocked between DevKit and SDK host | Use a channel in `neat --json` → `videoUDP`; check `api/ingest/stats` shows packets |
| `could not probe stream geometry` / stream won't open | The Insight source isn't playing, or the URL/port is wrong | Start the source (step A); check `rtsp://<sdk-host-ip>:8554/srcN` |
| Boxes but no plugin marks (Example 2) | Viewing another channel, or `--border 0 --no-bar --mode none` | Check the options and `--channel` |
| fps drops after changing the plugin | Per-frame code too slow | Measure with `log-interval`; optimise or process fewer pixels |

More detail: [tutorial Step 8](CUSTOM_PLUGIN_TUTORIAL.md#step-8-debugging).

---

## 12. FAQ

**Is `framemarker` a SiMa product plugin?**
No. It's a teaching sample written for this folder. SiMa's own plugins are named `neat*`, for
example `neatdecoder` and `neatprocessmla`.

**Is a Neat `Model` a plugin too?**
Sort of. A `Model` expands into *several* SiMa elements: `neatprocesscvu` (preprocess, CVU),
`neatprocessmla` (inference, MLA) and `neatobjectdecode` (boxes). Neat configures them from the
model archive. Your plugin is one element that you configure yourself.

**Can my plugin run on the MLA or CVU accelerators?**
Not through this route. A GStreamer plugin like this one runs on the ARM CPU. Model-side work
belongs in `Model::Options` and the model compiler tools.

**Can I use the plugin without Neat?**
Yes. It's a normal GStreamer element: `gst-launch-1.0 ... ! framemarker ! ...` works, as
`scripts/test_plugin.sh` shows.

**Can I use it from Python?**
Yes, with `pyneat.nodes.custom("framemarker ...")`; see
[apps/stream_to_insight/main.py](apps/stream_to_insight/main.py). Python can't subclass a Neat
Node, so the typed wrapper is C++ only.

**Where should the plugin go in my pipeline?**
Before the branch if the model should see your change (for example a privacy mask for
analytics too). On the video branch only if it's just for viewers (for example a watermark).
[Tutorial 5.4](CUSTOM_PLUGIN_TUTORIAL.md#54-where-to-put-it-in-the-graph) has a diagram.

**How do I install it permanently?**
Ship the `.so` with your app and set `GST_PLUGIN_PATH_1_0` and `SIMA_GST_NEAT_ONLY=0` in its
launcher, the same way `env.sh` does. See
[Deploying beyond this folder](CUSTOM_PLUGIN_TUTORIAL.md#deploying-beyond-this-folder).

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Neat** | SiMa's application library: Graph, Node, Model, Run |
| **Neat SDK** | The Docker-based development environment where you build (cross-compile) for the DevKit |
| **DevKit / Modalix** | The SiMa board that runs the apps |
| **Insight** | SiMa's browser tool: serves test videos as RTSP streams and shows output video with overlays |
| **RTSP** | The protocol for streaming video from a camera or server |
| **CVU / MLA** | SiMa accelerators: CVU for image pre/post-processing, MLA for neural-network inference |
| **Cross-compile** | Build on a PC (x86) for a different CPU (the DevKit's ARM64) |
| **`.so`** | A Linux shared library; a GStreamer plugin is one of these |
| **PTS** | Presentation timestamp: when a frame should be shown; used to match boxes to frames |

---

## 14. Where to go next

- **[CUSTOM_PLUGIN_TUTORIAL.md](CUSTOM_PLUGIN_TUTORIAL.md)**: the full guide, covering choosing
  a base class, each part of the source, the Node wrapper, graph placement, debugging, rules and
  deployment.
- **[plugin/gstframemarker.c](plugin/gstframemarker.c)**: read the code top to bottom; it's
  commented.
- **Neat library docs** (in the SDK under `/neat-resources/core-src/docs/`):
  `develop-apps/development-workflow/graph.mdx` for building graphs, and
  `develop-apps/advanced-concepts/execution-model/gstreamer_layer.md` for how Neat sits on
  GStreamer.
- **GStreamer plugin writer's guide:**
  <https://gstreamer.freedesktop.org/documentation/plugin-development/>
