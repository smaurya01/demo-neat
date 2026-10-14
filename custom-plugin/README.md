# Custom GStreamer Plugins on SiMa Neat: a hands-on starter

Part of the [demo-neat](../README.md) repository. This folder is a complete, working example of
**adding your own processing step to a SiMa video/AI pipeline**. You'll build a small plugin
called **`framemarker`**, load it on a Modalix DevKit, put it inside a real Neat application
(with and without AI detection), and watch the result live in Neat Insight.

If you have never written a GStreamer plugin before, start here. Every step is explained, and
every command is one you can copy.

> Requires: a Modalix DevKit on board software 2.1.3 (eLxr), Neat Library 0.5.0 and Neat SDK 2.1.3.1
> (GStreamer 1.22).

---

## Contents

1. [What this project is about](#1-what-this-project-is-about)
2. [What you will see at the end](#2-what-you-will-see-at-the-end)
3. [Plugin basics](#3-plugin-basics)
4. [What is in this folder](#4-what-is-in-this-folder)
5. [Prerequisites](#5-prerequisites)
6. [The whole journey at a glance](#6-the-whole-journey-at-a-glance)
7. [Step-by-step: build, run, verify](#7-step-by-step-build-run-verify)
8. [How to tell it is working](#8-how-to-tell-it-is-working)
9. [Troubleshooting](#9-troubleshooting)
10. [Where to go next](#10-where-to-go-next)

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

## 3. Plugin basics

**GStreamer** processes media as a chain of **elements** connected by `!`. A **plugin** is the `.so`
file that holds an element; ours is `libgstframemarker.so`, holding the `framemarker` element:

```
rtspsrc ! rtph264depay ! h264parse ! decoder ! framemarker ! encoder ! udpsink
```

A Neat app builds a **Graph** of **Nodes**, and Neat turns it into a GStreamer pipeline. Your element
joins the graph through a Node: `nodes::Custom("framemarker mode=grayscale")` (C++),
`pyneat.nodes.custom(...)` (Python), or a typed C++ Node such as
[apps/common/framemarker_node.h](apps/common/framemarker_node.h).

Two rules in a live pipeline: **be fast** (at 30 fps the whole pipeline has about 33 ms per frame),
and **don't touch timestamps** (Insight matches boxes to frames by them).

Often you don't need a plugin at all: see
[Do you need a plugin?](CUSTOM_PLUGIN_TUTORIAL.md#2-do-you-need-a-plugin). The terms are explained in
the tutorial's [Glossary](CUSTOM_PLUGIN_TUTORIAL.md#glossary).

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

| You need | Used for | Quick check |
|---|---|---|
| **Neat SDK** (Neat Development Environment) v2.1.3+ | Building the plugin and the apps | `echo $SDK_IMAGE_TAG` and `echo $SYSROOT` → `/opt/toolchain/aarch64/modalix` |
| **Modalix DevKit** reachable over SSH | Running everything | `ssh sima@<devkit-ip> uname -m` → `aarch64` |
| **Neat Insight** (bundled with the SDK) | The test video source (RTSP) and the viewer | `insight-admin status`; open `https://<sdk-host-ip>:9900` |
| **pyneat** on the DevKit (only for the Python example) | Python bindings | `source ~/pyneat/bin/activate && python3 -c "import pyneat"` |
| **OpenCV for Python** (`cv2`) in the same environment (only for the Python example) | Probing the stream's size and frame rate; not needed if you pass `--width --height --fps` | `python3 -c "import cv2"` |
| **sima-cli** on the DevKit (only for Example 2) | Downloading the YOLO26 model | `sima-cli --help` |

In the commands below:

- **`<sdk-host-ip>`** is the machine running the SDK and Insight: the IP on the **Insight (remote)**
  line of `neat`.
- **`<devkit-ip>`** is your DevKit's address.
- **`/path/to/demo-neat`** is where you cloned this repository. Clone it into your host workspace
  (`cd ~/workspace && git clone <repo-url> demo-neat`) so it is `/workspace/demo-neat` in the SDK,
  and on the DevKit too when the DevKit shares the SDK's workspace over NFS. Without NFS, the
  DevKit copy lives in `~/demo-neat` (see [step D](#d-put-the-folder-on-the-devkit)). Everything
  here lives in its `custom-plugin/` folder.
- The DevKit must be able to reach the SDK host on **TCP 8554** (RTSP in) and the `videoUDP` /
  `metadataUDP` ranges that `neat` reports, e.g. **UDP 9000–9023 / 9100–9123** (video and boxes out).

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
2. Go to **Media Sources** → **Import Media** → **Catalog** and import a 720p30 H.264 variant.
3. Go to **Streaming**, assign the video to `src1` and start it. This creates the RTSP stream.

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
to do: the board sees the same path. Otherwise copy the folder into your home folder on the
board, and use **`~/demo-neat`** wherever the DevKit commands below say `/path/to/demo-neat`:

```bash
ssh sima@<devkit-ip> mkdir -p demo-neat
rsync -a /path/to/demo-neat/custom-plugin/ sima@<devkit-ip>:demo-neat/custom-plugin/
```

Copy the folder again after every rebuild.

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
gst-inspect-1.0 framemarker      # "Frame marker", NV12/I420 pads, 5 framemarker properties
                                 # (plus the standard name, parent and qos)
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
frames=86 pulled_fps=17.2 last_frame_id=85
frames=241 pulled_fps=30.1 last_frame_id=240
frames=392 pulled_fps=30.0 last_frame_id=391
```

A progress line follows every 5 seconds; the first one is lower because it includes startup. Open
the viewer (section I). You should see grayscale video with a green border and the moving bar.
Stop with **Ctrl-C**: the app closes the pipeline and prints `stopped after N frames`.

Useful options: `--mode none|grayscale|invert`, `--border N` (0–256), `--color 0xRRGGBB`,
`--no-bar`, `--log-interval N` (the plugin itself prints `[framemarker ...] frames=... fps=...`
every N frames), `--frames N` (stop after N frames), `--print-graph` (print the node list and
the pipeline plan), and `--custom` (use `nodes::Custom` instead of the typed Node). `--help`
lists every option.

To see the exact GStreamer launch string, start the app with `SIMA_PIPELINE_STRING_DEBUG=1` in
front of the command. Neat then prints it while the graph builds, on lines starting with
`[PIPELINE:`.

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
export SIMA_CLI_CHECK_FOR_UPDATE=0   # otherwise a pending sima-cli update prompt can abort the download
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
Press Ctrl-C to stop.
frames=82 fps=16.4 avg_boxes=10.5
frames=233 fps=30.1 avg_boxes=11.2
frames=383 fps=30.0 avg_boxes=9.5
```

`--labels` can be left out: the app reads `coco_labels.txt` from its own folder (`apps/build/`).
The other options of Example 1 work here too, except `--custom`; `--mode` defaults to `none` and
`--color` to `0xFF0000`. Extra options: `--min-score 0.35` (detection threshold),
`--metadata-port-base 9100` and `--no-frame-timestamps` (below).

The viewer shows colour video with a **red border and bar** (the plugin) and **labelled boxes**
(YOLO26). Try `--mode grayscale`: the picture turns gray but the boxes don't change, because the
model receives the clean frames on its own branch.

> **60 fps sources.** The app stamps each metadata message with its frame's timestamp, so the boxes
> stay locked to the frame they belong to. That works on a 30 fps stream. With Insight 0.0.8 the
> viewer draws no boxes on a **60 fps** stream. Use a 30 fps source, or add
> **`--no-frame-timestamps`**: the viewer then pairs each message with the newest frame, and boxes
> trail fast-moving objects slightly.

### I. Watch it in Insight (any browser)

Open **`https://<sdk-host-ip>:9900`** → **Video Viewer** → channel **0**. It's HTTPS, so accept
the certificate warning. For a direct link, run
`curl -k "https://127.0.0.1:9900/api/viewer-url?src=0"` in the SDK.

> **Channels:** for a single stream, use **channel 0**. It's the apps' default (`--channel` can
> be left out), and Insight's default single-stream view shows channel 0. Other channels are only
> needed when you run several streams at once. The SDK exposes one channel per port in the `videoUDP` range
> (e.g. UDP 9000–9023 for video, 9100–9123 for boxes); a channel outside that range receives nothing. Check yours
> with `neat --json` → `videoUDP`.

---

## 8. How to tell it is working

Work down the list. Each level proves one more link in the chain.

| # | Question | How to check | Pass looks like |
|---|---|---|---|
| 1 | Is it built for the DevKit? | `file install/lib/gstreamer-1.0/libgstframemarker.so` (SDK) | `ELF 64-bit ... ARM aarch64` |
| 2 | Can GStreamer load it? | `gst-inspect-1.0 framemarker` (DevKit) | Element details, pads `NV12, I420`, 5 framemarker properties (plus `name`, `parent`, `qos`) |
| 3 | Does it process frames correctly? | `./scripts/test_plugin.sh` | `PASS`, plus JPEGs in `out/` showing the border, bar and mode |
| 4 | Does it run inside Neat? | `SIMA_PIPELINE_STRING_DEBUG=1 stream_to_insight ... --print-graph` | The graph lists `n7: FrameMarker`, and the `[PIPELINE:...]` launch string contains `framemarker name=n7_framemarker_<N> ...`; steady `pulled_fps` ≈ source fps; no `build.plugin_missing` |
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

## 9. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `build.plugin_missing ... framemarker` in a Neat app, although `gst-inspect-1.0` finds it | The Neat runtime replaced the plugin path | `source scripts/env.sh` in **this** shell (sets `GST_PLUGIN_PATH_1_0` + `SIMA_GST_NEAT_ONLY=0`) |
| `No such element or plugin 'framemarker'` in `gst-inspect-1.0` | Path not set, or the file isn't on the board | `source scripts/env.sh`; `ls install/lib/gstreamer-1.0/` |
| Plugin reported as blacklisted | An earlier broken build was cached | `rm -rf ~/.cache/gstreamer-1.0; rm -f /tmp/sima_gst_registry_*` (the second is the Neat apps' registry) and retry |
| `wrong ELF class` / cannot open shared object | Built for the PC instead of the DevKit | Rebuild in the SDK shell; `file` must say `ARM aarch64` |
| `not-negotiated` / caps error at build | Incoming format isn't NV12/I420 | Feed decoded NV12 (as the apps do), or add the format to the plugin |
| App runs at full fps but Insight shows nothing | Channel outside the `videoUDP` range `neat` reports (e.g. 0–23), or UDP blocked between DevKit and SDK host | Use a channel in `neat --json` → `videoUDP`; check `api/ingest/stats` shows packets |
| `could not probe stream geometry` / stream won't open | The Insight source isn't playing, or the URL/port is wrong | Start the source (step A); check `rtsp://<sdk-host-ip>:8554/srcN` |
| Boxes but no plugin marks (Example 2) | Viewing another channel, or `--border 0 --no-bar --mode none` | Check the options and `--channel` |
| fps drops after changing the plugin | Per-frame code too slow | Measure with `--log-interval 150` (the plugin's own fps); optimise or process fewer pixels |

More detail: [tutorial Step 8](CUSTOM_PLUGIN_TUTORIAL.md#step-8-debugging).

---

## 10. Where to go next

- **[CUSTOM_PLUGIN_TUTORIAL.md](CUSTOM_PLUGIN_TUTORIAL.md)**: the full guide. Start there for
  [how the plugin works inside](CUSTOM_PLUGIN_TUTORIAL.md#step-2-write-the-element) and
  [writing your own plugin from this one](CUSTOM_PLUGIN_TUTORIAL.md#28-start-your-own-element-from-this-one).
  It also covers the Node wrapper, where to put a plugin in the graph, debugging, the rules for a
  well-behaved plugin, deployment, an [FAQ](CUSTOM_PLUGIN_TUTORIAL.md#faq) and a
  [glossary](CUSTOM_PLUGIN_TUTORIAL.md#glossary).
- **[plugin/gstframemarker.c](plugin/gstframemarker.c)**: read the code top to bottom; it's
  commented.
- **Neat library docs** (in the SDK under `/neat-resources/core-src/docs/`):
  `develop-apps/development-workflow/graph.mdx` for building graphs, and
  `develop-apps/advanced-concepts/execution-model/gstreamer_layer.md` for how Neat sits on
  GStreamer.
- **GStreamer plugin writer's guide:**
  <https://gstreamer.freedesktop.org/documentation/plugin-development/>
