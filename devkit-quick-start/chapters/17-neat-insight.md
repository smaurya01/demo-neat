# Chapter 17 — Neat Insight

*The browser console for seeing what your application is actually doing.*

---

## What it is

Insight is a browser-based console for Neat vision applications. It has two main jobs:

1. **Visualising your application's output.** Your application sends Insight the video frames and,
   alongside them, the metadata for each frame — the frame ID and the detections (bounding boxes,
   labels, confidence scores). Insight pairs the metadata with its frame and **draws the overlay**
   in the browser. Your application never has to render boxes itself: it sends frames and
   detections, and Insight does the visualisation.

2. **Creating RTSP streams.** No camera on your desk? Import a video file and Insight turns it into
   a live RTSP source your application can consume, exactly as it would a real camera.

Around those two, it also shows what is in the workspace and what the runtime logged — useful when
the output is not what you expected.

The development loop it supports:

<p align="center">
  <img src="../images/insight-loop.jpg" alt="Insight loop: upload media, convert to RTSP sources, run the application, view output, read metrics" width="720">
</p>

---

## Opening it

Insight is **bundled with the Neat SDK** ([Chapter 15](15-install-neat-sdk.md)). When the SDK
container starts, Insight is installed, configured and supervised automatically — there is nothing
to install.

**Ask the SDK for the URL.** Run `neat` inside the SDK container:

```bash
sima-user@sdk:/workspace$ neat
```

```text
Components
  ...
  neat-insight           0.0.8              channel=main (assumed) latest=bedca8f560ca status=Running venv=/opt/neat-insight/venv
  ...

Web Access
  Local access uses a browser on this SDK host; remote access uses another machine.
  Prefer the local URL on this host; it remains valid even if the host network IP changes.
  Insight (local)    https://127.0.0.1:9900
  Insight (remote)   https://<host-ip>:9900
  ...

Exposed Ports
  Name               Protocol Host Port (Start) Host Port (End)
  ------------------ -------- ----------------- ---------------
  ...
  mainUI             tcp      9900              -
  metadataUDP        udp      9100              9123
  rtsp.tcp           tcp      8554              -
  videoUDP           udp      9000              9023
  ...
  webRTC             udp      40000             40047
  ...
```

**Insight (local)** is the address to open in a browser on the SDK host; use **Insight (remote)**
from another machine. `status=Running` confirms the service is up. Do not assume 9900 — the SDK
picks another host port if 9900 is already taken, and this output is the only place that tells you
which one it used.

The `Exposed Ports` table gives you the ports your application will send to: `videoUDP` from 9000 and
`metadataUDP` from 9100, one port per channel. [Chapter 18](18-cpp-video-app.md) uses both.

For scripting, `neat --json` gives the same information — read `insight.webUiUrl` and the
`exposedPorts` array.

| Browsing from | URL |
|---|---|
| The machine running the SDK | the **Insight (local)** URL from `neat`, usually <https://127.0.0.1:9900> |
| Another machine on the network | the **Insight (remote)** URL from `neat`, e.g. `https://<host-ip>:9900` |

It is **HTTPS with a self-signed certificate**, so accept the browser warning. If the page does not
load at all:

```bash
sima-user@sdk:/workspace$ insight-admin status
```

---

## The tabs

| Tab | What it does |
|---|---|
| **Workspace** | Browse project files and inspect generated model, package and profiling artifacts |
| **Media Sources** | Get test media in three ways — local upload, the built-in catalog, or a YouTube URL — then preview, inspect and delete it |
| **Streaming** | Turn uploaded media into live streams on numbered paths — `src1`, `src2`, and so on — port `8554` |
| **Video Viewer** | WebRTC output with metadata overlays — detection, classification, pose, segmentation, tracking |
| **Stats** | System and application metrics |

Before you start a stream, check two things: the **NFS mount** is present on the workspace (run
`dk status` in the SDK and check its `NFS mount path` line), and the paired **devkit-ip** shows in
the top-right corner of the UI.

---

## Getting test media in

**Media Sources → Import Media** offers three routes, as tabs across the top of the dialog:

<p align="center">
  <img src="../images/insight-import-catalog.jpg" alt="Insight Import Media dialog on the Catalog tab, showing assets, variants and the codec filter" width="760">
</p>

### Local files

Upload your own clip — `mp4`, `mov`, `avi`, `mkv` or `webm`. Use this when you are testing against
footage that matters to you: your own camera, a customer sample, the awkward scene that breaks
your pipeline.

### Catalog *(the easy option)*

A built-in library of scenes prepared for exactly this job — parking garages, gym activity,
highway and intersection traffic, people walking. Browse by **asset type** or **asset name**, or
search.

What makes the catalog worth reaching for first is that each scene ships as **22 variants**. Pick
the asset, then pick the encoding you actually want:

| Control | What it gives you |
|---|---|
| **Resolution & FPS** | From small previews up to `1080p120` — so you can choose `720p30` and match it exactly in your application config |
| **Codec** | H.264 or H.265. The [Chapter 18](18-cpp-video-app.md) app expects H.264 |
| Variant list | Every combination, each with its download size — `1080p120 / 120 fps / H.264 / 1920×1080`, 53.6 MB |

Selecting a variant shows its path and total size before you commit:

```text
parking_garage_cars/1080p120/1080p120_h264.mp4
Total selected: 53.6 MB
```

Press **Import**. No transcoding, no guessing at frame rates — you know the resolution, fps and
codec because you chose them, which is precisely what the application config needs to be told.

### YouTube

Paste a YouTube URL and Insight brings the video in. Handy for grabbing a scene resembling your
deployment when you have neither your own footage nor a catalog asset that fits.

---

## Turning a video file into an RTSP source

1. Open Insight in a browser and go to **Media Sources**. Import your video by any of the three
   routes above.

2. Go to **Streaming**. Assign files to `src1`, `src2`, … by hand, or use **Auto Assign**
   to spread unique files across the slots. Start one source, or use **Bulk Start** to start all
   of them.

   <p align="center">
     <img src="../images/insight-rtsp.jpg" alt="Insight Streaming tab with media assigned to source slots" width="720">
   </p>

3. Point your application at the source — the base URL the Streaming tab shows plus `/src1`,
   e.g. `rtsp://<host-ip>:8554/src1`. `<host-ip>` is the host IP the DevKit uses: the IP on the
   **Insight (remote)** line of `neat`, or the output of
   `curl -sk https://127.0.0.1:9900/api/server-ip`. Never `127.0.0.1` — on the board, that is the
   board itself. Then run the application.

4. Watch the result in **Video Viewer**, where you can confirm that frames are arriving on the
   channel you expect and that the overlays line up with them.

   <p align="center">
     <img src="../images/insight-viewer.jpg" alt="Insight Video Viewer showing application output with detection overlays" width="720">
   </p>

---

## When it goes wrong

| Symptom | Fix |
|---|---|
| Page will not load | `insight-admin status` to confirm it is up, then re-check the port with `neat --json` |
| Certificate warning | Expected — Insight generates its own certificate. Accept it |
| No video in the viewer | Confirm the RTSP source is started, and that the application is pointed at the same `src` path |
| No `devkit-ip` in the corner | The SDK is not paired. Re-run `sima-cli sdk setup --devkit <devkit-ip>` |

---

## Where to read more

- [Insight documentation](https://developer.sima.ai/software/tools/insight/) — the primary reference
- [Concepts](https://developer.sima.ai/software/tools/insight/concepts) ·
  [Install and upgrade](https://developer.sima.ai/software/tools/insight/install-upgrade) ·
  [User interface](https://developer.sima.ai/software/tools/insight/user-interface)
- [github.com/sima-neat/insight](https://github.com/sima-neat/insight) — source, releases and issues
- [neat_insight.md](../../installation/neat_insight.md) — the longer version of this chapter

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 16 · The `dk` command](16-devkit-tool-dk.md) | [All chapters](../README.md) | [Chapter 18 · A C++ video application](18-cpp-video-app.md) |
