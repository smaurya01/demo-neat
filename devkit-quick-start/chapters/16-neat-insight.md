# Chapter 16 — Neat Insight

*The browser console for seeing what your application is actually doing.*

---

## What it is

Insight is a browser-based inspection and test console for Neat vision applications. It answers the
questions that are tedious to answer from a terminal: is the video arriving, on which channel, is
the metadata aligned with the frames, what is in the workspace, and what did the runtime log while
it all happened.

It also solves the practical problem of having no camera on your desk — upload a video file and
Insight turns it into a live RTSP source your application can consume.

The development loop it supports:

<p align="center">
  <img src="../images/insight-loop.jpg" alt="Insight loop: upload media, convert to RTSP sources, run the application, view output, read metrics" width="720">
</p>

---

## Opening it

Insight is **bundled with the Neat SDK** ([Chapter 15](15-install-neat-sdk.md)). When the SDK
container starts, Insight is installed, configured and supervised automatically — there is nothing
to install.

| Browsing from | URL |
|---|---|
| The machine running the SDK | <https://localhost:9900> |
| Another machine on the network | `https://<host-ip>:9900` |

It is **HTTPS with a self-signed certificate**, so accept the browser warning. Confirm the service
is up with:

```bash
sima-user@sdk:/workspace$ insight-admin status
```

9900 is the default, but the SDK will pick another host port if 9900 is taken. To see what it
actually used:

```bash
sima-user@sdk:/workspace$ neat --json
```

Read `insight.webUiUrl` for the UI, and the `exposedPorts` array when pointing DevKit
applications, RTSP clients or external senders at it.

---

## The tabs

| Tab | What it does |
|---|---|
| **Workspace** | Browse project files and inspect generated model, package and profiling artifacts |
| **Media Library** | Upload, preview and delete test media — `mp4`, `mov`, `avi`, `mkv`, `webm` |
| **RTSP Source** | Turn uploaded media into live streams on three paths: `src1`, `src2`, `src3`, port `8554` |
| **Video Viewer** | WebRTC output with metadata overlays — detection, classification, pose, segmentation, tracking |
| **Stats** | System and application metrics |

Before you start a stream, check two things: the **NMS mount** is present on the workspace, and the
paired **devkit-ip** shows in the top-right corner of the UI.

---

## Turning a video file into an RTSP source

1. Open Insight in a browser and go to **Media Library**. Upload your video.

2. Go to **RTSP Source**. Assign files to `src1`, `src2` or `src3` by hand, or use **auto-assign**
   to spread unique files across the slots. Start one source, or bulk-start all of them.

   <p align="center">
     <img src="../images/insight-rtsp.jpg" alt="Insight RTSP Source tab with media assigned to source slots" width="720">
   </p>

3. Point your application at the source — `rtsp://127.0.0.1:8554/src1` — and run it.

4. Watch the result in **Video Viewer**, where you can confirm that frames are arriving on the
   channel you expect and that the overlays line up with them.

   <p align="center">
     <img src="../images/insight-viewer.jpg" alt="Insight Video Viewer showing application output with detection overlays" width="720">
   </p>

---

## Running Insight on the DevKit instead

Only worth doing when you want the console on the target device itself, or you are validating a
standalone DevKit with no SDK host:

```bash
sima@modalix:~$ sima-cli neat install insight@{release-tag}   # or insight@main
sima@modalix:~$ source ~/.simaai/neat-insight/venv/bin/activate
(venv) sima@modalix:~$ neat-insight --port 9900
```

Then browse to `https://<devkit-ip>:9900`.

Upgrading is the same install command. Inside the SDK, Insight lives in `/opt/neat-insight/venv`,
so upgrading the supervised instance means installing into that venv and restarting the service:

```bash
sima-user@sdk:/workspace$ NEAT_INSIGHT_VENV_DIR=/opt/neat-insight/venv sima-cli neat install insight@main
sima-user@sdk:/workspace$ supervisorctl restart neat-insight
```

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
| [Chapter 15 · Install the Neat SDK](15-install-neat-sdk.md) | [All chapters](../README.md) | [Troubleshooting](troubleshooting.md) |
