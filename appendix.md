# Appendix: Neat Insight and DevKit recovery

Operational recipes the apps in this repo need but none of them owns:

- **Neat Insight** serves the RTSP test sources the apps read, and displays the video and
  detections the apps send back, in a browser, with nothing to install on your machine.
- **Recovery:** how to un-wedge the DevKit when the MLA or a decoder stops responding.

← Back to the [repo README](README.md)

## Table of Contents

- [1. Neat Insight: sources in, video out](#1-neat-insight-sources-in-video-out)
  - [1.1 Open Insight](#11-open-insight)
  - [1.2 Serve a video as an RTSP source](#12-serve-a-video-as-an-rtsp-source)
  - [1.3 Confirm the source before blaming the app](#13-confirm-the-source-before-blaming-the-app)
  - [1.4 Send an app's output to Insight](#14-send-an-apps-output-to-insight)
  - [1.5 Watch it in the Video Viewer](#15-watch-it-in-the-video-viewer)
- [2. DevKit: recovery when the hardware wedges](#2-devkit-recovery-when-the-hardware-wedges)
  - [2.1 A leftover app is still holding the hardware](#21-a-leftover-app-is-still-holding-the-hardware)
  - [2.2 The MLA is blocked / the app hangs on model load](#22-the-mla-is-blocked--the-app-hangs-on-model-load)
  - [2.3 An LLiMa model stops loading after a few sessions](#23-an-llima-model-stops-loading-after-a-few-sessions)
  - [2.4 NEAT install blocked by `simaai-memory-lib`](#24-neat-install-blocked-by-simaai-memory-lib)
  - [2.5 Helper commands: find what is holding the hardware](#25-helper-commands-find-what-is-holding-the-hardware)
  - [Suggested order](#suggested-order)
- [3. Quick reference](#3-quick-reference)

---

## 1. Neat Insight: sources in, video out

Insight runs inside the SDK container. It hosts the RTSP test sources the apps read, and it
receives the H.264/RTP video (and, for some apps, JSON metadata) the apps send back, drawing both
in a browser viewer.

```
            RTSP  rtsp://<sdk-host-ip>:8554/srcN
Insight  ─────────────────────────────────────────►  DevKit app
(SDK)    ◄─────────────────────────────────────────
            video  UDP <sdk-host-ip>:9000+N   (channel N)
            metadata UDP <sdk-host-ip>:9100+N (apps that send it)
```

Full documentation: [developer.sima.ai/software/tools/insight](https://developer.sima.ai/software/tools/insight/).

### 1.1 Open Insight

In the SDK shell:

```bash
neat --json
```

- `insight.webUiUrl` is the Insight UI, `https://<host>:9900`, for your browser. The certificate is
  self-signed, so accept the browser warning the first time.
- `exposedPorts` lists the ports the DevKit must use: `rtsp.tcp` (RTSP sources, default 8554),
  `videoUDP` (video in, default from 9000) and `metadataUDP` (metadata in, default from 9100).
  **Use this map, not the defaults.** If a default port was taken, the SDK maps a different one,
  and the range also sets how many viewer channels you have.
- `curl -sk https://127.0.0.1:9900/api/server-ip` reports the host address the **DevKit** should
  connect to. That is `<sdk-host-ip>` everywhere below. It can differ from the host in `webUiUrl`
  when the host has more than one network, so always use `/api/server-ip` for RTSP URLs,
  `udp_host` and `insight_host`. Never give the DevKit `127.0.0.1`: that is the board itself.

### 1.2 Serve a video as an RTSP source

Insight's media sources are numbered slots, `src1`, `src2` and so on, each looping one video
file. In the UI, open **Media Sources**, assign a video to a slot, and start it. Use a catalog
video, or upload your own.

The same from the SDK shell:

```bash
curl -sk -F "file=@my_video.mp4" https://127.0.0.1:9900/api/upload/media     # optional: your own clip
curl -sk https://127.0.0.1:9900/api/mediasrc/videos                          # what can be assigned
curl -sk -H "Content-Type: application/json" \
  -d '{"index":1,"file":"my_video.mp4","transport":"rtsp"}' \
  https://127.0.0.1:9900/api/mediasrc/assign
curl -sk -H "Content-Type: application/json" -d '{"index":1}' \
  https://127.0.0.1:9900/api/mediasrc/start
```

The DevKit then reads `rtsp://<sdk-host-ip>:8554/src1` (8554, or the mapped `rtsp.tcp` port from
`neat --json`). That is the `<rtsp-url>` to put in an app's `config/default.conf`. For many sources at once, assign several slots and start them
together with `/api/mediasrc/start-bulk` (`{"count": N}` starts the first N *assigned* slots in
index order).

Use **H.264 with no B-frames** for the apps. Insight re-encodes uploaded H.264 `.mp4` files for
low-latency RTSP (baseline profile, no B-frames, frame rate kept), and its catalog clips are already
in that form. H.265 and MJPEG uploads are served as they are.

### 1.3 Confirm the source before blaming the app

**Always check the source before debugging an app.** The source frame rate is the hard ceiling
on any fps an app can claim, and a source that is not playing looks identical to a broken
pipeline.

- In **Media Sources**, the slot must show the right file and state **playing**.
- Check the clip's resolution and frame rate:

  ```bash
  curl -sk -H "Content-Type: application/json" -d '{"path":"my_video.mp4"}' \
    https://127.0.0.1:9900/api/media-info
  # {"codec":"H.264","frame_rate":"30/1","width":1280,"height":720,...}
  ```

  Apps that pin the source rate need an exact integer rate such as `30/1`. A `30000/1001`
  (29.97) clip fails caps negotiation.
- `curl -sk https://127.0.0.1:9900/api/mediasrc` lists every slot's file and state. The RTSP URLs
  it returns use `127.0.0.1`, which only works inside the SDK container. For the DevKit, replace
  that host with `<sdk-host-ip>`.

### 1.4 Send an app's output to Insight

Every app that streams video sends it to UDP port **9000** (single-stream apps) or **9000 + i** for
stream `i` (multi-stream apps; some also take a `udp_port_stride` or a per-stream port override).
`pcb-defect-detection-yolo26n` writes JPEGs and `benchmark` writes JSON, so they send nothing.
Those ports are Insight's viewer channels: port `9000 + N` is channel `N`. To see the output, set
the app's output host to the Insight host:

```
udp_host=<sdk-host-ip>            # most apps
insight_host=<sdk-host-ip>        # single-stream-yolo-insight, 16stream4model
```

- **Burned-in overlays:** most apps draw the boxes into the video before sending it, so video
  alone is enough.
- **Metadata overlays:** `single-stream-yolo-insight` and `16stream4model` send the original
  video untouched and the detections as JSON metadata on `9100 + N`. Insight matches the two on
  the RTP timestamp and draws the boxes itself. `usb-camera-yolo26m` can also send metadata
  (set `metadata_host`; off by default).
- Channel `N` exists only if the SDK maps port `9000 + N`. Check `videoUDP` in `neat --json`
  before running an app with more streams than you have channels.

### 1.5 Watch it in the Video Viewer

In the Insight UI, open **Video Viewer** and select the channels. Or get a direct link for, say,
channels 0–3:

```bash
curl -sk "https://127.0.0.1:9900/api/viewer-url?src=0,1,2,3"
```

The link it returns is on the viewer's own port (`videoUI`, default 8081) at `127.0.0.1`, so it
opens only in a browser on the SDK host. From another machine, replace `127.0.0.1` with the host's
address.

- A channel with no picture is not receiving. Check that the app is running, that its
  `udp_host` / `insight_host` is the Insight host's address as the DevKit sees it, and that the
  port is in the mapped range.
- **Keep one viewer open** while checking metadata overlays. Insight renders metadata reliably
  for a single viewer only.
- The fps each app prints is its own measurement. The viewer shows what actually arrived.

---

## 2. DevKit: recovery when the hardware wedges

A crashed or force-killed app can leave the MLA, its mailbox devices, or the hardware decoders
claimed. The next run then hangs, fails to load a model, or fails at decoder setup, and nothing
in the error message points at the real cause.

> **These are recovery commands, not routine ones.** Several kill processes bluntly. Know what
> else is running on the board before you fire them.

### 2.1 A leftover app is still holding the hardware

An app started with `dk` or over SSH can outlive the session that started it, for example when
it runs until Ctrl-C and the terminal closes. It keeps the decoders and the MLA, so the next run
fails, often at decoder setup (`runtime.element_failed` on a `decoder_*` stage). Look for it
first:

```bash
pgrep -fa <app-binary-or-script>     # e.g. pgrep -fa 16stream4model
kill <PID>                           # a clean stop first; kill -9 only if it will not exit
```

### 2.2 The MLA is blocked / the app hangs on model load

1. Find the stuck process and kill it:

   ```bash
   top          # find the hung app, note its PID
   sudo kill -9 <PID>
   ```

2. Reset the runtime:

   ```bash
   bash /usr/bin/fix_devkit_runtime.sh
   ```

This clears most "the model will not load" and "the MLA is stuck" symptoms.

### 2.3 An LLiMa model stops loading after a few sessions

Repeated load/unload cycles can leave the app-complex service in a bad state:

```bash
sudo systemctl restart simaai-appcomplex
```

### 2.4 NEAT install blocked by `simaai-memory-lib`

Only if installing NEAT core **fails with a conflict** against the memory library: remove it and
retry the install. A working board has it installed, so do not remove it otherwise. Run this on
the Modalix board:

```bash
sudo apt remove --purge simaai-memory-lib simaai-memory-lib-dev
```

### 2.5 Helper commands: find what is holding the hardware

Run on the Modalix board. Use these when `fix_devkit_runtime.sh` alone did not clear it.

| Command | What it does |
| --- | --- |
| `sudo fuser -v /dev/m4_lp_mbox` | Show which process holds the **MLA mailbox**. On a healthy board that is `mlashmcomplex`, the app-complex daemon: **never kill it**. Any other holder (a user app or a pyneat process) is the culprit. |
| `sudo fuser -v /dev/rpmsg*` | Show what holds the RPMsg devices. |
| `ps aux \| grep pyneat \| grep -v grep` | Find leftover `pyneat` processes still holding the runtime. |
| `sudo fuser -v <port>/tcp` | Show what holds a server port, for example `9998` for the GenAI server when a restart fails with "address in use". |
| `sudo fuser -k <port>/tcp` | Kill that process. **It kills the whole process, not just the socket.** |

### Suggested order

Escalate; do not start at the bottom.

1. `pgrep -fa <app>`: stop a leftover app.
2. `top`: kill the specific hung PID.
3. `bash /usr/bin/fix_devkit_runtime.sh`. It also stops and restarts `simaai-appcomplex` itself.
4. `sudo fuser -v /dev/m4_lp_mbox`: kill the PID it names **only if it is not `mlashmcomplex`**.
5. `ps aux | grep pyneat`: kill leftovers by PID.
6. Only then the blunt instrument: `sudo fuser -k <port>/tcp`.
7. Still stuck: reboot the board.

For LLiMa models that stop loading, restarting `simaai-appcomplex` (§2.3) is the direct fix.

---

## 3. Quick reference

| I want to… | Do this |
| --- | --- |
| Find the Insight UI and port map | `neat --json` (`insight.webUiUrl`, `exposedPorts`) |
| Find the address the DevKit should use | `curl -sk https://127.0.0.1:9900/api/server-ip` |
| Serve a video as RTSP | Insight **Media Sources**: assign a clip to `srcN` and start it → `rtsp://<sdk-host-ip>:8554/srcN` (or the mapped `rtsp.tcp` port) |
| Check a source's fps and resolution | `/api/media-info` (§1.3) |
| See an app's output | set `udp_host` / `insight_host` to the Insight host; open **Video Viewer** channel `N` (port `9000 + N`) |
| Get a viewer link | `curl -sk "https://127.0.0.1:9900/api/viewer-url?src=0,1,2,3"` |
| Next run fails at decoder setup | `pgrep -fa <app>`: a leftover run is holding the hardware |
| Un-wedge the MLA | kill the PID, then `bash /usr/bin/fix_devkit_runtime.sh` |
| Find what is holding the MLA | `sudo fuser -v /dev/m4_lp_mbox`  |
| LLiMa stopped loading models | `sudo systemctl restart simaai-appcomplex` |
| NEAT install blocked by the memory lib | `sudo apt remove --purge simaai-memory-lib simaai-memory-lib-dev` |
