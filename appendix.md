# Appendix — Host Setup & DevKit Recovery

Operational recipes that every app in this repo needs but none of them owns: how to stand up an
RTSP source on the host, how to watch the UDP/RTP output the apps publish, and how to un-wedge the
DevKit when the MLA stops responding.

← Back to the [repo README](README.md)

---

## 1. Host: a local RTSP source

The apps take an RTSP URL as input. If you do not have a camera, serve a video file instead. Two
pieces: an RTSP **server** (`mediamtx`) and an RTSP **publisher** (`ffmpeg`) that loops a file into it.

### 1.1 Start the RTSP server

```bash
docker run --rm -it --network=host -e MTX_RTSPADDRESS=:8555 bluenviron/mediamtx
```

- `--network=host` — the server must be reachable from the DevKit, not just from localhost.
- `MTX_RTSPADDRESS=:8555` — listen on 8555. Every config in this repo assumes port **8555**.
- `--rm -it` — foreground, and it cleans up on Ctrl-C. Leave it running in its own terminal.

### 1.2 Publish a video file into it, on a loop

```bash
ffmpeg -re -stream_loop -1 -i aa1.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -b:v 4M -maxrate 8M -bufsize 10M \
  -pix_fmt yuv420p -g 30 \
  -f rtsp rtsp://192.168.2.105:8555/stream
```

Replace `192.168.2.105` with **your host's LAN IP** — the address the DevKit will connect back to.
Not `127.0.0.1`: the board has to reach it.

Why each flag matters:

| Flag | Why |
| --- | --- |
| `-re` | Publish at **real time**, not as fast as ffmpeg can read. Without it you flood the server and the stream is meaningless as a frame-rate reference. |
| `-stream_loop -1` | Loop the file forever, so the stream does not die mid-test. |
| `-preset ultrafast -tune zerolatency` | Keep encoder latency out of your measurements. |
| `-pix_fmt yuv420p` | The pixel format the SiMa H.264 decoder expects. |
| `-g 30` | Keyframe every 30 frames, so a receiver joining late gets a picture within ~1 s instead of waiting. |

The stream is now at `rtsp://<host-ip>:8555/stream` — put that in the app's `config/default.conf`.

### 1.3 Confirm it before blaming the app

**Always probe the source before debugging an app.** The source frame rate is the hard ceiling on
any FPS an app can claim, and a stream that is not actually publishing looks identical to a broken
pipeline.

```bash
ffprobe -hide_banner -rtsp_transport tcp rtsp://192.168.2.105:8555/stream
```

---

## 2. Host: viewing the UDP/RTP output

The apps encode H.264 and push it out as RTP over UDP to `udp_host:udp_port` from their config. Run
these **on the machine you set as `udp_host`**.

### 2.1 One stream, with a live FPS readout

Example for port **5205**:

```bash
gst-launch-1.0 \
  udpsrc port=5205 buffer-size=2097152 \
    caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
  ! rtpjitterbuffer latency=100 \
  ! rtph264depay ! h264parse ! decodebin \
  ! videoconvert \
  ! fpsdisplaysink video-sink=autovideosink sync=false \
      text-overlay=true signal-fps-measurements=true
```

- `buffer-size=2097152` — a 2 MB socket buffer. At 1080p the default is too small and you lose
  packets, which shows up as a corrupt or undecodable picture rather than as an error.
- `rtpjitterbuffer latency=100` — reorders packets. Without it, out-of-order UDP looks like corruption.
- `decodebin` — picks whatever H.264 decoder the host has. More portable than naming `avdec_h264`.
- `sync=false` — render on arrival. With `sync=true` the sink paces to timestamps and appears to stall.
- `fpsdisplaysink … text-overlay=true` — burns the measured FPS into the window. This is your
  independent check on the app's own reported number.

### 2.2 Four streams at once, in a 2×2 grid

For the multi-stream apps. Each `udpsrc` is scaled to 640×360 and composited into one 1280×720 window:

```bash
gst-launch-1.0 -e \
  compositor name=mix background=black \
    sink_0::xpos=0   sink_0::ypos=0 \
    sink_1::xpos=640 sink_1::ypos=0 \
    sink_2::xpos=0   sink_2::ypos=360 \
    sink_3::xpos=640 sink_3::ypos=360 \
  ! videoconvert ! autovideosink sync=false \
  udpsrc port=5206 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! decodebin ! videoscale ! videoconvert ! video/x-raw,width=640,height=360 ! queue ! mix.sink_0 \
  udpsrc port=5208 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! decodebin ! videoscale ! videoconvert ! video/x-raw,width=640,height=360 ! queue ! mix.sink_1 \
  udpsrc port=5210 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! decodebin ! videoscale ! videoconvert ! video/x-raw,width=640,height=360 ! queue ! mix.sink_2 \
  udpsrc port=5212 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! rtph264depay ! decodebin ! videoscale ! videoconvert ! video/x-raw,width=640,height=360 ! queue ! mix.sink_3
```

- Ports step by **2** (5206, 5208, 5210, 5212) because each channel reserves a pair. Match them to the
  app's `udp_port` / `video_port_base` plus its channel index — check the app's config before assuming.
- `sink_N::xpos/ypos` place each tile. `queue` before each sink pad is required: without it, one slow
  stream stalls the whole compositor.
- `-e` sends EOS on Ctrl-C so the window closes cleanly instead of hanging.

A tile that stays black means **that** port is not receiving — the other three still render, which
makes this a quick way to see which stream died.

---

## 3. DevKit: recovery when the MLA wedges

A crashed or force-killed app can leave the MLA and its mailbox devices claimed. The next run then
hangs or fails to load a model, and nothing in the error message points at the real cause.

> **These are recovery commands, not routine ones.** Several kill processes bluntly. Know what else is
> running on the board before you fire them.

### 3.1 The MLA is blocked / the app hangs on model load

1. Find the stuck process and kill it:

   ```bash
   top          # find the hung app, note its PID
   kill -9 <PID>
   ```

2. Reset the runtime:

   ```bash
   bash /usr/bin/fix_devkit_runtime.sh
   ```

This is the first thing to try. Most "the model will not load" and "the MLA is stuck" symptoms clear here.

### 3.2 An LLiMa model stops loading after a few sessions

Repeated load/unload cycles can leave the app-complex service in a bad state:

```bash
sudo systemctl restart simaai-appcomplex
```

### 3.3 NEAT install blocked by `simaai-memory-lib`

If installing NEAT core fails because of a conflict with the memory library, remove it and retry the
install. **Run this on the Modalix board:**

```bash
sudo apt remove --purge simaai-memory-lib simaai-memory-lib-dev
```

### 3.4 Helper commands — find and clear what is holding the hardware

Run on the Modalix board. Use these when `fix_devkit_runtime.sh` alone did not clear it.

| Command | What it does |
| --- | --- |
| `sudo fuser -v /dev/m4_lp_mbox` | Show which process holds the **MLA mailbox**. This is usually the culprit. |
| `sudo fuser -v /dev/rpm*` | Show what holds the RPM devices. |
| `ps aux \| grep pyneat \| grep -v grep` | Find leftover `pyneat` processes still holding the runtime. |
| `sudo pkill -9 python3` | Kill **every** Python 3 process on the board. |
| `fuser -k 5001/tcp` | Kill whatever holds TCP **5001** (a stale server socket blocking a restart). |

**Two blast-radius warnings, because both of these bite:**

- `sudo pkill -9 python3` kills *all* Python on the board — including a Jupyter kernel you are running
  the tutorial notebooks from, and any other user's session. Identify the process with `fuser -v` or
  the `ps aux` line first and kill that PID specifically; reach for `pkill -9` only when that fails.
- `fuser -k 5001/tcp` kills the process holding the port, not just the socket.

### Suggested order

Escalate — do not start at the bottom.

1. `top` → kill the specific hung PID.
2. `bash /usr/bin/fix_devkit_runtime.sh`
3. `sudo fuser -v /dev/m4_lp_mbox` → kill the PID it names.
4. `ps aux | grep pyneat` → kill leftovers by PID.
5. `sudo systemctl restart simaai-appcomplex` (LLiMa / app-complex issues).
6. Only then the blunt instruments: `sudo pkill -9 python3`, `fuser -k 5001/tcp`.
7. Still stuck → reboot the board.

---

## 4. Quick reference

| I want to… | Do this |
| --- | --- |
| Serve a video file as RTSP | `docker run --rm -it --network=host -e MTX_RTSPADDRESS=:8555 bluenviron/mediamtx`, then the `ffmpeg -re -stream_loop -1 …` publisher |
| Check the RTSP source is alive, and its true FPS | `ffprobe -hide_banner -rtsp_transport tcp rtsp://<host>:8555/stream` |
| Watch one app's output, with FPS | `udpsrc port=<udp_port> … ! fpsdisplaysink` (§2.1) |
| Watch four streams at once | the `compositor` 2×2 pipeline (§2.2) |
| Un-wedge the MLA | kill the PID, then `bash /usr/bin/fix_devkit_runtime.sh` |
| Find what is holding the MLA | `sudo fuser -v /dev/m4_lp_mbox` |
| LLiMa stopped loading models | `sudo systemctl restart simaai-appcomplex` |
| NEAT install blocked by the memory lib | `sudo apt remove --purge simaai-memory-lib simaai-memory-lib-dev` |
