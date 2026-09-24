# Troubleshooting

*Symptom first. Find yours, work down the list in order.*

---

## Power and display

### The DevKit does not power on

The **red power LED** should light as soon as the board has power — there is no power button. Check
the 12 V adapter is fully seated in the barrel jack and that the outlet is live. No LED means a
power problem, and nothing else in this guide will help until it is lit.

### No HDMI output

1. Monitor on, and set to the correct input. This is the most common cause by a wide margin.
2. Re-seat the HDMI cable and **reboot** — HDMI is detected at boot, so a cable plugged in
   afterwards may not be seen.
3. Confirm the resolution is within limits: 4K @ 30 Hz or 1080p @ 60 Hz.
4. Confirm the board is alive over [serial](04-serial-console.md).
   A serial prompt means the board is fine and the fault is purely in the display path.

### Serial console output is garbled

Baud rate mismatch. Earlier firmware uses 921600 rather than 115200:

```bash
sima-user@host:~$ sima-cli serial -b 921600
```

---

## Network

### `ssh sima@modalix.local` does not resolve

`modalix.local` depends on **mDNS**, which is unreliable on Windows and blocked on many corporate
networks. Use the IP instead:

```bash
sima-user@host:~$ sima-cli device discover      # from the host
sima@modalix:~$   ip a | grep inet              # from the serial console
```

Then `ssh sima@<devkit-ip>`.

### `sima-cli device discover` finds nothing

- Host and board are on different subnets, or discovery traffic is blocked.
- The board is on **older firmware** that does not advertise itself. Reach it over
  [serial](04-serial-console.md), run
  `cat /etc/buildinfo`, and [update it](09-check-and-update-image.md).

### The board has an IP but no internet

Establish which half is broken first:

```bash
sima@modalix:~$ ping -c3 8.8.8.8              # routing
sima@modalix:~$ ping -c3 developer.sima.ai    # DNS
```

Address works, name fails → **DNS**. Both fail → **routing**.

Most common causes, in order:

1. No DNS server configured — set `ipv4.dns`
2. IP forwarding or NAT not enabled on the sharing host ([Chapter 5](05-internet-sharing.md))

### Internet sharing gives the board no address

- The board must be on the **DHCP** profile: `sudo nmcli connection up end0-dhcp`
  ([switching profiles](05-internet-sharing.md#switching-between-dhcp-and-static))
- The host's wired connection must be **Shared to other computers**, then toggled off and on
- The cable must be in the host's own Ethernet port, not a dock presenting a different interface
- Scan the sharing subnet: `nmap -sn 10.42.0.0/24 | grep report` (Ubuntu),
  `192.168.137.0/24` (Windows), `192.168.2.0/24` (macOS)

---

## sima-cli

### `sima-cli: command not found` after installing

Open a new shell, or:

```bash
sima@modalix:~$ source ~/.bashrc
sima@modalix:~$ which sima-cli
```

If `which` still finds nothing, call it by absolute path:

```bash
sima@modalix:~$ /data/.sima-cli/.venv/bin/sima-cli --version
```

### `sima-cli update` or an install fails to download

Three things, in order:

1. `sima-cli login` — are you authenticated?
2. Is your Developer Portal account **approved**? Registering is not enough, and approval is not
   instant.
3. Does the board have working internet? `ping -c3 developer.sima.ai`

Nearly every download failure is one of these three, usually the third.

---

## Storage

### `df -h` shows no NVMe

```bash
sima@modalix:~$ lsblk | grep nvme       # is the device present at all?
```

| Result | Action |
|---|---|
| Present, not mounted | `sima-cli nvme remount` |
| Present, no usable filesystem | `sima-cli nvme format` — **erases the drive** |
| Absent | Power down and re-seat the M.2 module |

### The NVMe was mounted, and now it is not

An image update can change the root filesystem and drop the mount. Re-check after every update:

```bash
sima@modalix:~$ df -h | grep nvme
sima@modalix:~$ sima-cli nvme remount
```

### The root filesystem is full

The 16 GB eMMC fills fast, and the symptoms rarely mention disk space — installs die halfway, logs
stop, services misbehave.

```bash
sima@modalix:~$ df -h /
sima@modalix:~$ du -sh ~/* 2>/dev/null | sort -h | tail
```

Move models, datasets and outputs to `/media/nvme` and point your applications there
([Chapter 8](08-mount-nvme.md)).

---

## Software versions

### The board is on 1.7 / Yocto

```bash
sima@modalix:~$ cat /etc/buildinfo | grep DISTRO
```

`DISTRO = poky` means a legacy Yocto image. `sima-cli update` **cannot** cross to eLxr — follow the
[eLxr Conversion Guide](https://developer.sima.ai/hardware/reference/tech-notes/elxr-conversion)
first. Units shipped before mid-December 2025 are typically Yocto.

### "It built but it won't run"

Almost always a version mismatch between board software, Neat Library and Neat SDK. Check all three
against the [compatibility table](09-check-and-update-image.md#which-version-should-you-be-on) and
upgrade them as a set.

```bash
sima@modalix:~$ cat /etc/buildinfo | grep DISTRO_VERSION
sima@modalix:~$ dpkg -l | grep -E 'neat|sima-neat' | awk '{print $2, $3}'
```

### Forgotten credentials

Defaults are `sima` / `edgeai`. If they were changed, contact your administrator or SiMa support.

---

## Neat SDK and `dk`

### Setup reinstalled everything, or there are now two SDK containers

You pressed `y` at setup prompt 6. Run setup again and press `n` to reuse the container you
already have:

```bash
sima-user@host:~$ sima-cli sdk setup --devkit <devkit-ip>
```

### Pairing fails

The host and the board must be on the same network and able to reach each other:

```bash
sima-user@host:~$ ping -c3 <devkit-ip>
```

If the ping fails, fix the network first ([Chapter 5](05-internet-sharing.md) or
[Chapter 6](06-connect-to-router.md)). If the board's IP has changed, run setup again with the new
one.

### `dk: command not found`

You are not in the SDK container. Use the terminal in the VS Code window attached to the
`sima-neat/sdk` container ([Chapter 15](15-install-neat-sdk.md)). If you are in the container and it
is still missing, reload your shell:

```bash
sima-user@sdk:/workspace$ source ~/.bashrc
```

### `dk status` shows `SSH status : not reachable`

The board is off, or its IP has changed since pairing. Check it is up, then run setup again on the
host with the current IP.

### `Path must be inside /workspace`

`dk` only runs files under `/workspace`, because that is the folder the board can see. Move the
script or binary into your project under `/workspace` and run it from there.

### `Refusing to run non-ARM64 binary`

The binary was built for your host, not for the board. Build it inside the SDK container, which
cross-compiles for the board ([Chapter 18](18-cpp-video-app.md), Step 5).

---

## Running apps on the board

### The next run hangs, or fails to load the model

A previous run is still going on the board and holding the MLA. This happens easily: stopping an
`ssh` command with `Ctrl-C` or `timeout` only stops it on your side, and the app keeps running on
the board. The next run then hangs while the graph builds, or fails with errors such as
`gstreamer.unclassified_element_failure` or `Input buffer allocation failed`.

Find it from a board shell (`dk shell`):

```bash
sima@modalix:~$ ps -eo pid,etime,cmd | grep -E 'python|rtsp_detector' | grep -v grep
```

Stop only the process you recognise as yours:

```bash
sima@modalix:~$ kill <pid>
```

**Never kill `mla_rt_service`** — it is the system service that runs the MLA.

---

## Building C++ apps

### `Could not find a package configuration file provided by "SimaNeat"`

CMake cannot see the SDK's copy of Neat. Configure with the prefix path, after deleting the failed
build:

```bash
sima-user@sdk:/workspace/rtsp-detector$ rm -rf build
sima-user@sdk:/workspace/rtsp-detector$ cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
```

### A binary that used to work no longer starts

It was built against an older Neat Library. Delete `build/` and build it again.

---

## Neat Insight

### The page will not load

Run `neat` in the SDK and check the `neat-insight` line says `status=Running`. Open the exact
address on the **Insight Web UI** row — it is `https://`, not `http://`, and the port is not
always 9900. Accept the certificate warning.

### No `devkit-ip` in the top-right corner

The SDK is not paired with a board. Run `sima-cli sdk setup --devkit <devkit-ip>` on the host.

### The application times out waiting for frames

The RTSP source is not running. In Insight's **RTSP Source** tab, start `src1` again, and check the
URL in your config matches the one Insight shows.

### No video in the Video Viewer

- `insight_host` in your config must be the Insight host's IP — the address on the Insight Web UI
  row of `neat` — and the board must be able to reach it.
- `video_port` must match the `videoUDP` row of `neat`.

### Video, but no boxes

- `metadata_port` must match the `metadataUDP` row of `neat`.
- Lower `min_score` in the config — the threshold may be filtering out every detection.

---

## Still stuck?

- [Miscellaneous](miscellaneous.md) — command reference and links
- [SiMa Developer Portal](https://developer.sima.ai/)
- [Firmware Update](https://developer.sima.ai/hardware/getting-started/firmware-update)

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 19 · Agentic development](19-agentic-development.md) | [All chapters](../README.md) | [Miscellaneous](miscellaneous.md) |
