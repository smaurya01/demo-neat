# Chapter 11 — Install simaai-sentinel

*See what the chip is actually doing: temperature, power, CPU, memory and MLA usage.*

---

## Why bother

Without it you are guessing. When an application runs slower than expected, the useful question is
whether the MLA is saturated, the CPU is pinned, or the board is thermally throttling —
`simaai-sentinel` answers that in one screen. It is the first thing to install after the board is
on the network.

---

## Install

```bash
sima@modalix:~$ sima-cli neat install sentinel
```

## Run

```bash
sima@modalix:~$ simaai-sentinel
```

A live view of temperature, power draw, CPU load, memory and MLA utilisation. Leave it in a second
terminal while your application runs.

For a single snapshot — useful in scripts, or to capture a number for a bug report:

```bash
sima@modalix:~$ simaai-sentinel table --once
```

---

## What to look at

| Reading | What it tells you |
|---|---|
| **MLA utilisation** | Low while your app runs? The bottleneck is elsewhere — pre-processing, decode, or the CPU. |
| **CPU load** | A pinned core usually means CPU-side work that could move to the CVU or a hardware codec. |
| **Temperature** | Sustained high temperature leads to throttling, and throughput that drifts down over minutes. |
| **Power** | Typical draw is 8–12 W. |

Remember the board reports **16 logical threads** — 8 dual-threaded Cortex-A65 cores. A load of 8
is half the machine, not all of it.

---

## Also worth having: htop

Not SiMa-specific, but the standard tool for seeing what is running:

```bash
sima@modalix:~$ sudo apt update
sima@modalix:~$ sudo apt install -y htop
sima@modalix:~$ htop
```

`htop` shows **16 bars** for the reason above. Nothing is wrong.

The two tools answer different questions: `htop` tells you which *process* is busy,
`simaai-sentinel` tells you which *part of the chip* is busy.

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 10 · Check and update the board image](10-check-and-update-image.md) | [All chapters](../README.md) | [Chapter 12 · Install pyneat](12-install-pyneat.md) |
