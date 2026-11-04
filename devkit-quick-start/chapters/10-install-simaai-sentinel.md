# Chapter 10 — Install simaai-sentinel

*See what the chip is actually doing: temperature, power, CPU and memory, including MLA memory.*

---

## Why bother

Without it you are guessing. When an application runs slower than expected, the useful question is
whether a CPU core is pinned, the board is thermally throttling, or memory is running out —
`simaai-sentinel` answers that in one screen. It is the first thing to install after the board is
on the network.

Neat uses it too: from Neat Library 0.6.0, `Model::benchmark()` and the board telemetry APIs read
power, temperature, CPU and memory from Sentinel. Without it, benchmarks still report latency and
FPS, but no power or temperature.

---

## Install

```bash
# On the board
sima-cli neat install sentinel
```

## Run

```bash
# On the board
simaai-sentinel
```

A live view of temperature, power draw, CPU load and memory, including the memory the MLA holds. Leave it in a second
terminal while your application runs.

For a single snapshot — useful in scripts, or to capture a number for a bug report:

```bash
# On the board
simaai-sentinel table --once
```

```text
Metric             Group          Value  Unit   Status
-----------------------------------------------------------
MLA-0              MLA            56.15  C      normal
...
CVU-5              CVU            54.27  C      normal
...
LM96-1             Board          49.00  C      normal
CPU%               CPU              5.7  %      normal
Load1              CPU             0.39  load   normal
Mem%               Memory           2.9  %      normal
MLAMB              MLA            105.8  MB     normal
EV74MB             EV74            14.1  MB     normal
CMAFree            Memory       17951.3  MB     normal
...
c0                 CPU              2.0  %      normal
...
```

A trimmed snapshot, taken with no application running. Temperatures are per block (MLA, CVU, APU, board sensors), `MLAMB` and
`EV74MB` are memory held by the accelerators, and `c0`, `c1` … are the individual CPU threads.

---

## What to look at

| Reading | What it tells you |
|---|---|
| **MLA** | The `MLA-*` rows are temperatures and `MLAMB` is the memory the MLA holds. Sentinel does not show an MLA busy percentage; to see how much of each frame the MLA works, time it with `Model.benchmark()` and `include_mla_stage_stats` (see the [Neat measurement docs](https://developer.sima.ai/software/develop-apps/development-workflow/model)). |
| **CPU load** | A pinned core usually means CPU-side work that could move to the CVU or a hardware codec. |
| **Temperature** | Sustained high temperature leads to throttling, and throughput that drifts down over minutes. |
| **Power** | Typical draw is 8–12 W. |

Remember the board reports **16 logical threads** — 8 dual-threaded Cortex-A65 cores. A load of 8
is half the machine, not all of it.

---

## Also worth having: htop

Not SiMa-specific, but the standard tool for seeing what is running:

```bash
# On the board
sudo apt update
sudo apt install -y htop
htop
```

`htop` shows **16 bars** for the reason above. Nothing is wrong.

The two tools answer different questions: `htop` tells you which *process* is busy,
`simaai-sentinel` tells you which *part of the chip* is busy.

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 9 · Check and update the board image](09-check-and-update-image.md) | [All chapters](../README.md) | [Chapter 11 · Install pyneat](11-install-pyneat.md) |
