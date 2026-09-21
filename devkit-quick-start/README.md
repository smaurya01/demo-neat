# Modalix DevKit — Quick Start Guide

A chapter-by-chapter walkthrough that takes a Modalix DevKit 3.0 from a sealed box to running
object detection and a local LLM.

<p align="center">
  <img src="images/qsg-devkit-board.png" alt="Modalix DevKit 3.0" width="520">
</p>

Work through it in order the first time. After that, each chapter stands on its own — jump
straight to the one you need.

> **Versions referenced throughout:** DevKit software **2.1.3** · Neat SDK **2.1.3.0** ·
> Neat Library **0.4.0** · Neat Apps **0.5.0**
>
> **Default login:** `sima` / `edgeai`

---

## Chapters

### Part 1 — The hardware

| # | Chapter | What you get out of it |
|---|---|---|
| 1 | [Know the hardware](chapters/01-know-the-hardware.md) | What the MLSoC actually contains, and why the layout matters for your code |
| 2 | [The board and its connections](chapters/02-board-and-connections.md) | Every port, what's in the box, what you must supply yourself |
| 3 | [HDMI, keyboard and mouse](chapters/03-hdmi-and-peripherals.md) | Using the DevKit as a standalone desktop *(optional)* |
| 4 | [The serial console](chapters/04-serial-console.md) | The connection that works when the network doesn't — Linux, macOS, Windows, browser |

### Part 2 — Getting it on the network

Pick **one** of these two. They are alternatives, not steps.

| # | Chapter | Choose it when |
|---|---|---|
| 5 | [Share the host's internet](chapters/05-internet-sharing.md) | One cable to your laptop, no router available |
| 6 | [Connect to a router](chapters/06-connect-to-router.md) | **Recommended.** Normal LAN with DHCP and internet |

### Part 3 — Software setup

| # | Chapter | What you get out of it |
|---|---|---|
| 7 | [Install sima-cli](chapters/07-install-sima-cli.md) | The tool everything else depends on |
| 8 | [Mount the NVMe](chapters/08-mount-nvme.md) | Somewhere to actually put models and data |
| 9 | [Check and update the board image](chapters/09-check-and-update-image.md) | Get onto a known-good software version |
| 10 | [Install simaai-sentinel](chapters/10-install-simaai-sentinel.md) | See temperature, power, CPU and MLA usage |
| 11 | [Install pyneat](chapters/11-install-pyneat.md) | The Python API for the accelerator |

### Part 4 — Running something

| # | Chapter | What you get out of it |
|---|---|---|
| 12 | [Object detection on images](chapters/12-object-detection.md) | Your first real inference on the MLA |
| 13 | [LLiMa — search, list, pull](chapters/13-llima.md) | Run a language model on the board |

### Reference

- [Troubleshooting](chapters/troubleshooting.md) — symptom-first, for when something doesn't work
- [Miscellaneous](chapters/miscellaneous.md) — command reference, defaults, links, PCIe card notes

---

## Before you begin

You will need a **SiMa Developer Portal account**, and it must be approved before you can download
the SDK or firmware images. Approval is not instant — [sign up](https://community.sima.ai/signup)
now, then carry on reading while you wait.

---

| Contents | Next → |
|:---|---:|
| You are here | [Chapter 1 · Know the hardware](chapters/01-know-the-hardware.md) |
