# Chapter 4 — The serial console

*The connection that works when the network doesn't. Set this up before you need it.*

---

## Why this comes first

Every other way of reaching the board depends on networking, and networking is exactly what you are
about to change. Set a static address wrong, mistype a subnet, run an update that reboots the board —
and SSH is gone.

The serial console does not care. It is a direct wire to the board's console, independent of IP
addresses, DHCP, routing and firewalls. It is how you:

- Read the board's IP address when you don't know it
- Change network settings safely ([Chapters 5](05-internet-sharing.md)–[7](07-static-connection.md))
- Watch an update that reboots partway through ([Chapter 10](10-check-and-update-image.md))
- Recover a board you have locked yourself out of

Set it up now, while nothing is broken.

---

## The cable

| Board | Cable | Connects to |
|---|---|---|
| **Modalix DevKit** | USB-A → USB-C *(supplied in the box)* | The UART port |
| Modalix EA Kit | Flat connector — align the triangle mark to the rightmost pin | — |
| Modalix PCIe Card | USB-A → UART | The ARM UART port |

The other end goes into a USB-A port on your computer. A hub is fine.

---

## Linux and macOS

With `sima-cli` installed on your host ([Chapter 8](08-install-sima-cli.md)):

```bash
sima-user@host:~$ sima-cli serial
```

That is the whole thing — it finds the port and sets the baud rate for you.

**To exit:** `Ctrl` + `A`, then `Ctrl` + `X`.

Log in with `sima` / `edgeai`.

### If the output is garbled

Unreadable characters mean the baud rate is wrong. Earlier firmware runs at 921600 rather than
115200:

```bash
sima-user@host:~$ sima-cli serial -b 921600
```

### Without sima-cli

Any serial terminal works. The port is usually `/dev/ttyUSB0` on Linux or `/dev/tty.usbserial-*` on
macOS, at **115200 8N1**:

```bash
sima-user@host:~$ sudo screen /dev/ttyUSB0 115200
```

If you get a permissions error on Linux, add yourself to the `dialout` group and log out and back in:

```bash
sima-user@host:~$ sudo usermod -aG dialout $USER
```

---

## Windows

`sima-cli serial` is not the path here. Use a terminal program against the COM port.

### 1. Find the COM port

1. Open **Device Manager**
2. Expand **Ports (COM & LPT)**
3. Look for **USB Serial Port (COMx)** — note the number, e.g. `COM3`

If nothing appears there, the USB-serial driver is missing. Check under **Other devices** for an
unknown device, and install the driver for your adapter's chipset (commonly FTDI or Silicon Labs
CP210x).

### 2. Install a terminal

[PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) or Tera Term. Either is fine.

### 3. Connect

| Setting | Value |
|---|---|
| Connection type | **Serial** |
| Serial line | your COM port, e.g. `COM3` |
| Speed | **115200** |
| Data / parity / stop | 8 / none / 1 |

Click **Open**, press Enter, and log in with `sima` / `edgeai`.

> If the window fills with garbage, change the speed to **921600** and reopen.

---

## The browser console

There is a **Web Serial Console** that needs nothing installed at all:

**<https://developer.sima.ai/tools/serial/index.html>**

Open it in a Chromium-based browser (Chrome or Edge), pick the serial port when prompted, and you
have a console. Firefox and Safari do not support the Web Serial API.

This is the quickest option on a locked-down machine where you cannot install PuTTY or `sima-cli`.

---

## First things to do once you are in

```bash
sima@modalix:~$ ip a | grep inet          # what address does it have?
sima@modalix:~$ cat /etc/buildinfo        # what software is it running?
```

Those two answers determine which chapter you go to next.

---

## Quick reference

| | |
|---|---|
| Linux / macOS | `sima-cli serial` |
| …at the older baud rate | `sima-cli serial -b 921600` |
| Exit `sima-cli serial` | `Ctrl`+`A`, then `Ctrl`+`X` |
| Windows | PuTTY / Tera Term → Serial → `COMx` @ 115200 |
| Browser | <https://developer.sima.ai/tools/serial/index.html> (Chromium only) |
| Login | `sima` / `edgeai` |
| Standard parameters | 115200 8N1 (or 921600 on older firmware) |

Full reference: [Configure Serial Connection](https://developer.sima.ai/hardware/getting-started/setup-serial)

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 3 · HDMI, keyboard and mouse](03-hdmi-and-peripherals.md) | [All chapters](../README.md) | [Chapter 5 · Share the host's internet](05-internet-sharing.md) |
