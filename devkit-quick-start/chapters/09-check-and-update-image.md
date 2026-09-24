# Chapter 9 — Check and update the board image

*Find out what the board is running, then get it onto a known-good version.*

---

## Read the current version

On the board:

```bash
sima@modalix:~$ cat /etc/buildinfo
```

`/etc/buildinfo` is the authoritative record of what is installed. A healthy 2.1.3 board looks like:

```text
DISTRO = eLxr
DISTRO_VERSION = 2.1.3
MACHINE = modalix
SIMA_BUILD_VERSION = 2.1.3_master_B4837
```

The field that matters most is `DISTRO_VERSION`.

From your host, without logging in interactively:

```bash
sima-user@host:~$ ssh sima@<devkit-ip> 'uname -m; cat /etc/buildinfo'
```

### Check the distribution first

```bash
sima@modalix:~$ cat /etc/buildinfo | grep DISTRO
```

| Output | Meaning |
|---|---|
| `DISTRO = eLxr` | Current Debian-derived runtime. Everything in this guide applies. |
| `DISTRO = poky` | Legacy **Yocto** image. `sima-cli update` will **not** carry you across to eLxr — see [Special cases](#special-cases). |

Units shipped before mid-December 2025 are typically Yocto.

---

## Which version should you be on?

| DevKit software | Neat SDK | Neat Library | Neat Apps | Model Compiler |
|---|---|---|---|---|
| **2.1.3** ← this guide | 2.1.3.0 | 0.4.0 | 0.5.0 | 2.1.3 |
| 2.1.2 | 2.1.2.3 | — | — | 2.1.2 |
| 2.0.0 | 2.0.0 | — | — | 2.0.0 |

**The rule:** the board software, the SDK and the Neat Library must be compatible with each other.
Upgrade them as a set, not one at a time. Version mismatch is the root cause of most "it built but
it won't run" problems.

> This table goes stale faster than anything else here. Confirm against the live
> [compatibility matrix](https://developer.sima.ai/software/getting-started/compatibility) before
> upgrading.

---

## Update the board

Log in first — the update downloads from SiMa's servers:

```bash
sima@modalix:~$ sima-cli login
```

Then run the update:

```bash
sima@modalix:~$ sima-cli update
```

During the update:

- When the menu appears, choose **Update all packages to the latest**
- At `Proceed with eLxr update (y/N)`, answer **`y`**

**The board may reboot partway through, and your serial connection will drop.** That is expected.
Once it is back, reconnect over serial with `sima-cli serial`, or over SSH with `ssh sima@<board-ip>`.

---

## Confirm, then re-check the NVMe

```bash
sima@modalix:~$ sudo reboot
```

When it comes back:

```bash
sima@modalix:~$ cat /etc/buildinfo
```

`DISTRO_VERSION` should now read `2.1.3`.

Then check the NVMe is still mounted — **an image update can change the root filesystem**:

```bash
sima@modalix:~$ df -h | grep nvme
sima@modalix:~$ sima-cli nvme remount     # only if the mount is gone
```

This catches people out. Do not skip it.

---

## Special cases

| Situation | What to do |
|---|---|
| `DISTRO = poky` (Yocto), or software 1.7 | Convert to eLxr first — follow the [eLxr Conversion Guide](https://developer.sima.ai/hardware/reference/tech-notes/elxr-conversion). `sima-cli update` cannot cross that boundary. |
| Downgrade, or install a pre-release | [Net Boot Recovery](https://developer.sima.ai/hardware/getting-started/firmware-update/net-boot). `sima-cli update` only moves forward to published releases. |
| Board will not boot | Net Boot Recovery, or write fresh boot media with `sima-cli bootimg`. |
| Board has no network at all | Write an SD card / USB / NVMe boot image on the host with `sima-cli bootimg`. |

More detail: [Firmware Update](https://developer.sima.ai/hardware/getting-started/firmware-update).

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 8 · Mount the NVMe](08-mount-nvme.md) | [All chapters](../README.md) | [Chapter 10 · Install simaai-sentinel](10-install-simaai-sentinel.md) |
