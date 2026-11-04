# Chapter 9 — Check and update the board image

*Find out what the board is running, then get it onto a known-good version.*

---

## Read the current version

On the board:

```bash
# On the board
cat /etc/buildinfo
```

`/etc/buildinfo` is the authoritative record of what is installed. A 3.0.0 board looks like this:

```text
DISTRO = eLxr
DISTRO_VERSION = 3.0.0
CODENAME = agate
MACHINE = modalix
SIMA_BUILD_VERSION = 3.0.0_<build>
```

The field that matters most is `DISTRO_VERSION`.

From your host, without logging in interactively:

```bash
# On the host
ssh sima@<devkit-ip> 'uname -m; cat /etc/buildinfo'
```

### Check the distribution first

```bash
# On the board
cat /etc/buildinfo | grep DISTRO
```

| Output | Meaning |
|---|---|
| `DISTRO = eLxr`, `DISTRO_VERSION = 3.0.0` | Current Debian 13-based runtime. Everything in this guide applies. |
| `DISTRO = eLxr`, `DISTRO_VERSION = 2.1.x` | Previous Debian 12-based runtime. Move it to 3.0.0 — see [Upgrade from 2.1.3 to 3.0.0](#upgrade-from-213-to-300). |
| `DISTRO = poky` | Legacy **Yocto** image. `sima-cli update` will **not** carry you across to eLxr — see [Special cases](#special-cases). |

Units shipped before mid-December 2025 are typically Yocto.

---

## Which version should you be on?

| DevKit software | Neat SDK | Neat Library | Neat Apps | Model Compiler |
|---|---|---|---|---|
| **3.0.0** ← this guide | 3.0.0.0 | 0.6.0 | 0.6.0 | 3.0.0 |
| 2.1.3 | 2.1.3.1 | 0.5.0 | 0.5.1 | 2.1.3.1 |
| 2.1.3 | 2.1.3.0 | 0.4.0 | 0.5.0 | 2.1.3 |
| 2.1.2 | 2.1.2.3 | — | — | 2.1.2 |
| 2.0.0 | 2.0.0 | — | — | 2.0.0 |

**The rule:** the board software, the SDK and the Neat Library must be compatible with each other.
Upgrade them as a set, not one at a time. Version mismatch is the root cause of most "it built but
it won't run" problems.

> This table goes stale faster than anything else here. Confirm against the live
> [compatibility matrix](https://developer.sima.ai/software/getting-started/compatibility) before
> upgrading.

---

## Upgrade from 2.1.3 to 3.0.0

3.0.0 moves the board from a Debian 12 image to Debian 13, with a new root-filesystem layout.
It is not compatible with 2.1, so **`sima-cli update` cannot take you there.** The path is: bring
the board to 2.1.3 with `sima-cli update` (next section), then write the 3.0.0 image over the
network from your host with `sima-cli bootimg --netboot`.

Follow SiMa's step-by-step guide:
**[Convert 2.1.3 to 3.0.0](https://developer.sima.ai/hardware/reference/tech-notes/convert-2-1-3-to-3-0-0)**.
Before you start:

- **Back up** anything on the board you want to keep. Writing the image replaces the board's
  software.
- Use **`sima-cli` 2.1.19 or later** on the host ([Chapter 7](07-install-sima-cli.md)).
- Connect host and board with Ethernet, and keep the [serial console](04-serial-console.md) open to
  watch progress.

Afterwards, `cat /etc/buildinfo` must show `DISTRO_VERSION = 3.0.0`. Then carry on with this
guide from [Chapter 7](07-install-sima-cli.md): the new image needs `sima-cli`, the NVMe mount,
Sentinel and the Neat Library set up again.

---

## Update the board within a release

`sima-cli update` moves the board to the latest published build of its release series — for
example, an older 2.1.x board to 2.1.3. Log in first — the update downloads from SiMa's servers:

```bash
# On the board
sima-cli login
```

Then run the update:

```bash
# On the board
sima-cli update
```

During the update:

- When the menu appears, choose **Update all packages to the latest**
- At `Proceed with eLxr update (y/N)`, answer **`y`**

**The board may reboot partway through, and your serial connection will drop.** That is expected.
Once it is back, reconnect over serial with `sima-cli serial`, or over SSH with `ssh sima@<board-ip>`.

---

## Confirm, then re-check the NVMe

```bash
# On the board
sudo reboot
```

When it comes back:

```bash
# On the board
cat /etc/buildinfo
```

`DISTRO_VERSION` should now read the version you updated to.

Then check the NVMe is still mounted — **an image update can change the root filesystem**:

```bash
# On the board
df -h | grep nvme
sima-cli nvme remount     # only if the mount is gone
```

This catches people out. Do not skip it.

---

## Special cases

| Situation | What to do |
|---|---|
| Board on 2.1.x, you want 3.0.0 | [Upgrade from 2.1.3 to 3.0.0](#upgrade-from-213-to-300) — update to 2.1.3 first, then netboot the 3.0.0 image |
| `DISTRO = poky` (Yocto), or software 1.7 | Convert to eLxr first — follow the [eLxr Conversion Guide](https://developer.sima.ai/hardware/reference/tech-notes/elxr-conversion). `sima-cli update` cannot cross that boundary. |
| Downgrade, or install a pre-release | [Net Boot Recovery](https://developer.sima.ai/hardware/getting-started/firmware-update/net-boot). `sima-cli update` only moves forward to published releases. |
| Board will not boot | Net Boot Recovery, or write fresh boot media with `sima-cli bootimg`. |
| Board has no network at all | Write an SD card / USB / NVMe boot image on the host with `sima-cli bootimg`. |

More detail: [Firmware Update](https://developer.sima.ai/hardware/getting-started/firmware-update).

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 8 · Mount the NVMe](08-mount-nvme.md) | [All chapters](../README.md) | [Chapter 10 · Install simaai-sentinel](10-install-simaai-sentinel.md) |
