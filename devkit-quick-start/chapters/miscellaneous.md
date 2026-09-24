# Miscellaneous

*Command reference, defaults, the PCIe card, and where to read more.*

---

## Defaults

| | Value |
|---|---|
| Username | `sima` |
| Password | `edgeai` |
| Hostname (mDNS) | `modalix.local` |
| Network | DHCP on `end0` |
| Serial baud | 115200 (921600 on older firmware) |
| NVMe mount point | `/media/nvme` |
| pyneat virtualenv | `~/pyneat` |
| LLiMa model store | `/media/nvme/llima/models` |

### Version set used throughout this guide

| Component | Version |
|---|---|
| DevKit software | 2.1.3 |
| Neat SDK | 2.1.3.0 |
| Neat Library | 0.4.0 |
| Neat Apps | 0.5.0 |

Confirm against the live
[compatibility matrix](https://developer.sima.ai/software/getting-started/compatibility) — this is
the fastest-moving information in the guide.

---

## Command reference

### On your host

| Command | Purpose |
|---|---|
| `sima-cli serial` | Open the serial console |
| `sima-cli serial -b 921600` | …at the older baud rate |
| `sima-cli device discover` | Find boards on the network |
| `sima-cli login` | Authenticate against SiMa's artifact server |
| `sima-cli neat install sdk@release-2.1` | Install the SDK and pair it with a board |
| `sima-cli sdk setup --devkit <ip>` | Start the SDK (and re-pair after an IP change) |
| `ssh sima@modalix.local` | Connect by hostname |

### On the board

| Command | Purpose |
|---|---|
| `cat /etc/buildinfo` | What software the board is running |
| `sima-cli login` | Authenticate for downloads |
| `sima-cli update` | Update the board software |
| `sima-cli network` | Guided network configuration |
| `sima-cli nvme remount` | Mount an existing NVMe partition |
| `sima-cli nvme format` | Partition and format the NVMe — **erases it** |
| `sima-cli neat install core@v0.4.0` | Install the Neat Library |
| `sima-cli neat install apps@v0.5.0` | Install the example applications |
| `sima-cli neat install sentinel` | Install the monitoring tool |
| `sima-cli modelzoo` | Browse pre-compiled models |
| `simaai-sentinel` | Live temperature / power / CPU / MLA view |
| `llima list` · `search` · `pull` · `run` | Generative model runtime |
| `ip a \| grep inet` | Read the board's addresses |
| `df -h \| grep nvme` | Is the NVMe mounted? |

---

## Verification checklist

Run these on the board. All five should succeed before you consider the setup done.

```bash
# 1. sima-cli is installed
sima-cli --version

# 2. the board is on the expected software version
cat /etc/buildinfo | grep DISTRO_VERSION      # expect 2.1.3

# 3. the NVMe is mounted
df -h | grep nvme                             # expect /media/nvme

# 4. the board has internet
ping -c3 developer.sima.ai

# 5. pyneat imports
source ~/pyneat/bin/activate && python3 -c "import pyneat; print('pyneat OK')"
```

---

## Modalix HHHL PCIe card

If you have the PCIe card rather than the DevKit, the board installs into a host machine and is used
for offloaded inference. Chapters 1–6 of this guide do not apply.

**Installing the card**

1. Power down the host and open the chassis
2. Fit the card in a PCIe slot — the black slot, if your board colour-codes them
3. Secure the bracket and reassemble
4. Optionally connect the serial header for debug access
5. Connect the Ethernet port for network access

**Host driver setup**

```bash
# Linux host
sudo modprobe simaai_pcie

# Windows host — install the SiMa PCIe driver package, then reboot
```

**Verify enumeration**

```bash
sima-user@host:~$ lspci | grep -i sima
```

Once enumerated, the software chapters ([7](07-install-sima-cli.md) onward) apply as written.

---

## Links

| | |
|---|---|
| Developer Portal | <https://developer.sima.ai/> |
| Hardware overview | <https://developer.sima.ai/hardware> |
| Getting started (software) | <https://developer.sima.ai/software/getting-started/> |
| Quick start guide | <https://developer.sima.ai/tools/qsg/index.html> |
| Compatibility matrix | <https://developer.sima.ai/software/getting-started/compatibility> |
| Firmware update | <https://developer.sima.ai/hardware/getting-started/firmware-update> |
| eLxr conversion | <https://developer.sima.ai/hardware/reference/tech-notes/elxr-conversion> |
| Tutorials | <https://developer.sima.ai/software/tutorials> |
| Python API | <https://developer.sima.ai/software/reference/pythonapi/> |
| C++ API | <https://developer.sima.ai/software/reference/cppapi/> |
| SOM Carrier Board Data Sheet (PDF) | <https://docs.sima.ai/pkg_downloads/datasheets_product_briefs/SiMa_SOM_Carrier_Board_Data_Sheet_Rev1.2_1-24-2026.pdf> |
| Sign up | <https://community.sima.ai/signup> |

---

| ← Previous | Contents |
|:---|---:|
| [Troubleshooting](troubleshooting.md) | [All chapters](../README.md) |
