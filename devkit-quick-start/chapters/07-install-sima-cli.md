# Chapter 7 — Install sima-cli

*The tool everything else depends on. Install it twice: once on your host, once on the board.*

---

## What it does

`sima-cli` is SiMa's command-line tool. It discovers boards, opens the serial console, configures
networking, mounts the NVMe, updates the board image, and installs the SDK and Neat packages. Every
remaining chapter uses it.

It runs in two places, and they are separate installs:

| Where | What you use it for |
|---|---|
| **Host PC** | `sima-cli serial`, `device discover`, installing the SDK, pairing |
| **DevKit** | `sima-cli update`, `nvme`, `network`, installing Neat packages on the board |

---

## Install on the host

```bash
sima-user@host:~$ curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
```

Then open a new terminal, or reload your shell:

```bash
sima-user@host:~$ source ~/.bashrc
```

Check it:

```bash
sima-user@host:~$ sima-cli --version
```

---

## Install on the board

The board needs its own copy. Get a shell on it — over SSH if the network is up
([Chapter 6](06-connect-to-router.md)), otherwise over
[serial](04-serial-console.md) — and run the **same**
command:

```bash
sima@modalix:~$ curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
sima@modalix:~$ source ~/.bashrc
sima@modalix:~$ sima-cli --version
```

> This downloads from the internet. If it fails, the board does not have working internet access —
> go back to your networking chapter and confirm `ping -c3 developer.sima.ai` succeeds before
> retrying. Almost every failure at this step is a network problem, not a `sima-cli` problem.

---

## Log in

Before downloading anything — board updates, SDK images, Neat packages, models — authenticate
against SiMa's artifact server:

```bash
sima@modalix:~$ sima-cli login
```

Follow the prompt. Do this **on both machines**, host and board.

> This needs an **approved** SiMa Developer Portal account. Approval is not instant. If you have not
> signed up yet, do it at [community.sima.ai/signup](https://community.sima.ai/signup) — you cannot
> get past this point without it.

---

## What you can now do on the board

| Command | Purpose |
|---|---|
| `sima-cli --version` | Confirm the install |
| `sima-cli login` | Authenticate for downloads |
| `sima-cli update` | Update the board image — [Chapter 9](09-check-and-update-image.md) |
| `sima-cli nvme remount` / `format` | Mount the NVMe — [Chapter 8](08-mount-nvme.md) |
| `sima-cli network` | Switch DHCP / default static IP — [Chapter 5 walkthrough](05-internet-sharing.md#guided-configuration-with-sima-cli-optional) |
| `sima-cli neat install <pkg>` | Install Neat packages on the board |
| `sima-cli modelzoo` | Pre-compiled models for the MLA |

And on the host:

| Command | Purpose |
|---|---|
| `sima-cli serial` | Open the serial console |
| `sima-cli device discover` | Find boards on the network |
| `sima-cli neat install sdk@release-2.1` | Install the SDK and pair it with a board |

---

## If the command is not found after installing

The installer adds it to your `PATH` through your shell profile. If a new terminal still cannot find
it:

```bash
sima@modalix:~$ source ~/.bashrc
sima@modalix:~$ which sima-cli
```

If `which` finds nothing, see
[Troubleshooting](troubleshooting.md#sima-cli-command-not-found-after-installing).

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 6 · Connect to a router](06-connect-to-router.md) | [All chapters](../README.md) | [Chapter 8 · Mount the NVMe](08-mount-nvme.md) |
