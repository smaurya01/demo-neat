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

*On the host*

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
```

Then open a new terminal, or reload your shell:

*On the host*

```bash
source ~/.bashrc
```

Check it:

*On the host*

```bash
sima-cli --version
```

---

## Install on the board

The board needs its own copy. Get a shell on it — over SSH if the network is up
([Chapter 6](06-connect-to-router.md)), otherwise over
[serial](04-serial-console.md) — and run the **same**
command:

*On the board*

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
source ~/.bash_profile
sima-cli --version
```

On the board the installer puts `sima-cli` in its own virtual environment under `/data` and adds it
to your `PATH` through `~/.bash_profile` — which is why you source that file here, not `~/.bashrc`:

```text
sima-cli successfully installed.

   Virtual env: /data/sima-cli/.venv
   Binary:      /data/sima-cli/.venv/bin/sima-cli
   Shell config: /home/sima/.bash_profile
```

> This downloads from the internet. If it fails, the board does not have working internet access —
> go back to your networking chapter and confirm `ping -c3 developer.sima.ai` succeeds before
> retrying. Almost every failure at this step is a network problem, not a `sima-cli` problem.

---

## Log in

Before downloading anything — board updates, SDK images, Neat packages, models — authenticate
against SiMa's artifact server:

*On the board*

```bash
sima-cli login
```

Do this **on both machines**, host and board. On the board, no browser opens; `sima-cli` prints a
link and a code instead:

```text
🔐 sima-cli needs you to authorize this device login.
...
  1. Open this URL in a browser on your workstation:
     https://auth.sima.ai/activate?user_code=<code>
  2. Sign in with your SiMa Developer Portal account.
  3. If prompted, confirm this code: <code>
  4. Return to this terminal. sima-cli will continue automatically after authorization.
```

Open the link on any computer, sign in, confirm the code, and the board's terminal carries on by
itself. The code expires after 15 minutes.

> If any `sima-cli` command asks `Do you want to update sima-cli now? [Y/n]`, either answer
> works: `Y` updates sima-cli first, `n` carries on with the installed version.

> This needs an **approved** SiMa Developer Portal account. Approval is not instant. If you have not
> signed up yet, do it at [community.sima.ai/signup](https://community.sima.ai/signup) — you cannot
> get past this point without it.

---

## What you can now do on the board

| Command | Purpose |
|---|---|
| `sima-cli --version` | Confirm the install |
| `sima-cli login` | Authenticate for downloads |
| `sima-cli update` | Update the board within a release series — [Chapter 9](09-check-and-update-image.md) |
| `sima-cli nvme remount` / `format` | Mount the NVMe — [Chapter 8](08-mount-nvme.md) |
| `sima-cli network` | Switch DHCP / default static IP — [Chapter 5 walkthrough](05-internet-sharing.md#guided-configuration-with-sima-cli-optional) |
| `sima-cli neat install <pkg>` | Install Neat packages on the board |
| `sima-cli modelzoo` | Pre-compiled models for the MLA |

And on the host:

| Command | Purpose |
|---|---|
| `sima-cli serial` | Open the serial console |
| `sima-cli device discover` | Find boards on the network |
| `sima-cli neat install sdk@release-3.0` | Install the SDK and pair it with a board |
| `sima-cli bootimg` | Write a board image over the network — used for the [2.1.3 → 3.0.0 upgrade](09-check-and-update-image.md#upgrade-from-213-to-300) |

---

## If the command is not found after installing

The installer adds it to your `PATH` through your shell profile. If a new terminal still cannot find
it, reload the profile the installer wrote to, then look for it — `~/.bash_profile` on the board,
`~/.bashrc` on a Linux host:

*On the board*

```bash
source ~/.bash_profile
which sima-cli
```

```text
/data/sima-cli/.venv/bin/sima-cli
```

If `which` finds nothing, see
[Troubleshooting](troubleshooting.md#sima-cli-command-not-found-after-installing).

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 6 · Connect to a router](06-connect-to-router.md) | [All chapters](../README.md) | [Chapter 8 · Mount the NVMe](08-mount-nvme.md) |
