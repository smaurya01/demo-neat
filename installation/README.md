# NEAT Installation & Setup Guide

Install the NEAT SDK on your host, pair it with a Modalix DevKit, and attach VS Code.

> **On Windows?** Follow [`neat_on_windows.md`](neat_on_windows.md) instead — same stack, via WSL2.

**Also in this folder:** [`components.md`](components.md) (what each piece of the stack is) ·
[`neat_insight.md`](neat_insight.md) (browser-based RTSP sources, viewer, runtime metrics)

---

## 1. Prerequisites

- **Host OS:** Ubuntu 22.04 / 24.04 (recommended), Windows 11 via WSL (x86_64), or macOS 15.5+
  Apple Silicon (aarch64).
- **Tools:** `sudo` access, Git, `curl` / `wget`, and a container runtime (Docker or Colima).
- **Disk:** ~10 GB free (add ~10 GB more if you install the Model Compiler).
- **Network:** the host and the Modalix DevKit must reach each other — confirm you can `ping` the
  DevKit IP from the host.

Install `sima-cli` on the host:

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
```

---

## 2. Install

**On the host**, run:

```bash
sima-cli neat install sdk@release-2.1
```

This is the default install for current boards. It is one continuous flow — there is no separate
setup command to run afterwards.

**What the command does**

1. Pulls the NEAT SDK Docker image.
2. Creates the SDK container and installs the SDK software into it.
3. Pairs with the Modalix DevKit.
4. Shares one workspace folder between the host, the SDK container, and the DevKit — so you never
   copy files by hand.
5. Installs Neat Core on the SDK container, and on the paired DevKit if you paired one.

`release-2.1` tracks the latest patch in the 2.1 series: **SDK 2.1.2.2**, which pairs with **DevKit
software 2.1.2**. The first install takes several minutes.

**Not sure which SDK your board needs?** Check what it's running with `cat /etc/buildinfo` on the
DevKit, then look it up in the
[compatibility matrix](https://developer.sima.ai/software/getting-started/compatibility).

> **Older SDK releases use the legacy two-step install flow.** For SDK **2.0.0**, **2.1.2.0**, or
> **2.1.2.1**, install with the legacy image pull and setup commands. See
> [Two Step SDK Installation](https://developer.sima.ai/software/reference/two-step-sdk-installation/).
> Either way, setup then continues with the same prompts below.

---

## 3. The decisions you'll make

Setup is a series of prompts. Everything you have to decide is in this table — the rest is `Enter`.

| # | Prompt | Choose | Notes |
|---|---|---|---|
| 1 | **Pair this SDK with a DevKit now?** `[y/N]` | `y`, then enter the DevKit IP | e.g. `192.168.XXX.XXX`. Press `N` if you don't have the IP yet — the workspace is still created and you can pair later with `sima-cli sdk setup --devkit <devkit-ip>`. |
| 2 | **Some system checks failed — continue anyway?** `[y/N]` | `y` — *if* the only failure is a `Firewall` `WARNING` | Any other failure is worth reading before you continue. |
| 3 | **SDK Docker image** | `Enter` | The image you just downloaded is pre-selected. |
| 4 | **Host workspace path** | `Enter` for the default | Or type a custom path. This folder is shared with the DevKit, so you never copy files by hand. |
| 5 | **SDK extension** | `Enter` | |
| 6 | **Create a new SDK container?** | **First install:** follow the prompt to create it. **Every later run:** press `n`. | ⚠️ The one prompt that actually matters. Pressing `y` on a repeat run creates a second container and reinstalls from scratch. |
| 7 | **Install the Model Compiler extension?** | `n`, unless you need to compile ONNX/GenAI models | Optional. ~15 minutes and ~10 GB. You can [add it later](#optional-model-compiler). |
| 8 | **DevKit workspace path** | `Enter` for `/workspace` | Mounts your host workspace onto the board. If it asks for a password, it's `edgeai`. |

<img src="images/devkit-workspace.svg" alt="Host, SDK container, and DevKit sharing one workspace folder" width="900">

**Running setup again later?** You don't reinstall or re-pull anything. Just:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

…and press `n` at prompt 6 to reuse the existing container.

---

## 4. Attach VS Code

1. Install [VS Code](https://code.visualstudio.com/download) on the host. On Ubuntu:

   ```bash
   sudo apt update && sudo snap install code --classic
   ```

2. Install the **Dev Containers** extension by Microsoft.

3. Open the Command Palette (`Ctrl+Shift+P`) → **Dev Containers: Attach to Running Container…** →
   pick the `sima-neat/sdk` container.

4. In the attached window, open `/workspace` (or wherever you cloned this repo).

5. Optionally install the **Claude** or **Codex** extension for agentic development, and confirm it
   picks up `sima-skill`.

Screenshots for each of these steps are in the [detailed walkthrough](#walkthrough) below.

---

## 5. Run something on the DevKit

From inside the attached SDK container, use `dk` to execute on the paired board:

Smoke test — save as `hello_neat.py`, then `dk hello_neat.py`:

```python
import neat
print("PyNeat import successful")
```

```bash
dk hello_neat.py            # a PyNeat script
#dk build/<binary-name>      # a compiled C++ binary
```

If that prints, the runtime is live on the DevKit. Next: the
[tutorial notebooks](../tutorial/README.md), or straight to an [app](../README.md#apps).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Version mismatch | `cat /etc/buildinfo` on the board, then pin the SDK and model-compiler versions to match. |
| Pairing fails | Confirm host and DevKit are on the same network and can `ping` each other; check firewall rules. |
| Setup reinstalls everything | You pressed `y` at prompt 6. Press `n` to reuse the existing container. |
| Copying files by hand | Don't — use the shared `/workspace` set up by pairing. |

**NEAT Insight** runs at `https://localhost:9900` from inside the SDK — use it to inspect streams,
files, and runtime logs. See [neat_insight.md](neat_insight.md).

---

<a id="walkthrough"></a>

<details>
<summary><h3>Detailed walkthrough with screenshots</h3></summary>

<br>

Every prompt from Section 3, in order, as it actually appears.

**Install starts.** `sima-cli neat install sdk@release-2.1` validates the release metadata and
downloads the SDK image.

<img src="images/linux/neat-install.png" alt="Terminal running sima-cli neat install sdk@release-2.1" width="600">

**Prompt 1 — pair with a DevKit.** Setup starts automatically once the image is local. Press `y`,
then type the DevKit IP inline.

<img src="images/linux/neat-pair-devkit.png" alt="Prompt: Do you want to pair this SDK with a DevKit now? followed by Enter DevKit IP address" width="600">

**Prompt 2 — system requirements report.** If the only red mark is a `Firewall` `WARNING`, press
`y` to continue.

<img src="images/linux/Install-setup2.png" alt="System Requirements Report with a firewall warning and a continue-anyway prompt" width="600">

**Prompt 3 — SDK Docker image.** The image you downloaded is pre-selected; press `Enter`.

<img src="images/linux/Install-setup3.png" alt="Image selection list with the downloaded SDK image pre-selected" width="600">

**Prompt 4 — host workspace.** Accept the default or type a custom path.

<img src="images/linux/Install-setup4.png" alt="Workspace path prompt showing the default path" width="600">

**Prompt 5 — SDK extension.** Press `Enter`.

<img src="images/linux/Install-setup5.png" alt="SDK extension prompt" width="600">

**Prompt 6 — container reuse.** ⚠️ On a first install, follow the prompt to create the container.
On every later run, press `n` to reuse it.

<img src="images/linux/Install-setup6.png" alt="Prompt asking whether to create a new SDK container" width="600">

**Prompt 7 — Model Compiler.** Skip unless you need to compile models. ~15 min, ~10 GB.

<img src="images/linux/Install-setup7.png" alt="Optional Model Compiler extension prompt" width="600">

**Prompt 8 — DevKit workspace.** `Enter` accepts `/workspace` on the board. Password, if asked:
`edgeai`.

<img src="images/linux/Install-setup8.png" alt="DevKit workspace path prompt defaulting to /workspace" width="600">

**Done.** Confirm the installation reports success.

<img src="images/linux/Install-setup9.png" alt="Setup complete, installation successful" width="600">

### VS Code, step by step

Command Palette → **Dev Containers: Attach to Running Container…**

<img src="images/VS-1.png" alt="VS Code Command Palette showing Dev Containers: Attach to Running Container" width="600">

Select the `sima-neat/sdk` container.

<img src="images/VS-2.png" alt="Container picker listing the sima-neat/sdk container" width="600">

In the attached window, open `/workspace`.

<img src="images/VS-3.png" alt="VS Code attached to the SDK container with /workspace open" width="600">

Install the `Claude` or `Codex` extension for agentic development.

<img src="images/VS-4.png" alt="VS Code extensions pane showing the Claude and Codex extensions" width="600">

Confirm `sima-skill` is picked up.

<img src="images/VS-5.png" alt="Agent extension showing sima-skill loaded" width="600">

</details>

---

<details>
<summary><h3>Older SDKs: two-step install</h3></summary>

<br>

Only for SDK **2.0.0**, **2.1.2.0**, or **2.1.2.1**. Newer boards use the single command in
Section 2. Full reference:
[Two Step SDK Installation](https://developer.sima.ai/software/reference/two-step-sdk-installation/).

1. **Pull the SDK image** matching your board:

   ```bash
   sima-cli install ghcr:sima-neat/sdk:v2.0.0   # v2.0.0 for a 2.0.0 board
   ```

   <img src="images/linux/Download.png" alt="Pulling the SDK container image" width="600">

2. **Run setup and pair**, passing the IP:

   ```bash
   sima-cli sdk setup --devkit <devkit-ip>
   ```

   <img src="images/linux/Install-setup1.png" alt="Running sima-cli sdk setup with the DevKit IP" width="600">

From here, setup continues with the same prompts — 2–8 in the table above.

</details>

---

<a id="optional-model-compiler"></a>

<details>
<summary><h3>Optional: Model Compiler</h3></summary>

<br>

Only needed to compile or quantize ONNX / GenAI models for Modalix. Match the architecture
(amd64/arm64) and version to your board and SDK.

```bash
# amd64, v2.1.2
sima-cli install -v 2.1.2 tools/model-compiler/amd64
```

References:
[Installation Guide](https://developer.sima.ai/software/getting-started/dev-environment/install-model-compiler) ·
[Compatibility](https://developer.sima.ai/software/getting-started/compatibility#model-compiler) ·
[Compile a model](https://developer.sima.ai/software/compile-a-model/)

</details>

---

## References

- [Getting started](https://developer.sima.ai/software/getting-started/) — primary reference
- [Install the environment](https://developer.sima.ai/software/getting-started/dev-environment/install-the-environment/)
- [Pair with a DevKit](https://developer.sima.ai/software/getting-started/dev-environment/pair-with-a-devkit/)
- [Compatibility matrix](https://developer.sima.ai/software/getting-started/compatibility/)
- [Compile a model](https://developer.sima.ai/software/compile-a-model/)
- [Hello NEAT](https://developer.sima.ai/software/develop-apps/hello-neat/minimal/)
