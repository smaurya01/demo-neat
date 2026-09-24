# Chapter 13 — LLiMa: search, list, pull

*Run a language or vision-language model directly on the board.*

---

## What LLiMa is

LLiMa is the on-board runtime for generative models — LLMs, VLMs and speech models — compiled for
the MLA. It is installed on the board together with **Neat core** — the `sima-cli neat install
core` step in [Chapter 11](11-install-pyneat.md) — so there is nothing extra to install.

Check it is there:

```bash
sima@modalix:~$ llima --help
```

```text
usage: llima [-h] {run,search,pull,list,rm,benchmark-server} ...
```

---

## The commands

| Command | Does |
|---|---|
| `llima search` | Search the remote model catalogue |
| `llima pull` | Download a model to the board |
| `llima list` | Show models already downloaded |
| `llima run` | Run a model interactively |
| `llima rm` | Delete a local model |
| `llima benchmark-server` | Start a server for benchmarking |

---

## Search

```bash
sima@modalix:~$ llima search
```

Filter by name:

```bash
sima@modalix:~$ llima search qwen
```

Model names encode their quantisation — `a16w4` means 16-bit activations with 4-bit weights, which
is the common choice on this hardware. Lower weight precision means a smaller download and less
memory.

---

## Pull

```bash
sima@modalix:~$ llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

> **Check where it lands.** Models are large. The default location is
> `/media/nvme/llima/models` — on the NVMe, which is where you want it. If your NVMe is not mounted
> ([Chapter 8](08-mount-nvme.md)), a download will fill the root filesystem and cause the confusing
> failures described there. Mount it first.

---

## List

```bash
sima@modalix:~$ llima list
```

A board with a few models pulled looks like:

```text
gemma-4-E2B-it-GPTQ-a16w4
LFM2-VL-1.6B-a16w4
Llama-3.2-3B-Instruct-a16w4
Qwen3-4B-Instruct-2507-GPTQ-a16w4
Qwen3-VL-4B-Instruct-Autoround-a16w4
Qwen3-VL-4B-Instruct-GPTQ-a16w4
whisper-small-a16w8
```

Note the mix: text models, `-VL-` vision-language models, and `whisper-` for speech.

---

## Run

```bash
sima@modalix:~$ llima run Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

First load takes noticeably longer than later ones while the model is unpacked and staged.

---

## Remove

Models are large and the NVMe is not infinite:

```bash
sima@modalix:~$ llima rm <model-name>
sima@modalix:~$ df -h /media/nvme
```

---

## Watch it work

Run `simaai-sentinel` ([Chapter 10](10-install-simaai-sentinel.md)) in a second terminal. Generative
models show a very different profile from vision models — memory-bound rather than compute-bound,
with MLA utilisation that rises and falls per token rather than sitting flat.

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 12 · Object detection on images](12-object-detection.md) | [All chapters](../README.md) | [Chapter 14 · The Neat stack](14-neat-components.md) |
