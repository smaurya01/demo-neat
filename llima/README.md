# LLiMa — LLM / VLM / ASR on SiMa Neat

This folder teaches **LLiMa** — the toolchain that prepares and runs generative models (LLM / VLM /
ASR) on a Modalix DevKit — and the **`pyneat.genai`** Python API you use to call those models from an
application.

Everything here is written to run on the **Modalix DevKit**. The notebooks follow the same house
style as `../tutorial/I-easy`: a markdown concept cell, then a short runnable code cell, then a brief
interpretation.

## Table of Contents

- [Notebooks](#notebooks)
- [Models](#models)
- [How to run these notebooks](#how-to-run-these-notebooks)
- [Reference](#reference)

---

## Notebooks

Work through them in order; each section builds on the one before.

| Section | Notebook | What you learn |
| --- | --- | --- |
| **01 · Basics** | [`llima-introduction`](01-llima-basics/llima-introduction.ipynb) | What LLiMa is, the model families, quantization suffixes, and every `llima` subcommand |
| **02 · Run** | [`01_run_llm`](02-run-llm-vlm/01_run_llm.ipynb) | `pyneat.genai.GenAIModel`: load, generate, stream, options |
|  | [`02_run_vlm`](02-run-llm-vlm/02_run_vlm.ipynb) | `VisionLanguageModel`: image + prompt → text |
|  | [`03_audio_input_asr`](02-run-llm-vlm/03_audio_input_asr.ipynb) | `ASRModel`: audio file → transcription |
| **03 · Apply** | [`01_detection_to_vlm`](03-yolo-plus-vlm/01_detection_to_vlm.ipynb) | Detection → trigger-gated VLM captions (pairs with the [`detection-vlm-assistant`](../apps/detection-vlm-assistant/README.md) app) |
| **04 · Compile** | [`01_llm_compilation`](04-llm-vlm-compilation/01_llm_compilation.ipynb) | Bring your own LLM with the host-side `llima-compile` tool |
|  | [`02_vlm_compilation`](04-llm-vlm-compilation/02_vlm_compilation.ipynb) | VLM specifics: vision encoder + LM, common failures. Plus a [triage checklist](04-llm-vlm-compilation/notes/triage_checklist.md) |
| **05 · Serve** | [`01_genai_server`](05-genai-server/01_genai_server.ipynb) | `GenAIServer`: one OpenAI-compatible HTTP server for LLM/VLM + ASR |

> **There is no `llima compile` subcommand.** The `llima` CLI on the board only runs and manages
> models. Compiling is done on the host with the separate `llima-compile` tool; see section 04 and
> the [SiMa LLiMa compilation docs](https://developer.sima.ai/software/genai-llima/compilation_genai).

## Models

The notebooks use these model IDs as their worked examples. A deployed model lives at
`/media/nvme/llima/models/<model-id>`. Run `llima list` on the DevKit to see which models are present,
and `llima pull <model-id>` for any you need.

| Role | Example model ID |
| --- | --- |
| LLM (text) | `Qwen3-4B-Instruct-2507-GPTQ-a16w4` |
| VLM (image+text) | `Qwen3-VL-4B-Instruct-GPTQ-a16w4` |
| VLM in `02_run_vlm` | `gemma-4-E2B-it-GPTQ-a16w4` |
| ASR (audio) | `whisper-small-a16w8` |


## How to run these notebooks

Run these **on the DevKit board**, so the notebook kernel can import and execute `pyneat` — `pyneat`
and `llima` live on the DevKit, not in the SDK container, and so do the models. When the SDK is
paired with the board, `/workspace` is NFS-mounted on the board at the same path (check with
`dk status` in the SDK: its `NFS mount path` line), so you can edit host-side and run board-side
with no copying. Without the mount, copy this folder to the board first.

Activate the pyneat environment, install Jupyter if needed, and start the notebook server:

```bash
ssh sima@<devkit-ip>

source $HOME/pyneat/bin/activate

python -m pip install notebook   # only once

cd /workspace/demo-neat   # your clone of this repo under /workspace
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

Then open the notebook URL the server prints, from your machine:

```text
http://modalix:8888/tree?token=<token>
```

If `modalix` does not resolve on your network, use the DevKit's IP instead:
`http://<devkit-ip>:8888/tree?token=<token>`.

From the file tree, open `llima/` and start at
[`01-llima-basics/llima-introduction.ipynb`](01-llima-basics/llima-introduction.ipynb).

If you would rather not run Jupyter, lift the code cells into a script and run that in the same
`pyneat` environment.


## Reference

- GenAI C++ headers: [`include/genai/`](https://github.com/sima-neat/core/tree/main/include/genai)
- `pyneat` bindings: [`module.cpp`](https://github.com/sima-neat/core/blob/main/python/src/module.cpp)
- Core tutorials: [`019_run_an_llm`](https://github.com/sima-neat/core/tree/main/tutorials/019_run_an_llm),
  [`020_run_a_vlm`](https://github.com/sima-neat/core/tree/main/tutorials/020_run_a_vlm),
  [`021_serve_genai_models`](https://github.com/sima-neat/core/tree/main/tutorials/021_serve_genai_models)
- Concept doc: [`genai-model/`](https://github.com/sima-neat/core/tree/main/docs/develop-apps/development-workflow/genai-model)
- Official docs: [Getting started](https://developer.sima.ai/software/getting-started/),
  [GenAI model workflow](https://developer.sima.ai/software/develop-apps/development-workflow/genai-model),
  [LLiMa runtime](https://developer.sima.ai/software/genai-llima/runtime)
