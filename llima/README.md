# LLiMa — LLM / VLM / ASR on SiMa Neat

This folder teaches **LLiMa** — the toolchain that prepares and runs generative models (LLM / VLM /
ASR) on a Modalix DevKit — and the **`pyneat.genai`** Python API you use to call those models from an
application.

Everything here is written to run on the **Modalix DevKit**. The notebooks follow the same house
style as `../tutorial/I-easy`: a markdown concept cell, then a short runnable code cell, then a brief
interpretation.

## Folder map

```text
llima/
  README.md                       # you are here
  01-llima-basics/
    llima-introduction.ipynb      # what LLiMa is, the model families, quantization suffixes,
                                   # and every llima subcommand with its on-disk effect
  02-run-llm-vlm/
    01_run_llm.ipynb              # pyneat.genai.GenAIModel: load, generate, stream, options
    02_run_vlm.ipynb              # pyneat.genai.VisionLanguageModel: image + prompt -> text
    03_audio_input_asr.ipynb      # pyneat.genai.ASRModel: audio file -> transcription
    assets/audio.wav              # sample audio for the ASR notebook
  03-yolo-plus-vlm/
    01_detection_to_vlm.ipynb     # detection -> trigger-gated VLM captions (pairs with
                                   # apps/detection-vlm-assistant)
  04-llm-vlm-compilation/
    01_llm_compilation.ipynb      # bring-your-own LLM: host-side llima-compile flow,
                                   # supported formats, artifact contract
    02_vlm_compilation.ipynb      # VLM specifics: vision encoder + LM, common failures
    notes/triage_checklist.md     # "is this model LLiMa-able?" decision checklist
  05-genai-server/
    01_genai_server.ipynb         # pyneat.genai.GenAIServer: one HTTP server for LLM/VLM/ASR,
                                   # OpenAI-compatible endpoints, memory budgeting + concurrency
```

The five sections build on each other:

- **[`01-llima-basics`](./01-llima-basics/llima-introduction.ipynb)** — what LLiMa is and the `llima` CLI, subcommand by subcommand.

- **[`02-run-llm-vlm`](./02-run-llm-vlm)** — run LLMs, VLMs, and ASR from Python via `pyneat.genai`.
  See [`02-run-llm-vlm/01_run_llm.ipynb`](02-run-llm-vlm/01_run_llm.ipynb).

- **[`03-yolo-plus-vlm/`](03-yolo-plus-vlm/01_detection_to_vlm.ipynb)** — trigger-based
  detection → VLM captioning (pairs with `../apps/detection-vlm-assistant`).

- **[`04-llm-vlm-compilation/`](04-llm-vlm-compilation/01_llm_compilation.ipynb)** — bring-your-own
  LLM/VLM compilation via the host-side `llima-compile` tool, plus a
  [triage checklist](04-llm-vlm-compilation/notes/triage_checklist.md).

- **[`05-genai-server/`](05-genai-server/01_genai_server.ipynb)** — `pyneat.genai.GenAIServer`:
  one OpenAI-compatible HTTP server hosting an LLM/VLM + ASR, with a runnable start/client/stop walkthrough.

> **On `04-llm-vlm-compilation/`:** there is **no `llima compile` subcommand** — the on-board `llima`
> CLI is runtime + model-manager only (`run, search, pull, list, rm, benchmark-server`). GenAI
> compilation is a separate **host-side `llima-compile` (Model Compiler)** tool. Its exact flags
> should be confirmed against the official SiMa documentation. The load-bearing, testable part — the
> deployed model-directory contract — is verified against `core/src/genai/GenAIInternal.cpp`.

## Models

The notebooks use these model IDs as their worked examples. A deployed model lives at
`/media/nvme/llima/models/<model-id>`. Run `llima list` on the DevKit to see which models are present,
and `llima pull <model-id>` for any you need.

| Role | Example model ID |
| --- | --- |
| LLM (text) | `Qwen3-4B-Instruct-2507-GPTQ-a16w4` |
| VLM (image+text) | `Qwen3-VL-4B-Instruct-GPTQ-a16w4` |
| ASR (audio) | `whisper-small-a16w8` |

## How to run these notebooks

Run these **on the DevKit board**, so the notebook kernel can import and execute `pyneat` — `pyneat`
and `llima` live on the DevKit, not in the SDK container, and so do the models. `/workspace` is
NFS-mounted on the board at the same path, so you can edit host-side and run board-side with no
copying.

Activate the pyneat environment, install Jupyter if needed, and start the notebook server:

```bash
ssh sima@<devkit-ip>

source $HOME/pyneat/bin/activate

python -m pip install notebook (only once)

cd /workspace/demo-neat
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

Then open the DevKit notebook URL from your machine (host system):

```bash
http://modalix:8888/tree?token=************************
```

Replace the `modalix` name in above URL with devkit ip.

```bash
http://192.168.135.203:8888/tree?token=**************
```

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
- Concept doc: [`genai-model.mdx`](https://github.com/sima-neat/core/blob/main/docs/develop-apps/development-workflow/genai-model.mdx)
- Official docs: developer.sima.ai — `/software/getting-started/`,
  `/software/develop-apps/development-workflow/genai-model`, `/software/genai-llima/runtime`
