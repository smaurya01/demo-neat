# LLiMa — LLM / VLM / ASR on SiMa Neat

This folder is the GenAI (LLM / VLM / ASR) training track for the NEAT program. It teaches
**LLiMa** — the toolchain that prepares and runs generative models on a Modalix DevKit — and the
**`pyneat.genai`** Python API you use to call those models from an application.

Everything here is written to be run on the **Modalix DevKit** (board `192.168.135.203`, user
`sima`). The notebooks are structured markdown-concept → runnable-code → interpretation, the same
house style as `../tutorial/I-easy`.

> **Important — nothing in this folder auto-runs a model.** Every LLM/VLM/ASR run is a heavy,
> hardware-bound operation and is left for you to run manually. Cells and scripts that would load or
> run a model are labelled **`MANUAL RUN — not executed by tooling`** and are paired with an
> **Expected output** section so you can confirm a manual run went as intended. The cheap CLI probes
> (`llima --help`, `llima list`, `llima search`) were captured on the board on 2026-07-09 and are
> embedded inline.

## Folder map

```text
llima/
  README.md                       # you are here
  01-llima-basics/
    01_llima_concepts.ipynb       # what LLiMa is, where it sits in the NEAT stack,
                                   # supported model families, memory/size constraints
    02_llima_cli.ipynb            # llima search / pull / list / run / rm / benchmark-server
                                   # each explained, with real captured output + on-disk effect
  02-run-llm-vlm/
    01_run_llm.ipynb              # pyneat.genai.GenAIModel: load, generate, stream, options
    02_run_vlm.ipynb              # pyneat.genai.VisionLanguageModel: image + prompt -> text
    03_audio_input_asr.ipynb      # pyneat.genai.ASRModel: audio file -> transcription
    scripts/
      run_llm.py                  # minimal CLI version of 01_run_llm
      run_vlm.py                  # minimal CLI version of 02_run_vlm
      run_asr.py                  # minimal CLI version of 03_audio_input_asr
```

Sections `03-yolo-plus-vlm/`, `04-llm-vlm-compilation/`, and `05-genai-server/` are owned by other
tracks and are not part of this folder's basics + run material.

## Two layers, do not confuse them

| Layer | What it is | Where it runs | Covered in |
| --- | --- | --- | --- |
| **`llima` CLI** | Prepares models: search a catalog, pull weights, list/remove local models, and quick-test with `run` / `benchmark-server`. | `/usr/bin/llima` on the DevKit. | `01-llima-basics/02_llima_cli.ipynb` |
| **`pyneat.genai` API** | The application-facing Python API you call from your own code (`GenAIModel`, `VisionLanguageModel`, `ASRModel`, `GenAIServer`). | Your Python process on the DevKit (`pyneat` in `$HOME/pyneat`). | `02-run-llm-vlm/*` |

LLiMa **prepares** a model directory; `pyneat.genai` **runs** that directory from an app. The
notebooks in `02-run-llm-vlm` point at the same model directories the `llima` CLI produced.

## Three models are already on the board

Every happy path in this folder uses a model that is **already pulled** — no `llima pull` needed:

| Role | Model ID | Deployed directory on the board |
| --- | --- | --- |
| LLM (text) | `Qwen3-4B-Instruct-2507-GPTQ-a16w4` | `/media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4` |
| VLM (image+text) | `Qwen3-VL-4B-Instruct-GPTQ-a16w4` | `/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4` |
| ASR (audio) | `whisper-small-a16w8` | `/media/nvme/llima/models/whisper-small-a16w8` |

`llima list` on the board (captured 2026-07-09) confirms exactly these three.

## How to run these notebooks and scripts on the DevKit

`pyneat` and `llima` live **on the DevKit**, not in the SDK container. `/workspace` is NFS-mounted on
the board at the identical path, so you edit host-side and run board-side with no copying.

**Human, at a real terminal (intended UX):** use `dk`.

```bash
# one-time, from the SDK container shell, to define the dk helper:
source /usr/local/bin/devkit.sh 192.168.135.203 sima 22

# run a script on the board:
dk /workspace/demo-neat/llima/02-run-llm-vlm/scripts/run_llm.py \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**Automation / CI (no TTY):** `dk` needs a TTY and hangs in scripted contexts, so use ssh (this is
the documented fallback). Always wrap board commands in `timeout` so a hang cannot stall you.

```bash
timeout 300 ssh -o BatchMode=yes sima@192.168.135.203 \
  'source $HOME/pyneat/bin/activate; \
   python /workspace/demo-neat/llima/02-run-llm-vlm/scripts/run_llm.py \
     --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4'
```

**Notebooks:** run Jupyter on the board and open these `.ipynb` files, or lift the code cells into a
script and run them with the command above. Every heavy cell is marked `MANUAL RUN — not executed by
tooling`; run those yourself.

Board runtime observed 2026-07-09: `aarch64`, Python 3.11.2, `pyneat 0.3.0+develop`.

## Board disk is small — check before you pull

The board root filesystem has only **~5.9 GB free of 14 GB** (`df -h /`, 2026-07-09). GenAI weights
are large, so **always `df -h /` before any `llima pull`.** If you need a fresh, small model, the
smallest viable variants in the catalog are:

- LLM: `LFM2.5-230M-a16w4`, `Qwen3-0.6B-GPTQ-a16w4`
- VLM: `LFM2-VL-450M-a16w4`

## Three correctness traps taught throughout

1. The serve subcommand is **`benchmark-server`, NOT `serve`**.
2. ASR is run through **`llima run --stt_model_path <elf-dir> <model>`** (the model needs its
   speech-to-text ELF directory passed explicitly).
3. Quantization suffixes: **`a16w4`** = 16-bit activations / 4-bit weights;
   **`a16w8`** = 16-bit activations / 8-bit weights.

## Sources of truth

- GenAI C++ headers: `/workspace/core/include/genai/`
- `pyneat` bindings: `/workspace/core/python/src/module.cpp`
- Core tutorials: `/workspace/core/tutorials/019_run_an_llm`, `020_run_a_vlm`, `021_serve_genai_models`
- Concept doc: `/workspace/core/docs/develop-apps/development-workflow/genai-model.mdx`
- Official docs: developer.sima.ai — `/software/getting-started/`,
  `/software/develop-apps/development-workflow/genai-model`, `/software/genai-llima/runtime`
