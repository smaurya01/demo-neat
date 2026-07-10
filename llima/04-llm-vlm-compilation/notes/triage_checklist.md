# Is this model LLiMa-able? — a triage checklist

A practical yes / no / maybe flow for deciding whether a given LLM or VLM can be
brought onto the Modalix DevKit through LLiMa **before** you spend time on a
compile you cannot run yourself.

> **Honesty boundary.** The on-board `llima` CLI has **no `compile` subcommand**
> — its subcommands are exactly `{run, search, pull, list, rm, benchmark-server}`.
> Bring-your-own compilation is a **separate, host-side Model Compiler step**,
> documented by SiMa as the `llima-compile` tool
> (see [genai-llima/compilation_genai](https://developer.sima.ai/software/genai-llima/compilation_genai)).
> `llima-compile` is a host-side tool, not part of the on-board CLI. Treat the
> exact compile flags and specifics below as things to confirm against the current
> official SiMa documentation before relying on them, rather than as tested truth.

---

## Step 0 — Do you even need to compile?

- **Is the model already in the catalog?** Run `llima search` (cheap, safe) and
  check the catalog listing. If a suitable `...-a16w4` /
  `...-a16w8` variant exists, **STOP — just `llima pull` it.** No compilation.
- The catalog already covers Llama 2/3.1/3.2, Gemma 1/2/3/4, Phi-3.5, Qwen2.5/3
  (LLM + VL), Mistral, LFM2/2.5 (LLM + VL), Whisper-small. Only go BYO when your
  model is genuinely outside this list or is a private fine-tune.

## Step 1 — Architecture family  → **hard gate**

| Answer | Verdict |
| --- | --- |
| Base architecture is one of the **supported families**: Llama 2 / 3.1 / 3.2, Gemma 1 / 2 / 3 / 4, Phi-3.5, Qwen2.5 / Qwen3 (and `-VL`), Mistral, LFM2 / LFM2.5 (and `-VL`); ASR = Whisper | **YES** — proceed |
| A **fine-tune or re-quant of a supported family** (same layer structure, different weights) | **MAYBE** — usually fine; the family, not the checkpoint name, is what matters |
| A brand-new / exotic architecture (novel attention, custom ops, non-transformer) | **NO / ASK** — not on the supported list; do not assume it compiles |

## Step 2 — Source format  → **hard gate**

Accepted input formats:

- **Hugging Face safetensors** (full-precision) → pipeline: DEVKIT → ONNX → QUANTIZE → COMPILE
- **GGUF** → pipeline: direct → COMPILE
- **Pre-quantized compressed-tensors (GPTQ / AWQ)** → pipeline: SOURCE_TO_QUANT → COMPILE
  - **must use symmetric quantization** — asymmetric pre-quant is not accepted

| Answer | Verdict |
| --- | --- |
| HF safetensors, GGUF, or symmetric GPTQ/AWQ | **YES** |
| Pre-quantized but asymmetric, or an unlisted format (e.g. raw PyTorch `.pt` only) | **NO / convert first** |

## Step 3 — Required companion files (HF path)  → **hard gate**

A Hugging Face model directory must contain:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`
- weights in **safetensors**

| Answer | Verdict |
| --- | --- |
| All present | **YES** |
| Any missing | **NO** — obtain them from the model card first |

## Step 4 — Size & memory budget  → **soft gate (plan, don't block)**

Two very different budgets — do not confuse them:

- **Host compile budget** — `llima-compile` runs on a host/workstation, not the
  board. The quantize step is heavy and a **GPU is recommended**. This is
  where big models are actually processed.
- **Board runtime budget** — the DevKit root filesystem is small. Deployed weights
  land under `/media/nvme/llima/models/<id>` (nvme, larger than `/`), but **always
  run `df -h /` and `df -h /media/nvme` before deploying** to confirm free space.

Rules of thumb:

- Prefer the smallest viable variant. Catalog smallest: `LFM2.5-230M-a16w4`,
  `Qwen3-0.6B-GPTQ-a16w4` (LLM), `LFM2-VL-450M-a16w4` (VLM).
- `a16w4` (4-bit weights) is ~half the footprint of `a16w8` (8-bit weights) —
  choose the lowest precision that still meets your accuracy bar.
- Default max context is **1024 tokens**, configurable at compile time.
  Longer context = more KV-cache memory at runtime.

| Answer | Verdict |
| --- | --- |
| Fits after quantization + leaves headroom on `/media/nvme` | **YES** |
| Marginal | **MAYBE** — drop to `a16w4`, pick a smaller variant, or shorten context |
| A large (`7B`+ at high precision) model on top of an already-full board | **NO** — free space or pick smaller first |

## Step 5 — Output must satisfy the runtime contract  → **verify after compile**

A model directory only loads if it matches what the Neat GenAI runtime inspects
(`inspect_model_directory` in `/workspace/core/src/genai/GenAIInternal.cpp`).
The deployed `<model-id>/` directory **must** contain:

- `devkit/` — Python runtime + config files. **Missing → runtime throws**
  `"GenAI model directory missing devkit/"`.
- `elf_files/` — the compiled MLA binaries. **Missing → runtime throws**
  `"GenAI model directory missing elf_files/"`.
- Inside `devkit/`, **exactly one** of:
  - `vlm_config.json` → task = VisionLanguage (text LLM *or* image VLM)
  - `whisper_config.json` → task = ASR
  - Having **both** or **neither** → runtime throws.
- For an **image-capable VLM**, `vlm_config.json` must carry non-null `vm_cfg`,
  non-null `mm_cfg`, and a non-empty `vision_model_name` — that is exactly how the
  runtime sets `accepts_image()`. A VLM compiled without these loads as
  a **text-only** model (silent capability loss — see `02_vlm_compilation.ipynb`).

(The compiler's own output tree is `output_directory/{onnx_files,sima_files/{devkit,mpk,npy_files}}`;
deployment reorganizes `sima_files` into the `devkit/` + `elf_files/`
layout the runtime expects. The `mpk/` compiled binaries become `elf_files/`.)

---

## Decision summary

```
Step 0  in catalog? ----------------- yes --> pull it, done (no compile)
                       no
Step 1  supported family? ----------- no  --> NOT LLiMa-able (check official docs)
                       yes
Step 2  accepted source format? ----- no  --> convert, or NOT LLiMa-able
                       yes
Step 3  HF companion files present? - no  --> fix inputs first
                       yes
Step 4  fits board/nvme budget? ----- no  --> smaller variant / lower precision / free space
                       yes/maybe
Step 5  compile, then VERIFY:
        devkit/ + elf_files/ present, exactly one config,
        VLM vision keys present -------------> LLiMa-able ✅
```

**Yes** = every hard gate (1, 2, 3) passes and Step 5 verifies after compile.
**Maybe** = a soft gate (Step 4) is tight; mitigate and re-check.
**No** = any hard gate fails; document why and confirm against the official SiMa
documentation rather than guessing.
