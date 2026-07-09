# DECISIONS

Autonomous wave execution of `priority-task-and-implementation-plan.md`.
Owner away 2026-07-09. Append-only: each entry = what was decided, why, the alternative,
how to reverse it.

---

## D0 — Orchestrator: session bootstrap (2026-07-09)

**Decided:** Reinstalled `ultralytics==8.4.90` into `/sdk-extensions/model-compiler`.
**Why:** It was absent this session (the plan recorded it as installed; the venv had lost it).
Owner explicitly pre-approved reinstalling it there.
**Verified after install:** `onnx 1.17.0`, `afe` imports, `afe.apis.defines` imports — pins intact.
**Alternative:** A separate venv (as `overall-learning.md` originally advised). Rejected because
the plan deliberately supersedes that: one env for export+surgery+quantize+compile.
**Reverse:** `source /sdk-extensions/model-compiler/bin/activate && pip uninstall ultralytics ultralytics-thop polars polars-runtime-32 nvidia-ml-py`

## D1 — Orchestrator: regenerated the llima ground truth (2026-07-09)

**Decided:** Re-captured `llima --help`, per-subcommand help, `llima list`, and the full
36-model `llima search` catalog into `<scratchpad>/llima_ground_truth.md`.
**Why:** The previous session's scratchpad was wiped with its session ID, so the file the plan
told agents to reuse no longer existed.
**Alternative:** Let each agent re-run `llima search` — rejected: repeated board hits, and the
plan explicitly says capture once and reuse.
**Reverse:** N/A (read-only capture; only `--help`/`list`/`search` were run, all cheap and safe).

## D2 — Orchestrator: global compile slot enforced with `flock` (2026-07-09)

**Decided:** All compile/quantize jobs run through `<scratchpad>/compile_slot.sh`, an
`flock`-based wrapper on `/tmp/neat_compile_slot.lock` (90 min wait cap).
**Why:** The plan requires exactly one compile at a time across ALL agents. Agents run in
parallel, so the constraint needed a mechanical enforcement, not a per-agent instruction.
**Alternative:** Orchestrator hand-serializes every compile. Rejected: agents would idle waiting
on the orchestrator and the wave would not fit the ~2 h budget.
**Reverse:** Delete the wrapper; the lock file is disposable.

## D3 — Orchestrator: branch `training-material-waves` (2026-07-09)

**Decided:** Branched from `feature/python-examples`. Nothing pushed; no existing branch touched.
**Preserved as-is:** the uncommitted edit to `tutorial/README.md`, and the untracked
`model-compilation/`, `training/`, `video/`, `tutorial/III-advanced/`, and the pptx.
**Reverse:** `git checkout feature/python-examples && git branch -D training-material-waves`

## D-B1 — Agent B (T2/T4): API names verified real; used the `pyneat.genai.*` namespace (2026-07-09)

**Decided:** The plan's names `GenAIModel`, `VisionLanguageModel`, `ASRModel` are CORRECT — no
discrepancy. All GenAI notebooks/scripts call them via the `pyneat.genai.*` submodule
(`neat.genai.GenAIModel`, `neat.genai.GenerationRequest`, ...), not the bare `pyneat.*` aliases.
**Why:** The bindings (`/workspace/core/python/src/module.cpp`) register these classes both directly
on the module AND alias them under a `genai` submodule; the shipped tutorials (019/020/021) use the
`neat.genai.*` form, so I matched that for one coherent, tutorial-consistent voice.
**Verified (no model run):** on the board via ssh, `import pyneat; pyneat.genai` exposes GenAIModel,
VisionLanguageModel, ASRModel, GenerationRequest, ChatMessage, GenAITask, GenAIServer,
VisionLanguageOptions (all True), and a fresh `GenerationRequest` has fields prompt, system_prompt,
messages, images, audio_file, language, max_new_tokens, use_cached_images, tools, tool_choice.
`pyneat 0.3.0+develop`.
**Alternative:** bare `pyneat.GenAIModel` (also valid). Rejected for tutorial consistency.
**Reverse:** the two forms are interchangeable; a find/replace `neat.genai.` -> `neat.` would switch.

## D-B2 — Agent B: direct ASRModel path sourced from headers/doc, not a shipped example (2026-07-09)

**Decided:** `03_audio_input_asr.ipynb` and `scripts/run_asr.py` use the DIRECT
`pyneat.genai.ASRModel(dir).run(request)` path with `request.audio_file` + `request.language`.
**Why:** The only shipped ASR *example* in core (tutorial 021 `request_audio_transcription.py`) is
server-based (HTTP POST to `/v1/audio/transcriptions`). The direct-model API is authoritative in the
headers (`ASRModel.h`, `GenAITypes.h` audio_file/audio/language fields), the bindings (module.cpp),
and `genai-model.mdx` (which shows exactly this request shape). I documented the server path as an
alternative and cited every source; nothing invented.
**Correctness note captured in the material:** trap #2 (`llima run --stt_model_path <elf-dir>`) is a
CLI-only requirement; the Python `ASRModel` takes the model dir in its constructor and needs no such
flag — I called this out explicitly so readers don't copy the CLI flag into Python.
**Reverse:** N/A (additive teaching content).
