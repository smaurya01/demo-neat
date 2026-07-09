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

## T5 phase 1 (Agent C) — 2026-07-09

- **Zoo availability check**: `https://docs.sima.ai/pkg_downloads/SDK2.1.2/model_zoo/metadata_gen2.json`
  now returns HTTP 302 to `auth.sima.ai/authorize` (auth-gated), so it cannot be parsed
  programmatically without a login token. Documented and proceeded to compile all four T5
  models (plan already expected none of the four to be in the 2.1.2 zoo). To reverse: obtain a
  `sima-cli login` token and re-check the metadata before compiling.
- **New scripts 13-16 instead of editing 09/11/12** (frozen 01-12 per ownership rules):
  `13_export_t5_models.py` (export yolo11s / yolo11s-seg / yolo26s-pose),
  `14_t5_compile_ready_surgery.py` (generalized head-exposing surgery: detection + seg + pose),
  `15_compile_t5_int8.py` (compile; output names read from surgery report),
  `16_yolox_surgery.py` (YOLOX-specific surgery). Alternative was editing the frozen scripts;
  reversed by deleting 13-16.
- **models.yaml NOT edited** (not an owned path). Input name/shape hardcoded as `images` /
  `1,3,640,640` in scripts 14-16 (true for all YOLO variants). Reverse: add the four models to
  models.yaml if the registry-driven scripts are preferred later.
- **YOLOX source**: used Megvii's official pre-exported `yolox_s.onnx` (0.1.1rc0 release,
  opset 11) instead of installing the `yolox` pip package (would perturb the model-compiler
  venv). The exported ONNX is already NMS-free with only MLA-supported ops. Reverse: `pip install
  yolox` in a separate venv and re-export from `.pth` if a different opset/head is needed.
- **Surgery = expose raw per-scale heads, remove CPU decode tail** for all four (same pattern as
  proven yolo11n/yolo26n). Seg adds mask_coeff + proto outputs; pose adds kpt outputs; YOLOX
  exposes the 3 decoupled [1,85,H,W] heads. Task-specific decode (seg mask assembly, pose kpt
  mapping, YOLOX grid+stride) stays on the host — Neat has no built-in seg/pose/YOLOX decode
  type. Documented per-model in work/<model>/reports/SURGERY.md.
- **All four audited clean**: 0 unsupported ops for int8 (incl. seg ConvTranspose stride-2,
  YOLO26 Einsum). No `.so` fallback expected; strict one-ELF/no-.so policy achievable.

## T1 / Agent A — YOLO11 verification + 2x RTSP app (2026-07-09)

- Decision: Ran the fresh YOLO11n verification chain in a NEW scratch subdir
  `model-compilation/work/yolo11n/t1_verify/` instead of the existing
  `work/yolo11n/`. Why: the brief forbids deleting/overwriting existing owner
  artifacts, and scripts 11/09/12 write to hard-coded `work/<id>/` paths.
  Alternative: `--force` in place (would overwrite the prior run). Reverse:
  delete the `t1_verify/` subdir; nothing else is touched. Implementation: copies
  of scripts 11/09/12 in `<scratchpad>/t1_scripts/` with `ROOT` pinned to the real
  model-compilation dir and `base` redirected to the `t1_verify` subdir. The
  committed scripts under `scripts/` were NOT modified.
- Decision: multi-stream app uses ONE shared model Run handle serviced
  round-robin by the two streams (push frame -> pull result before next stream),
  rather than two independent model graphs. Why: matches the "shared YOLO11 model
  stage" requirement and keeps one MLA ELF resident; identity is preserved because
  each result belongs to the frame just pushed. Alternative: per-stream model
  graph (needed only if the two inputs have different resolutions — the app
  raises a clear error and documents that fallback in the README). Reverse: call
  `build_model_graph` per context in `run()`.
- Decision: app decode uses `BoxDecodeType.YoloV26` for the yolo11 archive
  (not YoloV8). Why: the compile_ready surgery exposes the YoloV26 6-tensor
  grouped contract (bbox_0..2 + class_logit_0..2), confirmed at runtime by the
  board log "Configuring for the decoding type: 8:yolo26 / Configured for
  subtensors: 6". No deprecated boxdecode_original_width/height set (Model.h
  deprecates them).
- Note: `scripts/06_neat_smoke_test.py` fails on this archive with
  `misconfig.caps ... not-negotiated` because its default-ModelOptions raw-tensor
  route mis-negotiates appsrc caps. Verification instead used
  `scripts/10_run_yolo_sample_pipeline.py` (proper tensor ModelOptions route) which
  passes, and the 2x RTSP app (image route). Not a model defect. If 06 is meant to
  be the canonical smoke test, its route/options need fixing (owned by scripts/,
  not modified here).
