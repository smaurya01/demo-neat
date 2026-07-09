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

## D4 — Orchestrator: pre-existing bug found in `scripts/06_neat_smoke_test.py` (2026-07-09)

**Found:** Agent A could not use `scripts/06_neat_smoke_test.py` on a valid yolo11n archive — it
fails with `misconfig.caps ... not-negotiated`. Root cause: it feeds a raw NCHW tensor using
*default* `ModelOptions`, so the appsrc caps mis-negotiate. This is a **defect in the existing
script, not in the model or the archive** — `scripts/10_run_yolo_sample_pipeline.py` (proper tensor
`ModelOptions` route) passes on the same archive, and the new RTSP app (image route) runs it
end to end.
**Decided:** Do NOT fix script 06 in this wave. Agent A used script 10 for the smoke test instead.
**Why:** `scripts/01-12` are pre-existing owner files; Agent C was concurrently working in
`scripts/`, and a same-wave edit risked a write conflict on a file neither agent owned.
**Alternative:** Patch 06 to set an explicit tensor route (`InputKind.Tensor` + EV74 tensor memory).
That is the correct fix and is a small, self-contained change — recommended as a follow-up.
**Reverse:** N/A — nothing was changed.

## D5 — Orchestrator: compile slot must be held from a background process (2026-07-09)

**Found:** the agent harness kills a foreground shell call at ~2 minutes. When that call held the
`flock` compile slot, the SIGTERM released the lock — which would let a second agent start a
concurrent compile and silently break the "exactly one compile at a time" invariant.
**Decided:** long compiles are launched as background processes that hold the lock for their whole
lifetime; agents poll instead of blocking in the foreground.
**Why:** the invariant must survive harness timeouts, not just agent good behaviour.
**Reverse:** N/A (operational practice, recorded in overall-learning.md).

## T6 (Agent E) — LLM/VLM compilation + GenAI server — 2026-07-09

- DECIDED: Documented BYO LLM/VLM compilation as the host-side `llima-compile` (Model Compiler)
  step, NOT a `llima` subcommand. WHY: the on-board `llima` CLI has no `compile` verb
  ({run,search,pull,list,rm,benchmark-server}); the official docs
  (developer.sima.ai/software/genai-llima/compilation_genai) name `llima-compile`, and
  core/docs/getting-started/compatibility.mdx confirms the Model Compiler is required for GenAI
  compile/quantize. ALTERNATIVE: invent a `llima compile` command — rejected as fabrication.
  REVERSE: if the compiler tool is later confirmed to have a different name/flags, edit the printed
  command strings in 04-*/*.ipynb (all heavy commands are inert printed strings).
- DECIDED: Every `llima-compile` fact is marked [docs] (WebFetch of official docs, NOT executed) vs
  [core] (verified in /workspace/core/src/genai/GenAIInternal.cpp). WHY: honesty requirement — we
  could not run llima-compile (not on board, not in core). The deployed artifact contract
  (devkit/ + elf_files/, exactly one of vlm_config.json/whisper_config.json, VLM vision keys
  vm_cfg/mm_cfg/vision_model_name) is core-verified and is the load-bearing, testable part.
- DECIDED: 05-genai-server built entirely on tutorial 021 (Python + C++ + all 3 request scripts) and
  GenAIServer.h / module.cpp bindings. serve_multi_model.py adapts serve_genai_models.py; defaults to
  the 3 board models. client_examples.py merges the 3 tutorial request clients behind a --run gate so
  import/py_compile never fires a request. WHY: real source material; no invented APIs.
- DECIDED: Taught the two-servers distinction explicitly — `llima benchmark-server` CLI (on-board,
  one model, benchmarking) vs in-process `pyneat.genai.GenAIServer` (multi-model app server). WHY:
  brief flagged conflation risk. Concurrency guidance grounded in tutorial 021 "In Practice" (models
  share one MLA hardware gatekeeper; one process/many served names does not multiply throughput).
- NOTE for Agent F: llima/README.md folder map currently says 04/05 are "owned by other tracks and
  not part of this folder's basics". Those sections now exist and could be linked from the README.

## T3 — detection-vlm-assistant app + llima/03 notebook (Agent D, 2026-07-09)

- **New app `apps/detection-vlm-assistant/`** (`main.py`, `src/vlm_commenter.py`,
  `config/default.conf`, `README.md`, `assets/models/` git-ignored with the T1 yolo11n
  archive copied in as `yolo_11n_mpk.tar.gz`). Adapted from
  `apps/examples/genai/detection-to-vlm-assistant` (crop-to-VLM) + Agent A's
  `apps/multi-stream-yolo-yolo11` (detector idioms). Detection leg validated LIVE; VLM leg
  code-complete + API-checked, NOT executed (per split-validation rule).

- **Decision: detector uses `push([tensor])` + `pull("detections", ...)`, NOT `run([...])`.**
  Why: on the compile_ready yolo11n archive, the reference app's synchronous
  `detector_run.run([tensor])` returned a sample with **0 extractable tensors** -> 0
  detections, even though the raw 6-tensor YoloV26 head was present. Switching to Agent A's
  named-output push/pull pattern surfaced the model-managed box-decode output (a 1-tensor
  TensorSet) which `pyneat.decode_bbox` turned into real boxes (PERSON/CHAIR/TV/... verified
  live). Alternative (reference's `run()` + `parse_boxes`) left on the table because it
  produced nothing here. Reverse: revert `detect()` to `run([...])` if a future archive
  surfaces decode output through the sync helper.

- **Decision: explicit letterbox resize in ModelOptions** (`resize.enable=On`,
  `width/height=model_width/height=640`, `mode=Letterbox`, `pad_value=114`, plus
  `color_convert` BGR->RGB, `num_classes=80`). Why: the reference's auto-only preprocess did
  not resize full-size frames to the 640 model input on this archive. Mirrors Agent A's proven
  block. Reverse: drop the resize block to test auto-planner behaviour.

- **Decision: VLM via direct `pyneat.genai.VisionLanguageModel` (in-process), not the
  upstream OpenAI-compatible `GenAIServer` HTTP path.** Why: one process owns both detector and
  VLM, so a function call beats an HTTP round-trip; matches the house `02-run-llm-vlm` usage.
  Server path documented in the notebook + README as the alternative (use when the boundary is
  a network). Reverse: swap `_call_vlm` for an HTTP client if the VLM becomes a separate service.

- **Decision: dry-run (`--no-vlm`) is the default whenever the VLM dir is missing / disabled.**
  The bounded background worker logs the crop + the exact prompt that WOULD be sent instead of
  calling the VLM. This is how the detection leg was validated live without touching the VLM,
  and is genuinely useful for tuning triggers/prompts. Colour trap (`cv2.cvtColor BGR2RGB` at
  the request boundary) called out in code, README, and notebook.

- **Trigger/dedup/rate-limit knobs** (class allow-list, min score, min area frac, IoU dedup +
  cooldown, wall-clock interval, bounded queue) live in `config/default.conf` so operators
  retune without editing code. `vlm_interval_seconds` is wall-clock (noted in README): over a
  fast still-image batch only the first qualifying crop fires — intended for the always-on RTSP
  case.

## T5 phase 1 (Agent C) — completion + learnings — 2026-07-09

- **RESULT: 4/4 compiled, all archives pass strict one-ELF/zero-.so.** yolo11s (MLA:1,EV74:12,
  A65:0), yolo11s-seg (MLA:1,EV74:16,A65:0), yolo26s-pose (MLA:1,EV74:21,A65:0), yolox_s
  (MLA:1,EV74:9,A65:0). A65:0 on every model = nothing fell back to the host CPU. Compile times
  ~6-8 min each. Zero blockers (yolox_s genuinely-new surgery landed).
- **Gotcha (fixed in reporting, not a code bug): `pgrep -f "15_compile_t5_int8.py --model-id X"`
  self-matches the polling loop's own command line**, because the loop's cmdline literally
  contains that pattern string. It made every wait loop falsely report "STILL RUNNING". The
  reliable completion signal was the wrapper's `RELEASED slot ... rc=0` line and the
  fully-written compile log ("Compilation complete"). For future waiters, grep the wrap log for
  RELEASED, or use `pgrep -f quantize_compile.py` (the actual child), not the launcher pattern.
- **12_compile_yolo_int8.py buffers all child stdout and writes the log only at process exit**
  (subprocess.run(..., stdout=PIPE) then write). So mid-compile progress is invisible in the log
  file; watch the wrapper's stderr wrap-log instead for the acquire/release bracket.

## T7 (Agent G) — transformer / difficult models — 2026-07-09

- **05_validate_archive.py extended (SANCTIONED).** Added `--min-elf`/`--max-elf`
  (default 1/1) and `--allow-so`. Defaults reproduce the original strict
  "exactly one ELF, zero .so" behaviour byte-for-byte, so all T1/T5 callers are
  unaffected (verified: strict run on yolo11n archive still `status: pass`). T7
  invokes with `--max-elf 3 --allow-so`; a `.so` then yields
  `status: pass_requires_justification` (not an auto-fail) and the model's surgery
  report must carry the written justification. Reverse: revert the file; the extra
  flags are additive.
- **Calibration set = /workspace/calibration_images (20 real COCO images)** for the
  T7 INT8 quantization, not assets/calibration (only 8 synthetic gradients). We are
  re-quantizing from scratch anyway (see maxvit note), and the brief lists
  /workspace/calibration_images as a sanctioned standard set. Better calibration →
  better INT8 top-k. Reverse: pass `--calib-dir assets/calibration` to script 18.
- **maxvit_t "resume" = full re-run of quantize+compile.** quantize_compile.py has no
  load-a-saved-.sima-and-only-compile path; `quant_model` must be produced in-process
  before `.compile()`. So the cheapest correct "resume" is re-running the whole
  command through the compile slot. The interrupted run's leftover .sima under
  work/maxvit_t/compile/ is untouched; the fresh build lands in work/maxvit_t/compile_int8/.
- **ViT / DINOv2 attention rewrite (scripts/19_vit_attention_surgery.py).** Rewrote the
  two per-block rank-4 batched attention MatMuls (Q·Kᵀ and A·V, both operands
  activations) to `Einsum "bhmk,bhkn->bhmn"` — the same MLA-friendly rewrite the YOLO
  walkthrough uses, generalized to token-sequence attention. Equation is
  layout-agnostic for any 4D batched matmul and is validated by onnx.checker before any
  compile slot is spent. Linear q/k/v/proj/mlp MatMuls (one weight operand) are left as
  ordinary supported GEMMs. 24 rewrites each (12 blocks x 2). Alternative considered:
  trust the compiler's native batched-MatMul support and skip the rewrite; rejected as
  the higher-risk first attempt given only ~1 compile slot per model.
- **DINOv2 masks/Where removal.** torch.hub DINOv2 exports an unused rank-0 `masks`
  input feeding masks→Unsqueeze→Where(cond, mask_token, patch_embed)→Concat. At
  inference masks is all-false so Where is the identity on the patch-embed tensor; we
  rewire Concat to the patch-embed output and delete masks/Unsqueeze/Where, leaving a
  single `input`. Without this the compiler importer (bound to only `input`) would
  choke on the second graph input. Reverse: re-run 03_surgery.py to regenerate
  surgery.onnx with masks intact.
- **DETR attention left as-is.** detr_resnet50's generic surgery (03_surgery.py) already
  produced a static compile-ready graph (pred_logits[1,100,92], pred_boxes[1,100,4], 0
  unsupported ops); its attention is rank-3 (nn.MultiheadAttention folds heads into
  batch: [8,625,32]x[8,32,625]), which the audit lists as a supported batched MatMul, so
  the rank-4 Einsum rewrite deliberately does not touch it. No Hungarian matching exists
  at inference (training-only loss); DETR postprocess is pure CPU softmax+threshold+box-decode.

## T5 phases 2-3 — quad-stream-quad-model app (Agent C2, 2026-07-09)

- **Decided:** one independent 3-graph shuttle (RTSP source / model / video sink)
  PER stream, four different compiled archives, single-process single-thread
  round-robin. Alternative was `graphs.combine(ByFrame)` (tutorial 015) — rejected
  because the 4 streams run 4 different models + 4 different UDP sinks with nothing
  to join, and per-stream stats must stay separable. Reversible: swap the loop for
  a combine graph if a future variant shares one model.
- **Decided:** detection (yolo11s) uses on-device `BoxDecodeType.YoloV26` +
  `pyneat.decode_bbox`; seg/pose/YOLOX leave `decode_type=Unspecified` and decode
  the RAW heads on the host in NumPy (`src/decoders.py`). Confirmed on board that
  the raw-head route delivers ALL model outputs (10 seg / 9 pose / 3 yolox tensors)
  through a single named `"heads"` endpoint, and that the MLA emits them **NHWC**
  `(1,H,W,C)` (not NCHW) — decoders transpose HWC→CHW in `_squeeze_batch`.
- **Finding (contradicts the T5 "no built-in seg/pose decode" note, worth
  recording):** `core/include/pipeline/BoxDecodeType.h` DOES define
  `YoloV26Seg(19)`, `YoloV26Pose(18)`, `YoloX(21)`, and `DetectionTypes.h` exposes
  `decode_pose`/`decode_segmentation`. BUT those host helpers parse a BoxDecode
  *wire payload*, which our raw-head archives (decode tail surgically removed) do
  NOT produce. So on THESE archives host NumPy decode is still required — the paid
  knowledge holds for the archives as built. A future variant could instead set
  `decode_type=YoloV26Seg/Pose` and let Neat attach the fused stage; untested here.
- **Measured (DevKit, 20 frames/stream, round-robin):** per-stream service FPS
  detection ~15.7, yolox ~12.7, seg ~3.6, pose ~0.5; **aggregate ~1.7 FPS across 4
  streams.** Bottleneck is A65 host decode (pose ~1.9 s/frame), NOT the MLA. Single
  Python thread ⇒ agg ≈ 1/Σ(per-frame times). 2-stream config ~8 agg FPS. Honest
  limit documented in README + TEACHING; fix = per-stream threads + on-device decode.
- **Known gap:** YOLOX host decoder runs end-to-end but emits 0 boxes — decoupled-
  head channel order needs one more on-board verification (activation auto-adapt
  already added). Detection/seg/pose validate the raw-head→host-decode design.
- **Did NOT commit** (orchestrator commits). Referenced compiled archives in place
  under model-compilation/work/ (NFS, no copy); assets/models/ is override-only.

## Wave 3 integration pass (Agent F) — root README, llima map, training alignment, REVIEW — 2026-07-09

- **Root `README.md` rewritten as the repo front door, preserving what was still true.** Kept the
  original layout/requirements/build-artifacts/per-app-README spirit; added the 4-day arc table, a
  navigation table (folder → what you learn → entry point), a prominent **"Verified vs
  documented-but-unrun"** section (3 honesty tiers: live-validated / not-executed / docs-derived),
  the SDK+venv+DevKit+RTSP prereqs, and the `dk` (human) vs `ssh+timeout` (automation) rule. WHY: a
  day-1 trainee needs one honest map; the old README predated `llima/`, `model-compilation/`,
  `training/`, and the three new apps. ALTERNATIVE: minimal patch of the old README — rejected, it had
  no notion of the new tracks or the verified/unrun distinction (the single most important thing).
  REVERSE: `git checkout` the file; prior version is in history.
- **`llima/README.md` folder map updated (minimal edit, kept Agent B's voice).** The map claimed
  03/04/05 were "owned by other tracks and not part of this folder's basics"; those sections now
  exist, so I extended the map to list 03/04/05 with cross-links and carried over Agent E's honesty
  note about `llima-compile` / no `llima compile` / HTTP 403 flags. Did NOT touch 01/02 content.
- **`training/` alignment is ADDITIVE, not a rewrite.** Appended an "Alignment With This Repo" section
  mapping each day/session to the concrete in-repo app/track that now realizes it (with validation
  status), carried the verified/unrun honesty note, and listed reference paths in the doc that do NOT
  resolve: `/workspace/apps/examples/model-benchmark` (real dir is `.../benchmarking`) and ~11
  abbreviated core-tutorial names that don't match the real directory names. WHY: brief says
  ADD/ALIGN, never delete/rewrite owner content; the fixes are flagged, not silently patched into the
  body. REVERSE: delete the appended section.
- **`REVIEW.md` created (owned, new file).** One substantive finding: `sima-cli modelzoo ... get
  yolo_11n` in `apps/multi-stream-yolo-yolo11/README.md:54-60` (and pre-existing
  `apps/single-stream-yolo-yolo11/README.md:35`) contradicts the fact that yolo11 is compiled via
  surgery, not fetched from the 2.1.2 Modalix zoo (which only exposed yolo_v8n/yolo_v8n_seg/open_pose)
  — medium, reported not fixed (not my file). All other categories clean: no invented APIs, no
  deprecated boxdecode fields, correct NV12/BGR/RGB colour handling, correct YoloV26 decode, honest
  verification claims. All relative links (mine + reviewed non-owned docs) tested with `test -e` and
  resolve.
- **Did NOT git commit** (orchestrator commits). Did NOT write anything under `model-compilation/`
  (Agent G active there) — read-only review only.

## T7 (Agent G) — final compile outcomes + corrected understanding — 2026-07-09

All four T7 models are **documented blockers** (fail-forward); none met the relaxed
1–3 `.elf` policy. Verified outcomes:

- **maxvit_t**: rc=0, **58 `.elf` / 78 `.so`** (136 segments). Cause: windowed/grid
  attention partition/departition non-4D reshapes + "unsupported einsum equation" on the
  native MaxViT einsums, no `--any_shape_on_mla`. Fix to try: recompile with
  `--any-shape-on-mla` (not attempted — out of 60-min compile-start budget).
- **dinov2_vits14**: rc=0, **99 `.elf` / 195 `.so`** (294 segments) **even with
  `--any_shape_on_mla`**. Cause (from log, ranked): batch-axis reshapes ("Reshape
  affecting the batch axis is not supported", ~260 hits) + token-sequence
  LayerNormalization ("Input is not 4D/5D", "reduction over channel axis, dim<128" — the
  384-dim channel). The `MatMul→Einsum` rewrite was **accepted** (not a fallback cause).
- **vit_b_16**: compile **errored in quantization** — TVM conv2d channel mismatch
  (64 vs 768) on the Einsum-rewritten graph's attention out-projection. Next: compile the
  plain-MatMul `surgery.onnx`.
- **detr_resnet50**: not compiled (budget); surgery ONNX + pipeline + decode analysis
  ready. Verified `BoxDecodeType.Detr=13` has no raw-head decoder (only YoloV26*/YoloV6/
  YoloX do) → CPU postprocess is correct.

**Corrected understanding (supersedes my earlier optimistic notes):**
1. The attention `MatMul→Einsum` rewrite (the YOLO house pattern) is **not the lever**
   for pure ViTs and can hurt (vit quantize error). For a stock ViT the batched MatMul is
   already a supported op; only rewrite where the compiler actually rejects the matmul.
2. The real blocker for pure token-sequence transformers on SiMa gen2 / SDK 2.1.0 is
   **placement, not op legality**: 3-D `[1,N,C]` LayerNorm (channel-axis reduction >128)
   and batch-axis head reshapes are not MLA-placeable, so the graph fragments regardless
   of surgery. Conv-hybrid backbones that stay 4-D NCHW (fastvit_t8 → 1 elf/0 so) do not
   hit this. This corrects my earlier "leave LayerNorm, the compiler lowers it" note.
3. An rc=0 compile is not a passing artifact — always check the real `.elf`/`.so` count
   (the fixed `05_validate_archive.py` excludes `.so` from the ELF count; a `.so` also
   carries ELF magic and was double-counted before the fix — my first maxvit read said
   "136 elf" when the truth was 58 elf / 78 so).
