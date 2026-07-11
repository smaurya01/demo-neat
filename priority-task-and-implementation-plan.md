# Priority Task And Implementation Plan

Date: 2026-07-09
Owner: Suraj Maurya
Goal: close the highest-value gaps in the 4-day NEAT training material, prioritizing the
current weak points: **LLiMA (LLM/VLM)** and **model compilation with graph surgery**.

Ground rules for all tasks:

- `tutorial/III-advanced` is unreviewed and NOT a source of truth. Ignore it.
- Source-of-truth learning references: official docs, `/workspace/core` (tutorials, docs,
  include, python) and `/workspace/apps/examples`.
- **`/workspace/overall-learning.md` is local memory/skill**: every agent reads it BEFORE
  starting work (validated commands, pyneat best practices, ModelZoo facts, dk usage,
  RTSP debugging). New durable, cross-project lessons discovered during this work are
  appended back to it (via the orchestrator, to avoid write conflicts).
- New apps go under `apps/` and follow the existing app layout
  (`README.md`, `main.py`, `config/default.conf`, `assets/models/`, `src/` when needed).
- New LLiMA learning material goes under a new top-level `llima/` folder.
- Default model precision is **INT8** (BF16 only where explicitly comparing).
- Calibration: 20 images from `/workspace/demo-neat/model-compilation/assets/yolo_calibration`
  (or `/workspace/calibration_images`).
- **Do NOT run or verify any LLM/VLM/ASR model** — these runs are heavy and will be done
  manually by the owner later. Agents prepare notebooks/scripts/configs with exact,
  copy-paste-ready run commands and expected-output documentation, but never execute
  LLM/VLM inference, `llima pull`, `llima run`, or LLM/VLM compilation jobs.
- **One model compilation at a time, strictly.** Never start a compile while another is
  running — the next compile starts only after the previous one finishes (across ALL
  agents, not just within one agent's task).
- **Model size cap: nothing larger than the medium (`m`) YOLO variant.** Allowed: `n`,
  `s`, `m` (e.g. yolo11n/s/m, yolov8n/s/m, yolo26n/s/m, YOLOX-s/m). Never compile `l`
  or `x` variants (yolov8l, yolov8x, yolo11l, yolo11x, yolo26l, yolo26x, YOLOX-l/x, …).

## Environment And Access

| Item | Value |
| --- | --- |
| RTSP stream | `rtsp://192.168.132.129:8555/stream` — for multi-stream apps, use this SAME stream N times for now |
| DevKit board | `192.168.135.203`, user `sima`, password `edgeai` |
| Running on DevKit | `dk` works in a real terminal; **agents/automation must use ssh** (see verification below) |
| llima CLI | On the board at `/usr/bin/llima` — **WORKING** (PyYAML installed by owner 2026-07-09) |
| RTSP sanity check | `ffprobe -hide_banner -rtsp_transport tcp rtsp://192.168.132.129:8555/stream` before blaming model/pipeline (see overall-learning.md) |
| DevKit etiquette | Never kill user-owned processes on the board; report the owner (`/tmp/rpmsg_lock_rpmsg*.owner`, `ps -ef`) instead |

### Environment verification (run 2026-07-09, all checks executed)

| Check | Result |
| --- | --- |
| Host = SiMa SDK container | yes — `/opt/toolchain/aarch64/modalix` present |
| DevKit reachable | yes — ping 16ms; **passwordless ssh works** (`ssh sima@192.168.135.203`) |
| RTSP stream live | yes — H.264 1280x720 @ 59.94 fps + AAC audio (verified with `ffprobe`) |
| `dk` availability | bash **function**, visible via `bash -lic 'type dk'` (the `sima-neat` skill's prescribed check). No `devkit-run` on PATH |
| `dk` non-interactive | **FAILS/hangs — needs a TTY.** `dk shell "cmd"`, `dk shell /bin/bash -c ...`, `dk shell /bin/bash script.sh`, and `dk /workspace/script.py` all blocked inside `source devkit.sh` with stdin=`/dev/null` and had to be killed. In a login shell it also mis-detected sync and aborted: `Local sync root not found` / `rsync fallback sync failed`. Works fine for a human in a real terminal |
| Board execution (verified working) | `ssh -o BatchMode=yes sima@192.168.135.203 'source $HOME/pyneat/bin/activate; python /workspace/<script>.py'` → `aarch64`, python 3.11.2, **pyneat 0.3.0+develop** |
| `/workspace` on board | **NFS-mounted from host** (`172.16.1.24:/home/surajmaurya/workspace2`). Files written on the host appear instantly on the board at the same path — no copying needed |
| `/neat-resources` | present: `core-src` and `apps-src` (per skill, prefer these for source navigation in SDK) |
| Model compiler env | `/sdk-extensions/model-compiler/bin/activate` → `afe` OK, `onnx` 1.17.0 |
| `ultralytics` | **INSTALLED** into the model-compiler venv (8.4.90, owner-approved); `afe` + onnx 1.17.0 verified intact afterwards |
| Host disk | 83 GB free on `/workspace` — enough for vision compiles |
| Board disk | **only 5.9 GB free of 14 GB on `/`** — a real constraint for fresh LLM/VLM pulls |
| Board internet + sudo | pypi reachable; `sudo` is NOPASSWD |
| `llima` on board | **WORKING** at `/usr/bin/llima`. Subcommands: `run, search, pull, list, rm, benchmark-server` (note: `benchmark-server`, NOT `serve`) |
| llima models already pulled | `Qwen3-4B-Instruct-2507-GPTQ-a16w4` (LLM), `Qwen3-VL-4B-Instruct-GPTQ-a16w4` (VLM), `whisper-small-a16w8` (ASR) — covers text/image/audio for T4 with **no pull needed** |

Full captured CLI signatures + the 36-model remote catalog are saved for the agents at
`<scratchpad>/llima_ground_truth.md`. Agents must reuse that instead of re-running `llima`.

### Consequences (binding on all agents)

- **Agents use ssh, not `dk`, for board commands.** `dk` needs a TTY and hangs in the agent
  harness. This is the `sima-neat` skill's own documented recovery path ("fall back to direct
  SSH only if `dk` is unavailable"). Verified working:

  ```bash
  ssh -o BatchMode=yes sima@192.168.135.203 'source $HOME/pyneat/bin/activate; python /workspace/<script>.py'
  ```

  **In app READMEs, still document the `dk` command** — the human reader has a terminal and
  `dk` is the intended UX. Document ssh only as the automation/CI fallback.
- **No file copying to the board.** `/workspace` is NFS-mounted; write host-side, run board-side
  at the identical path. Keep any script the board must run under `/workspace`.
- **`llima` is working.** Use the captured signatures and model IDs in
  `<scratchpad>/llima_ground_truth.md`. Two correctness traps for the teaching material:
  the serve subcommand is `benchmark-server` (not `serve`), and ASR goes through
  `llima run --stt_model_path`.
- **Three models are already on the board** (LLM `Qwen3-4B-Instruct-2507-GPTQ-a16w4`,
  VLM `Qwen3-VL-4B-Instruct-GPTQ-a16w4`, ASR `whisper-small-a16w8`). Write T4's
  text/image/audio material against exactly these — no `llima pull` in any lesson's happy path.
- **Board has only ~5.9 GB free.** T2/T4/T6 must document a `df -h /` precheck before any
  `llima pull`, and name the smallest viable models (`LFM2.5-230M-a16w4`, `Qwen3-0.6B-GPTQ-a16w4`,
  `LFM2-VL-450M-a16w4`) for anyone pulling fresh.
- **`ultralytics` 8.4.90 is installed in the model-compiler venv** (owner-approved). T1/T5 export
  scripts should `source /sdk-extensions/model-compiler/bin/activate` — one env for export,
  surgery, quantize, and compile. `afe` and `onnx` 1.17.0 verified intact after the install.
- **Explain the LLiMa quantization suffixes** in the material: `a16w4` = 16-bit activations /
  4-bit weights, `a16w8` = 8-bit weights.

## Priority Order

| Priority | Task | Why this order | Size |
| --- | --- | --- | --- |
| P0 | T1 — YOLO11 surgery verification + 2x RTSP Python app | Confidence check on an already-working flow; unblocks T3 and T5; small effort | S-M |
| P1 | T2 — `llima/` foundations (concepts + CLI) | Weak point #1; foundation every other GenAI task builds on | M |
| P1 | T4 — Run LLM/VLM: notebook + scripts (text/image/audio) | Weak point #1; direct continuation of T2, same folder | M |
| P2 | T3 — YOLO + GenAI (detection-to-VLM) app + teaching material | Needs a compiled YOLO (T1) and a pulled VLM (T2); high demo value | M |
| P2 | T6 — LLM/VLM compilation + GenAI server (multi-model) | Weak point #1 and #2 combined; builds on T2/T4 concepts | M-L |
| P3 | T5 — 4x model / 4x stream pipeline + teaching material | Largest effort (3+ new model compiles incl. YOLOX surgery); you already know multi-stream CNN well, so learning value is in the material, not the doing. Model compilation starts early because it is the long pole. | L |
| P4 | T7 — Transformer/difficult-model surgery + INT8 + pipelines (finish `model-compilation`) | Weak point #2 at its hardest (non-CNN graph surgery); runs LAST, after T1–T6 are done, because it is open-ended iteration work and shares the single compile slot | L |

Note on T5 priority: P3 for *your* learning, but its model-compilation phase starts in
Wave 1 because compiling four models is the longest-running dependency in the whole plan.

---

## T1 — YOLO11: .pt → ONNX → surgery → INT8 → 2x RTSP Python App (P0)

### Current status (verified 2026-07-09)

Already working in `model-compilation/`:

- `scripts/11_export_fresh_yolo.py` — download Ultralytics `.pt`, export ONNX
- `scripts/09_yolo_compile_ready_surgery.py` — `compile_ready` surgery (MLA-friendly
  attention rewrite, decode/postprocess removed, exposes the 6 tensors for
  `BoxDecodeType.YoloV26`)
- `scripts/12_compile_yolo_int8.py` — INT8 calibration + compile
- Result: `yolo11n` and `yolo26n` archives pass validation (exactly one ELF, no `.so`)

### Deliverables

1. **Verification run**: re-run the full chain fresh (`.pt` → ONNX → surgery → INT8 →
   archive validation → Neat smoke test) for `yolo11n` and record the log in
   `model-compilation/results/`. Confirms "we are good with yolov11 graph surgery".
2. **New app** `apps/multi-stream-yolo-yolo11/` (Python):
   - 2x RTSP input streams → shared YOLO11 model stage → Neat box decode →
     annotated UDP/RTP output per stream (aligned with existing single-stream apps)
   - both streams default to `rtsp://192.168.132.129:8555/stream` (same source twice
     for now), configurable in `config/default.conf`
   - `README.md` (run instructions, `dk shell` run command, host viewer pipeline),
     `main.py`, `config/default.conf`, `assets/models/` (downloaded, git-ignored)
   - follow pyneat best practices from `overall-learning.md` (ModelOptions preprocess
     presets, `BoxDecodeType`, no deprecated boxdecode width/height fields)
3. **Walkthrough doc** `model-compilation/README.md`: turn the project log into a
   teachable step-by-step (what surgery does and why, output contract, how to redo it
   for any YOLO11/26 variant). This is the Day-2 Session-1 teaching backbone.

### Validation

- Archive check: one `.elf`, zero `.so`.
- Smoke test on still images (detections match expected classes).
- App runs against 2 RTSP streams on DevKit; both output streams viewable; stream
  identity preserved.

Dependencies: none. Reference: `apps/single-stream-yolo-yolo11`, `core/tutorials/015`, `018`.

---

## T2 — `llima/` Folder: LLiMA Foundations (P1)

### Deliverables

Create top-level `llima/` with:

```text
llima/
  README.md                     # folder map + how to run notebooks/scripts on DevKit
  01-llima-basics/
    01_llima_concepts.ipynb     # what LLiMa is, where it sits in the NEAT stack,
                                # supported model families, memory/size constraints
    02_llima_cli.ipynb          # llima search / pull / list / run — each explained
                                # with expected output and what happens on disk
```

- Every concept backed by the official docs:
  - https://developer.sima.ai/software/getting-started/
  - https://developer.sima.ai/software/develop-apps/development-workflow/genai-model
  - https://developer.sima.ai/software/genai-llima/runtime
- Cross-checked against `/workspace/core/docs/develop-apps/development-workflow/genai-model.mdx`
  and GenAI headers in `/workspace/core/include/genai/`.
- First step: locate the `llima` CLI on the DevKit via `dk shell` (or ssh fallback,
  `sima@192.168.135.203`) and document exactly where it lives and how to invoke it.
- Notebook style: markdown concept cell → runnable code cell → interpretation cell
  (same pattern as `tutorial/I-easy`).

### Validation (no heavy runs)

- Light CLI commands only: `llima --help`, `llima search`, `llima list` executed via
  `dk shell`; outputs captured in the notebook.
- `llima pull` and `llima run` are DOCUMENTED with exact commands and expected output
  (from official docs / core material) but NOT executed — owner runs them manually.

Dependencies: none (runs parallel with T1).

---

## T4 — Run LLM And VLM From Python: Text, Image, Audio (P1)

### Deliverables

Extend `llima/`:

```text
llima/
  02-run-llm-vlm/
    01_run_llm.ipynb            # GenAIModel: load, generate, stream tokens, options
    02_run_vlm.ipynb            # VisionLanguageModel: image + prompt → description
    03_audio_input_asr.ipynb    # ASRModel: audio file → transcription
    scripts/
      run_llm.py                # minimal CLI versions of each notebook
      run_vlm.py
      run_asr.py
```

- APIs covered: `GenAIModel`, `VisionLanguageModel`, `ASRModel`, generation options,
  prompt/chat formats, input handling for text, image, and audio.
- Ported/adapted from `core/tutorials/019_run_an_llm`, `020_run_a_vlm`, and
  `021_serve_genai_models/request_audio_transcription.py` — verified against the
  official runtime docs, not copied blindly.
- Each notebook explains the concept (what the API does, memory implications, when to
  use direct model vs server) before the code.

### Validation (no heavy runs)

- Scripts and notebooks are API-checked against `/workspace/core/include/genai/` headers,
  `core/python` bindings, and core tutorial sources — no invented APIs.
- Each notebook includes an "expected output" section (taken from core tutorial READMEs
  and official docs) so the owner can verify when running manually.
- NOT executed on device — LLM/VLM inference is run manually by the owner later.

Dependencies: T2 (folder conventions). Same agent as T2, executed
sequentially, to keep one coherent voice in the `llima/` folder.

---

## T3 — YOLO + GenAI: Detection-to-VLM App + Teaching Material (P2)

### Deliverables

1. **New app** `apps/detection-vlm-assistant/` (Python), adapted from
   `/workspace/apps/examples/genai/detection-to-vlm-assistant`:
   - YOLO detection pipeline; selected detections/crops trigger a VLM query;
     output = detections + natural-language description
   - Standard app layout aligned with other `apps/` entries
2. **Teaching notebook** `llima/03-yolo-plus-vlm/01_detection_to_vlm.ipynb`:
   - why detection + VLM (trigger-based multimodal), crop selection strategy,
     prompt design, direct `VisionLanguageModel` vs `GenAIServer` trade-off,
     performance considerations (VLM latency vs detection FPS)

### Validation (split: detection live, VLM deferred)

- Detection leg validated live on DevKit via `dk`: RTSP/image in → detections out.
- VLM leg is code-complete and API-checked, with exact run commands documented, but the
  VLM query itself is NOT executed — owner validates manually after pulling a VLM.
- Notebook cells runnable on DevKit (VLM cells marked "manual run").

Dependencies: T1 (compiled YOLO artifact), T2 (llima CLI documented; VLM pull is a
manual owner step before final validation).
Reference: `/workspace/apps/examples/genai/detection-to-vlm-assistant`, `core/tutorials/020`, `022`.

---

## T6 — LLM/VLM Compilation + GenAI Server With Multiple Models (P2)

### Deliverables

Extend `llima/`:

```text
llima/
  04-llm-vlm-compilation/
    01_llm_compilation.ipynb    # bring-your-own LLM: supported architectures,
                                # formats, size limits, compile flow, artifact layout
    02_vlm_compilation.ipynb    # VLM specifics: vision encoder + LM, source format
                                # requirements, common failure modes
    notes/triage_checklist.md   # is this model LLiMa-able? decision checklist
  05-genai-server/
    01_genai_server.ipynb       # GenAIServer concepts, OpenAI-compatible endpoints
    02_multi_model_server.ipynb # serving multiple LLM/VLM: memory budgeting,
                                # model switching/concurrency behavior
    scripts/
      serve_multi_model.py
      client_examples.py        # text chat, image chat, transcription requests
```

- Compilation content grounded in the official genai-llima docs plus whatever the
  `llima` CLI supports for model preparation. The full compile flow is written up as a
  step-by-step runnable procedure with exact commands — but NOT executed (heavy; owner
  runs it manually and the log gets added as teaching evidence afterwards).
- Server content adapted from `core/tutorials/021_serve_genai_models` (server +
  request scripts already exist there in Python and C++).

### Validation (no heavy runs)

- Compilation notebooks reviewed against official docs + CLI `--help` output; every
  command copy-paste-ready with expected artifacts described.
- Server/client scripts API-checked against `core/tutorials/021` and GenAI headers;
  multi-model memory-budget guidance documented. Live serving validated manually by
  the owner later.

Dependencies: T2/T4 (concepts, folder).

---

## T5 — 4x Model / 4x Stream Pipeline + Teaching Material (P3, starts early)

### Models (all INT8)

Known ModelZoo facts from `overall-learning.md` (SDK 2.1.2, Modalix): the zoo exposes
`yolo_v8n`, `yolo_v8n_seg`, `open_pose` — none of the four targets below — so expect
compile work for all four. `sima-cli modelzoo list` is interactive and aborts without a
TTY; check availability programmatically via
`https://docs.sima.ai/pkg_downloads/SDK2.1.2/model_zoo/metadata_gen2.json` first.

| Model | Task | Expected source |
| --- | --- | --- |
| YOLO11s | detection | not in 2.1.2 zoo metadata → compile via T1 flow (scale-up of proven yolo11n) |
| YOLO26s-pose | pose | compile: export + `compile_ready`-style surgery for pose heads |
| YOLO11s-seg | segmentation | compile: surgery for seg heads (proto + mask coefficients) |
| YOLOX-s | detection | compile + surgery (different head structure than Ultralytics YOLO — new surgery work) |

Calibration: 20 images from `model-compilation/assets/yolo_calibration`.

### Phases

1. **Model preparation** (starts in Wave 1 — long pole):
   - confirm zoo availability via the metadata JSON; download anything that exists
   - export/surgery/compile the rest through `model-compilation/` scripts, extending
     them where the head structure differs (pose, seg, YOLOX); document each surgery
     decision in `model-compilation/work/<model>/reports/`
2. **Pipeline app** `apps/quad-stream-quad-model/` (Python):
   - 4 RTSP streams → 4 different models → per-stream annotated output (UDP/RTP)
   - reference for structure: `apps/multi-model-load-probe`
3. **Teaching material** `apps/quad-stream-quad-model/TEACHING.md` (+ optional notebook):
   - how to design a complex Graph: stream/model routing, named endpoints
   - best use of `ModelOptions` (pre/post target EV74/A65), `RunOptions`
     (queue depth, overflow policy, low latency vs reliable), `InputOptions`,
     `OutputOptions`, model-route options
   - efficient output streaming (encoder settings, per-stream sinks)
   - measuring: per-stream FPS, drops, end-to-end latency

### Validation

- All 4 archives pass the one-ELF/no-`.so` check + still-image smoke test per task type.
- App sustains 4 streams with all 4 models on DevKit; per-stream stats reported.

Dependencies: T1 (proven compile flow). Surgery for pose/seg/YOLOX is genuinely new
work — budget for iteration.

---

## T7 — Transformer / Difficult Models: Surgery, INT8, Compile, Pipelines (P4, after T1–T6)

Objective: finish the hard models in `model-compilation/` so the folder becomes shareable
REFERENCE code — "how to get a transformer / non-CNN / difficult model onto SiMa Neat" —
with the surgery reasoning documented, not just the artifacts.

### Current status (from `model-compilation/results/summary.md`)

| Model | Status | Remaining work |
| --- | --- | --- |
| `vit_b_16` | ONNX pass, MPK not started | static-shape/attention surgery → INT8 → compile → smoke test |
| `maxvit_t` | quantized, compile interrupted | resume/re-run compile (long CPU-bound job) → package → validate |
| `dinov2_vits14` | ONNX pass, MPK not started | surgery (attention patterns) → INT8 → compile → embedding sanity check |
| `detr_resnet50` | ONNX pass, MPK not started | hardest: export + postprocess split (Hungarian matching stays on CPU) → surgery → INT8 → compile |

### Artifact policy (relaxed for T7 only)

The strict single-ELF/no-`.so` rule applies to the YOLO/CNN work (T1, T5). For these
harder transformer/non-CNN models it is relaxed:

- **1–3 `.elf` members are acceptable** (a model may legitimately split into multiple
  compiled subgraphs).
- **`.so` is acceptable only as a last resort, and only WITH a documented reason** —
  which op(s)/subgraph forced a host fallback, why surgery could not eliminate it, and
  what the runtime implication is. No unexplained `.so`.
- Before doing surgery, check each op against the official supported-layer list:
  https://developer.sima.ai/software/compile-a-model/model-compatibility — the surgery
  report must cite which ops were supported, which were rewritten to supported ones, and
  which (if any) fell back. Cross-check against
  `model-compilation/scripts/02_audit_onnx.py` output and the local
  `supported_operators.json` referenced by the `sima-model-surgery` skill.

### Deliverables

1. **Compiled artifacts** for the four models above (INT8; 1–3 `.elf`, `.so` only with a
   documented reason per the artifact policy above), worked ONE AT A TIME through the
   compile queue. If a model is genuinely blocked (unsupported op that surgery cannot fix
   and that cannot be pushed to an acceptable host fallback), stop and write a triage
   report instead of burning time — a documented blocker is also valid reference material.
2. **Per-model surgery documentation** in `model-compilation/work/<model>/reports/`:
   what was changed in the graph, WHY, and how to recognize the same pattern in other
   transformer models (attention rewrite, LayerNorm handling, dynamic-shape fixes,
   postprocess extraction).
3. **Reference pipelines** (Python, minimal, in `apps/` or `model-compilation/pipelines/`):
   - ViT/MaxViT: image classification (top-k) pipeline
   - DINOv2: embedding extraction + nearest-label sanity demo
   - DETR: detection pipeline with CPU postprocess (box + class decode)
4. **Updated `model-compilation` docs**: summary table refreshed; README section
   "Transformer and difficult models — patterns and gotchas" distilled from the
   per-model reports (this is the teaching payload for Day-4 Session-1).

### Validation

- Archive check per model: 1–3 `.elf` members; any `.so` present must have a written
  justification in that model's surgery report (otherwise it is a failure).
- Smoke test on sample images via `dk`.
- Classification models: ImageNet top-5 sanity on known images. DINOv2: embedding shape
  + nearest-label check. DETR: boxes/classes on a COCO sample.
- INT8 only; calibration from the standard calibration sets.
- `scripts/05_validate_archive.py` needs updating: its current pass criterion is
  "exactly one ELF and zero `.so`". Extend it to accept a configurable ELF count (1–3)
  and to report `.so` members as "requires justification" rather than an automatic fail.

Dependencies: starts only after T1–T6 are complete (shares the single compile slot and
the DevKit). Sequencing within T7: `maxvit_t` first (only needs its compile resumed),
then `vit_b_16`, `dinov2_vits14`, and `detr_resnet50` last (hardest).

---

## Execution Model: Agents And Waves

File-ownership is partitioned so agents never write the same paths.

| Agent | Task | Owns (writes) | Wave |
| --- | --- | --- | --- |
| A | T1 | `apps/multi-stream-yolo-yolo11/`, `model-compilation/README.md`, `model-compilation/results/` | 1 |
| B | T2 then T4 | `llima/README.md`, `llima/01-llima-basics/`, `llima/02-run-llm-vlm/` | 1 |
| C | T5 phase 1 (models) | `model-compilation/work/<new models>/`, `model-compilation/scripts/13+_*` (new files only) | 1 |
| D | T3 | `apps/detection-vlm-assistant/`, `llima/03-yolo-plus-vlm/` | 2 (after A + B basics) |
| E | T6 | `llima/04-llm-vlm-compilation/`, `llima/05-genai-server/` | 2 (after B) |
| C | T5 phases 2–3 (app + teaching) | `apps/quad-stream-quad-model/` | 2–3 |
| F | Integration pass | root `README.md`, `training/` program alignment, cross-links, code review | 3 |
| G | T7 transformer/difficult models | `model-compilation/work/<vit_b_16, maxvit_t, dinov2_vits14, detr_resnet50>/`, `model-compilation/pipelines/` (or new `apps/` entries), `model-compilation/results/` updates | 4 (only after Waves 1–3 complete) |

Shared-resource rules:

- **DevKit (`192.168.135.203`) is a single shared device**: agents prepare everything
  host-side and queue on-device validation; device runs are serialized (one agent on the
  board at a time, coordinated by the orchestrator). All board execution goes through
  `dk` / `dk shell` — no ad-hoc ssh sessions for running scripts.
- **Never kill user-owned processes on the DevKit**; report the owning process instead.
- **No LLM/VLM execution anywhere** (inference, `llima pull/run`, GenAI compilation) —
  applies to every agent; those steps are documented for manual runs by the owner.
- **Model compilation is strictly serialized: exactly ONE compile job at any moment,
  across all agents.** The compile queue is: A's yolo11n verification first, then C's
  T5 models one by one, then G's T7 transformer models one by one (each compile starts
  only after the previous finishes).
- **No model above the medium variant is ever compiled** (`n`/`s`/`m` only — never
  `l`/`x` variants of yolov8/yolo11/yolo26/YOLOX or similar).
- Agents must read `/workspace/overall-learning.md` first, then `/workspace/core`
  headers/tutorials/docs and `/workspace/apps/examples` before writing any API call;
  no invented APIs. `tutorial/III-advanced` is off-limits as a reference.
- **Artifact policy differs by task**: T1/T5 (YOLO/CNN) keep the strict one-ELF/no-`.so`
  rule. T7 (transformer/difficult) allows 1–3 `.elf` and permits `.so` only with a
  documented reason. Supported-op ground truth:
  https://developer.sima.ai/software/compile-a-model/model-compatibility
- Model-surgery work should use the `sima-model-surgery` and `sima-model-quantize-compile`
  skills (unsupported-op detection against `supported_operators.json`, calibration,
  target-specific compilation).
- Each agent's final report includes a "durable learnings" section; the orchestrator
  appends the cross-project ones to `/workspace/overall-learning.md`.

Review checkpoints — **owner is unavailable, so these become self-review gates**, not
blocking waits. At each boundary the orchestrator verifies the wave's deliverables,
records outcomes, and proceeds:

1. End of Wave 1: T1 app + verification log, `llima/` 01+02 sections, T5 model status table.
2. End of Wave 2: T3 app, T6 material, T5 pipeline.
3. End of Wave 3: integrated repo, updated READMEs, final review.
4. End of Wave 4 (T7): per-model artifact-or-triage-report, reference pipelines, and the
   "transformer patterns and gotchas" README section.

## Autonomous Operation Rules (owner away)

- **Do not block on approval.** Waves run back to back. Every non-obvious decision is
  logged to `DECISIONS.md` at the repo root (what was decided, why, what the alternative
  was, how to reverse it) so the owner can audit in one pass on return.
- **Git**: work is committed locally on a dedicated branch
  (`training-material-waves`, branched from the current `feature/python-examples`),
  one commit per completed task. **Nothing is pushed** and no existing branch is
  modified. Note: `tutorial/README.md` currently has an uncommitted modification and
  `model-compilation/`, `training/`, and the pptx are untracked — these are preserved,
  not reverted.
- **Hard stops — the orchestrator halts and reports instead of improvising:**
  - any action that would delete or overwrite existing owner work
  - board becomes unreachable, or a compile corrupts a shared artifact
  - a task requires running an LLM/VLM (never allowed — document and move on)
  - a model needs an `l`/`x` variant to succeed (out of policy — document and move on)
- **Blocked tasks fail forward**: if a model cannot be compiled, the agent writes a
  triage report and the wave continues. A documented blocker is a valid deliverable.
- **Learning capture**: durable cross-project facts are appended to
  `/workspace/overall-learning.md` by the orchestrator only (single writer).
