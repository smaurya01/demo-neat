# REVIEW — Wave 3 integration review (Agent F, 2026-07-09)

Reviewer's pass over the material built this wave (T1–T6 apps + `llima/` notebooks), plus a
cross-link check. Findings are grouped by the categories in the review brief. Each finding is
`file:line — what's wrong — severity — suggested fix`. **Fix only what I own; everything below in
files I do not own is reported for the owner, not changed.**

Scope note: `model-compilation/` (incl. `pipelines/`) is being actively written by Agent G (T7) and
was reviewed **read-only and not deeply** — anything there may change under me.

---

## 1. Invented APIs

**None found in the new material.** Every `pyneat` / `pyneat.genai` call in
`apps/multi-stream-yolo-yolo11`, `apps/detection-vlm-assistant`, `apps/quad-stream-quad-model`, and
`llima/*` matches the APIs verified in `overall-learning.md` / `DECISIONS.md`
(`ModelOptions`/`decode_bbox`/push-pull for detection; `VisionLanguageModel` / `GenerationRequest` /
`GenerationResult.metrics` / `accepts_image()` / `model_id()` for VLM).

Positive note on honesty: `llima/04-llm-vlm-compilation/*` correctly and repeatedly states there is
**no `llima compile` subcommand** and that GenAI compilation is a host-side `llima-compile` tool
(`01_llm_compilation.ipynb:24,35`, `02_vlm_compilation.ipynb:13`). This is the right call, not a
fabrication.

## 2. Contradictions between documents

- **`apps/multi-stream-yolo-yolo11/README.md:54-60`** — "Option A — download the prebuilt YOLO11n
  archive from the model zoo" with `sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_11n`.
  This **contradicts `overall-learning.md`** (the SDK 2.1.2 Modalix zoo metadata exposed only
  `yolo_v8n`, `yolo_v8n_seg`, `open_pose` — **not** `yolo_11n`) **and the same README's Option B**,
  which says to compile yolo11n yourself "because this is the flow T1 verifies." If yolo11 were in the
  zoo, the whole `model-compilation/` surgery flow would be unnecessary. The `get yolo_11n` command is
  unverified and most likely fails. **Severity: medium.** Suggested fix: delete Option A, or relabel
  it "unverified — yolo11 was not present in the 2.1.2 Modalix zoo metadata; use Option B (compile)."
  Owner: Agent A.
- **`apps/single-stream-yolo-yolo11/README.md:35`** — same `get yolo_11n` command (pre-existing app,
  not built this wave). **Severity: low** (pre-existing). Report to that app's owner; same fix.

## 3. Claims of verification the commits don't support

**None found.** The apps are conservative and match what `DECISIONS.md` records:

- `apps/detection-vlm-assistant` explicitly splits "detection leg validated live / VLM leg NOT
  executed" in its module docstring and README.
- `apps/quad-stream-quad-model/README.md` honestly reports the measured **~1.7 fps aggregate**, names
  the **A65 host-decode bottleneck**, and flags the **YOLOX decoder emits 0 boxes** known limitation.
- `llima/*` marks every heavy cell `MANUAL RUN — not executed by tooling`.

## 4. Deprecated fields (`boxdecode_original_width` / `_height`)

**Clean — none are set.** Every occurrence in the new material is a comment explicitly documenting
that they are deprecated and intentionally NOT set (`apps/multi-stream-yolo-yolo11/main.py:22,368`,
`apps/detection-vlm-assistant/main.py:317`, `apps/quad-stream-quad-model/TEACHING.md:127`). Correct.

## 5. Colour formats (RTSP=NV12, OpenCV=BGR, VLM images=RGB)

**Clean.**

- `apps/multi-stream-yolo-yolo11/main.py:353-354` — RTSP route sets
  `color_convert.input_format = NV12` → output RGB. Correct for the decoded-frame source.
- `apps/detection-vlm-assistant/main.py:304-305` — OpenCV/BGR route sets `input_format = BGR` → RGB,
  and `decoded_tensor_to_bgr()` converts NV12/I420 → BGR before feeding the BGR-configured detector.
  Consistent.
- `apps/detection-vlm-assistant/src/vlm_commenter.py:228` — converts the crop `BGR2RGB` before
  building the `GenerationRequest`, with a comment calling out the exact colour trap. Correct.
- `apps/quad-stream-quad-model/src/decoders.py` — operates on raw head tensors, no colour handling
  needed (correct); it transposes NHWC→CHW per the on-device raw-head layout.

## 6. `BoxDecodeType.YoloV8` where `YoloV26` is required

**Clean for the shipped default.** Both YOLO apps select `BoxDecodeType.YoloV26` for
`model_name in {yolo11, yolo26n}` (the default is `yolo11`), and only fall back to `YoloV8` in an
`else` branch for other model names (`apps/multi-stream-yolo-yolo11/main.py:360-363`,
`apps/detection-vlm-assistant/main.py:309-312`). The default path is correct.
- Minor (severity: low, informational): the `YoloV8` else-branch is effectively dead for the archives
  this repo ships and `YoloV8` is known to reject the compile_ready grouped head at build time. It is
  harmless (guarded, documented) but a future user setting `model_name=something_else` would hit a
  build error rather than a clear message. Optional: replace the else-branch with an explicit
  "unsupported model_name" error. Not a bug in current use.

## 7. Broken links

- **In files I own** (`README.md`, `llima/README.md`, `training/NEAT_4_DAY_TRAINING_PROGRAM.md`): all
  relative links were tested with `test -e` and **all resolve** (see Agent F report). No broken links.
- **In files I do not own** (new app READMEs, `TEACHING.md`, `model-compilation/README.md`,
  `model-compilation/pipelines/README.md`): all markdown links extracted and tested — **none broken.**
- **Reference discrepancies in `training/NEAT_4_DAY_TRAINING_PROGRAM.md`** (these are not markdown
  links but path citations, and I own this file — handled additively, not by rewriting the body):
  - `/workspace/apps/examples/model-benchmark` does **not exist**; the real dir is
    `/workspace/apps/examples/benchmarking` (cited Day 3 S1, Day 4 S3, Recommended Labs).
  - Several core-tutorial short names are abbreviated and don't match the real directory names
    (e.g. `001_run_first_model` → `001_run_your_first_model`, `019_run_llm` → `019_run_an_llm`,
    `020_run_vlm` → `020_run_a_vlm`, `021_serve_genai` → `021_serve_genai_models`, and ~7 more).
  Both are documented in the new "Alignment With This Repo" section I appended to that file, rather
  than edited inline, per the "don't rewrite owner content wholesale" rule.

---

## Summary

The Wave-1/2 deliverables are in good shape: no invented APIs, no deprecated-field usage, correct
colour handling, correct `YoloV26` decode, and honest verification claims. The one substantive
finding is the **`sima-cli modelzoo get yolo_11n`** command (medium; contradicts the established fact
that yolo11 is compiled, not fetched from the zoo) in the multi-stream app README, echoed in the
pre-existing single-stream app README. All cross-links (mine and the reviewed non-owned docs) resolve.
