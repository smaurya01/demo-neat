# NEAT 4-Day Training Program

This program is designed as a reusable NEAT enablement course. Days 1 and 2 are optimized for the Korea team and customer profile: Ubuntu host, Modalix DevKit, YOLOv11, VLM, two-camera pipelines, and agentic development with Codex or Claude. Days 3 and 4 are for all attendees and add the broader Japan/general field needs: Modalix SoM DevKit, Modalix EA DevKit, PCIe HHHL, conventional workflows, model compilation triage, GenAI, and production support patterns.

## Training Outcomes

By the end of the four days, attendees should be able to:

- Explain the NEAT stack: SDK container, DevKit runtime, Neat Library, Model Compiler, LLiMa, Insight, and example applications.
- Build Python and C++ NEAT applications using `Model`, `Graph`, `Run`, `Tensor`, `Sample`, and GenAI APIs.
- Compile, quantize, package, inspect, and validate CNN/YOLO model artifacts for Modalix.
- Triage unseen customer models and decide whether to use ONNX Model Compiler, graph surgery, LLiMa, or stop early due to unsupported architecture.
- Build single-stream, two-camera, and multi-stream YOLO pipelines with runtime queue tuning and metadata/video output.
- Integrate detection with VLM workflows for practical multimodal applications.
- Debug runtime, model, graph, queue, input/output, and device issues using structured NEAT diagnostics.
- Support both agentic workflows and conventional manual workflows for customer-facing work.

## Audience Split

| Days | Primary Audience | Focus |
| --- | --- | --- |
| Day 1-2 | Korea team, optional for others | Modalix DevKit, YOLOv11, two-camera input, VLM, agentic app development |
| Day 3-4 | All teams | model compilation/surgery, unknown-model triage, GenAI/LLiMa, PCIe, conventional workflow, production diagnostics |

Day 3 and Day 4 intentionally do not repeat Day 1 and Day 2 labs. They build on them with model preparation, production architecture, GenAI, PCIe, and customer-support playbooks.

## Pre-Training Setup

Host and access:

- Ubuntu 22.04 or 24.04 host.
- NEAT SDK container installed.
- Modalix DevKit access for Days 1-2.
- Modalix SoM DevKit, Modalix EA DevKit, or PCIe HHHL access for Day 3 or Day 4 when available.
- DevKit shared workspace mounted or synchronized.
- Access to `/workspace/core`, `/workspace/apps`, and `/workspace/demo-neat`.

Models and data:

- A compiled image-classification model for first-run exercises.
- YOLOv11 model source and one already compiled YOLO artifact for app exercises.
- A YOLO26 artifact or example config for comparison with existing app examples.
- Representative calibration images for INT8 quantization.
- Sample videos or cameras for two RTSP streams.
- A supported VLM model package or a running OpenAI-compatible VLM server for multimodal labs.

Recommended instructor preparation:

- Verify `python -c "import neat"` works inside the target environment.
- Verify `dk /workspace/...` can run a small NEAT script on the DevKit.
- Verify one model-only run with `core/tutorials/001_run_first_model`.
- Verify one graph pipeline from `core/tutorials/004_build_inference_pipeline`.
- Verify one app example from `/workspace/apps/examples/object-detection`.
- Verify one GenAI or VLM example when Day 2 or Day 4 includes a live VLM lab.

## Day 1: NEAT Foundations And First YOLO Application

Primary audience: Korea team.

Main theme: understand the NEAT application surface and build confidence with a first Modalix application.

### Session 1: NEAT Platform Orientation

Topics:

- Palette Neat development path: prepare model, build app, validate on hardware.
- SDK container, DevKit runtime, DevKit Sync, and host-vs-board responsibilities.
- Modalix DevKit workflow on Ubuntu.
- Where to find source-of-truth information: official docs, core headers, core tutorials, and apps examples.
- When to use PyNeat vs C++.

Hands-on:

- Inspect `/workspace/core/tutorials`.
- Run a first model-only example.
- Confirm environment, board connectivity, and artifact paths.

Reference material:

- Official Getting Started documentation.
- Official Neat SDK documentation.
- `core/tutorials/001_run_first_model`
- `core/tutorials/003_benchmark_model`
- `/workspace/apps/examples/README.md`

### Session 2: Core NEAT APIs

Topics:

- `Model`: load a compiled `.tar.gz` model artifact.
- `Tensor`: input and output buffer contract.
- `Sample`: payload plus metadata for pipeline edges.
- `Graph`: multi-stage app composition.
- `Run`: executable graph instance.
- `ModelOptions`, `RunOptions`, `InputOptions`, `OutputOptions`, and model-route configuration at a conceptual level.
- Named endpoints for multi-input and multi-output apps.

Hands-on:

- Run a model directly.
- Build a simple graph.
- Print model input/output specs.
- Compare model-only execution with graph execution.

Reference material:

- `core/tutorials/004_build_inference_pipeline`
- `core/tutorials/005_configure_model_options`
- `core/tutorials/009_pass_numpy_to_model`
- `core/tutorials/010_feed_multi_input_model`
- `core/include/model/Model.h`
- `core/include/pipeline/Graph.h`
- `core/include/pipeline/Run.h`

### Session 3: YOLO Detection Application Basics

Topics:

- YOLO app shape in NEAT: input, preprocessing, model, decode, postprocess, annotated output.
- Difference between model tensor output and decoded detection output.
- Bounding-box format and score/class filtering.
- YOLOv11 as the Korea-team reference model.
- YOLO26 app examples as a local reference pattern when useful.

Hands-on:

- Run an image-folder detector.
- Inspect detection outputs and annotated images.
- Change confidence threshold and NMS threshold.
- Compare raw model output with decoded boxes.

Reference material:

- `core/tutorials/006_preprocess_images`
- `core/tutorials/007_read_detection_boxes`
- `core/tutorials/011_interpret_output`
- `/workspace/apps/examples/object-detection/yolo26-object-detector`

### Session 4: Agentic Development Workflow

Topics:

- How to use Codex or Claude against the NEAT repos without inventing APIs.
- Prompting pattern: ask the agent to read official docs, core headers, tutorials, and app examples first.
- Asking for small runnable changes, then asking for validation.
- Keeping customer work reproducible: commands, logs, configs, model artifacts, and sample inputs.
- How to review agent output: API names, artifact paths, board commands, and runtime logs.

Hands-on:

- Use an agent to locate the right tutorial for a requested pipeline change.
- Ask the agent to modify a threshold/config path.
- Ask the agent to produce a short debug report from a failed run.

Day 1 deliverable:

- A working image-based YOLO detector on Modalix DevKit, with a short note explaining the APIs used and where the code came from.

## Day 2: Two-Camera Pipelines, Runtime Tuning, And Detection + VLM

Primary audience: Korea team.

Main theme: build the practical customer demo: two-camera YOLO pipeline with optional VLM integration.

### Session 1: Video And RTSP Inputs

Topics:

- RTSP decoded input pipeline.
- Camera and stream identity with metadata.
- Input sample contracts: image/tensor payload plus `stream_id`, `frame_id`, and timestamps.
- NV12/RGB preprocessing expectations.
- When to use RTSP input, camera input, still image input, or custom input nodes.

Hands-on:

- Start two RTSP streams from sample video files.
- Confirm both streams can be consumed.
- Print stream/frame metadata from incoming samples.

Reference material:

- `core/tutorials/018_consume_rtsp`
- `core/tutorials/023_run_mipi_camera_model`
- `core/docs/advanced-concepts/mipi-camera-input.md`
- `core/include/nodes/io/RTSPInput.h`
- `core/include/nodes/io/CameraInput.h`

### Session 2: Two-Camera YOLO Pipeline

Topics:

- Building a graph with two named camera inputs.
- Routing two streams through one model stage or through separate branches.
- Shared model execution vs independent model routes.
- Metadata and output naming.
- Insight video and metadata output concepts.

Hands-on:

- Build or modify a two-camera YOLO pipeline.
- Run both streams through YOLOv11.
- Save annotated frames or send output to a viewer.
- Confirm detections preserve stream identity.

Reference material:

- `core/tutorials/015_run_multiple_streams`
- `core/tutorials/017_production_pipeline`
- `/workspace/apps/examples/object-detection/multi-stream-object-detector`
- `/workspace/apps/examples/tracking/multi-stream-people-tracker`

### Session 3: Queue Tuning And Runtime Behavior

Topics:

- `RunOptions` presets: realtime, balanced, and reliable behavior.
- `queue_depth` and backpressure.
- Overflow policies: block, drop incoming, keep latest.
- `input_timeout_ms` and behavior when inputs do not arrive in time.
- Drop callbacks and how to log frame loss.
- Output memory and performance considerations.

Hands-on:

- Run the two-camera pipeline under different queue settings.
- Simulate slow postprocess or blocked output.
- Capture drop events.
- Compare latency and throughput behavior.

Reference material:

- `core/tutorials/016_tune_throughput_and_queues`
- `core/tutorials/004_build_inference_pipeline`
- `core/include/pipeline/Run.h`

### Session 4: Detection + VLM Workflow

Topics:

- Why detection plus VLM is useful in real apps.
- Direct VLM call vs detection-triggered VLM.
- Crop selection from detections.
- Prompt design for VLM scene/object questions.
- When to use `GenAIModel`, `VisionLanguageModel`, `GenAIServer`, or an OpenAI-compatible server.

Hands-on:

- Run a VLM example on an image.
- Use detected crop or selected frame as VLM input.
- Produce a short natural-language description from a detected object or scene.

Reference material:

- `core/tutorials/020_run_vlm`
- `core/tutorials/021_serve_genai`
- `core/tutorials/022_compose_genai_into_graph`
- `/workspace/apps/examples/genai/detection-to-vlm-assistant`
- `/workspace/apps/examples/genai/multimodal-assistant`

Day 2 deliverable:

- A two-camera YOLO pipeline with runtime queue settings documented, plus an optional detection-to-VLM path.

## Day 3: Model Preparation, Unknown-Model Triage, And PCIe Workflow

Primary audience: all teams.

Main theme: support customer models and platforms beyond the happy path.

### Session 1: Model Compiler Workflow

Topics:

- Model Compiler role in the NEAT stack.
- ONNX model preparation flow: compatibility check, graph preparation, quantization, validation, compilation.
- INT8 as the default deployment target for many CNN/YOLO models.
- BF16 as a fallback or accuracy-oriented format when INT8 is not ready or not appropriate.
- Representative calibration data and why it matters.
- Generated `.tar.gz` artifact structure and what to inspect.
- Requirement for deployment package sanity: expected ELF payload, avoid unexpected `.so` dependencies when targeting that packaging goal.

Hands-on:

- Compile or inspect a known CNN model.
- Quantize with a small representative image set.
- Inspect the generated package.
- Run a model-only validation.

Reference material:

- Official Compile a Model documentation.
- Official Compile Your First Model documentation.
- Official Quantization documentation.
- Official Model Compilation documentation.
- `core/tutorials/003_benchmark_model`
- `/workspace/apps/examples/model-benchmark`

### Session 2: Graph Surgery And Unsupported Model Triage

Topics:

- What graph surgery means in practice: shape fixes, unsupported op replacement, output-head extraction, postprocess removal, and static-shape cleanup.
- Common model issues: dynamic shapes, unsupported operators, framework-export artifacts, custom decode/NMS layers, and layout mismatches.
- YOLO-specific issues: raw heads vs decoded boxes, model output contract, and where decode should live.
- How to produce a useful support report: original model, export command, ONNX checker result, compiler log, unsupported-op list, sample inputs, expected outputs.

Decision guide:

| Model Type | First Path | Continue If | Stop Or Escalate If |
| --- | --- | --- | --- |
| CNN classification/segmentation/detection | ONNX Model Compiler | ops and static shapes are compatible or fixable | unsupported custom ops dominate the graph |
| YOLO detection | export to ONNX, graph surgery as needed, compile, decode in app or supported stage | heads can be exposed cleanly and decode contract is understood | export includes opaque postprocess/custom plugin ops |
| Vision Transformer or hybrid CNN/Transformer | ONNX compatibility audit first | attention and tensor ops map to supported compiler capabilities | architecture depends on unsupported dynamic attention patterns |
| LLM/VLM/ASR | LLiMa path | architecture is listed as supported and model size/format constraints are met | unsupported architecture, unsupported VLM format, or too-large model |
| Customer-unknown model | triage checklist | minimal reproducible compile issue can be produced | no model provenance, no sample input, no expected output |

Hands-on:

- Take one unknown or intentionally broken ONNX model.
- Run compatibility checks.
- Identify whether it is compiler-ready, surgery-ready, LLiMa-ready, or should be escalated.
- Write a one-page triage report.

Reference material:

- Official Model Compatibility documentation.
- Official Graph Surgery documentation.
- `core/docs/reference/troubleshooting.md`
- `core/docs/reference/error_codes.md`
- `/workspace/core/include/model`

### Session 3: Conventional Workflow Without Agents

Topics:

- Manual source inspection: headers, tutorials, docs, and examples.
- Building Python apps.
- Building C++ apps for the target runtime.
- Using config files for repeatable apps.
- Capturing logs and command history.
- Producing customer-friendly reproduction packages.

Hands-on:

- Run a known app example manually.
- Modify config by hand.
- Re-run and capture output.
- Produce a minimal reproduction note.

Reference material:

- Official Development Workflow documentation.
- Official Develop Apps documentation.
- `core/tutorials/012_diagnose_pipeline`
- `core/docs/development-workflow`

### Session 4: PCIe HHHL And Platform Variants

Topics:

- Modalix DevKit vs Modalix SoM DevKit vs Modalix EA DevKit vs PCIe HHHL.
- Where application code runs and where inference runs.
- PCIe tensor source/sink concepts.
- Host application responsibilities for PCIe-style deployment.
- When PCIe is the right customer architecture.

Hands-on options:

- If PCIe hardware is available: run a minimal PCIe tensor path or inspect an existing PCIe sample.
- If hardware is unavailable: review the PCIe node interfaces and produce an architecture diagram for a host-to-card inference flow.

Reference material:

- `core/include/nodes/sima/PCIeSink.h`
- `core/include/nodes/sima/PCIeSrc.h`
- `core/docs/advanced-concepts/graphs.md`
- `core/docs/model-runtime/mpk_contract.md`

Day 3 deliverable:

- A model triage report plus either a compiled model package inspection or a PCIe workflow note.

## Day 4: Production Apps, GenAI, Diagnostics, And Capstone

Primary audience: all teams.

Main theme: turn working pieces into supportable customer applications.

### Session 1: Production Graph Design

Topics:

- Choosing between direct `Model.run`, built model runner, and `Graph`.
- Graph naming and endpoint conventions.
- Multi-input and multi-output app design.
- Metadata sender, video sender, UDP output, still image output, and custom nodes.
- Error boundaries and observability.
- Packaging configs and sample data with the app.

Hands-on:

- Convert a small model-only script into a graph-style app.
- Add named inputs and outputs.
- Add metadata or image output.
- Validate graph structure before runtime.

Reference material:

- Official Advanced Concepts documentation.
- `core/tutorials/008_plug_model_into_pipeline`
- `core/tutorials/013_custom_data_graph`
- `core/tutorials/014_embed_model_inside_graph`
- `core/tutorials/017_production_pipeline`
- `core/include/nodes/io`
- `core/include/nodes/groups`

### Session 2: GenAI And LLiMa

Topics:

- LLiMa role for LLM, VLM, and ASR on Modalix.
- Supported model-family thinking: architecture, parameter size, model format, tokenizer/config requirements.
- `GenAIModel`, `VisionLanguageModel`, `ASRModel`, and `GenAIServer`.
- Direct GenAI calls vs OpenAI-compatible server.
- Combining GenAI with classic vision pipelines.
- Practical limitations: unsupported architectures, model size, VLM source format, and deployment memory.

Hands-on:

- Run or inspect a VLM example.
- Run or inspect a GenAI server example.
- Connect a detection event to a GenAI/VLM query path.

Reference material:

- Official GenAI with LLiMa documentation.
- `core/tutorials/019_run_llm`
- `core/tutorials/020_run_vlm`
- `core/tutorials/021_serve_genai`
- `core/tutorials/022_compose_genai_into_graph`
- `/workspace/apps/examples/genai/multimodal-assistant`
- `/workspace/apps/examples/genai/detection-to-vlm-assistant`

### Session 3: Diagnostics, Performance, And Support Playbooks

Topics:

- `Graph::validate` and validation before runtime.
- `Run::last_error` and structured runtime errors.
- Neat error categories and diagnostics.
- Input caps, tensor shape, dtype, layout, and memory placement.
- Queue depth, overflow policy, input timeout, dropped frames, and latency.
- Model benchmark vs end-to-end app performance.
- Power, FPS, latency, and customer-facing evidence.

Hands-on:

- Debug one broken pipeline.
- Tune queue settings for realtime vs reliable behavior.
- Benchmark a model and compare with app-level throughput.
- Produce a short customer support response from the evidence.

Reference material:

- `core/tutorials/012_diagnose_pipeline`
- `core/tutorials/016_tune_throughput_and_queues`
- `core/docs/reference/diagnostics.md`
- `core/docs/reference/env_vars.md`
- `/workspace/apps/examples/model-benchmark`

### Session 4: Capstone Lab

Each team chooses one capstone:

- Multi-stream YOLO detector with video and metadata output.
- Detection-to-VLM assistant using selected frames or crops.
- Unknown-model triage report with compile recommendation.
- PCIe host/card inference architecture or live PCIe sample.
- Production app hardening pass for an existing customer demo.

Expected capstone output:

- Runnable code or reviewed example path.
- Config file.
- Sample input.
- Output image, output stream, log, or benchmark report.
- Short explanation of APIs used.
- Known limitations and next steps.

Day 4 deliverable:

- A supportable mini-project that can be reused as a customer-facing reference.

## Recommended Labs By Topic

| Topic | Primary Lab Source |
| --- | --- |
| First model run | `core/tutorials/001_run_first_model` |
| Async/model benchmarking | `core/tutorials/002_async_inference`, `core/tutorials/003_benchmark_model`, `/workspace/apps/examples/model-benchmark` |
| Graph basics | `core/tutorials/004_build_inference_pipeline` |
| Model options | `core/tutorials/005_configure_model_options` |
| Preprocessing | `core/tutorials/006_preprocess_images` |
| Detection boxes | `core/tutorials/007_read_detection_boxes` |
| Model inside graph | `core/tutorials/008_plug_model_into_pipeline`, `core/tutorials/014_embed_model_inside_graph` |
| NumPy and multi-input | `core/tutorials/009_pass_numpy_to_model`, `core/tutorials/010_feed_multi_input_model` |
| Diagnostics | `core/tutorials/012_diagnose_pipeline` |
| Custom graphs | `core/tutorials/013_custom_data_graph` |
| Multi-stream | `core/tutorials/015_run_multiple_streams` |
| Queues/runtime tuning | `core/tutorials/016_tune_throughput_and_queues` |
| Production pipeline | `core/tutorials/017_production_pipeline` |
| RTSP | `core/tutorials/018_consume_rtsp` |
| LLM/VLM/GenAI server | `core/tutorials/019_run_llm`, `020_run_vlm`, `021_serve_genai`, `022_compose_genai_into_graph` |
| MIPI camera | `core/tutorials/023_run_mipi_camera_model` |
| YOLO app | `/workspace/apps/examples/object-detection/yolo26-object-detector` |
| Multi-stream detection | `/workspace/apps/examples/object-detection/multi-stream-object-detector` |
| Detection + VLM | `/workspace/apps/examples/genai/detection-to-vlm-assistant` |
| Full multimodal assistant | `/workspace/apps/examples/genai/multimodal-assistant` |

## API Coverage Checklist

Core application APIs:

- `Model`
- `ModelOptions`
- `Tensor`
- `Sample`
- `Graph`
- `Run`
- `RunOptions`
- `InputOptions`
- `OutputOptions`
- model route options
- detection/box decode helpers

Input and output nodes:

- still image input
- RTSP input
- camera/MIPI input
- custom input
- UDP output
- video sender
- metadata sender
- PCIe source/sink

GenAI APIs:

- `GenAIModel`
- `VisionLanguageModel`
- `ASRModel`
- `GenAIServer`
- OpenAI-compatible server examples

Compiler and deployment concepts:

- ONNX compatibility
- graph surgery
- calibration data
- INT8 quantization
- BF16 compilation
- compiled `.tar.gz` model package
- package inspection
- model benchmark
- end-to-end app validation

## Customer Support Decision Playbooks

### A Customer Wants YOLO On Modalix

1. Confirm model version, source framework, input size, class count, and expected postprocess.
2. Export to ONNX with static shape when possible.
3. Remove or bypass unsupported postprocess layers if needed.
4. Compile INT8 with representative images.
5. Keep decode/postprocess in the NEAT app when that is cleaner than embedding it in the model.
6. Validate on still images.
7. Move to RTSP or camera pipeline.
8. Add queue tuning and metadata output.

### A Customer Has An Unknown CNN Model

1. Request model file, export command, sample input, expected output, and accuracy target.
2. Run ONNX checker and compiler compatibility.
3. Identify unsupported ops and dynamic dimensions.
4. Decide whether graph surgery is small, large, or not practical.
5. Quantize with representative data.
6. Validate accuracy before app integration.
7. Document the exact blocker when escalation is needed.

### A Customer Wants VLM Or GenAI

1. Identify whether it is LLM, VLM, ASR, or a hybrid application.
2. Check whether the architecture and model format are supported by LLiMa.
3. Confirm model size and memory constraints.
4. Choose direct `GenAIModel`/`VisionLanguageModel` for app integration or `GenAIServer` for service-style integration.
5. For detection + VLM, keep detection as a classic NEAT pipeline and trigger VLM only on selected frames or crops.

### A Customer Wants Multi-Stream

1. Start with two streams before scaling up.
2. Use clear stream names and metadata.
3. Decide whether streams share a model stage or use independent branches.
4. Tune `queue_depth`, overflow policy, and input timeout based on realtime vs reliability goals.
5. Measure end-to-end latency and dropped frames, not only model FPS.
6. Use Insight or equivalent outputs for visual validation.

### A Customer Wants PCIe

1. Confirm where preprocessing, inference, postprocess, and display will run.
2. Identify tensor transfer boundaries.
3. Use PCIe source/sink APIs or existing PCIe examples as the starting point.
4. Benchmark host-card transfer plus inference, not only model execution.
5. Package a minimal reproducible host/card flow for support.

## Instructor Notes

- Keep Day 1 and Day 2 concrete and demo-driven; Korea’s immediate needs are YOLOv11, two-camera video, VLM, and agentic development.
- Keep Day 3 and Day 4 broader and support-oriented; Japan/general needs include varied customer CNNs, model triage, GenAI popularity, PCIe, and conventional workflow.
- Do not spend Day 3 repeating RTSP basics or first model-run basics. Assume participants have seen them or provide only a five-minute bridge.
- When teaching agentic workflow, require the agent to cite exact local files or official docs before editing code.
- When teaching conventional workflow, require attendees to run and modify examples manually so they can support customers who do not use agents.
- For every lab, capture: command, config, artifact path, input data path, output evidence, and failure log if applicable.

## Source Map

Official documentation:

- Getting Started: https://developer.sima.ai/software/getting-started/
- Neat SDK: https://developer.sima.ai/software/getting-started/dev-environment/
- Neat Library: https://developer.sima.ai/software/getting-started/neat-library/
- Compile a Model: https://developer.sima.ai/software/compile-a-model/
- Compile Your First Model: https://developer.sima.ai/software/compile-a-model/compile-your-first-model
- Model Compatibility: https://developer.sima.ai/software/compile-a-model/model-compatibility
- Graph Surgery: https://developer.sima.ai/software/compile-a-model/graph-surgery
- Quantization: https://developer.sima.ai/software/compile-a-model/quantization
- Model Compilation: https://developer.sima.ai/software/compile-a-model/model-compilation
- Validate Accuracy and Performance: https://developer.sima.ai/software/compile-a-model/validate-accuracy-performance
- Develop Apps: https://developer.sima.ai/software/develop-apps/
- Development Workflow: https://developer.sima.ai/software/develop-apps/development-workflow/
- Advanced Concepts: https://developer.sima.ai/software/develop-apps/advanced-concepts/
- Tutorials: https://developer.sima.ai/software/tutorials
- GenAI with LLiMa: https://developer.sima.ai/software/genai-llima/

Local source references:

- `/workspace/core/tutorials`
- `/workspace/core/docs`
- `/workspace/core/include`
- `/workspace/core/python`
- `/workspace/apps/examples`
- `/workspace/apps/examples/object-detection`
- `/workspace/apps/examples/genai`
- `/workspace/apps/examples/model-benchmark`

---

## Alignment With This Repo (added 2026-07-09)

The syllabus above is written against `/workspace/core` and `/workspace/apps/examples`. This
`demo-neat` repo now also contains **concrete, self-contained apps and tracks** built to realize
several of these sessions. Use this mapping to point trainees at runnable in-repo material. Nothing
above is removed; this section only adds the cross-links and corrects references that do not resolve.

### Sessions → in-repo material that now exists

| Day / Session | Realized by (in this repo) | Validation status |
| --- | --- | --- |
| Day 1 / S3 — YOLO detection basics | [`apps/single-stream-yolo-yolo11`](../apps/single-stream-yolo-yolo11/README.md) | live-validated |
| Day 2 / S1–S2 — RTSP + two-camera YOLO | [`apps/multi-stream-yolo-yolo11`](../apps/multi-stream-yolo-yolo11/README.md) (2× RTSP, one shared YOLO11 stage) | live-validated (~37 fps agg) |
| Day 2 / S4 — Detection + VLM | [`apps/detection-vlm-assistant`](../apps/detection-vlm-assistant/README.md) + [`llima/03-yolo-plus-vlm`](../llima/03-yolo-plus-vlm/01_detection_to_vlm.ipynb) | detection leg live; **VLM leg NOT executed** |
| Day 3 / S1–S2 — Model Compiler + graph surgery | [`model-compilation/`](../model-compilation/README.md) (YOLO surgery walkthrough + transformer models in `work/`) | YOLO chain live-validated; INT8 compiles verified |
| Day 4 / S1 — Production graph design | [`apps/quad-stream-quad-model`](../apps/quad-stream-quad-model/README.md) + [`TEACHING.md`](../apps/quad-stream-quad-model/TEACHING.md) (4 streams × 4 models) | live-validated (~1.7 fps agg; A65 host-decode bound) |
| Day 4 / S2 — GenAI + LLiMa | [`llima/`](../llima/README.md) — 01 basics, 02 run LLM/VLM/ASR, 04 compilation, 05 GenAI server | CLI probes live; **all model execution + compilation NOT executed** |

> **Honesty carry-over (matches the root README):** all LLM/VLM/ASR execution and all GenAI
> compilation in `llima/` are **documented for manual runs, not executed** by the material's authors.
> The `llima/04-llm-vlm-compilation` notebooks are **docs-derived** (the real tool is a host-side
> `llima-compile`; there is no `llima compile` subcommand). See the repo root
> [`README.md` → Verified vs documented-but-unrun](../README.md#verified-vs-documented-but-unrun).

### Reference corrections (paths in this doc that do not resolve as written)

Verified on 2026-07-09; the referenced concepts are real, but the exact paths differ:

- **`/workspace/apps/examples/model-benchmark` does not exist.** The actual example directory is
  **`/workspace/apps/examples/benchmarking`**. (Cited in Day 3 S1, Day 4 S3, and Recommended Labs.)
- **Several core-tutorial short names in the Reference/Source-Map lists are abbreviated and do not
  match the real directory names.** The real names carry extra words, e.g.
  `001_run_first_model` → `001_run_your_first_model`, `002_async_inference` → `002_run_inference_async`,
  `003_benchmark_model` → `003_benchmark_your_model`, `011_interpret_output` → `011_interpret_model_output`,
  `012_diagnose_pipeline` → `012_diagnose_a_pipeline`, `013_custom_data_graph` → `013_build_a_custom_data_graph`,
  `017_production_pipeline` → `017_build_production_pipeline`, `018_consume_rtsp` → `018_consume_rtsp_stream`,
  `019_run_llm` → `019_run_an_llm`, `020_run_vlm` → `020_run_a_vlm`, `021_serve_genai` → `021_serve_genai_models`.
  When directing trainees to a tutorial, confirm the full name in `/workspace/core/tutorials/`.
- `/workspace/apps/examples/tracking/multi-stream-people-tracker`,
  `/workspace/apps/examples/genai/detection-to-vlm-assistant`, and
  `/workspace/apps/examples/genai/multimodal-assistant` **do** exist and are cited correctly.
