# Model Compilation Project Plan

## Goal

Compile a set of open/source-available vision models into SiMa deployable model archives:

- final artifact is a `.tar.gz`
- archive contains exactly one compiled ELF model artifact
- archive contains no `.so` files
- artifact can be loaded by Neat/PyNeat
- artifact can run on representative sample images to verify input/output behavior

The work covers model acquisition, ONNX export, graph surgery, quantization, compilation, packaging validation, and Neat runtime smoke tests.

## Target Models

### Classification And Feature Models

| Family | Initial Variant | Upstream Source | Input | Verification Output |
| --- | --- | --- | --- | --- |
| ResNet-50 | `resnet50` | TorchVision models | `1x3x224x224` RGB/BGR normalized image | ImageNet top-5 |
| ConvNeXt | `convnext_tiny` first, then larger | TorchVision models | `1x3x224x224` | ImageNet top-5 |
| EfficientNetV2 | `efficientnet_v2_s` first | TorchVision models | `1x3x384x384` or static exported size | ImageNet top-5 |
| DenseNet-169 | `densenet169` | TorchVision models | `1x3x224x224` | ImageNet top-5 |
| ViT | `vit_b_16` first | TorchVision models | `1x3x224x224` | ImageNet top-5 |
| DINOv2 | `dinov2_vits14` first | Meta DINOv2 | `1x3x224x224` or `1x3x518x518` depending export | Embedding shape and nearest-label sanity check |
| FastViT | `fastvit_t8` or smallest available | Apple FastViT | model-specific static image size | ImageNet top-5 |
| MaxViT | `maxvit_t` | TorchVision models | model-specific static image size | ImageNet top-5 |

### Detection Models

| Family | Initial Variant | Upstream Source | Input | Verification Output |
| --- | --- | --- | --- | --- |
| YOLO11 | `yolo11n` first | Ultralytics | `1x3x640x640` | COCO boxes/classes on sample image |
| YOLO26 | `yolo26n` first | Ultralytics | `1x3x640x640` | COCO boxes/classes on sample image |
| DETR | `detr-resnet50` baseline | facebookresearch/detr or Torch/HF export path | static image tensor, likely `1x3x800x800` or chosen fixed size | COCO boxes/classes after postprocess |

## Source Policy

Use upstream sources that are public and reproducible. Each model entry must record:

- source repository or package
- exact commit, package version, or model weight tag
- license
- export command
- ONNX opset
- input names/shapes
- output names/shapes
- preprocessing contract
- postprocessing contract

Important license gate: do not redistribute weights or compiled artifacts until the model license is reviewed. Some public model sources are source-available or copyleft rather than permissive.

## Proposed Repository Layout

```text
model-compilation/
  PROJECT_PLAN.md
  README.md                         # quick project entry point, created later
  models.yaml                       # model registry, created in Phase 1
  scripts/
    00_check_env.sh
    01_download_or_export.py
    02_audit_onnx.py
    03_surgery.py
    04_quantize_compile.py
    05_validate_archive.py
    06_neat_smoke_test.py
  assets/
    calibration/
    sample_images/
    labels/
  work/
    <model_id>/
      source/
      onnx/
      surgery/
      compile/
      package/
      reports/
  results/
    summary.csv
    summary.md
```

Keep generated model files under `work/` and final reports under `results/`. Do not commit large downloaded weights, ONNX files, compiled artifacts, or calibration datasets unless explicitly required.

## Required Gates

Every model must pass these gates before being marked done.

| Gate | Requirement | Evidence |
| --- | --- | --- |
| G0 Source lock | Source URL, version, license, and checksum recorded | `work/<model>/reports/source.md` |
| G1 ONNX export | Static-shape ONNX produced and passes `onnx.checker` | `export.log`, `onnx_check.log` |
| G2 Operator audit | SiMa support audit run before surgery | `audit_before.json` |
| G3 Surgery | Minimal graph edits applied only if required | `surgery_notes.md`, `audit_after.json` |
| G4 Quantize | INT8 quantization completes with real or documented calibration data | `quantize.log` |
| G5 Compile | Compiler emits target artifacts | `compile.log` |
| G6 Package contract | `.tar.gz` contains exactly one ELF and zero `.so` files | `archive_manifest.txt` |
| G7 Neat load | `pyneat.Model(<archive>)` loads | `neat_load.log` |
| G8 Runtime smoke | Sample image inference succeeds | `smoke_test.json` |
| G9 Baseline compare | Output shape and rough behavior match PyTorch/ONNX baseline | `compare.json` |

## Workflow Per Model

1. **Register**
   - Add model to `models.yaml`.
   - Define `model_id`, task, source, license, expected input, expected output, sample image, and calibration settings.

2. **Acquire Or Export**
   - Prefer direct PyTorch export when source implementation is reliable.
   - Prefer ONNX export through official tools when provided by upstream.
   - Freeze dynamic dimensions during export whenever possible.

3. **Normalize ONNX**
   - Run ONNX checker.
   - Run shape inference.
   - Simplify if safe.
   - Staticify symbolic dimensions with a repeatable mapping.

4. **Audit**
   - Run `model_surgery_guard.py audit-model --dtype int8`.
   - On Modalix, also run `--dtype bfloat16` as information only unless the model is intentionally BF16.
   - Triage unsupported ops before attempting compile.

5. **Graph Surgery**
   - Prefer source-level changes and re-export.
   - Use ONNX surgery only for local, well-understood fixes.
   - Preserve output tensor names and shape contracts.
   - Re-run audit and ONNX checker after every surgery.

6. **Quantize**
   - Use real calibration images whenever possible.
   - Start with 50 calibration images per family.
   - Record calibration method and preprocessing exactly.

7. **Compile**
   - Compile for the active target, default `modalix` unless changed.
   - Capture full compiler logs.
   - Fail the gate if compile relies on emitted `.so` plugins for this project goal.

8. **Package Validation**
   - Inspect the `.tar.gz`.
   - Confirm one ELF-like compiled artifact.
   - Confirm no `.so`.
   - Confirm expected manifest files are present.

9. **Neat Runtime Smoke Test**
   - Load with `pyneat.Model`.
   - Run one or more sample images.
   - Validate output tensor count, shapes, dtype, and basic semantics.
   - For classification, print top-5.
   - For detection, draw and save boxes plus JSON detections.

10. **Report**
   - Write per-model summary.
   - Update `results/summary.csv` and `results/summary.md`.

## Classification Track

Start with models that are likely to compile cleanly and establish the repeatable pipeline:

1. ResNet-50
2. DenseNet-169
3. ConvNeXt Tiny
4. EfficientNetV2 Small
5. ViT-B/16
6. MaxViT Tiny
7. FastViT smallest variant
8. DINOv2 small variant

Expected risk:

- CNNs are the baseline path.
- ConvNeXt/EfficientNet/MaxViT may require activation or reshape audits.
- ViT/DINOv2 may require attention/reshape/transpose surgery and static sequence length.
- DINOv2 verification may be feature-shape based unless a classifier head is attached.

## Detection Track

Start with smallest variants and keep postprocessing explicit.

1. YOLO11n
2. YOLO26n
3. DETR ResNet-50

Expected risk:

- YOLO exports often include postprocess/NMS choices that affect compatibility. Prefer exporting raw heads first, then use Neat/model-side or Python postprocess if compile compatibility is better.
- YOLO graph surgery may be required around head reshapes, concat layout, sigmoid/grid decode, or NMS.
- DETR may require a hand-wired or staged path similar to `Input -> QuantTess -> MLA -> DetessDequant -> Output`, with Python-side postprocess for boxes/classes.

## Calibration Data

Use one shared calibration image set first:

- 50 to 200 ImageNet-like images for classification
- 50 to 200 COCO-like images for detection

Rules:

- Store only small sample/calibration references unless redistribution is approved.
- Record image preprocessing exactly.
- Keep calibration preprocessing identical to export/runtime preprocessing.

## Validation Scripts To Build

### Archive Validator

Checks:

- file is `.tar.gz`
- extractable
- contains exactly one ELF candidate
- contains no `.so`
- manifest/config files are readable
- output is written to `archive_manifest.txt`

### Neat Smoke Tester

Inputs:

- model archive
- task type
- sample image
- labels file
- preprocessing config

Outputs:

- `smoke_test.json`
- optional annotated image for detection
- top-k text for classification

### Summary Reporter

Columns:

- model_id
- source
- license
- opset
- input_shape
- export_status
- audit_status
- surgery_status
- quantize_status
- compile_status
- archive_contract_status
- neat_load_status
- smoke_status
- notes

## Milestones

### M0: Environment And Scaffolding

- Add `models.yaml`.
- Add env check script.
- Add archive validator.
- Add smoke-test skeleton.
- Confirm `activate-model-compiler` works.
- Confirm `sima-frontend` and ONNX tools are installed.

### M1: Golden Path With ResNet-50

- Export/download ResNet-50.
- Audit, quantize, compile.
- Validate `.tar.gz` single-ELF/no-`.so` contract.
- Load and run with PyNeat.
- Use this as the template for all classification models.

### M2: Complete CNN/Hybrid Classification

- DenseNet-169
- ConvNeXt
- EfficientNetV2
- MaxViT
- FastViT

### M3: Transformer Classification/Feature Models

- ViT
- DINOv2

### M4: YOLO Detection

- YOLO11
- YOLO26
- raw-head export first
- document whether Neat-managed decode, Python decode, or compiled decode is used

### M5: DETR Detection

- Export/staticify DETR.
- Decide whether full graph or staged graph is feasible.
- Validate Python-side postprocess path.

### M6: Final Report

- Summarize pass/fail by model.
- List surgery patterns.
- List unsupported blockers.
- List models that meet the exact single-ELF/no-`.so` archive requirement.

## Initial Priority Order

1. ResNet-50
2. DenseNet-169
3. ConvNeXt Tiny
4. EfficientNetV2 Small
5. YOLO11n
6. YOLO26n
7. ViT-B/16
8. DETR ResNet-50
9. MaxViT Tiny
10. FastViT
11. DINOv2 small

This order gets one clean classification path first, then one clean detection path, then higher-risk transformer-style models.

## Open Decisions

- Target device: assume `modalix` until changed.
- Quantization mode: start INT8; BF16 only if INT8 is blocked or model family requires it.
- Exact variant sizes: start smallest stable variant per family unless accuracy goals require larger.
- YOLO postprocess location: compiled graph, Neat decode, or Python decode.
- DETR postprocess location: compiled graph or Python decode.
- Redistribution policy for weights and compiled archives.

## References

- TorchVision model registry and weights: https://docs.pytorch.org/vision/main/models.html
- Meta DINOv2: https://github.com/facebookresearch/dinov2
- Apple FastViT: https://github.com/apple/ml-fastvit
- Ultralytics YOLO11: https://docs.ultralytics.com/models/yolo11/
- Ultralytics YOLO/ONNX export: https://docs.ultralytics.com/integrations/onnx/
- DETR official repository: https://github.com/facebookresearch/detr
