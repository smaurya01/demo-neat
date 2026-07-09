# YOLO Compile-Ready INT8 Progress

Date: 2026-07-08

## What Changed

- Deleted the generated YOLO11n/YOLO26n work artifacts and rebuilt both models from fresh Ultralytics `.pt` downloads.
- Copied exactly 20 real calibration images from `/workspace/calibration_images` into `assets/yolo_calibration`.
- Copied 5 real images for inference smoke tests into `assets/yolo_inference`.
- Replaced the old raw-head helper with `scripts/09_yolo_compile_ready_surgery.py`.
- Added `scripts/11_export_fresh_yolo.py` for repeatable `.pt` download and ONNX export.
- Added `scripts/12_compile_yolo_int8.py` for repeatable INT8 calibration/compile.
- Updated `scripts/10_run_yolo_sample_pipeline.py` to use the new INT8 archives.

## Naming

The old `raw_heads` name meant the graph exposed YOLO head tensors before final YOLO decode/postprocess. The new name is `compile_ready`: attention is rewritten for MLA, final decode/postprocess is removed from ONNX, and the graph exposes the six tensors expected by Neat `BoxDecodeType.YoloV26`.

## Output Contract

Both models expose:

```text
bbox_0         [1, 4, 80, 80]
bbox_1         [1, 4, 40, 40]
bbox_2         [1, 4, 20, 20]
class_logit_0  [1, 80, 80, 80]
class_logit_1  [1, 80, 40, 40]
class_logit_2  [1, 80, 20, 20]
```

## INT8 Compile Result

- `yolo11n`: pass
  - `work/yolo11n/compile_int8/yolo11n.compile_ready/yolo11n.compile_ready_mpk.tar.gz`
  - Archive validation: one `.elf`, zero `.so`
- `yolo26n`: pass
  - `work/yolo26n/compile_int8/yolo26n.compile_ready/yolo26n.compile_ready_mpk.tar.gz`
  - Archive validation: one `.elf`, zero `.so`

Both compile logs show one MLA segment and A65 plugin count `0`.

## DevKit Smoke Test

The sample pipeline is present and was attempted through `dk`:

```bash
dk /workspace/demo-neat/model-compilation/scripts/10_run_yolo_sample_pipeline.py \
  --model-id yolo26n \
  --output-dir /workspace/demo-neat/model-compilation/work/sample_runs
```

The model loaded and graph build started, but runtime failed in the EV74 quant stage with transport timeout code `110`. The failure is captured in:

```text
work/sample_runs/yolo26n_sample_run.json
```

This is a runtime transport issue on the DevKit path, not a compile/archive failure.
