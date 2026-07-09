# Run Log

Date: 2026-07-08

## Completed

- Created reusable pipeline scripts under `scripts/`.
- Created model registry at `models.yaml`.
- Generated synthetic sample/calibration images under `assets/`.
- Installed optional export dependencies in the model-compiler environment:
  - `timm`
  - `ultralytics`
- Exported ONNX and ran audit/surgery for:
  - `resnet50`
  - `densenet169`
  - `convnext_tiny`
  - `efficientnet_v2_s`
  - `vit_b_16`
  - `maxvit_t`
  - `fastvit_t8`
  - `dinov2_vits14`
  - `yolo11n`
  - `yolo26n`
  - `detr_resnet50`
- Added fixed-shape ONNX simplification and constant-index `Gather` to `Slice`/`Squeeze` surgery, including static negative-index handling.
- Cleared unsupported `Gather` from `vit_b_16`, `maxvit_t`, `dinov2_vits14`, `yolo11n`, and `detr_resnet50`.
- Quantized and compiled MPK archives for:
  - `resnet50`
  - `densenet169`
  - `convnext_tiny`
  - `efficientnet_v2_s`
  - `fastvit_t8`
- Validated each compiled MPK archive contains exactly one ELF and no `.so` files.
- Compiled `yolo11n`, but its MPK fails the archive contract because it contains multiple MLA ELF stages and A65 `.so` stages.
- Started `maxvit_t` quantize/compile; it produced `.sima` quantization artifacts, but the long compile was interrupted before MPK packaging.
- Exported DETR with a wrapper that returns `pred_logits` and `pred_boxes`; post-surgery audit is unsupported `0`, unknown `5`.
- Rebuilt `yolo11n` and `yolo26n` from fresh Ultralytics `.pt` downloads.
- Added compile-ready YOLO surgery for `yolo11n` and `yolo26n` so the graphs expose grouped `bbox_0..2` and `class_logit_0..2` outputs for `BoxDecodeType.YoloV26`.
- Reused the PCB YOLO26 attention rewrite pattern and applied the matching YOLO11 attention rewrite. YOLO11 also gets DFL-to-4-channel bbox conversion.
- Audited both compile-ready YOLO graphs with int8: unsupported `0`, unknown `0`.
- Quantized/calibrated and compiled `yolo11n` compile-ready INT8 with 20 real calibration images. Archive validation passed: one `.elf`, zero `.so`.
- Quantized/calibrated and compiled `yolo26n` compile-ready INT8 with 20 real calibration images. Archive validation passed: one `.elf`, zero `.so`.
- Updated `scripts/10_run_yolo_sample_pipeline.py` to run the compile-ready YOLO MPKs on 5 copied inference images through pyneat and Neat YOLO box decode.

## Compiled Archives

```text
work/resnet50/compile/resnet50.surgery/resnet50.surgery_mpk.tar.gz
work/densenet169/compile/densenet169.surgery/densenet169.surgery_mpk.tar.gz
work/convnext_tiny/compile/convnext_tiny.surgery/convnext_tiny.surgery_mpk.tar.gz
work/efficientnet_v2_s/compile/efficientnet_v2_s.surgery/efficientnet_v2_s.surgery_mpk.tar.gz
work/fastvit_t8/compile/fastvit_t8.surgery/fastvit_t8.surgery_mpk.tar.gz
work/yolo11n/compile_int8/yolo11n.compile_ready/yolo11n.compile_ready_mpk.tar.gz
work/yolo26n/compile_int8/yolo26n.compile_ready/yolo26n.compile_ready_mpk.tar.gz
```

## Pending

- DevKit/runtime smoke test was attempted through `dk`. The model loaded, but the board returned EV74 transport timeout code `110` in the quant stage. See `work/sample_runs/yolo26n_sample_run.json`.
- `vit_b_16`, `maxvit_t`, `dinov2_vits14`, and `detr_resnet50` now have unsupported `0` after surgery but still include transformer-style unknowns such as `LayerNormalization`, `Gemm`, `Squeeze`, and `Unsqueeze`.
- Compile-ready `yolo11n` and `yolo26n` both pass the requested single-ELF/no-`.so` archive contract.
- Compile-ready YOLO INT8 status is tracked in `results/yolo_compile_ready_int8.md`.
- Calibration images are synthetic placeholders. Replace them with representative ImageNet/COCO images before quality validation.

## Next Recommended Steps

1. Run the five passing MPKs on a DevKit/PyNeat environment with `scripts/06_neat_smoke_test.py`.
2. Add real calibration images and rerun quantization for the passing classification models.
3. Resume/finish `maxvit_t` compile from the saved command in `work/maxvit_t/reports/quantize_compile.command.txt`.
4. Resolve the DevKit EV74 transport timeout seen while running the YOLO sample pipeline.
5. Continue transformer compile experiments for `vit_b_16`, `dinov2_vits14`, and `detr_resnet50`.
