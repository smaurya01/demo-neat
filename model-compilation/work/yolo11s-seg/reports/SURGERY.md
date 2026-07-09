# yolo11s-seg — compile_ready surgery report (T5)

- Task: instance segmentation (COCO 80-class + 32 prototype masks)
- Source: Ultralytics `yolo11s-seg.pt` (exported opset 17, static 1x3x640x640)
- Precision: INT8, target `modalix`
- Surgery: `scripts/14_t5_compile_ready_surgery.py --model-id yolo11s-seg`
- Compile: `scripts/15_compile_t5_int8.py --model-id yolo11s-seg`

## What is different vs plain detection

A YOLO11 segmentation head is the **detection head plus two extra pieces**:

1. **Mask-coefficient branch `cv4`.** Alongside `cv2` (bbox) and `cv3` (class)
   there is a fourth per-scale branch `cv4.k.2` producing `[1,32,H,W]` — 32 mask
   coefficients per anchor. These select/weight the prototype masks.
2. **Prototype-mask branch `proto`.** A small FCN
   (`Conv -> SiLU -> ConvTranspose(stride2) -> Conv -> Conv`) that produces
   `output1 = [1,32,160,160]`: 32 shared prototype masks at 1/4 resolution.
   A detection's instance mask = sigmoid(sum_k coeff_k * proto_k) cropped to box.

## Surgery applied

- **Detection part:** identical to yolo11s — attention `/model.10/m/m.0/attn`
  MatMul->Einsum; DFL(16) -> 4 distance channels on `cv2.k.2`; expose class
  logits from `cv3.k.2`.
- **Mask coefficients:** expose `cv4.{0,1,2}/cv4.{0,1,2}.2/Conv_output_0` as
  `mask_coeff_{0,1,2}` `[1,32,H,W]` via Identity.
- **Prototypes:** the proto branch already terminates at graph `output1`. We
  re-expose it as a named output `proto` `[1,32,160,160]` (Identity) so it
  survives the graph-output replacement. **The whole proto FCN stays on the
  MLA** — its `ConvTranspose` (stride 2, group 1) is MLA-supported (support DB:
  ConvTranspose int8=Y, stride in [1,2,4,8,16] non-depthwise), so there is **no
  host fallback and no `.so`**.
- **Removed tail:** the detection decode/NMS and the mask-assembly matmul/crop
  (which are dynamic, per-detection, data-dependent) are cut; those run on host.

## Output contract (10 tensors)

| output | shape | meaning |
| --- | --- | --- |
| `bbox_0/1/2` | `[1,4,80/40/20,·]` | box distances (post-DFL) |
| `class_logit_0/1/2` | `[1,80,80/40/20,·]` | class logits |
| `mask_coeff_0/1/2` | `[1,32,80/40/20,·]` | 32 mask coefficients per anchor |
| `proto` | `[1,32,160,160]` | 32 shared prototype masks |

Host-side seg decode: box-decode + NMS on the detection tensors, gather the 32
coefficients of the surviving boxes, `mask = sigmoid(coeff @ proto)`, then crop
to box and threshold. (This is the standard Ultralytics YOLO-seg postprocess;
Neat does not yet expose a built-in seg decode type, so this is a host step —
documented here for the phase-2 pipeline app.)

## How to recognize / reproduce

- A seg head adds a `cv4.*` branch (32 channels) and a `proto` subgraph ending
  in a graph output shaped `[1,32,160,160]`. Detect it by: (a) two ONNX graph
  outputs (`output0` detection, `output1` proto), and (b) `cv4` conv names.
- The proto branch's `ConvTranspose` is the one op to check against the support
  DB — it is supported at stride 2, but a non-standard stride/group would force
  a fallback. Here it passes.

## Audit result

`reports/audit_compile_ready_int8.json`: **0 unsupported ops** for int8
(ConvTranspose count 1, listed supported).
