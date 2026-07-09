# yolo11s — compile_ready surgery report (T5)

- Task: object detection (COCO 80-class)
- Source: Ultralytics `yolo11s.pt` (downloaded fresh, exported opset 17, static 1x3x640x640)
- Precision: INT8, target `modalix`
- Surgery script: `scripts/14_t5_compile_ready_surgery.py --model-id yolo11s`
- Compile script: `scripts/15_compile_t5_int8.py --model-id yolo11s`

## What this model is

yolo11s is the small-size sibling of the already-proven yolo11n. It is the same
graph topology (355 ONNX nodes, identical node names) with wider channels. This
is why it is the cheapest, most-certain compile in the T5 set: the surgery specs
from the proven yolo11n flow apply **byte-for-byte**.

## Surgery applied (identical to yolo11n)

1. **Attention MatMul -> Einsum rewrite.** The single C2PSA attention block
   `/model.10/m/m.0/attn` has two `MatMul` nodes. Both are rewritten to `Einsum`
   (`bhnc,bhck->bhnk` and `bhcn,bhnm->bhcm`). The MLA compiler maps batched
   Einsum to batch-matmul cleanly; a bare 4D `MatMul` is riskier. (Supported-op
   DB: Einsum int8=Y, "converted to batch matmul".)
2. **DFL bbox decode moved on-graph.** yolo11 emits 16 DFL bins per box side, so
   each `cv2.k.2` conv output is `[1,64,H,W]` (= 4 sides x 16 bins). For every
   scale we insert `Split(4x16) -> Softmax(axis=1) -> 1x1 Conv(weights=arange16)
   -> Concat` to collapse the bins into 4 distance channels `[1,4,H,W]`. This is
   the expected-value DFL reduction; all ops are MLA-supported.
3. **Postprocess/decode tail removed.** The graph outputs are replaced with the
   six raw head tensors so the compiler never sees the CPU-side
   Concat/Reshape/Transpose/anchor-decode/NMS tail.

## Output contract (6 tensors, Neat `BoxDecodeType.YoloV26` grouped layout)

| output | shape | meaning |
| --- | --- | --- |
| `bbox_0/1/2` | `[1,4,80/40/20,H]` | l,t,r,b distances per scale (post-DFL) |
| `class_logit_0/1/2` | `[1,80,80/40/20,H]` | per-class logits per scale |

## How to recognize / reproduce this pattern

- Node names `/model.23/cv2.k/cv2.k.2/Conv_output_0` (bbox) and
  `/model.23/cv3.k/cv3.k.2/Conv_output_0` (class) are stable across **all**
  Ultralytics YOLO11 detection sizes (n/s/m/l/x share the same graph, only
  channel widths differ). yolo11m would reuse this spec unchanged.
- `dfl_bins=16` marks a YOLO11/YOLOv8-style head. If the bbox conv output has 4
  channels instead of 64, it is a YOLO26-style head (set `dfl_bins=0`, identity).

## Audit result

`reports/audit_compile_ready_int8.json`: **0 unsupported ops** for int8.
