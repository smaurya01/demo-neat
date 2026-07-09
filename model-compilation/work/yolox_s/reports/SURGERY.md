# yolox_s — compile_ready surgery report (T5)

- Task: object detection (COCO 80-class), **non-Ultralytics** head
- Source: Megvii official pre-exported `yolox_s.onnx`
  (`github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx`,
  opset 11, ir 6, input `images` 1x3x640x640, output `[1,8400,85]`)
- Precision: INT8, target `modalix`
- Surgery: `scripts/16_yolox_surgery.py` (YOLOX-specific — the shared script 14
  does NOT apply)
- Compile: `scripts/15_compile_t5_int8.py --model-id yolox_s`

## Why this is genuinely new surgery

YOLOX is anchor-free with a **decoupled head** and a structure unlike the
Ultralytics YOLO family, so none of the yolo11/yolo26 node-name specs apply:

- **No transformer attention** anywhere -> no MatMul->Einsum rewrite.
- **No DFL** -> box regression is 4 raw channels (cx,cy,w,h) decoded with
  grid+stride offsets on the host.
- **Decoupled head per scale:** separate reg / obj / cls convs, then a per-scale
  `Concat` to `[1,85,H,W]` (85 = 4 reg + 1 obj + 80 cls). In this exported ONNX
  the obj and cls channels are **already Sigmoid-activated**; reg is raw.
- **Numeric node names** (`Conv_261`, `Concat_265`, tensor `798`) instead of the
  Ultralytics `/model.23/...` scheme, so names must be rediscovered per export
  (see below) rather than copied.
- The exported ONNX ships an **NMS-free flatten tail**:
  `[1,85,H,W] --Reshape--> [1,85,N] --Concat--> [1,85,8400] --Transpose-->
  [1,8400,85]`. That is pure layout postprocess.

## Surgery applied

Cut the flatten tail (Reshape/Concat/Transpose) and expose the three per-scale
decoupled-head tensors directly, so the compiler keeps a clean NCHW conv-head
boundary and the YOLOX grid+stride decode stays on the host:

| exported tensor | exposed output | shape |
| --- | --- | --- |
| `798` (Concat_265) | `yolox_head_0` | `[1,85,80,80]` |
| `824` (Concat_286) | `yolox_head_1` | `[1,85,40,40]` |
| `850` (Concat_307) | `yolox_head_2` | `[1,85,20,20]` |

Channel layout within each head: `0:4` reg (cx,cy,w,h raw), `4` obj (sigmoid),
`5:85` cls (sigmoid).

Note: the raw ONNX is already fully MLA-compatible (0 unsupported ops, see
below), so it *could* be compiled unchanged. We still do the head-exposing
surgery for two reasons: (1) it removes an output-side Transpose to `[1,8400,85]`
that is an awkward MLA output layout, and (2) it gives the phase-2 pipeline the
same "per-scale raw head + host decode" contract as the other three models.

## Host-side YOLOX decode (phase-2 note)

For each scale s with stride {8,16,32}: build the grid; `xy = (reg_xy + grid) *
stride`, `wh = exp(reg_wh) * stride`; `score = obj * cls`; then flatten all
scales, threshold, and NMS. Neat has no YOLOX decode type, so this is a host
step (documented for the pipeline app).

## How to rediscover the head tensors on another YOLOX export

Trace back from the graph output: final `Transpose` -> `Concat` (across scales)
-> per-scale `Reshape` -> per-scale `Concat`; the three `Concat` outputs shaped
`[1,85,80,80]/[1,85,40,40]/[1,85,20,20]` are the head tensors. The numeric names
(798/824/850) are export-specific; the shapes and the trace are stable.

## Audit result

`reports/audit_compile_ready_int8.json`: **0 unsupported ops / 0 unknown** for
int8. Ops: Conv, Sigmoid, Mul (SiLU), Add, Concat, MaxPool (SPP), Resize
(upsample), Slice — all MLA-supported. No `.so` fallback expected.
