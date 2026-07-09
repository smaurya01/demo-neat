# detr_resnet50 — surgery report (T7)

**Model:** `facebookresearch/detr` `detr_resnet50`, COCO detection (set prediction).
**Input:** `input` `[1,3,800,800]` · **Outputs:** `pred_logits[1,100,92]`,
`pred_boxes[1,100,4]` (100 queries × 91 classes + no-object; boxes cxcywh
normalised). **Artifact policy:** relaxed T7. **Order:** hardest, done last.

## What this model is, in graph terms

ResNet-50 backbone (`Conv`/`Relu`/`MaxPool`/`Add`) → flatten the feature map to a
`[625, ...]` token sequence (25×25 from 800/32) + sinusoidal position embeddings →
6-layer transformer **encoder** (self-attention) → 6-layer **decoder** (self- +
cross-attention over 100 learned object queries) → two small heads: a `Gemm`
classifier (`pred_logits`) and a 3-layer MLP box regressor (`pred_boxes`). It is
NMS-free by design.

## Surgery, step by step

### 1. Static-shape simplify + constant `Gather`→`Slice` (generic)

The raw DETR export is the messiest of the four models: it carries the NestedTensor
padding-mask machinery (`Shape`/`Gather`/`Where`/`Expand`/`ConstantOfShape`/`Tile`/
`CumSum` for position encoding over a dynamic mask) and **dynamic output shapes**
(`pred_logits` exported as `[0,0,92]`). `scripts/03_surgery.py` with
`overwrite_input_shapes={input:[1,3,800,800]}` folds all of it: the op count drops
from ~30 distinct ops (incl. `Shape`×144, `Gather`×128, `Where`×24, `Expand`×13,
`ConstantOfShape`×28) to a clean static graph, outputs become `[1,100,92]` and
`[1,100,4]`, and 2 constant-index `Gather`s become `Slice`. This single generic pass
did the heavy lifting — the dynamic-mask tail is what makes DETR "hard," and static
simplification removes it. Output: `surgery/detr_resnet50.surgery.onnx`.

### 2. Attention — left as supported rank-3 batched `MatMul` (NO rewrite)

DETR uses `torch.nn.MultiheadAttention`, which exports attention with the head
dimension **folded into the batch**, giving **rank-3** batched matmuls:

```
encoder self-attn:  [8,625,32] x [8,32,625] -> [8,625,625]   (8 = batch*heads)
                    [8,625,625] x [8,625,32] -> [8,625,32]
```

The rank-4 ViT rewrite (`scripts/19`, equation `bhmk,bhkn->bhmn`) deliberately does
**not** touch these — its guard requires rank-4 operands. The op audit lists these
as **supported** batched `MatMul` (33 two-activation matmuls, all rank-3). If a
`.so` were to appear pinned to these, the fix would be the rank-3 analogue
`Einsum "bmk,bkn->bmn"`; documented here so the next person does not have to
rediscover the layout.

## Op-support cross-check (audit INT8, release 2.1)

**0 unsupported.** "unknown": `LayerNormalization`×35, `Gemm`×17, `Constant`×10,
`Squeeze`×3, `Unsqueeze`×7 — all compiler-lowered composites (same reasoning as the
other transformers). Supported set: `Conv`, `MatMul`, `MaxPool`, `Relu`, `Add`,
`Mul`, `Sigmoid`, `Softmax`, `Slice`, `Reshape`, `Transpose`, `Concat`.

## Decode: why postprocess is CPU, verified against the runtime

The plan said "Hungarian matching stays on CPU." Two precise corrections belong in
the teaching material:

1. **There is no Hungarian matching at inference.** Hungarian matching is a
   *training-only* loss component (bipartite matching of predictions to targets). At
   inference DETR postprocess is just: softmax over the 92-way class axis, drop the
   no-object class, threshold, `cxcywh(normalised)→xyxy(pixels)`. That is what
   `pipelines/detr_detect.py` does, entirely on the host.

2. **`BoxDecodeType.Detr = 13` exists but is not a usable raw-head decoder for this
   archive.** Verified in `/workspace/core`: the enum token is defined in
   `include/pipeline/BoxDecodeType.h` and name-mapped in
   `src/pipeline/internal/sima/BoxDecodeTypeUtils.cpp:111`, but the actual runtime
   box-decode backend only implements the raw-head *contract* for the YOLO families —
   `class_depth_for(...)` in `stagesemantics/BoxDecodeStageSemantics.cpp` has cases
   only for `YoloV26/YoloV26Pose/YoloV26Seg/YoloV6/YoloX` and sends everything else
   (incl. `Detr`) to `default`; `BoxDecodeStaticContractExtractor.cpp` likewise
   raw-decodes only the YOLO26/YoloV6/YoloX set. So DETR's raw `pred_logits`/
   `pred_boxes` cannot be handed to `BoxDecodeType.Detr` and decoded — **keep the two
   heads as plain model outputs and decode on the CPU.** (General rule: a
   `BoxDecodeType` enum value existing is NOT proof the backend decodes that family
   from raw heads; only YoloV26* / YoloV6 / YoloX have implemented raw-head extractors.)

## Runtime gotcha (applies to all raw-head T7 models)

The MLA emits dequantized outputs as **NHWC (1,H,W,C), not NCHW**, and delivers all
raw outputs through **one named endpoint as a single multi-tensor sample**. A host
decoder that assumes NCHW or fixed index order silently returns nothing. Therefore
`detr_detect.py` **routes tensors by shape** (the 92-axis is logits, the 4-axis is
boxes) and moves the feature axis last, rather than trusting `outputs[0]`/`outputs[1]`.

## How to recognize this pattern in other models

- **Dynamic-mask / NestedTensor exports** (DETR, deformable-DETR, some segmentation
  models) look unmanageable until static-shape simplification collapses the
  `Shape`/`Gather`/`Where` padding machinery. Try that *first* before hand-surgery.
- **`nn.MultiheadAttention` ⇒ rank-3 batched matmuls** (heads folded into batch);
  the rank-4 ViT rewrite won't match — use `"bmk,bkn->bmn"` only if a fallback forces it.
- **Set-prediction detectors need no in-graph decode and no NMS** — keep the heads,
  decode on the host, and do not assume a same-named `BoxDecodeType` will consume them.

## Compile result

- Command (if the compile slot / 60-min budget allowed):
  `scripts/18_compile_transformer_int8.py --model-id detr_resnet50 --output-names pred_logits pred_boxes --any-shape-on-mla`
  (picks `surgery/detr_resnet50.surgery.onnx`; note `models.yaml` has
  `enabled: false` for detr — script 18 reads the entry directly, it does not gate on
  `enabled`). INT8, 20 real calib images, modalix, via the global compile slot.
- Status: _see `results/summary.md` and `reports/compile_int8.log`._
