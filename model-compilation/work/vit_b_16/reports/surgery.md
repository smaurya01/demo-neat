# vit_b_16 — surgery report (T7)

**Model:** torchvision `vit_b_16` (ViT-Base/16), ImageNet-1k classification.
**Input:** `input` `[1,3,224,224]` NCHW · **Output:** `logits` `[1,1000]`.
**Artifact policy:** relaxed T7 (1–3 `.elf`; `.so` only with justification).

## What this model is, in graph terms

A pure vision transformer: `Conv` patch-embed (16×16 stride-16) → flatten to a
`[1,197,768]` token sequence (196 patches + 1 CLS) → 12 identical encoder blocks
(multi-head self-attention + MLP, each wrapped in `LayerNormalization`) → take the
CLS token → `Gemm` classifier head → 1000 logits. There is no spatial conv stack
after the patch embed and no task-specific decode tail, so — unlike YOLO — nothing
has to be *cut* from the head. The whole difficulty is the attention.

## Surgery, step by step

Two passes were applied. The first is generic (`scripts/03_surgery.py`, run by the
earlier CNN/transformer sweep); the second is the T7 attention rewrite
(`scripts/19_vit_attention_surgery.py`).

### 1. Static-shape simplification + constant `Gather` → `Slice` (generic)

ViT's ONNX export carries dynamic `Shape`/`Gather` index math around the token
reshapes. `onnxsim.simplify(..., overwrite_input_shapes={input:[1,3,224,224]})`
folds those to constants, and 37 constant-index `Gather` nodes were rewritten to
`Slice`(+`Squeeze`). **Why:** the MLA compiler needs fully static shapes; a
constant-index `Gather` on a static tensor is a disguised `Slice`, and `Slice` is a
first-class supported op (`Converted to Conv2d` per the op table) whereas dynamic
`Gather` is not. Result: `work/vit_b_16/surgery/vit_b_16.surgery.onnx`.

### 2. Attention batched-`MatMul` → `Einsum` (the T7 rewrite)

Each encoder block exports its scaled-dot-product attention as two rank-4 batched
`MatMul`s whose **both** operands are activations:

```
Q·Kᵀ : [1,12,197,64] x [1,12,64,197] -> [1,12,197,197]
A·V  : [1,12,197,197] x [1,12,197,64] -> [1,12,197,64]
```

Both were replaced by `Einsum(equation="bhmk,bhkn->bhmn")`. That single equation is
the *layout-agnostic* form of any 4-D batched matmul `A[b,h,m,k]·B[b,h,k,n]`, so it
is correct for both the Q·Kᵀ and the A·V matmul without case analysis — the script
verifies `A.shape[-1] == B.shape[-2]` and that the output equals `[b,h,m,n]` before
rewriting, and `onnx.checker` re-validates the equation against the operand ranks.
**24 rewrites** total (12 blocks × 2).

**Why:** this is the same rewrite the YOLO walkthrough (README §3a) applies to the
C2PSA attention — an explicit-equation `Einsum` states the batched dims and the
contraction axis unambiguously, so the MLA tessellator maps the block onto MLA tiles
instead of falling back to the host. The math is identical (same numbers); only the
op form changes. The **linear** projections (q/k/v, attention out-proj, and the two
MLP matmuls) each have one weight operand and are left as ordinary supported GEMMs —
36 `MatMul` remain in the graph, which is expected and fine.

## Op-support cross-check (`sima-model-surgery` audit, INT8, release 2.1)

Cross-checked against the local `supported_operators.json` (same table as the
public model-compatibility list). After surgery: **0 unsupported**, 4 "unknown".

| Op | Count | Verdict |
| --- | ---: | --- |
| `Einsum` | 24 | **supported** — the attention rewrite target ("converted to batch matmul") |
| `MatMul` | 36 | **supported** — linear projections (batched matmul) |
| `Conv`, `Add`, `Mul`, `Div`, `Erf`, `Softmax`, `Reshape`, `Slice`, `Transpose`, `Concat` | — | supported |
| `LayerNormalization` | 25 | "unknown" in the table but **lowered by the compiler** to supported primitives (mean/var/normalize/affine). Not a real gap. |
| `Gemm` | 13 | "unknown" in the table; the classifier head + fused linears. Compiler lowers `Gemm`→matmul+add. |
| `Squeeze` / `Unsqueeze` | 49 / 12 | "unknown"; shape ops the compiler folds. |

"unknown" here means "not an atomic entry in the audit table," **not** "will fall
back to host." `LayerNormalization`/`Gemm`/`Squeeze`/`Unsqueeze` are standard
composites the SiMa compiler decomposes; the true fallback signal is a `.so` in the
archive, checked at validation.

## How to recognize this pattern in other transformers

- **Sequence-token attention** shows up as two rank-4 `MatMul`s per block where
  *both inputs are activations* and the middle two dims are `(heads, seq)`; the
  linear projections always have a weight operand. Rewrite only the two-activation
  ones to `Einsum "bhmk,bhkn->bhmn"`.
- **LayerNorm is fine** — do not try to "fix" it; leave it for the compiler.
- **Dynamic token-reshape `Gather`s** are the usual first-day blocker; kill them with
  static-shape `onnxsim` + constant-`Gather`→`Slice`.
- The classifier tail (`Gemm` on the CLS token) needs **no** surgery — keep the
  natural `logits` output and do softmax/top-k on the CPU (see
  `pipelines/vit_maxvit_classify.py`).

## Compile result — FAIL (quantization type-inference error), diagnosed

- Command: `scripts/18_compile_transformer_int8.py --model-id vit_b_16
  --any-shape-on-mla` (picks `surgery/vit_b_16.compile_ready.onnx`, INT8, 20 real COCO
  calib images, modalix), via the global compile slot.
- The ONNX **loads** ("Model successfully loaded for modalix") but INT8 **quantization
  fails immediately** in TVM type inference (`reports/compile_int8.log`):

  ```
  conv2d: requires that `64`, the input channels (64) divided by groups (1),
   must match the input channels of the weight `768`, where the weight shape is ([768, 768, 1, 1]).
  The type inference pass was unable to infer a type for this expression ... under constrained
  ```

  The 768→768 1×1 conv is the attention **out-projection** (a `Linear` the importer maps
  to conv2d); it received a **64-channel** tensor — the per-head dim (768/12) — instead
  of the reshaped 768-channel context. So the layout around the attention context tensor
  is wrong *after import*.

### Diagnosis and recommended next step

The ONNX itself is valid (passes `onnx.checker` + `shape_inference`; the surgery report
above shows matched Einsum equations and correct shapes), so this is an **importer
layout-conversion** problem, not a bad graph. **`dinov2_vits14` uses the identical
`MatMul→Einsum` rewrite and compiled**, which points at a **vit_b_16-specific** trigger:
torchvision `vit_b_16` exports attention with separate q/k/v/out projections and a
particular reshape order that, once the two batched matmuls become `Einsum`, the
importer's Einsum→batch-matmul layout conversion pairs with the wrong axis.

**Next step (highest-value):** recompile from the **plain-MatMul** graph to isolate the
cause — the generic `surgery/vit_b_16.surgery.onnx` has 0 unsupported ops and static
shapes; the audit lists `MatMul` (batched) as supported, so the Einsum rewrite may be
unnecessary here:

```
scripts/18_compile_transformer_int8.py --model-id vit_b_16 \
  --onnx work/vit_b_16/surgery/vit_b_16.surgery.onnx \
  --calib-dir /workspace/calibration_images --num-calib-samples 20 --any-shape-on-mla
```

If the plain-MatMul graph compiles, the teaching conclusion is: *unlike YOLO, a plain
torchvision ViT's attention MatMuls are natively supported and should NOT be rewritten
to Einsum — the rewrite is only needed where the compiler actually rejects the matmul.*
This did not fit inside the T7 60-minute compile-start budget (spent on maxvit + the
vit/dinov2 Einsum attempts); it is queued as the recommended follow-up.
