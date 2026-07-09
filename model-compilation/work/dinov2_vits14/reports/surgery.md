# dinov2_vits14 — surgery report (T7)

**Model:** `facebookresearch/dinov2` `dinov2_vits14` (ViT-S/14), self-supervised
feature extractor. **Input:** `input` `[1,3,224,224]` · **Output:** `features`
`[1,384]` (CLS-token embedding). No classifier head — DINOv2 produces *embeddings*.
**Artifact policy:** relaxed T7.

## What this model is, in graph terms

Same shape of network as `vit_b_16` but smaller (12 blocks, 6 heads, dim 384) and
with a 14×14 patch → `[1,257,384]` token sequence (256 patches + 1 CLS). Two things
differ from a plain torchvision ViT and both needed surgery: (a) a leftover
image-masking input from the DINOv2 forward signature, and (b) the same
attention-matmul layout issue.

## Surgery, step by step

### 1. Static-shape simplify + constant `Gather` → `Slice` (generic)

`scripts/03_surgery.py`: fixed input shape, 1 constant-index `Gather`→`Slice`.
→ `surgery/dinov2_vits14.surgery.onnx`.

### 2. Remove the unused `masks` input + `Where` (DINOv2-specific)

`torch.hub` traces DINOv2's `forward(x, masks=None)` into the graph as a **second
graph input** `masks` (rank-0) feeding:

```
masks → Unsqueeze → Where(cond=masks, X=mask_token, Y=patch_embed) → Concat(CLS, …)
```

At inference `masks` is all-false, so `Where` selects its Y branch — i.e. it is the
identity on the real patch-embed tensor. The surgery **rewires the `Concat` input
from the `Where` output to the patch-embed tensor directly**, deletes the `Where`
and the masks `Unsqueeze`, and drops the `masks` graph input.

**Why this is mandatory (not cosmetic):** the compiler importer is bound to exactly
one input (`input`) with one shape. A second, rank-0, unbound graph input makes
`load_model` fail. Removing the dead masks path leaves a clean single-input graph.
(This is the DINOv2 flavour of the general lesson: *torch.hub exports often carry
dead optional-argument branches — trace them and cut them.*)

### 3. Attention batched-`MatMul` → `Einsum`

Identical to the `vit_b_16` rewrite. Per block:

```
Q·Kᵀ : [1,6,257,64] x [1,6,64,257] -> [1,6,257,257]
A·V  : [1,6,257,257] x [1,6,257,64] -> [1,6,257,64]
```

→ `Einsum "bhmk,bhkn->bhmn"`, **24 rewrites** (12 blocks × 2). Linear projections
(48 `MatMul`) untouched. Output: `surgery/dinov2_vits14.compile_ready.onnx`.

## Op-support cross-check (audit INT8, release 2.1)

**0 unsupported**, 2 "unknown": `LayerNormalization`×25 and `Squeeze`×37 — both
compiler-lowered composites, same reasoning as vit_b_16. `Einsum`×24 and `MatMul`×48
are supported. The `Where`/`Unsqueeze`/`Gather` dynamic-shape ops that would have
been problematic are gone after surgery.

## How to recognize this pattern in other models

- **`torch.hub` models with optional forward args** (`masks`, `return_features`,
  attention-mask, …) frequently leave a **dead second input + `Where`/`If`** in the
  graph. Symptom: importer complains about an unbound/extra input, or a rank-0 input
  appears. Fix: find the `Where`/`If`, pick the branch taken at inference, rewire its
  consumers to that branch's tensor, delete the branch and the dead input.
- Everything else (attention `Einsum`, LayerNorm handling) is the shared ViT recipe.
- **No classifier head** → keep the `features` output and do nearest-neighbour /
  downstream heads on the CPU (see `pipelines/dinov2_embedding.py`). The T7 sanity
  check is "embedding shape == 384" + "an image is most similar to itself / a
  near-duplicate," which needs no labels baked into the graph.

## Compile result — FAIL (fragmented), root cause is token-sequence LayerNorm

- Command: `scripts/18_compile_transformer_int8.py --model-id dinov2_vits14
  --any-shape-on-mla` (INT8, 20 real calib images, modalix), via the global compile slot.
- Compiler **rc=0** but the archive is **NOT acceptable**: **MLA 99 / EV74 844 /
  A65 195** → **99 `.elf` + 195 `.so`** (294 segments). Validator (`--max-elf 3
  --allow-so`) → `status: fail` (elf out of range). This is worse fragmentation than
  maxvit even *with* `--any_shape_on_mla`.

### Root cause (from `reports/compile_int8.log` "Cannot assign … to MLA")

Ranked fallback reasons — the Einsum rewrite is **not** among them (it was accepted):

1. **`Reshape affecting the batch axis is not supported` / `Zero axis of the input
   shape must have a value of 1`** (≈260 occurrences). The token reshapes that fold
   heads into/out of the batch dimension around attention cannot go on the MLA.
2. **`LayerNormalization` → `Input is not 4D or 5D tensor` + `Mean or sum reduction
   operation over batch or channel axis is not supported and dimension should be less
   than 128`** (every `/blocks.N/norm1|norm2/LayerNormalization`, 25 nodes). DINOv2
   normalizes over the **384-dim channel** of a **3-D** `[1,257,384]` token tensor;
   the MLA reduction unit does not support a reduction over the channel axis of a
   non-4-D tensor when that dim is > 128. **This is not fixable by op-rewriting** —
   the reduction itself is the unsupported primitive on this HW/SDK.

### Why this is the important finding (and a correction)

**Pure token-sequence transformers (vit_b_16, dinov2_vits14) fragment on SiMa gen2 /
SDK 2.1.0**, because their two structural signatures — LayerNorm over a large channel
dim of a 3-D `[1,N,C]` tensor, and batch-axis head reshapes — are exactly what the MLA
placement rejects. The **conv-hybrid "ViTs"** in this same folder — `fastvit_t8`
compiled to **1 `.elf` / 0 `.so`** — do *not* hit this, because FastViT keeps 4-D
NCHW spatial tensors and uses conv token-mixing, so its norms/reshapes stay 4-D and
MLA-placeable. That contrast is the teaching payload: **on this toolchain, "can a ViT
compile cleanly?" is decided by whether it stays in a 4-D spatial layout, not by the
attention op.** The attention `MatMul→Einsum` rewrite (which helped YOLO) is neither
sufficient nor the bottleneck here.

### What the next person should try

- This is a **documented blocker**, valid per the T7 fail-forward policy. A clean 1–3
  `.elf` is likely **not achievable for a stock torchvision/torch.hub pure ViT** on
  SDK 2.1.0 without toolchain support for 3-D LayerNorm + batch-axis reshapes.
- If a clean compile is required, options (all unverified, in rough order of promise):
  1. a newer SDK that lists 3-D `LayerNormalization` / batch-axis reshape as MLA ops;
  2. a conv-hybrid backbone (FastViT/MaxViT-conv-only) instead of a pure ViT when the
     use-case allows — those already compile clean here;
  3. accept the fragmented archive **only** if every `.so` is enumerated and justified
     (195 host stages cannot be reasonably justified → treat as a blocker, not a pass).

### Board smoke test — attempted, timed out (inconclusive)

`pipelines/dinov2_embedding.py` was run against this archive on the DevKit
(`ssh`, `timeout 420`). It **timed out (rc=124)** with no output: the 294-stage
archive (99 MLA + 195 host `.so`) is too heavy to load+run in a 7-minute window, and a
first (harness-killed) attempt left orphaned board processes contending for the MLA.
So the embedding-shape / nearest-label check is **not verified on device**. The pipeline
itself is **code-verified** (`python -m py_compile` clean; uses the same
`pyneat.Model(archive).run([tensor])` route as `scripts/06_neat_smoke_test.py`). A
device smoke test should be re-run against a **de-fragmented** archive (i.e. after the
compile blocker is resolved), not this one.
