# maxvit_t — surgery report (T7)

**Model:** torchvision `maxvit_t` (MaxViT-Tiny), ImageNet-1k classification.
**Input:** `input` `[1,3,224,224]` · **Output:** `logits` `[1,1000]`.
**Artifact policy:** relaxed T7.

## What this model is, in graph terms

MaxViT is a **hybrid**: a conv stem + MBConv blocks (depthwise/pointwise `Conv`,
`Squeeze-Excite`, `BatchNormalization`) interleaved with *two* attention flavours —
**block attention** (within a local window) and **grid attention** (a strided
global window). Crucially, torchvision exports MaxViT's attention already as
`Einsum` (44 of them in the audit), not as bare `MatMul`. So MaxViT is the
counter-example to the ViT rewrite: **no attention surgery is needed** — the export
is already in the MLA-friendly form the YOLO/ViT rewrite *produces*.

## Surgery

Only the generic pass (`scripts/03_surgery.py`): static-shape `onnxsim` with
`input:[1,3,224,224]`; **0** constant-`Gather`→`Slice` replacements were required
(MaxViT's windowed reshapes fold cleanly once shapes are static). Output:
`surgery/maxvit_t.surgery.onnx`. No head cutting (classification tail is kept), no
attention rewrite.

## Op-support cross-check (audit INT8, release 2.1)

**0 unsupported**, 5 "unknown". The supported set includes `Einsum`×44 (attention),
`MatMul`×89, `Conv`×61, `AveragePool`, `Erf`, `Sigmoid`, `Tanh`, `Softmax`, `Slice`,
`Transpose`, `Reshape`, `Add/Mul/Div`. The 5 "unknown" are the usual
compiler-lowered composites: `BatchNormalization`×11 (fused into conv),
`GlobalAveragePool`×12, `LayerNormalization`×45, `Gemm`×1 (classifier), `Flatten`×1.
None is a real gap.

Two op-table constraints worth noting for hybrids like this:
- `AveragePool`: kernel < 128 unless it is a *global* pool (any shape). MaxViT's
  pools are fine.
- `BatchNormalization` is only "unknown" because the table lists it as
  fuse-into-conv rather than a standalone primitive.

## How to recognize this pattern in other models

- **Check whether attention is already `Einsum` before rewriting.** Some exporters
  (torchvision MaxViT, and models converted through certain toolchains) emit
  attention as `Einsum` directly; running the ViT MatMul→Einsum pass then finds
  nothing to do, which is correct — do not force `MatMul`s that aren't there.
- **Conv-heavy hybrids mostly "just work"** after static-shape simplification; the
  attention is the only transformer-specific risk, and here it is pre-handled.
- This model was also the **longest CPU-bound compile** of the four (windowed
  attention expands into many tessellation tiles). Budget compile time accordingly
  and always run it through the serialized compile slot.

## Compile result — POLICY FAILURE (fragmented), diagnosed

- Command: `scripts/18_compile_transformer_int8.py --model-id maxvit_t` (INT8, 20 real
  COCO calib images, modalix, **no `--any_shape_on_mla`**), via the global compile slot.
- Compiler **rc=0** but the archive is **NOT acceptable**:
  `maxvit_t.surgery_mpk.tar.gz` = **58 `.elf` + 78 `.so`** (verified `tar tzf`; the
  compiler log says *"The model is split into 136 segments for MLA and APU"*,
  *"A65 : 78"*). The relaxed T7 policy is 1–3 `.elf` and `.so` only with a per-`.so`
  justification, so this **fails** (`scripts/05_validate_archive.py --max-elf 3
  --allow-so` → `status: fail`, elf out of range).

### Root cause (from `reports/compile_int8.log`)

The graph shattered at MaxViT's **windowed / grid attention** `partition_op` /
`departition_op`. Those blocks `Reshape`+`Transpose` the 4-D feature map into
**window-partitioned 5-D/non-4-D tensors** to do local/global attention, and the
compiler logged, repeatedly:

```
Cannot assign node .../window_attention/partition_op/Reshape to MLA
Cannot assign node .../grid_attention/partition_op/Transpose  to MLA
Some tensors are not 4D. Support for tensors that are not 4D is disabled in the
compilation settings.  (also: "Unsupported einsum equation", "Input is not 4D tensor")
```

Every partition/departition op that produced a non-4-D tensor was pushed to the
**A65 host** (`.so`), and each host detour split the surrounding MLA work into its
own `.elf` — 58 MLA islands separated by 78 host stages.

### The fix (not yet re-run — see budget note)

**Compile with `--any_shape_on_mla`.** That flag lifts exactly the "tensors that are
not 4D is disabled" restriction the log names, letting the window-partition reshapes
and the windowed-attention `Einsum`s stay on the MLA and collapse the 136 segments
back toward 1–3 `.elf`:

```
scripts/18_compile_transformer_int8.py --model-id maxvit_t \
    --calib-dir /workspace/calibration_images --num-calib-samples 20 --any-shape-on-mla
```

(Note: the earlier interrupted run also omitted `--any_shape_on_mla`, so it would
have fragmented identically had it finished — the interruption hid the real problem.)
This recompile did not fit inside the T7 60-minute compile-start budget after the
higher-priority vit/dinov2 attempts; it is the **first thing the next person should
run**. If `--any_shape_on_mla` alone does not fully de-fragment, the residual host
ops will be a short, enumerable list (the partition/departition `Transpose`/`Reshape`)
and each remaining `.so` must then be justified per policy.

**Teaching point:** windowed/shifted-window attention (MaxViT, Swin) partitions the
spatial grid into non-4-D tensors; on SiMa that needs `--any_shape_on_mla` or the
model fragments massively. This is the single most important gotcha for this model
family, and it is invisible until you inspect the ELF/`.so` count — an rc=0 compile
is **not** a passing artifact.

**Caveat (measured on the sibling model):** `dinov2_vits14` fragmented to 99 `.elf` /
195 `.so` **even with `--any_shape_on_mla`**, because of its 45+ token-sequence
`LayerNormalization`s (channel-axis reduction over a dim > 128 on a 3-D tensor — not
MLA-placeable; see `work/dinov2_vits14/reports/surgery.md`). MaxViT also carries 45
LayerNorms, so `--any_shape_on_mla` may only *partially* de-fragment it — the windowed
reshapes should re-place, but the LayerNorms likely will not. Set expectations
accordingly: a fully clean 1–3 `.elf` may not be reachable for this model on SDK 2.1.0.
