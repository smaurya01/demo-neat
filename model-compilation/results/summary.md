# Model Compilation Status

> ## ⚠️ CORRECTION (2026-07-11): transformers ARE supported
>
> The T7 headline below ("pure token-sequence transformers fragment on SiMa gen2") is **WRONG**.
> SiMa ships **validated 1-`.elf` / 0-`.so`** archives for both **DETR** and **ViT/DINOv2
> (`vits14`)**, and both now run end-to-end on the DevKit from `pipelines/`.
> The real lever is a **source-prepared model** (SiMa's `detr_..._modified_class_embed_bbox_embed`
> and `image_classification_vits14.onnx`) — **not** ONNX surgery, `MatMul→Einsum`, or
> `--any_shape_on_mla`. Our from-scratch exports fragmented because of how *we* prepared them.
> **Check the model zoo before compiling any transformer.**
> Full write-up: [`T7_CORRECTION_transformers_are_supported.md`](T7_CORRECTION_transformers_are_supported.md)


| Model | ONNX | Unsupported | Unknown | MPK | Archive | Notes |
| --- | --- | ---: | ---: | --- | --- | --- |
| `resnet50` | pass | 0 | 3 | pass | pass |  |
| `densenet169` | pass | 0 | 4 | pass | pass |  |
| `convnext_tiny` | pass | 0 | 5 | pass | pass |  |
| `efficientnet_v2_s` | pass | 0 | 3 | pass | pass |  |
| `vit_b_16` | pass | 0 | 4 | fail | **SUPERSEDED** | Use official `vits14` (1 elf/0 so). T7: static-shape + attention Einsum surgery done (compile_ready.onnx, 0 unsupported); INT8 compile **errors during quantization** with a conv2d channel mismatch (64 vs 768). See T7 detail + `work/vit_b_16/reports/surgery.md`. Diagnosed, not compiled. |
| `maxvit_t` | pass | 0 | 5 | rc0_fragmented | fail | Retry 2026-07-11 with --any-shape-on-mla: OOM-killed (rc=-9) at stage 25/113, still fragments. BLOCKER. T7: compiled rc=0 but **58 `.elf` + 78 `.so`** (136 segments) — windowed-attention partition/departition reshapes are non-4D and fell to A65 without `--any_shape_on_mla`. POLICY FAILURE; fix = recompile with `--any-shape-on-mla`. See `work/maxvit_t/reports/surgery.md`. |
| `fastvit_t8` | pass | 0 | 6 | pass | pass |  |
| `dinov2_vits14` | pass | 0 | 2 | rc0_fragmented | **SUPERSEDED** | Use official `vits14` (1 elf/0 so). T7: masks/Where removed + attention Einsum surgery; INT8 compiled rc=0 but **99 `.elf` + 195 `.so`** (MLA 99/EV74 844/A65 195). Root cause = token-sequence LayerNorm (channel-dim reduction over 384, not MLA-placeable) + batch-axis head reshapes. Blocker. See `work/dinov2_vits14/reports/surgery.md`. |
| `yolo11n` | pass | 0 | 0 | pass | pass | compile_ready_int8; one ELF, no `.so` |
| `yolo26n` | pass | 0 | 0 | pass | pass | compile_ready_int8; one ELF, no `.so` |
| `detr_resnet50` | pass | 0 | 5 | **OFFICIAL** | **pass (1 elf/0 so)** | Official SiMa archive downloaded + running on DevKit. T7: generic surgery already yields a static compile-ready graph (`pred_logits[1,100,92]`, `pred_boxes[1,100,4]`, 0 unsupported); attention is rank-3 supported batched MatMul. Not compiled (out of T7 compile-start budget). Ready to compile; pipeline + decode analysis done. See `work/detr_resnet50/reports/surgery.md`. |

Archive `pass` (strict, T1/T5 CNN/YOLO) means exactly one ELF and no `.so`. For T7
transformer models the relaxed contract is 1–3 `.elf` with any `.so` justified in the
model's surgery report; validate with
`scripts/05_validate_archive.py --max-elf 3 --allow-so`.

## T7 transformer / difficult-model status (Agent G, 2026-07-09)

Worked strictly one compile at a time through the global slot, easiest first. Honest
outcomes (no fabricated passes):

| Model | Surgery | Compile outcome | Deliverable |
| --- | --- | --- | --- |
| `maxvit_t` | generic static-shape (attention already `Einsum`, no rewrite needed) | **FAIL (fragmented)**: rc=0 but **58 `.elf` / 78 `.so`** (136 segments); windowed-attention partition/departition non-4D reshapes → A65 (compiled without `--any_shape_on_mla`) | diagnosed blocker + fix to try |
| `dinov2_vits14` | masks/Where removal + 24 attention `MatMul→Einsum` | **FAIL (fragmented)**: rc=0 but **99 `.elf` / 195 `.so`** (294 segments) **even with `--any_shape_on_mla`**; root cause = token-sequence LayerNorm (channel reduction over 384) + batch-axis head reshapes | diagnosed blocker + surgery report + pipeline |
| `vit_b_16` | static-shape (37 Gather→Slice) + 24 attention `MatMul→Einsum` | **FAIL (quantize error)**: TVM conv2d channel mismatch 64 vs 768 during quantization type-inference (importer layout on the Einsum'd graph) | diagnosed blocker + next step |
| `detr_resnet50` | generic static-shape collapsed the dynamic NestedTensor mask; rank-3 attention left as supported MatMul | **not compiled** (out of 60-min compile-start budget) | ready-to-compile ONNX + pipeline + verified decode analysis |

**Bottom line (honest):** none of the four met the relaxed 1–3 `.elf` policy. Two
compiled but fragmented into a blocker-level number of host `.so` stages, one errored in
quantization, one was not reached in budget. Per the T7 fail-forward policy these are
**documented blockers** — each with a verified root cause, which is the real teaching
value. The single most important, verified finding:

> **Pure token-sequence transformers (ViT, DINOv2) fragment on SiMa gen2 / SDK 2.1.0**
> because their per-block LayerNorm reduces over a large channel dim of a **3-D**
> `[1,N,C]` tensor (rejected: reduction over channel axis, dim > 128, non-4-D) and their
> attention reshapes fold heads into the **batch axis** (rejected: "reshape affecting the
> batch axis"). These are unsupported *placements*, not unsupported *ops*, so no ONNX
> surgery removes them. The **conv-hybrid** models in this folder that keep a 4-D NCHW
> spatial layout — `fastvit_t8` (**1 `.elf` / 0 `.so`**), and the CNNs — do not hit this.
> On this toolchain, "does a ViT compile cleanly?" is decided by 4-D-spatial layout, not
> by the attention op. The attention `MatMul→Einsum` rewrite (which fixed YOLO) is not
> the lever here.

**Key next actions (priority order):**

1. **maxvit_t**: recompile with `--any-shape-on-mla` — its fragmentation is windowed
   partition/departition reshapes; the flag may re-place them. Lower confidence than for
   a conv model (windowed attention still produces non-4-D tensors), but cheap to try and
   fully diagnosed.
2. **vit_b_16**: recompile from the **plain-MatMul** `surgery/vit_b_16.surgery.onnx`
   (`--onnx …surgery.onnx`) — dinov2 shows the Einsum is accepted elsewhere, so vit's
   quantize error is likely the Einsum interacting with vit's separate-q/k/v export.
   NOTE: even if it then compiles, expect the same LayerNorm fragmentation as dinov2.
3. **detr_resnet50**: `scripts/18_compile_transformer_int8.py --model-id detr_resnet50
   --output-names pred_logits pred_boxes --any-shape-on-mla`. `BoxDecodeType.Detr=13` is
   only an enum token (no raw-head decoder implemented); DETR postprocess stays on CPU
   (`pipelines/detr_detect.py`).
4. **Strategic**: for a clean transformer archive on this SDK, prefer a **conv-hybrid**
   backbone (FastViT / conv-only MaxViT) over a pure ViT, or a newer SDK that lists 3-D
   LayerNorm + batch-axis reshape as MLA placements. Always verify with the real
   `.elf`/`.so` count — an rc=0 compile with dozens of `.so` is a failure.
