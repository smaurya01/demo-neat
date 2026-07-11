# CORRECTION: transformers ARE supported — T7's "they fragment" conclusion was wrong

Date: 2026-07-11. Supersedes the T7 headline in `summary.md` and the matching section in
`/workspace/overall-learning.md`.

## What T7 concluded (WRONG)

> "Pure token-sequence transformers (ViT, DINOv2) fragment on SiMa gen2 / SDK 2.1.0 … These are
> unsupported *placements*, not unsupported *ops*, so no ONNX surgery removes them."

That was inferred from our own compiles fragmenting. It generalised a limitation of **our model
preparation** into a claim about **the hardware/toolchain**. It is false.

## What is actually true (verified 2026-07-11)

SiMa ships **validated, cleanly-compiled** archives for these exact models:

| Model | SiMa official archive | Our T7 from-scratch attempt |
| --- | --- | --- |
| DETR-R50 | **1 `.elf` / 0 `.so`** | not compiled (budget) |
| DINOv2 ViT-S/14 (`vits14`) | **1 `.elf` / 0 `.so`** | **99 `.elf` / 195 `.so`** |

Both pass even the STRICT one-ELF/no-`.so` policy. Both now **run end-to-end on the DevKit** from
this repo (see `pipelines/`). So ViT and DETR are fully supported; the MLA compiles them to a single
ELF.

## The actual lever: a source-prepared model, NOT ONNX surgery or compile flags

The difference is entirely upstream of the compiler:

- **DETR**: the shipped model is `detr_resnet50_**modified_class_embed_bbox_embed**.onnx` — SiMa
  rewrote the `class_embed` / `bbox_embed` heads at the **source** level before export. Its archived
  compile script uses the plain `afe` API (`load_model` → `quantize(default_quantization)` →
  `compile`) with **no** special flags and **no** `--any_shape_on_mla`. Input `[1,3,800,1333]`.
- **ViT (`vits14`)**: the zoo recipe (`vits14.yaml`) points at SiMa's own prepared ONNX
  (`sz://models/image_classification_vits14.onnx`), 224x224 ImageNet classification.

We took stock exports (torchvision / HF) and tried to fix them with **ONNX-level surgery**. That is
the wrong lever, and it fragments. **`MatMul→Einsum`, `--any_shape_on_mla`, and static-shape passes
do not rescue a stock ViT/DETR export.**

## Practical rule

1. **Before compiling any transformer, check the model zoo / published models first.**
   `sima-cli download` handles the auth; the catalog is
   `.../SDK<ver>/model_zoo/metadata_gen2.json` (136 models).
2. If SiMa publishes it, **use their archive** — it is a clean single ELF and is validated.
3. Only attempt a from-scratch transformer compile if it is genuinely absent, and expect that a
   **source-level rewrite** (not ONNX surgery) is what makes it compile.

## Where the models came from

```bash
# DETR (used by apps/examples/object-detection/detr-object-detector)
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz

# ViT / DINOv2 ViT-S/14
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/model_zoo/gen2/image_classification/vits14/vits14_mpk.tar.gz
```

## Verified on the DevKit (2026-07-11)

- `pipelines/detr_detect.py` → **63 detections across 3 COCO images**, correct classes
  (chair/bottle/vase on 139; person/tennis racket/sports ball on the tennis frames).
  Raw outputs `(1,1,100,92)` logits + `(1,1,100,4)` boxes.
- `pipelines/vit_classify.py` → correct ImageNet top-5 (tennis frame → `racket` 0.77,
  `tennis ball` 0.05). Output `(1,1,1,1000)`.

## Status of the from-scratch attempts (unchanged, still blockers)

- `maxvit_t` — retried 2026-07-11 **with `--any-shape-on-mla`**: still fragments (**113 stages**) and
  the compile was **OOM-killed (rc=-9)** at stage 25/113. The flag does not fix it. Blocker.
- `dinov2_vits14` (our export) — 99 `.elf` / 195 `.so`. **Superseded**: use the official `vits14`.
- `vit_b_16` (our export) — quantize error. **Superseded**: use the official `vits14`.
- `detr_resnet50` (our export) — never compiled. **Superseded**: use the official DETR archive.

These remain useful only as a documented example of what *not* to do.
