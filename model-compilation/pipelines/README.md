# Reference pipelines — transformer / difficult models (T7)

Minimal, board-runnable Python pipelines for the compiled INT8 transformer
archives. Each one follows the same contract: **the MLA graph ends at the model's
natural tensor output(s); all task postprocess is CPU-side, here in the pipeline.**
That is the whole point of the T7 surgery — nothing task-specific (no argmax, no
NMS, no Hungarian matching) was baked into the accelerator graph.

All three use the same house runtime pattern as `scripts/06_neat_smoke_test.py`:

```python
import pyneat
tensor  = pyneat.Tensor.from_numpy(chw_float32_nchw, copy=True)
model   = pyneat.Model(archive_path)
outputs = model.run([tensor], timeout_ms=...)   # list of output tensors
```

## Running on the DevKit

`/workspace` is NFS-mounted on the board at the identical path, so **do not copy
files** — write host-side, run board-side. `dk` needs a TTY; for automation use
ssh, wrapped in `timeout`:

```bash
timeout 200 ssh -o BatchMode=yes sima@192.168.135.203 \
  'source $HOME/pyneat/bin/activate; python <one of the commands below>'
```

The human-facing `dk` form (for READMEs / a real terminal) is:
`dk /workspace/demo-neat/model-compilation/pipelines/<script>.py --archive ... --image ...`

## 1. `vit_maxvit_classify.py` — ImageNet top-k (ViT / MaxViT)

Single `logits[1,1000]` output → CPU softmax + top-k + ImageNet label lookup.

```bash
python pipelines/vit_maxvit_classify.py \
  --archive work/vit_b_16/compile_int8/vit_b_16.compile_ready/vit_b_16.compile_ready_mpk.tar.gz \
  --image assets/yolo_inference/000000000139.jpg --input-size 224 --topk 5
```

`--input-size 224` for both `vit_b_16` and `maxvit_t`. Labels:
`assets/labels/imagenet_classes.txt` (index-ordered ImageNet-1k).

## 2. `dinov2_embedding.py` — embedding + nearest-label sanity (DINOv2)

Single `features[1,384]` output (CLS embedding). No classifier head, so "label" =
nearest neighbour in embedding space. Two modes:

```bash
# gallery/query retrieval (label = gallery file stem)
python pipelines/dinov2_embedding.py --archive <dinov2 archive> \
  --gallery assets/yolo_inference --query assets/yolo_inference/000000000139.jpg --topk 3

# pairwise cosine-similarity matrix (sanity: self-similarity ~1.0)
python pipelines/dinov2_embedding.py --archive <dinov2 archive> \
  --images assets/yolo_inference/000000000139.jpg assets/yolo_inference/000000000885.jpg
```

The T7 validation is: embedding dim == 384, and the similarity structure is
sensible (an image is most similar to itself / a near-duplicate).

## 3. `detr_detect.py` — detection with CPU postprocess (DETR)

Two heads `pred_logits[1,100,92]` + `pred_boxes[1,100,4]` → CPU softmax over classes,
drop the no-object class, threshold, `cxcywh(normalised)→xyxy(pixels)`. NMS-free.

```bash
python pipelines/detr_detect.py \
  --archive work/detr_resnet50/compile_int8/detr_resnet50.compile_ready/detr_resnet50.compile_ready_mpk.tar.gz \
  --image assets/yolo_inference/000000000139.jpg --input-size 800 --threshold 0.7
```

Labels: `assets/labels/coco91_detr.txt` (91-slot COCO order with `N/A` gaps, the
mapping DETR's class index uses).

## Runtime gotcha — NHWC outputs, single multi-tensor sample

The MLA emits dequantized outputs as **NHWC (1,H,W,C), not NCHW**, and delivers all
raw outputs through **one named endpoint as a single multi-tensor sample**. A decoder
that assumes NCHW or a fixed index order silently returns nothing (no error). For
classification/embedding this is harmless (the output flattens to 1-D:
`logits[1000]`, `features[384]`). For DETR it matters: `detr_detect.py` **routes by
shape** (locate the size-92 axis = logits, the size-4 axis = boxes; move it last)
rather than trusting `outputs[0]`/`outputs[1]`. Also note `BoxDecodeType.Detr = 13`
is only an enum token — the runtime box-decode backend implements raw-head decode
only for the YOLO families (`YoloV26*`, `YoloV6`, `YoloX`), so DETR is decoded here
on the CPU, not by a Neat decode stage (see `work/detr_resnet50/reports/surgery.md`).

## Notes

- Preprocess is ImageNet mean/std `[0.485,0.456,0.406]/[0.229,0.224,0.225]` on a
  `[0,1]`-scaled RGB image, NCHW — matching each model's `models.yaml` entry and the
  calibration normalization, so runtime input matches the quantized expectation.
- Only run a pipeline for a model whose archive actually compiled; see
  `results/summary.md` for per-model status.
