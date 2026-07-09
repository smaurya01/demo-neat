# yolo26s-pose — compile_ready surgery report (T5)

- Task: human pose estimation (person box + 17 COCO keypoints)
- Source: Ultralytics `yolo26s-pose.pt` (exported opset 17, static 1x3x640x640)
- Precision: INT8, target `modalix`
- Surgery: `scripts/14_t5_compile_ready_surgery.py --model-id yolo26s-pose`
- Compile: `scripts/15_compile_t5_int8.py --model-id yolo26s-pose`

## What is different vs plain detection

Two things: it is a **YOLO26** head (NMS-free `one2one_*` branches, no DFL), and
it is a **pose** head (adds a keypoint branch).

1. **YOLO26 one2one head.** Box and class come from `one2one_cv2.k.2`
   (`[1,4,H,W]`, already 4 distance channels — **no DFL**) and `one2one_cv3.k.2`
   (`[1,1,H,W]`, single "person" class). This matches the proven yolo26n specs
   (`dfl_bins=0`, identity on bbox).
2. **Two attention blocks.** yolo26 has an extra C2PSA in the head:
   `/model.10/m/m.0/attn` and `/model.22/m.0/m.0.1/attn`. Both MatMul pairs are
   rewritten to Einsum (same as yolo26n).
3. **Keypoint branch.** `one2one_cv4_kpts.k` produces `[1,51,H,W]` = 17 keypoints
   x (x, y, visibility). (The intermediate `one2one_cv4.k.*` convs feed this
   branch; only the final `one2one_cv4_kpts.k` conv is exposed.)

## Surgery applied

- Attention MatMul->Einsum on both blocks.
- Bbox: Identity from `one2one_cv2.k.2` (`[1,4,H,W]`, no DFL).
- Class: Identity from `one2one_cv3.k.2` (`[1,1,H,W]`).
- Keypoints: Identity from `one2one_cv4_kpts.k` (`[1,51,H,W]`).
- Removed tail: the one2one top-300 select + decode (`output0=[1,300,57]`),
  which is data-dependent postprocess, runs on host.

## Output contract (9 tensors)

| output | shape | meaning |
| --- | --- | --- |
| `bbox_0/1/2` | `[1,4,80/40/20,·]` | person box distances (no DFL) |
| `class_logit_0/1/2` | `[1,1,80/40/20,·]` | person-class logit |
| `kpt_0/1/2` | `[1,51,80/40/20,·]` | 17 keypoints x (x,y,vis) per anchor |

Host-side pose decode: decode/NMS the person boxes, gather the 51 keypoint
channels of surviving anchors, and map each (x,y) through the anchor grid +
stride to image coordinates (visibility = sigmoid of the 3rd channel). Neat has
no built-in pose decode type, so this is a documented host step for phase 2.

## How to recognize / reproduce

- YOLO26 heads use the `one2one_*` name prefix and emit 4 raw box channels (no
  `dfl` conv) — set `dfl_bins=0`. A YOLO11/v8 head would use `cv2/cv3` and 64
  box channels.
- A pose head adds a keypoint conv `*_cv4_kpts.k` with `3 * num_keypoints`
  channels (51 for 17 COCO keypoints). Segmentation instead adds a 32-channel
  `cv4` + a `proto` output — same "extra per-scale branch" surgery shape, only
  the channel count and the presence of `proto` differ.

## Audit result

`reports/audit_compile_ready_int8.json`: **0 unsupported ops** for int8
(Einsum count 4 = 2 blocks x 2 matmuls, listed supported).
