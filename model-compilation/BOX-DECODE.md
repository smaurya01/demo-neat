# On-device box decode

`test_model.py` prints the raw head **shapes**. It does not tell you whether Neat can turn those
heads into boxes itself. That is worth knowing: an app that host-decodes raw heads pays 143–337
ms/frame, while Neat's on-device `neatobjectdecode` stage costs ~0.6 ms.

`compile/test_box_decode.py` checks it, on the DevKit:

```bash
dk compile/test_box_decode.py --all               # every detection, segmentation and pose model
dk compile/test_box_decode.py --model-id yolov8s
```

It takes the same `--precision` flag as `compiler.py`. It skips batched archives (see
[`MULTI-BATCH.md`](MULTI-BATCH.md#8-what-neat-050-can-do-with-a-batched-archive)).

## The decode type per model

**Every family decodes with its own Neat `BoxDecodeType`**, which `models.yaml` records per model as
`box_decode` / `box_decode_option` (set exactly this pair in your app's `ModelOptions`):

| Archive | `decode_type` | `decode_type_option` | Head the surgery exposes |
| --- | --- | --- | --- |
| `yolov8s`, `yolov8l`, `yolo11n`, `yolo11s`, `yolov8s-worldv2` | `YoloV8` | `GroupedByRoleLogit` | raw 64-channel DFL bbox heads + class logits |
| `yolo11s-seg` | `YoloV8Seg` | `GroupedByRoleLogit` | the same, plus mask coefficients and protos |
| `yolo26n`, `yolo26s`, `yolo26m` | `YoloV26` | `Auto` | one-to-one heads, 4 distance channels |
| `yolo26s-pose` | `YoloV26Pose` | `Auto` | one-to-one heads plus keypoints |
| `yolox_s` | `YoloX` | `Split3Interleaved` | decoupled bbox / objectness / class heads |

Two more settings go with the pair: `num_classes = 1` for `yolo26s-pose` (the pose decoder rejects
any other value), and the preprocess preset `NormalizePreset.None` for `yolox_s`, which takes raw
0–255 pixels (the Ultralytics models use `COCO_YOLO`).

Why the family matters: YOLOv8 and YOLO11 heads are **one-to-many** (several anchors fire per
object) and need NMS, which `YoloV8` runs on the board. YOLO26 heads are **one-to-one**, and since
Neat 0.5.0 `YoloV26` skips NMS by design. A YOLO11 decoded as `YoloV26` therefore returns a stack of
near-identical boxes per object. `GroupedByRoleLogit` tells `YoloV8` the class heads are logits (the
surgery cuts before Ultralytics' sigmoid; keeping logits through INT8 preserves recall), so the
decoder applies the sigmoid itself. Without it, scores come out above 1.

## What the script checks

The script builds each archive with exactly that pair, runs real images, and **fails** if the build
is rejected, if no image produces a box, if a score falls outside [0, 1], or if same-class boxes
overlapping above the NMS IoU are systematic (more than one pair per five boxes: NMS did not run). An
isolated overlapping pair is printed as a warning.

YOLO26's one-to-one head, compiled INT8 or bf16, occasionally returns one or two near-identical
pairs that FP32 Ultralytics does not (a second anchor's score crosses the threshold after
quantization). `YoloV26` runs no NMS, so they reach the app; measured on `yolo26s`: 0–3 such pairs
per COCO image. An app that must never show one can run a light class-aware NMS on the decoded
boxes.
