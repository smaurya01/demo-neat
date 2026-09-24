# Chapter 12 — Object detection on images

*Your first real inference: a folder of images in, annotated images out.*

---

## Before you start

| Requirement | Chapter |
|---|---|
| Board on the network with internet | [5](05-internet-sharing.md) · [6](06-connect-to-router.md) |
| `sima-cli` installed and logged in | [7](07-install-sima-cli.md) |
| NVMe mounted at `/media/nvme` | [8](08-mount-nvme.md) |
| Board software 2.1.3 | [9](09-check-and-update-image.md) |
| `pyneat` installed | [11](11-install-pyneat.md) |

---

## Get a compiled model

Models for the MLA are distributed as compiled `.tar.gz` archives.

Set the Model Zoo release first — it can differ from your board software version:

```bash
sima@modalix:~$ export MODELZOO_VERSION="2.1.3"
```

Download into the NVMe, not your home directory:

```bash
sima@modalix:~$ mkdir -p /media/nvme/example/models
sima@modalix:~$ cd /media/nvme/example/models
sima@modalix:~$ sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz"
```

`yolo26m-det-bf16-mla_tess-b1` is a good default: BF16 weights with MLA tessellation, which is what
the example applications use. The `n`, `s`, `l` and `x` sizes are published alongside it — swap the
letter after `yolo26` to trade accuracy for speed.

---

## Get the COCO labels

The model returns class **ids**; the label file turns them into names — the standard 80-class COCO
list, one name per line, in class-id order.

**Option 1 — create the file on the board.** Copy the whole block below and paste it into the
board's terminal. It writes the file in one go:

<details>
<summary>Show the command (80 labels)</summary>

```bash
cat > /media/nvme/example/models/coco_labels.txt << 'EOF'
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
EOF
```

</details>

**Option 2 — copy it from your host.** If you have this repository checked out on your host, the
same list is at [`tutorial/assets/coco_labels.txt`](../../tutorial/assets/coco_labels.txt):

```bash
sima-user@host:~$ scp demo-neat/tutorial/assets/coco_labels.txt \
                      sima@<devkit-ip>:/media/nvme/example/models/coco_labels.txt
```

Check it has 80 lines:

```bash
sima@modalix:~$ wc -l /media/nvme/example/models/coco_labels.txt
80 /media/nvme/example/models/coco_labels.txt
```

---

## Prepare some input images

This repository ships six sample images for detection at
[`tutorial/assets/images/`](../../tutorial/assets/images/). Create the folder on the board:

```bash
sima@modalix:~$ mkdir -p /media/nvme/example/images
```

Then copy the images across from your host checkout:

```bash
sima-user@host:~$ scp demo-neat/tutorial/assets/images/image{,1,2,3,4,5}.png \
                      sima@<devkit-ip>:/media/nvme/example/images/
```

Check they arrived:

```bash
sima@modalix:~$ ls /media/nvme/example/images
image.png  image1.png  image2.png  image3.png  image4.png  image5.png
```

Your own JPEGs or PNGs work too — anything with recognisable objects (people, vehicles, laptops,
cups) suits a COCO-trained detector.

---

## The application

Both files go in `/media/nvme/example`, next to the folders you just created:

```text
/media/nvme/example/
├── config.yaml
├── detect.py
├── models/
│   ├── yolo26m-det-bf16-mla_tess-b1.tar.gz
│   └── coco_labels.txt
├── images/                 # your input images
└── output/                 # created on the first run
```

### `config.yaml`

Everything you are likely to change lives here, so the script itself stays fixed:

```yaml
model:
  path: /media/nvme/example/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
  labels: /media/nvme/example/models/coco_labels.txt
  input_size: 640         # the detector's input frame; YOLO26 is 640x640

io:
  input_dir: /media/nvme/example/images
  output_dir: /media/nvme/example/output

decode:
  score_threshold: 0.40   # minimum confidence to keep a box
  nms_iou: 0.60           # overlap above which duplicate boxes are merged
  max_detections: 50      # cap per image

runtime:
  timeout_ms: 20000
```

### `detect.py`

```python
#!/usr/bin/env python3
"""Detect objects in a folder of images with YOLO26 on the MLA."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PAD_VALUE = 114  # the grey YOLO pads with
BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def letterbox(bgr, size):
    """Fit an image into a square `size` frame, preserving its aspect ratio.

    Returns the padded frame plus the scale and padding needed to map
    detections back to the original image.
    """
    height, width = bgr.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2

    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
        bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, pad_x, pad_y


def to_original(boxes, scale, pad_x, pad_y, width, height):
    """Map boxes from the letterboxed frame back to original-image pixels."""
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - pad_x) / scale, 0, width)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - pad_y) / scale, 0, height)
    return out


def to_tensor(bgr):
    """Wrap a BGR OpenCV frame as a Neat tensor the MLA route can consume."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def draw_boxes(frame, boxes, labels):
    for x1, y1, x2, y2, score, class_id in boxes:
        class_id = int(class_id)
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        name = labels[class_id] if class_id < len(labels) else str(class_id)
        text = f"{name} {score:.2f}"
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), color, -1)
        cv2.putText(frame, text, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    parser = argparse.ArgumentParser(description="YOLO26 folder detection")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) or {}
    model_path = cfg["model"]["path"]
    labels = Path(cfg["model"]["labels"]).read_text().splitlines()
    input_size = int(cfg["model"].get("input_size", 640))
    input_dir = Path(cfg["io"]["input_dir"])
    output_dir = Path(cfg["io"]["output_dir"])
    decode = cfg.get("decode", {})
    max_detections = int(decode.get("max_detections", 50))
    timeout_ms = int(cfg.get("runtime", {}).get("timeout_ms", 20000))

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"No images in {input_dir}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(images)} images")

    # ModelOptions describes both ends of the model: the format of the frame
    # you push, and how the raw output is turned into boxes.
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = float(decode.get("score_threshold", 0.40))
    opt.nms_iou_threshold = float(decode.get("nms_iou", 0.60))
    opt.top_k = max_detections

    model = pyneat.Model(model_path, opt)

    # The graph is fixed to the input size it was built with, so build it once
    # for a square detector frame and letterbox every image into that frame.
    seed = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    runner = model.build([to_tensor(seed)])

    processed = 0
    try:
        for index, path in enumerate(images, start=1):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable: {path.name}", file=sys.stderr)
                continue
            height, width = bgr.shape[:2]

            frame, scale, pad_x, pad_y = letterbox(bgr, input_size)
            outputs = runner.run([to_tensor(frame)], timeout_ms=timeout_ms)

            # decode_bbox turns Neat's BBOX payload into [N, 6] rows of
            # x1, y1, x2, y2, score, class_id — in the coordinates of the
            # frame that was pushed, so undo the letterbox to draw on the
            # original image.
            boxes = pyneat.decode_bbox(
                outputs, clamp_to=(input_size, input_size), top_k=max_detections
            )[0].to_numpy()
            boxes = to_original(boxes, scale, pad_x, pad_y, width, height)

            draw_boxes(bgr, boxes, labels)
            out_path = output_dir / f"{path.stem}.png"
            cv2.imwrite(str(out_path), bgr)

            processed += 1
            print(f"[{index}/{len(images)}] {path.name} -> {out_path.name} "
                  f"({len(boxes)} detections)")
    finally:
        runner.close()

    print(f"Done: {processed}/{len(images)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### What the code is actually doing

- **The graph is fixed to the size it was built with.** Seeding `model.build(...)` lets the
  planner tighten the pipeline's input to that exact width and height. Push a differently sized
  frame into that route and nothing comes back out. Building for a fixed 640×640 frame and
  letterboxing every image into it keeps one graph valid for the whole folder, whatever sizes the
  images are.
- **Letterbox, not a plain resize.** `cv2.resize` straight to 640×640 stretches the image and
  distorts the objects in it, which costs you confidence and box accuracy. Letterboxing scales by
  the smaller ratio and pads the remainder, so nothing is squashed.
- **`ModelOptions` still covers normalisation and decode.** Your code does the geometry; Neat
  applies the COCO/YOLO normalisation, runs the MLA and turns the raw head output into boxes,
  without round-tripping through Python.
- **`BoxDecodeType.YoloV26` must match the model.** A YOLOv8 archive needs `YoloV8`, a YOLOX
  archive `YoloX`. A mismatch either fails when the graph is built or gives wrong boxes.
- **Boxes come back in the coordinates you pushed.** They describe the 640×640 letterboxed frame,
  so `to_original(...)` subtracts the padding and divides by the scale to put them back on the
  full-size image before drawing.

---

## Run it

```bash
sima@modalix:~$ source ~/pyneat/bin/activate
(pyneat) sima@modalix:~$ cd /media/nvme/example
(pyneat) sima@modalix:/media/nvme/example$ python3 detect.py --config config.yaml
```

The pyneat environment already includes everything the script imports. If you do get an import
error, install the missing packages:

```bash
(pyneat) sima@modalix:~$ pip install numpy opencv-python pyyaml
```

---

## What you should see

A per-image line naming the detections found, then one annotated image written per input:

```text
Found 6 images
[1/6] image.png -> image.png (10 detections)
...
Done: 6/6 images
```

Open the output images in `/media/nvme/example/output` to confirm the boxes land on the right objects.

**No detections at all?** Check `decode_type` against the model you downloaded, then lower
`score_threshold`. **Boxes in the wrong places?** Either a decode-type mismatch, or the letterbox
maths was changed without the inverse in `to_original` being changed to match.
**`timeout waiting for output` partway through the folder?** A frame reached the graph at a size
it was not built for — every frame must be letterboxed to `model.input_size` before it is pushed.

---

## If it is slower than you expect

Run `simaai-sentinel` ([Chapter 10](10-install-simaai-sentinel.md)) in a second terminal while it
works. For single-image inference the usual finding is that the MLA is barely busy and the time
goes on image decode and PNG encode — inference itself is often a small fraction of wall-clock time.
That is normal and not a problem to fix.

---

## Where to go next

| Next step | Where |
|---|---|
| The same detector as a packaged app, with C++ source and profiling | `~/prebuilt-apps/examples/object-detection/yolo26-object-detector/` |
| Other ready-to-run examples | `sima-cli neat install apps@v0.5.0`, then browse `prebuilt-apps/examples/` |
| Run a language model on the board | [Chapter 13](13-llima.md) |
| The full tutorial series | [Tutorials](https://developer.sima.ai/software/tutorials) |
| Compile your own ONNX model | [Model Compiler](https://developer.sima.ai/software/getting-started/) |

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 11 · Install pyneat](11-install-pyneat.md) | [All chapters](../README.md) | [Chapter 13 · LLiMa](13-llima.md) |
