# Accuracy study: scripts and how to run them

These scripts measure what compiling to bf16 or INT8 costs in accuracy against FP32, on COCO
val2017. **The results, the method and how they compare with Ultralytics' published numbers are in
[`../ACCURACY.md`](../ACCURACY.md).** This page is only how to rerun it.

| File | What it does |
| --- | --- |
| [`prepare_coco_subset.py`](prepare_coco_subset.py) | Downloads the COCO val2017 annotations and images into `data/`, skipping the INT8 calibration images (`--num N`, default all) |
| [`eval_coco.py`](eval_coco.py) | `detect`: runs one model at one precision — `--backend onnx` (FP32, host) or `--backend neat --precision int8\|bf16` (DevKit) — and writes `results/<model>_<precision>.json`. `score`: the COCO mAP table |
| [`eval_ultralytics.py`](eval_ultralytics.py) | The independent reference: the `.pt` through Ultralytics' own `predict()`, written as `results/<model>_pytorch.json` |
| `data/`, `results/` | Created by the scripts; git-ignored |

---

## Run it

Everything on the SDK host runs in the model-compiler environment from the
[main README](../README.md#setup):

```bash
source /sdk-extensions/model-compiler/bin/activate
cd model-compilation
```

### 1. Download the model and build the archives (SDK host)

```bash
python compile/convert_to_onnx.py --model-id yolo26s      # downloads yolo26s.pt from Ultralytics, exports ONNX
python compile/graph_surgery.py   --model-id yolo26s
python compile/compiler.py        --model-id yolo26s --precision int8   # ≈6 min
python compile/compiler.py        --model-id yolo26s --precision bf16   # ≈10 min
```

Compile one at a time. Details and expected output:
[COMPILE-COMMANDS.md, section 14](../COMPILE-COMMANDS.md#14-yolo26s--detection-int8-and-bf16-accuracy-study).

### 2. Download COCO val2017 (SDK host)

```bash
python accuracy/prepare_coco_subset.py            # all 5,000 minus calibration: ≈800 MB, ≈7 min
```

Add `--num 500` for a quick 500-image check instead. That replaces the downloaded subset, and
`score` can then no longer read results from the full set; run the script again without `--num` to
go back. Only compare results scored on the same images.

### 3. FP32 and the Ultralytics reference (SDK host)

```bash
python accuracy/eval_coco.py detect --model-id yolo26s --backend onnx      # ≈25 min on CPU
python accuracy/eval_ultralytics.py --model-id yolo26s                     # ≈15 min on CPU
```

### 4. Copy the images to the DevKit (once)

The DevKit runs read the images locally instead of over the shared workspace: the same results,
about a third faster. On the DevKit:

```bash
mkdir -p /media/nvme/coco
cp -r /path/to/demo-neat/model-compilation/accuracy/data/val2017_subset /media/nvme/coco/
```

### 5. bf16 (DevKit)

On the DevKit, from the `model-compilation` folder:

```bash
cd /path/to/demo-neat/model-compilation
/home/sima/pyneat/bin/python accuracy/eval_coco.py detect --model-id yolo26s --backend neat \
    --precision bf16 --image-dir /media/nvme/coco/val2017_subset
```

About 8 minutes. It runs the bf16 archive from `work/yolo26s/compile_bf16/` and writes
`accuracy/results/yolo26s_bf16.json`. Expected last lines:

```text
   4980/4980  median inference 18.4 ms
[detect] wrote 635003 detections -> .../accuracy/results/yolo26s_bf16.json
```

### 6. INT8 (DevKit)

On the DevKit, from the `model-compilation` folder:

```bash
cd /path/to/demo-neat/model-compilation
/home/sima/pyneat/bin/python accuracy/eval_coco.py detect --model-id yolo26s --backend neat \
    --precision int8 --image-dir /media/nvme/coco/val2017_subset
```

About 6 minutes. It runs the INT8 archive from `work/yolo26s/compile_int8/` and writes
`accuracy/results/yolo26s_int8.json`. Expected last lines:

```text
   4980/4980  median inference 11.1 ms
[detect] wrote 621464 detections -> .../accuracy/results/yolo26s_int8.json
```

With `dk` from the SDK, steps 5 and 6 also work as `dk accuracy/eval_coco.py detect ...` with the
same arguments.

### 7. Score (SDK host)

```bash
python accuracy/eval_coco.py score                  # ≈4 min
```

```text
model     precision   mAP50-95   mAP50  vs FP32  images   median inference
yolo26s   pytorch        47.41   63.85    -0.30    4980   171.9 ms (host CPU, Ultralytics)
yolo26s   fp32           47.71   64.31        -    4980   231.6 ms (host CPU)
yolo26s   bf16           47.64   64.26    -0.07    4980   18.4 ms (DevKit MLA)
yolo26s   int8           45.05   61.82    -2.67    4980   11.1 ms (DevKit MLA)
```

`score` reads every file in `results/`, or only the files you name. What these numbers mean:
[`../ACCURACY.md`](../ACCURACY.md#results).

---

## Another model

Any YOLO26 detection model works. Add it to [`models.yaml`](../models.yaml) as `yolo26s` is, and
give it the YOLO26 surgery spec in [`compile/_surgery_ultralytics.py`](../compile/_surgery_ultralytics.py)
(next to the `yolo26s` line: `YOLO_SPECS["<id>"] = copy.deepcopy(YOLO_SPECS["yolo26n"])`). Then
compile it and pass its id to `eval_coco.py` and `eval_ultralytics.py`. The decoder expects YOLO26's
six raw output heads.
