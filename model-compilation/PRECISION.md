# Precision: INT8 or bf16

The MLA does not run FP32, so every model is compiled to one of two precisions. This page explains
the difference, which one each model uses and why, and how to build or check the other one.

For the commands, go back to the [README](README.md).

## Table of Contents

- [INT8 and bf16 in one minute](#int8-and-bf16-in-one-minute)
- [What it costs: measured](#what-it-costs-measured)
- [Which one to pick](#which-one-to-pick)
- [Each model's default](#each-models-default)
  - [Why these three are bf16](#why-these-three-are-bf16)
- [How `compiler.py` decides](#how-compilerpy-decides)
- [Build a model at the other precision](#build-a-model-at-the-other-precision)
- [Check which precision you built](#check-which-precision-you-built)

---

## INT8 and bf16 in one minute

| | **INT8** | **bf16** |
| --- | --- | --- |
| Numbers | 8-bit integers | 16-bit floats |
| Speed and size | faster, smaller archive | slower, bigger archive |
| Accuracy | loses some; how much depends on the calibration | close to FP32 |
| Calibration images | **decide the accuracy**: the compiler learns each tensor's 8-bit range from them | still read, but there is no 8-bit range to learn, so they matter far less |

INT8 squeezes every tensor into 256 values. The compiler picks each tensor's range from the
calibration images, so they must be **real images from the target domain**; synthetic images give a
model that compiles cleanly and is quietly wrong (see
[Calibration](MODEL-COMPILATION.md#calibration-the-mistake-that-silently-ruins-a-model)).

bf16 keeps the same exponent range as FP32 with a shorter mantissa, so there is nothing to
calibrate and little to lose.

---

## What it costs: measured

On `yolo26s` over COCO val2017 (4,980 images, all but the 20 calibration images):

| | mAP50-95 vs FP32 | latency |
| --- | --- | --- |
| bf16 | −0.07 | 18.4 ms |
| INT8 | −2.67 | 11.1 ms |

For `yolo26m`, bf16 matched FP32 (+0.13 mAP50-95, on 500 COCO images). Method, all results, and why
they differ from Ultralytics' published numbers: [`ACCURACY.md`](ACCURACY.md).

---

## Which one to pick

| If | Use |
| --- | --- |
| Accuracy matters most, or INT8 fails the compile | **bf16** |
| You need the speed and a smaller archive, and a small accuracy loss is acceptable | **INT8**, then measure it on real images |
| Not sure | Build both and compare them with [`accuracy/`](accuracy/README.md) |

---

## Each model's default

| `precision:` in `models.yaml` | Models |
| --- | --- |
| **`bf16`** | `convnext_tiny`, `yolo26m`, `yolov8s-worldv2` |
| **`int8`** | `resnet50`, `densenet169`, `efficientnet_v2_s`, `yolov8s`, `yolov8l`, `yolo11n`, `yolo11s`, `yolo26n`, `yolo26s`, `yolo11s-seg`, `yolo26s-pose`, `yolox_s` |

### Why these three are bf16

- **`convnext_tiny`**: with Model Compiler 2.1.3 its INT8 build compiles cleanly (1 `.elf`, 0 `.so`)
  but loses most of its accuracy. On `000000000885.jpg` it scores racket 0.20, against 0.73 for the
  FP32 ONNX; the bf16 build scores 0.75. The other three classifiers are fine in INT8.
- **`yolov8s-worldv2`**: INT8 **fails the compile**. The graph places as a single MLA segment in both
  precisions, but INT8 fails the compiler's sim/bit-accuracy check: the rewritten attention exposes
  wide-range text-similarity score maps as intermediate INT8 tensors (saturation warnings on the attn
  `proj_conv` bias). bf16 has no quantization step and passes. Keep it bf16. The full story:
  [INT8 for YOLO-World](MODEL-COMPILATION.md#-int8-for-yolo-world--passes-placement-fails-the-sim-check).
- **`yolo26m`**: bf16 keeps FP32 accuracy, and it is the format SiMa's Model Zoo ships its YOLO26
  archives in. It also builds in INT8 (35 MB instead of 69 MB); that build is the
  one used for [batch 6](MULTI-BATCH.md), because bf16 cannot batch.

---

## How `compiler.py` decides

**Every model's entry in [`models.yaml`](models.yaml) has a `precision:` line, `int8` or `bf16`, and
that is what a plain command builds.** The test scripts check the same precision.

```yaml
  - id: yolo26s
    ...
    precision: int8        # python compile/compiler.py --model-id yolo26s  ->  INT8

  - id: yolo26m
    ...
    precision: bf16        # python compile/compiler.py --model-id yolo26m  ->  bf16
```

For bf16, `compiler.py` adds `--bf16-weights --bf16-activations` to the SiMa compiler command;
without them the compiler quantizes to INT8. A model with no `precision:` line stops with a message
asking you to add one.

---

## Build a model at the other precision

**For one build**, add `--precision` to `compiler.py`, and the same flag to the test scripts so they
check that build:

```bash
python compile/compiler.py        --model-id yolo26n --precision bf16
python compile/test_model.py      --model-id yolo26n --precision bf16 --validate-only
dk compile/test_model.py          --model-id yolo26n --precision bf16
dk compile/test_box_decode.py     --model-id yolo26n --precision bf16
```

The older flags `--bf16-weights --bf16-activations` mean the same as `--precision bf16`.

**To change a model's default**, edit the `precision:` line in its `models.yaml` entry. The plain
commands, `--all`, and the test scripts then all use it.

Each precision is built into its own folder, so one model can have both builds side by side:

```text
work/<id>/compile_int8/<...>_mpk.tar.gz
work/<id>/compile_bf16/<...>_mpk.tar.gz
```

---

## Check which precision you built

- **The first line `compiler.py` prints:** `[compile] yolo26m: precision=bf16 ...`
- **The folder:** `compile_bf16/` or `compile_int8/`. The exact command it ran is in
  `work/<id>/reports/compile_<precision>.command.txt`.
- **The archive itself:** a bf16 build converts its input with `cast_transform`; an INT8 build uses
  `quantization_transform`. Count the kernels in the archive's contract:

  ```bash
  tar xzf work/yolo26m/compile_bf16/*/yolo26m.compile_ready_mpk.tar.gz -O --wildcards '*_mpk.json' \
    | grep -o '"kernel": "[a-z_]*"' | sort | uniq -c
  ```

  ```text
        7 "kernel": "cast_transform"
        6 "kernel": "detessellation_transform"
        1 "kernel": "pass_through"
        1 "kernel": "tessellation_transform"
        1 "kernel": "unpack_transform"
  ```

This build does its tessellation on the EV74 (the `tessellation_transform` step). SiMa's Model Zoo
archive `yolo26m-det-bf16-mla_tess-b1` does it inside the MLA instead, which is what `mla_tess` in
its name means; both run with the same Neat app code. The
[Quick Start Guide, Chapter 20](../devkit-quick-start/chapters/20-model-archive.md#how-to-tell-which-one-you-have)
explains every file in the archive.
