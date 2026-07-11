# Model Compilation for SiMa Modalix — reference & learning material

How to take a public model and turn it into a **single `.tar.gz` containing a single `.elf`** that
runs entirely on the MLA, and how to prove it actually works.

This folder is meant to be **read**, then run.

---

## The target contract: one `.tar.gz`, one `.elf`, zero `.so`

That is the whole game. Everything below exists to hit it.

| Archive contains | Meaning |
| --- | --- |
| exactly **1 `.elf`**, **0 `.so`** | the entire graph runs on the **MLA**. This is what you want. |
| a `.so` | part of the graph **fell back to the host CPU (A65)**. Slow, and a sign surgery is incomplete. |
| many `.elf` | the graph **fragmented** into many subgraphs. Something is unplaceable. |

**Two traps that make this easy to get wrong:**

1. **`rc=0` is NOT a passing artifact.** A compile can "succeed" and still hand you a fragmented,
   host-heavy archive. *Always check the members*, never the exit code.
2. **A `.so` file also carries the `\x7fELF` magic bytes.** A validator that sniffs magic bytes will
   count every `.so` as an `.elf` and report nonsense. **Count by file extension.**

The compile log tells you the same story up front — look for the plugin distribution line:

```text
MLA : 1     EV74 : 12     A65 : 0      <-- A65: 0 is the goal. Anything else means host fallback.
```

---

## The four steps

One registry (`models.yaml`) drives all four, so a model's input name, shape, normalization and
output contract can never drift between steps.

```
compile/
  convert_to_onnx.py   # 1. export the model  -> work/<id>/onnx/<id>.onnx
  graph_surgery.py     # 2. make it MLA-ready -> work/<id>/surgery/<id>.compile_ready.onnx
  compiler.py          # 3. INT8 quantize + compile (calibration lives here) -> *_mpk.tar.gz
  test_model.py        # 4. validate the contract, then run it on REAL images
  common.py            # shared registry/paths
  _surgery_*.py        # the per-family surgery implementations
```

Every script takes the same shape:

```bash
python compile/<step>.py --model-id yolo11n     # one model
python compile/<step>.py --all                  # every model in models.yaml
```

Full flow for one model:

```bash
source /sdk-extensions/model-compiler/bin/activate

python compile/convert_to_onnx.py --model-id yolo11n
python compile/graph_surgery.py   --model-id yolo11n
python compile/compiler.py        --model-id yolo11n     # strictly ONE compile at a time
python compile/test_model.py      --model-id yolo11n     # run this on the DevKit
```

---

## Step 2 is the interesting one: *why* surgery is needed

A stock YOLO export ends in a **CPU-shaped decode tail**: DFL softmax, anchor grids, concat,
transpose, NMS. The MLA cannot place those ops, so the compiler **splits the graph** and spills the
tail to the host as `.so` stages. That is where fragmentation and `.so` files come from.

The fix is not a compiler flag — it is to **cut the tail off and expose the raw per-scale heads**:

```
stock export:   backbone -> heads -> [DFL, anchors, concat, transpose, NMS]  -> 1 output
                                     ^^^^^^^^^^^ MLA can't place this ^^^^^^^^^^^

compile_ready:  backbone -> heads -> bbox_0..2, class_logit_0..2             -> 6 raw outputs
                (everything stays on the MLA -> ONE .elf, ZERO .so)
```

Neat then does the box decode itself (`BoxDecodeType`), or the app decodes the raw heads.

`graph_surgery.py` applies, per `models.yaml`:

| `surgery:` | what it does |
| --- | --- |
| `none` | CNNs. Nothing to do — they already compile to a single ELF. |
| `yolo_ultralytics` | attention `MatMul`→`Einsum`; expose `cv2.*`=bbox / `cv3.*`=class heads (YOLO26 uses `one2one_cv*`); **YOLO11 only**: rebuild DFL as `Split(64→16×4)→Softmax→Conv(arange)→Concat`. YOLO26 heads are already 4-channel, so DFL is skipped. Seg adds mask-coeff + proto; pose adds the 51-ch keypoint heads. |
| `yolox` | Megvii YOLOX: decoupled anchor-free head, numeric node names, no DFL/attention. Trace back from the output to the three `[1,85,H,W]` heads and cut the flatten/transpose tail. |

**Useful fact:** Ultralytics head node names are **scale-invariant** — `yolo11n` and `yolo11s` use
byte-for-byte the same names. Scaling a model up (n→s→m) is a *free* compile; the same surgery spec
works unchanged.

---

## Step 3: calibration — the mistake that silently ruins a model

INT8 quantization learns activation ranges from **calibration images**. They must be **real images
from the target domain**.

> Calibrating on synthetic/gradient images produces an archive that looks perfect — `rc=0`, one
> `.elf`, no `.so` — and is **quietly wrong**, because the learned ranges are meaningless.

This folder was itself guilty of that: the five CNN classifiers were originally calibrated on
`synthetic_calib_*.jpg`. They have been **recompiled on real COCO images**, and the synthetic data
is deleted.

To stop it happening again, `compiler.py` **refuses to run** if the calibration set looks synthetic:

```
[compile] REFUSING: calibration set looks SYNTHETIC (synthetic_calib_1.jpg, ...).
          Quantization must use real images from the target domain, or the archive
          will compile cleanly but be quietly wrong.
```

Calibration data lives in `assets/calibration/` (20 real COCO images); the smoke test uses
`assets/inference/` (5 real COCO images). Neither is synthetic.

---

## Step 4: prove it — contract **and** behaviour

```bash
python compile/test_model.py --all --validate-only     # host: archive contract only
python compile/test_model.py --model-id resnet50       # DevKit: contract + real inference
```

It checks both, because either alone can lie:

- **contract** — one `.elf`, zero `.so` (counted by extension, not magic bytes).
- **behaviour** — real inference on real images. Classification prints ImageNet top-k (a
  synthetic-calibrated model produces nonsense here even though its archive is "valid").
  Detection/seg/pose confirm the raw heads come back with the expected shapes.

---

## Models in this folder

All compile to **one `.elf`, zero `.so`**, calibrated on real images.

| Model | Task | Surgery | Why it's here |
| --- | --- | --- | --- |
| `resnet50`, `densenet169`, `convnext_tiny`, `efficientnet_v2_s` | classification | none | the CNN baseline — no surgery needed |
| `yolo11n`, `yolo11s`, `yolo26n` | detection | `yolo_ultralytics` | the surgery flow: raw heads, DFL rebuild (11) vs none (26) |
| `yolo11s-seg` | segmentation | `yolo_ultralytics` | + mask coefficients and the proto head |
| `yolo26s-pose` | pose | `yolo_ultralytics` | + 51-channel keypoint heads |
| `yolox_s` | detection | `yolox` | a **non-Ultralytics** head — different surgery entirely |

### Disabled: `fastvit_t8` — the cautionary case study

It **compiles perfectly** (1 `.elf`, 0 `.so`, `A65: 0`, `rc=0`) and is **quietly wrong**:

| | |
| --- | --- |
| FP32 ONNX (host) | **correct** — tennis frame → `racket 0.83` |
| INT8 archive (DevKit) | **wrong** — predicts an unrelated class |

Three calibration strategies all failed (`mse`/20 imgs → `coil 0.98` on every image; `mse`/100 →
`lampshade 0.08`; `min_max`/64 → `lampshade 0.53`). FastViT's reparameterized blocks do not survive
8-bit quantization — a **model-side sensitivity**, not a toolchain or calibration bug.

It is kept in `models.yaml` (`enabled: false`) precisely because it proves the point of step 4:
**a passing archive contract proves the graph is on the MLA. It proves nothing about accuracy.**
Only running it on real images caught this.

### Not here yet: ViT, DINOv2, MaxViT, DETR

Removed on 2026-07-11 because **we cannot yet compile them from source**. Be precise about why:

- **They ARE supported.** SiMa publishes archives for DETR and DINOv2 ViT-S/14 that compile to a
  clean **1 `.elf` / 0 `.so`** and run fine on the board.
- **We cannot build them.** SiMa ships **source-prepared** models
  (`detr_resnet50_modified_class_embed_bbox_embed.onnx`, `image_classification_vits14.onnx`) — the
  heads are rewritten *before* export. Their own compile script then uses the plain `afe` API with
  `default_quantization` and **no special flags**.
- Our stock exports + **ONNX-level surgery fragment badly** (DINOv2: 99 `.elf` / 195 `.so`).
  `MatMul→Einsum`, `--any_shape_on_mla` and static-shape passes do **not** rescue them.
  `maxvit_t` fragments into 113 stages and OOM-kills the compiler.

**The lever is a source-level model rewrite, not ONNX surgery.** Full write-up, plus the commands to
download and run the working DETR/ViT archives today:
[`results/T7_CORRECTION_transformers_are_supported.md`](results/T7_CORRECTION_transformers_are_supported.md).

**Rule: before compiling any transformer, check the published models first** — the zoo has 136
models, and `sima-cli download` handles the auth (plain `curl` hits an auth redirect loop).

---

## Layout

```
model-compilation/
  README.md                 # you are here
  models.yaml               # the single registry driving all four steps
  compile/                  # the 4-step flow
  assets/
    calibration/            # 20 REAL COCO images  (quantization)
    inference/              #  5 REAL COCO images  (smoke test)
    labels/
  work/<id>/                # artifacts: onnx/, surgery/, compile_int8/, reports/  (git-ignored)
  results/                  # status + write-ups
```

## Requirements

```bash
source /sdk-extensions/model-compiler/bin/activate   # afe + onnx + ultralytics
```

Steps 1–3 run on the **host** (SDK container). Step 4's inference runs on the **DevKit** —
`/workspace` is NFS-mounted there at the same path, so no copying is needed.

**Compile strictly one model at a time.** The compiler is memory-hungry; concurrent compiles OOM
(this is exactly how `maxvit_t` died).
