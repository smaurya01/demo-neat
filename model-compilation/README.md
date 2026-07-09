# Model Compilation — YOLO11/26 Graph Surgery Walkthrough

**Day-2, Session-1 teaching backbone.** This document turns the working YOLO
compile flow in this folder into a step-by-step you can follow even if you have
never done ONNX graph surgery. It answers three questions:

1. What does the `compile_ready` surgery actually change in the graph, and **why**?
2. What is the exact output contract the compiled model must satisfy?
3. How do I redo this for **any** YOLO11 / YOLO26 variant (n / s / m)?

If you just want to reproduce the artifact, jump to
[The four commands](#the-four-commands). If you want to understand it, read on.

---

## 0. Mental model: why surgery at all?

An Ultralytics YOLO `.pt` exported straight to ONNX is **not** in a shape the SiMa
MLA (the accelerator) can run end to end at full efficiency. Two problems:

- **The detection head does too much.** Stock YOLO ONNX ends with a big
  decode/postprocess tail: DFL expansion, anchor/stride math, box assembly, and
  sometimes NMS — fused into one `(1, 84, 8400)` output. That tail is control-flow
  and gather heavy. It runs poorly (or not at all) on the MLA and would be forced
  onto the host A65 CPU, producing `.so` host-fallback stages in the archive.
- **Attention uses `MatMul` patterns the MLA tessellator doesn't like.** YOLO11's
  C2PSA / attention block (and YOLO26's two attention blocks) contain batched
  `MatMul`s whose layout the compiler cannot map cleanly to MLA tiles.

The Neat runtime already ships a **hardware box-decode** unit
(`BoxDecodeType.YoloV26`) that does DFL, box assembly, score threshold, and NMS
*off* the ONNX graph. So the winning strategy is:

> Cut the postprocess tail out of ONNX, expose the raw per-scale head tensors in
> the exact order Neat's box decode expects, and rewrite the attention `MatMul`s
> into an MLA-friendly `Einsum`. Then the whole network compiles to **one MLA ELF
> with zero host `.so` stages**, and Neat does the decode.

That surgery is `scripts/09_yolo_compile_ready_surgery.py`. Everything below
explains what it does.

---

## 1. The full chain at a glance

```
 yolo11n.pt                                     Ultralytics weights
    │  scripts/11_export_fresh_yolo.py
    ▼
 yolo11n.onnx            (1,84,8400) single decoded head, opset 17, static shapes
    │  scripts/09_yolo_compile_ready_surgery.py   ← THE SURGERY
    ▼
 yolo11n.compile_ready.onnx   6 outputs: bbox_0..2 + class_logit_0..2
    │  scripts/12_compile_yolo_int8.py  (INT8, 20 real calib images)
    ▼
 yolo11n.compile_ready_mpk.tar.gz     one .elf, zero .so
    │  scripts/05_validate_archive.py   → assert single ELF / no .so
    │  scripts/06_neat_smoke_test.py    → load on board, run a still image
    ▼
 verified YOLO11 archive, ready for apps/*-yolo-yolo11
```

One Python environment covers all four steps:

```bash
source /sdk-extensions/model-compiler/bin/activate   # afe + onnx 1.17.0 + ultralytics 8.4.90
```

---

## 2. Step 1 — Export a fresh, static ONNX

`scripts/11_export_fresh_yolo.py` downloads the Ultralytics `.pt` and exports ONNX
with the settings that matter for compilation:

```python
yolo.export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False)
```

- `dynamic=False` + `imgsz=640` → **static** `1x3x640x640` input. The MLA compiler
  needs fully static shapes; dynamic axes are a common first-day failure.
- `opset=17` → a stable opset the SiMa toolchain supports.
- `simplify=False` here → we simplify *after* surgery instead, so the surgery
  script sees the original, predictable node names.

Output: `work/<model>/onnx/<model>.onnx`, input `images`, output `output0`
`(1, 84, 8400)` — the single fused decoded head we are about to remove.

---

## 3. Step 2 — The `compile_ready` surgery (the teaching core)

Script: `scripts/09_yolo_compile_ready_surgery.py`. It does exactly three things.

### 3a. MLA-friendly attention rewrite

The attention block(s) contain two batched `MatMul`s (`.../attn/MatMul` and
`.../attn/MatMul_1`). We replace each with a semantically identical `Einsum` whose
equation states the batched contraction explicitly:

```python
# Q·Kᵀ  : (batch, heads, n, c) x (batch, heads, c, k) -> (batch, heads, n, k)
Einsum(equation="bhnc,bhck->bhnk")
# attn·V: (batch, heads, c, n) x (batch, heads, n, m) -> (batch, heads, c, m)
Einsum(equation="bhcn,bhnm->bhcm")
```

**Why:** the `Einsum` form makes the batched dimensions and contraction axis
unambiguous to the MLA tessellator, so the block maps onto MLA tiles instead of
falling back to the host. The math is unchanged — same numbers, MLA-runnable
layout. YOLO11n has **one** attention block (`/model.10/m/m.0/attn`); YOLO26n has
**two** (`/model.10/...` and `/model.22/...`). The per-model list lives in
`YOLO_SPECS` in the script.

### 3b. Remove decode/postprocess, expose the raw heads

The detection head has two parallel 1x1-conv branches per feature scale:

- `cv2.*` → **bbox** branch (box geometry)
- `cv3.*` → **class** branch (per-class logits)

at three scales (strides 8/16/32 → 80x80, 40x40, 20x20 grids). The surgery grabs
the tensor **at the output of each branch's last conv** — *before* any DFL /
anchor / concat / NMS — and republishes those as the graph's new outputs:

| Neat expects | YOLO11 source node (`cv2`/`cv3` = bbox/class) |
| --- | --- |
| `bbox_0` | `/model.23/cv2.0/cv2.0.2/Conv_output_0` |
| `bbox_1` | `/model.23/cv2.1/cv2.1.2/Conv_output_0` |
| `bbox_2` | `/model.23/cv2.2/cv2.2.2/Conv_output_0` |
| `class_logit_0` | `/model.23/cv3.0/cv3.0.2/Conv_output_0` |
| `class_logit_1` | `/model.23/cv3.1/cv3.1.2/Conv_output_0` |
| `class_logit_2` | `/model.23/cv3.2/cv3.2.2/Conv_output_0` |

(YOLO26 uses the `one2one_cv2.*` / `one2one_cv3.*` NMS-free head — same idea,
different node names, listed in `YOLO_SPECS`.) After exposing these six, the old
`output0` decode tail is dropped: `del model.graph.output[:]` then extend with the
six new `ValueInfo`s. The dangling postprocess nodes become dead and are pruned by
`onnxsim.simplify(...)` at the end.

### 3c. DFL → 4 distance channels (YOLO11 only)

Here YOLO11 and YOLO26 differ, and it is the subtle part.

- **YOLO26** already emits **4** bbox distance channels per scale
  (`left, top, right, bottom`). Its bbox source is exposed with a plain `Identity`.
  Output channels: `bbox_* = 4`.
- **YOLO11** emits **DFL bins**: `4 x 16 = 64` channels per scale, where each of
  the 4 sides is a 16-way probability distribution over discrete distances. Neat's
  `YoloV26` decode wants 4 plain distances, so the surgery reproduces the DFL
  reduction *inside* the graph, per side:

  ```
  Split(64 → 16,16,16,16 on channel axis)
    → Softmax(axis=1)              # distribution over the 16 bins
    → Conv(weight = [0,1,2,...,15])# expectation = Σ bin·p(bin)
    → Concat(4 distances)          # → bbox_* = 4 channels
  ```

  The 1x1 conv weight `arange(16)` turns "probabilities over bins" into the
  expected distance — exactly what DFL means — using only MLA-supported ops
  (`Split`, `Softmax`, `Conv`, `Concat`). See `add_yolo11_dfl(...)`.

That is why the script carries `"dfl_bins": 16` for YOLO11 and `0` for YOLO26.
After surgery **both** models present identical 4-channel bbox outputs, so a single
`BoxDecodeType.YoloV26` decode handles both families.

### 3d. Finalize

`onnxsim.simplify(..., overwrite_input_shapes={images: [1,3,640,640]})` folds
constants, drops the now-dead decode subgraph, and locks static shapes;
`shape_inference` + `checker` confirm the graph is valid. Result:
`work/<model>/surgery/<model>.compile_ready.onnx`.

---

## 4. The output contract

Every `compile_ready` YOLO graph — 11 or 26, n / s / m — exposes these **six
tensors in this exact order**:

```text
bbox_0         [1, 4, 80, 80]     stride 8   (4 = l,t,r,b distances)
bbox_1         [1, 4, 40, 40]     stride 16
bbox_2         [1, 4, 20, 20]     stride 32
class_logit_0  [1, 80, 80, 80]    stride 8   (80 = COCO classes, pre-sigmoid)
class_logit_1  [1, 80, 40, 40]    stride 16
class_logit_2  [1, 80, 20, 20]    stride 32
```

Order matters: `scripts/12_compile_yolo_int8.py` passes exactly

```python
OUTPUT_NAMES = ["bbox_0","bbox_1","bbox_2","class_logit_0","class_logit_1","class_logit_2"]
```

to the compiler, and Neat's `BoxDecodeType.YoloV26` consumes them in that order.
If class count changes (a custom-trained model), the `80` in `class_logit_*` and
the app's `num_classes` must match. Geometry (`80/40/20`) is fixed by the 640 input
and the 8/16/32 strides; a different `imgsz` scales these grids accordingly.

**On the app side** (see `apps/multi-stream-yolo-yolo11`, `apps/single-stream-yolo-yolo11`):
set `opt.decode_type = pyneat.BoxDecodeType.YoloV26`, feed the six tensors to
`pyneat.decode_bbox(...)`, and — per `/workspace/core/include/model/Model.h` — do
**not** set the deprecated `boxdecode_original_width/height`; box decode reads
geometry from preprocess metadata.

---

## 5. Step 3 — INT8 quantize + compile

`scripts/12_compile_yolo_int8.py` drives the `sima-model-quantize-compile` helper:

- **INT8** default (project policy), calibrated on **20 real** images from
  `assets/yolo_calibration` (`--real_data --num_calib_samples 20`, `mse` method).
- `--mean 0 --std 1`: normalization is already handled by the Neat preprocess
  preset (`COCO_YOLO`) at runtime, so the compiled graph expects raw-scaled input.
- `--mla-tesselation --any_shape_on_mla`: keep the whole network on the MLA.
- `--device modalix`.

Because the decode tail is gone and attention was rewritten, the compiler maps the
**entire** graph to the MLA: the archive has **one MLA ELF and zero A65 `.so`**.
(Before the surgery, the stock YOLO11 graph produced multiple MLA ELF stages *and*
`.so` host stages — the exact contract failure this flow fixes.)

Output: `work/<model>/compile_int8/<model>.compile_ready/<model>.compile_ready_mpk.tar.gz`.

---

## 6. Step 4 — Validate the artifact + smoke test

**Archive contract** — exactly one ELF, zero `.so`:

```bash
python scripts/05_validate_archive.py \
  --archive work/<model>/compile_int8/<model>.compile_ready/<model>.compile_ready_mpk.tar.gz
# status: pass  (single_elf: true, no_so: true)
```

**Neat smoke test on the board** (still image → decoded tensors). The board runs
via ssh in automation (`dk` needs a TTY); `/workspace` is NFS-mounted so run the
same on-disk path, wrapped in `timeout`:

```bash
timeout 180 ssh -o BatchMode=yes sima@192.168.135.203 \
  'source $HOME/pyneat/bin/activate; \
   python /workspace/demo-neat/model-compilation/scripts/06_neat_smoke_test.py \
     --model-id <model> \
     --archive /workspace/.../<model>.compile_ready_mpk.tar.gz \
     --image /workspace/demo-neat/model-compilation/assets/yolo_inference/<img>.jpg'
```

A load + non-empty output tensor set is the pass signal. For a full
decode-to-boxes check, use `scripts/10_run_yolo_sample_pipeline.py`.

---

## 7. The four commands

```bash
source /sdk-extensions/model-compiler/bin/activate
cd /workspace/demo-neat/model-compilation

# 1. export fresh .pt -> static ONNX
python scripts/11_export_fresh_yolo.py --model-id yolo11n --imgsz 640

# 2. compile_ready surgery (attention rewrite + head exposure + DFL for YOLO11)
python scripts/09_yolo_compile_ready_surgery.py --model-id yolo11n --force

# 3. INT8 quantize + compile  (SERIALIZE: only one compile at a time)
python scripts/12_compile_yolo_int8.py --model-id yolo11n --num-calib-samples 20

# 4. validate the archive contract
python scripts/05_validate_archive.py \
  --archive work/yolo11n/compile_int8/yolo11n.compile_ready/yolo11n.compile_ready_mpk.tar.gz
```

Steps 1–2 are CPU-cheap. Step 3 is the long CPU-bound job and must be serialized
across everyone sharing the machine (one compile at a time).

---

## 8. Redo it for any YOLO11 / YOLO26 variant (n / s / m)

The flow is **variant-agnostic** for detection heads — the head node names are the
same across n/s/m; only channel widths differ, which the graph handles
automatically. To add a variant:

1. **Register it** in `models.yaml` (id, `arch: yolo11s.pt`, `input_name: images`,
   `input_shape: [1,3,640,640]`, `mean/std`).
2. **Add its arch** to `YOLO_ARCHES` in `11_export_fresh_yolo.py` and its head/attn
   spec to `YOLO_SPECS` in `09_yolo_compile_ready_surgery.py`. For a plain size
   scale-up (yolo11n → yolo11s/m) the node names and `dfl_bins: 16` are identical —
   copy the `yolo11n` entry. Confirm attention-block names with Netron if a new
   backbone adds or moves attention blocks.
3. Run the four commands with `--model-id yolo11s`.

**Size cap (policy): n / s / m only.** Never compile `l` or `x` variants.

**How to confirm the head node names** for a new model (open the exported ONNX in
[netron.app](https://netron.app) or):

```python
import onnx
m = onnx.load("work/<model>/onnx/<model>.onnx")
outs = {o for n in m.graph.node for o in n.output}
print([o for o in outs if "cv2" in o and "Conv_output_0" in o])  # bbox sources
print([o for o in outs if "cv3" in o and "Conv_output_0" in o])  # class sources
```

**Common first-day failures**

- Dynamic shapes → compiler rejects. Always export `dynamic=False`, static `imgsz`.
- `.so` in the archive → a postprocess/attention op fell back to host. Re-check the
  attention rewrite matched (report's `attention_rewrites` must be non-empty) and
  that you exposed the pre-decode head tensors, not `output0`.
- Wrong output order → boxes garbage. Keep the six-name order exactly.
- Class count mismatch → set the app's `num_classes` to your model's classes.

---

## 9. Status snapshot

Per-run verification logs live in `results/` (e.g.
`results/t1_yolo11n_verification.md`). Model-by-model status is tracked in
`results/summary.md`. The YOLO `compile_ready` INT8 details are in
`results/yolo_compile_ready_int8.md`.

For transformer / non-CNN models (ViT, MaxViT, DINOv2, DETR) the artifact policy
is relaxed (1–3 ELF, `.so` only with a written justification) — that is separate
work tracked in `results/summary.md`, not covered by this YOLO walkthrough.
