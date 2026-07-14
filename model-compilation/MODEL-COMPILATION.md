# Model compilation on Modalix — how it works, and what we learned

Why a public model does not just "compile" for the MLA, what has to change, and the traps that cost
us the most time.

For the commands, see **[`README.md`](README.md)**.

---

## The target contract: one `.tar.gz`, one `.elf`, zero `.so`

That is the whole game.

| Archive contains | Meaning |
| --- | --- |
| exactly **1 `.elf`**, **0 `.so`** | the entire graph runs on the **MLA**. This is what you want. |
| a `.so` | part of the graph **fell back to the host CPU (A65)**. Surgery is incomplete. |
| many `.elf` | the graph **fragmented**. Something is unplaceable. |

The compile log says the same thing up front:

```text
MLA : 1     EV74 : 12     A65 : 0      <-- A65: 0 is the goal. Anything else = host fallback.
```

**Two traps in the validation itself:**

1. **`rc=0` is NOT a passing artifact.** A compile can "succeed" and still hand you a fragmented,
   host-heavy archive. Always check the members, never the exit code.
2. **A `.so` also carries the `\x7fELF` magic bytes.** A validator that sniffs magic bytes counts
   every `.so` as an `.elf`, and reports a perfect archive. **Count by file extension.**

---

## Why graph surgery is needed

A stock YOLO export ends in a **CPU-shaped decode tail** — DFL softmax, anchor grids, concat,
transpose, NMS. The MLA cannot place those ops, so the compiler **splits the graph** and spills the
tail to the host as `.so` stages. That is where fragmentation and `.so` files come from.

The fix is not a compiler flag — **cut the tail off and expose the raw per-scale heads**:

```
stock export:   backbone -> heads -> [DFL, anchors, concat, transpose, NMS] -> 1 output
                                     ^^^^^^^^^^ MLA can't place this ^^^^^^^^^^

compile_ready:  backbone -> heads -> bbox_0..2, class_logit_0..2            -> 6 raw outputs
                (everything stays on the MLA -> ONE .elf, ZERO .so)
```

Neat then box-decodes itself (`BoxDecodeType`), or the app decodes the raw heads.

| `surgery:` | what `graph_surgery.py` does |
| --- | --- |
| `none` | CNNs — nothing to do; they already compile to a single ELF. |
| `yolo_ultralytics` | attention `MatMul`→`Einsum`; expose `cv2.*`=bbox / `cv3.*`=class (YOLO26 uses `one2one_cv*`); **YOLO11 only**: rebuild DFL as `Split(64→16×4)→Softmax→Conv(arange)→Concat` (YOLO26 heads are already 4-ch, so DFL is skipped). Seg adds mask-coeff + proto; pose adds 51-ch keypoint heads. |
| `yolox` | decoupled anchor-free head, numeric node names, no DFL/attention. Trace back from the output to the three `[1,85,H,W]` heads and cut the flatten/transpose tail. |

---

## Calibration: the mistake that silently ruins a model

INT8 quantization learns activation ranges from **calibration images**. They must be **real images
from the target domain**.

> Calibrating on synthetic/gradient images produces an archive that looks perfect — `rc=0`, one
> `.elf`, no `.so` — and is **quietly wrong**, because the learned ranges are meaningless.

This folder was guilty of exactly that. The CNNs have been recompiled on real COCO images and the
synthetic data is deleted. To stop it recurring, `compiler.py` **refuses to run** on a calibration
set that looks synthetic:

```
[compile] REFUSING: calibration set looks SYNTHETIC (synthetic_calib_1.jpg, ...).
          Quantization must use real images from the target domain, or the archive
          will compile cleanly but be quietly wrong.
```

Calibration set = `assets/calibration/` (20 real COCO images). Smoke test = `assets/inference/`
(5 real COCO images). Both are **tracked in git** — they are an *input*, not an artifact.

---

## Proving it: contract **and** behaviour

Both checks matter, because either one alone can lie:

- **Contract** — one `.elf`, zero `.so`, counted by extension. Proves the graph is **placed on the
  MLA**. Proves **nothing** about accuracy.
- **Behaviour** — real inference on real images. The only thing that catches a model that compiled
  perfectly and is numerically wrong.

---

## What worked, what didn't

The findings that cost the most time, and the ones worth reusing.

### ✅ The 209× pose fix: padding 51 → 64 channels

`yolo26s-pose` has a natural `{4, 1, 51}` channel mix across its bbox / class / keypoint heads. That
mix **defeats the compiler's output fusion** — each of the 9 outputs gets its own `slice_transform`
stage in the post-MLA tail:

| Keypoint head | Per frame | FPS |
| --- | --- | --- |
| natural 51 ch | 1782 ms | 0.6 |
| **padded to 64 ch** | **8.5 ms** | **117** |

**A 209× speedup, with identical weights and identical information.** The host decoder just slices
channels 51..63 (the zero padding) back off.

Two things make this generalize:

- It is the channel **mix**, not the 51 itself. Dropping *either* the class or the keypoint heads
  also makes it fast. Tile-aligned channel counts let the outputs fuse.
- **The MLA was never the bottleneck.** Pose is only 1.21× `yolo11s` in MLA cycles. All 1782 ms
  lived in the un-fused output tail.

The shipped archive is the padded build (verified: C=64). If you recompile, keep
`pad_channels_to: 64` in `compile/_surgery_ultralytics.py`.

### ✅ Ultralytics head names are scale-invariant

`yolo11n` and `yolo11s` use byte-for-byte the same head node names. Scaling n→s→m is a **free
compile** — the same surgery spec works untouched. This is why adding a model size costs nothing.

### ✅ YOLO26 needs no DFL rebuild

YOLO11 heads are 64-channel and need DFL reconstructed as
`Split(64→16×4)→Softmax→Conv(arange)→Concat`. YOLO26's heads are **already 4-channel**, so the whole
DFL step is skipped. Same surgery kind, different path through it.

### ❌ A green compile that was numerically garbage

We had a model compile to a flawless **1 `.elf` / 0 `.so`** archive, with `A65: 0` and `rc=0`, whose
FP32 ONNX was verifiably correct — and whose INT8 archive **predicted an unrelated class on every
image**.

**Three calibration strategies could not rescue it.** Some architectures simply do not survive 8-bit
quantization. There was no flag, no calibration set, and no surgery that fixed it.

This is the reason the DevKit behavioural test is not optional. Every contract check passed. The
model was still worthless.

### ❌ Synthetic calibration images

Covered above, but it belongs on this list: it is the failure mode that leaves **no trace in any log
or artifact**. The compile is green, the archive is well-formed, and the model is meaningless.
`compiler.py` now refuses outright rather than let it happen again.

---

## Provenance

Every archive in the [prebuilt bundle](README.md#download-the-prebuilt-archives) was built from its
**upstream original weights** — PyTorch / Ultralytics / Megvii — through this repo's own export →
graph-surgery → INT8 → compile chain.

**Nothing is downloaded pre-compiled from the SiMa model zoo.** (Some *apps* do use zoo models —
each app's own README says so — but those are not part of this flow.)

---

## Layout & what is committed

```
model-compilation/
  README.md              # download the archives, or set up to compile
  COMPILE-COMMANDS.md    # the copy-paste compile commands
  MODEL-COMPILATION.md   # you are here
  models.yaml            # the single registry driving all four steps
  compile/               # the 4-step flow (+ common.py, _surgery_*.py)
  assets/
    calibration/         # 20 REAL COCO images  -> quantization   (TRACKED: an input, not an artifact)
    inference/           #  5 REAL COCO images  -> smoke test     (TRACKED)
    labels/
    models/              # where the prebuilt bundle unpacks       (git-ignored)
  work/<id>/             # GENERATED build dir: onnx/ surgery/ compile_int8/ reports/  (git-ignored)
```

**Git policy: the repo holds only what you cannot regenerate.**

```
TRACKED                                    IGNORED
  README.md            get the models        work/           <- the entire build directory
  COMPILE-COMMANDS.md  the commands          *.pt *.onnx     <- weights & graphs (downloaded/exported)
  MODEL-COMPILATION.md the reasoning         *.tar.gz *.elf  <- compiled artifacts
  models.yaml          the registry          __pycache__/
  compile/             the 4 scripts
  assets/calibration/  REAL images
  assets/inference/    REAL images
```

`work/` is a **build directory** — every byte of it is regenerated by the four steps. Clone the repo,
run the flow, get identical archives.

**`assets/calibration/` is the exception, and it must be tracked**: those images are an *input*, not
an artifact. They cannot be re-derived, and swapping them silently changes every quantized model. A
compile is not reproducible without them.

Verify before committing — this must print nothing:

```bash
git add -A model-compilation/
git diff --cached --name-only | grep -E '\.tar\.gz$|\.elf$|\.so$|\.pt$|\.onnx$|\.sima$'
```
