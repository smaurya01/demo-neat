# Chapter 20 — Inside a model archive (.tar.gz)

*What is in the file you pass to `pyneat.Model`, which part of the stack reads each piece, and how
to check an archive before you build an app around it.*

---

## Before you start

| Requirement | Chapter |
|---|---|
| Board on the network, with internet | [5](05-internet-sharing.md) · [6](06-connect-to-router.md) |
| `sima-cli` installed and logged in | [7](07-install-sima-cli.md) |
| NVMe mounted at `/media/nvme` | [8](08-mount-nvme.md) |
| `pyneat` installed *(only for the last check)* | [11](11-install-pyneat.md) |

Nothing in this chapter changes the archive. The inspection commands need only `tar` and
`python3`, so they work the same on the board, on your host, or in the Neat SDK. The examples use
the board.

---

## What a model archive is

The MLA does not run ONNX. A model has to be **compiled** first, and the compiler writes the result
as one `.tar.gz` file — the **model archive**. It is the file Chapters
[12](12-object-detection.md) and [18](18-cpp-video-app.md) passed to `pyneat.Model(...)` and
`simaai::neat::Model(...)`.

Inside is a plain tar of flat files, no folders:

- **One MLA program** (`.elf`) — the compiled network.
- **One contract** (`*_mpk.json`) — the full description of the model: input, every processing
  step, every output, and which processor runs each step.
- **Per-stage config files** (`0_*.json`, `pipeline_sequence.json`) — settings for individual
  SiMa GStreamer plugins.
- **Two records of the compile** — a cycle report (`*_mla_stats.yaml`) and the script that built
  the archive (`archived_compile_script.*.py`).

You will also see the term **MPK**. It is the name of that inference contract, and it is where the
`_mpk` in names like `yolo_11n_mpk.tar.gz` comes from.

---

## Get an archive to look at

Use the same YOLO26 archive as [Chapter 12](12-object-detection.md). If you already downloaded it
there, skip this step.

```bash
sima@modalix:~$ export MODELZOO_VERSION="2.1.3"
sima@modalix:~$ mkdir -p /media/nvme/example/models
sima@modalix:~$ cd /media/nvme/example/models
sima@modalix:~$ sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz"
```

The name already tells you a lot: `yolo26m` model, `det` (detection), `bf16` precision,
`mla_tess` (the MLA does the tessellation itself — see below), `b1` (batch size 1). That naming is
a Model Zoo convention only. The files inside are what count.

---

## List what is inside

You do not need to unpack an archive to see what is in it:

```bash
sima@modalix:/media/nvme/example/models$ tar tzvf yolo26m-det-bf16-mla_tess-b1.tar.gz
```

```text
-rw-r--r-- root/root      2259 2026-05-16 03:57 0_preproc.json
-rw-r--r-- root/root     34307 2026-05-16 03:57 yolo26m_raw_supported_einsum_stage1_mla_stats.yaml
-rw-r--r-- root/root       793 2026-05-16 03:57 pipeline_sequence.json
-rw-r--r-- root/root 128348328 2026-05-16 03:57 yolo26m_raw_supported_einsum_stage1_mla.elf
-rw-r--r-- root/root     23480 2026-05-16 03:57 yolo26m_raw_supported_einsum_mpk.json
-rw-r--r-- root/root      2864 2026-05-16 03:57 0_process_mla.json
-rwxr-xr-x root/root      8183 2026-05-15 15:19 archived_compile_script.compile_yolo26_modelsdk.py
```

Seven files. Almost all of the 66 MB download is the `.elf`. The base name
(`yolo26m_raw_supported_einsum`) is the model name given to the compiler, not the name of the
archive — so the file names inside do not have to match the archive's name.

An INT8 archive of the same model has three more stage configs. You do not need to download it —
this is the listing of `yolo26m-det-int8-b1.tar.gz`, trimmed to the names:

```text
pipeline_sequence.json
0_boxdecoder.json
0_postproc.json
yolo26m_raw_supported_einsum_mpk.json
0_quanttess.json
0_preproc.json
0_process_mla.json
yolo26m_raw_supported_einsum_stage1_mla_stats.yaml
yolo26m_raw_supported_einsum_stage1_mla.elf
archived_compile_script.quantize_compile_model.py
```

Which stage configs appear depends on how the model was compiled — precision, and where
tessellation runs. Other archives you may meet also carry `0_tessellate.json` or
`0_detessellate.json`.

---

## What each file is

| File | What it stores | Who reads it |
|---|---|---|
| `<name>_mpk.json` (or `mpk.json`) | **The inference contract.** Model name, source ONNX path and checksum, compiler version, the input tensor, and every processing step in order with its processor (`EV74` or `MLA`), kernel, parameters, shapes and buffer sizes | **Neat `Model`.** The route planner builds the whole pipeline from this file. Loading fails without it |
| `<name>_stage1_mla.elf` | The compiled network — the program the MLA executes | The MLA stage (`neatprocessmla`). Neat takes its file name from the contract |
| `0_preproc.json` | Preprocess settings: resize, letterbox padding, colour format, normalisation, tile size, quantisation scale and zero point, output dtype | Written for the SiMa CVU preprocess plugin |
| `0_quanttess.json` | Quantise-and-tessellate settings: input size, `q_scale`, `q_zp`, tile size, `out_dtype` | Written for the SiMa CVU plugin. INT8 archives only |
| `0_process_mla.json` | MLA settings: `model_path` (the `.elf`), `batch_size`, and the size of each output buffer; most archives also list output shapes and `data_type` per output | Written for the SiMa MLA plugin |
| `0_postproc.json` | Detessellate-and-dequantise settings: per-output shape, `dq_scale`, `dq_zp` | Written for the SiMa CVU plugin |
| `0_boxdecoder.json` | Box-decode settings: `decode_type`, `num_classes`, thresholds, top-k, per-output shapes and dequant values | Written for the SiMa box-decode plugin |
| `0_tessellate.json`, `0_detessellate.json` | Layout-only steps, when the compiler emitted them as separate stages | Written for the SiMa CVU plugins |
| `pipeline_sequence.json` | The stage chain in plugin terms: `pluginId`, `processor`, `kernel`, `configPath` for each stage | Written for SiMa plugin tooling. Current Neat does not plan from it |
| `<name>_stage1_mla_stats.yaml` | The compiler's cycle estimate for each MLA segment (`name`, `start_cycle`, `end_cycle`) | You, when you want to see where MLA time goes. Neat extracts it but does not use it |
| `archived_compile_script.*.py` | The exact script that produced this archive | You, to see how it was quantised and compiled. Neat does not extract it |
| `*.so` | Part of the graph compiled for the host CPU instead of the MLA | See [the one-check health test](#the-one-check-health-test) |

**The rule that matters:** Neat treats **only** `*_mpk.json` as the description of the model. Its
documentation puts it plainly: *if it's not in the MPK contract, it doesn't exist.* The `0_*.json`
files are still extracted — you can find them with `model.find_config_path_by_plugin(...)` while
debugging — but for a normal `Model` route the planner takes the MLA settings, shapes and
quantisation values from the contract, not from them.

That is why editing `0_boxdecoder.json` does not change how Neat decodes boxes. Look at its
contents in any archive and you see the same generic values (`"decode_type": "yolo"`,
`"num_classes": 80`) — even in a ResNet-50 classification archive. Box decoding is chosen in your
code with `decode_type` in `ModelOptions` (see [Chapter 12](12-object-detection.md)).

---

## Read the contract

The contract is a single JSON document, 20–30 KB for a YOLO model. This short script reads it
straight out of the archive, without unpacking anything, and prints one line per step:

```bash
sima@modalix:/media/nvme/example/models$ cat > inspect_archive.py << 'EOF'
import json, sys, tarfile

with tarfile.open(sys.argv[1]) as tar:
    names = tar.getnames()
    mpk_name = next(n for n in names if n.endswith("mpk.json"))
    mpk = json.load(tar.extractfile(mpk_name))

print("contract:", mpk_name, "| Model SDK", mpk["model_sdk_version"])
print("elf:", sum(n.endswith(".elf") for n in names),
      "| so:", sum(n.endswith(".so") for n in names))
for node in mpk["input_nodes"]:
    print("input:", node["name"], node["size"], "bytes, range", node["input_range"])
for p in mpk["plugins"]:
    cp = p.get("config_params") or {}
    params = cp.get("params") or {}
    step = cp.get("kernel") or (p.get("resources") or {}).get("executable")
    dtype = params.get("out_dtype") or params.get("output_data_type") or ""
    shapes = params.get("output_shapes") or []
    shape = str(shapes[0]) if len(shapes) == 1 else ""
    print(f'{p["sequence"]:>3}  {p["processor"]:<4}  ' + "  ".join(x for x in (step, dtype, shape) if x))
EOF
sima@modalix:/media/nvme/example/models$ python3 inspect_archive.py yolo26m-det-bf16-mla_tess-b1.tar.gz
```

```text
contract: yolo26m_raw_supported_einsum_mpk.json | Model SDK 2.0.0
elf: 1 | so: 0
input: images 4915200 bytes, range [6.556510925292969e-07, 0.9999999403953552]
  1  EV74  cast_transform  bfloat16  [1, 640, 640, 3]
  2  MLA   yolo26m_raw_supported_einsum_stage1_mla.elf
  3  EV74  unpack_transform
  4  EV74  slice_transform  [1, 80, 80, 4]
  5  EV74  cast_transform  float32  [1, 80, 80, 4]
  6  EV74  slice_transform  [1, 40, 40, 4]
  7  EV74  cast_transform  float32  [1, 40, 40, 4]
  8  EV74  slice_transform  [1, 20, 20, 4]
  9  EV74  cast_transform  float32  [1, 20, 20, 4]
 10  EV74  cast_transform  float32  [1, 80, 80, 80]
 11  EV74  cast_transform  float32  [1, 40, 40, 80]
 12  EV74  cast_transform  float32  [1, 20, 20, 80]
 13  EV74  pass_through
```

Read it top to bottom — it is the model's whole data path:

1. **Input.** One tensor called `images`, 4,915,200 bytes. That is 1 × 640 × 640 × 3 values at
   4 bytes each: a 640×640 RGB image as `float32`. The `input_range` is the value range the model
   saw during calibration — here 0 to 1, so pixels divided by 255.
2. **Before the MLA** (`EV74` = the CVU vector processor). One step converts `float32` to
   `bfloat16`.
3. **The MLA.** One step, running the `.elf`.
4. **After the MLA**, back on the EV74. `unpack_transform` splits the MLA's single output buffer
   into six tensors. The rest trims and converts each one back to `float32`.
5. **The outputs.** Six tensors: three box heads (`[1, 80, 80, 4]`, `[1, 40, 40, 4]`,
   `[1, 20, 20, 4]` — four box values per grid cell at three scales) and three class heads with 80
   channels, one per COCO class. These are raw heads, not finished boxes. Neat's box-decode stage
   turns them into detections when you set `decode_type`.

The `input_range` is worth checking on any model you did not compile yourself. A ResNet-50
archive compiled by this repository shows `range [-2.018..., 2.570...]`: it was calibrated on
images normalised with ImageNet mean and standard deviation, and it expects the same at run time.

### The fields worth knowing

To open the contract directly, pull it out of the archive and pretty-print it:

```bash
sima@modalix:/media/nvme/example/models$ tar xzf yolo26m-det-bf16-mla_tess-b1.tar.gz -O \
    yolo26m_raw_supported_einsum_mpk.json | python3 -m json.tool | less
```

If `jq` is installed, it can pick out single fields — for example the input:

```bash
sima@modalix:/media/nvme/example/models$ tar xzf yolo26m-det-bf16-mla_tess-b1.tar.gz -O \
    yolo26m_raw_supported_einsum_mpk.json | jq '.input_nodes'
```

| Where in `*_mpk.json` | What it tells you |
|---|---|
| `name`, `model_path`, `model_checksum` | Which ONNX file was compiled, and its checksum |
| `model_sdk_version` | The compiler version that built the archive |
| `input_nodes[].name`, `.size`, `.input_range` | Input tensor name, size in bytes, calibration range |
| `plugins[]` | Every step, in order. Each has `sequence`, `processor` (`EV74` or `MLA`), `input_nodes`, `output_nodes` (with byte sizes) and `config_params` |
| `plugins[].config_params.kernel` | What the step does: `quantization_transform`, `tessellation_transform`, `cast_transform`, `unpack_transform`, `detessellation_transform`, `dequantization_transform`, `slice_transform`, `pass_through` |
| `plugins[].config_params.params.channel_params` | Quantisation scale and zero point, e.g. `[[255.12458585747225, -128]]` |
| `plugins[].config_params.params.input_shapes` / `output_shapes` | Tensor shapes in and out of that step |
| `plugins[].config_params.params.slice_shape` | Tile size for tessellation steps |
| `plugins[].resources.executable` | The `.elf` file name, on the `MLA` step |

---

## Tessellation, in one paragraph

The MLA does not read images row by row. It reads them in **tiles**, so data has to be reshuffled
into tile order on the way in (**tessellation**) and back into normal row order on the way out
(**detessellation**). Same bytes, different order. In an INT8 archive, and in a BF16 archive
compiled without MLA tessellation, you see this as separate `tessellation_transform` and
`detessellation_transform` steps on the EV74. In the `mla_tess` BF16 archive above there is no
tessellation step at all — the `.elf` handles it inside the MLA.

---

## INT8 and BF16

The compiler can build a model at two precisions:

- **INT8** — 8-bit integers, one byte per value. The compiler learns a scale and zero point for
  each tensor from **calibration images**. Bad calibration images give a model that loads and runs
  but gives wrong answers.
- **BF16** — 16-bit floating point (bfloat16), two bytes per value. No 8-bit scale and zero point
  per tensor, at the cost of a bigger program. It is the fallback when INT8 loses too much
  accuracy — this repository compiles YOLO-World as BF16 for that reason.

How much accuracy each costs was measured for `yolo26s` on COCO val2017 (4,980 images): BF16
stays within 0.07 mAP50-95 of FP32, while INT8 costs 2.67 mAP50-95 but runs about 1.7× faster than
BF16. The method and full table are in
[`model-compilation/ACCURACY.md`](../../model-compilation/ACCURACY.md).

Run the inspection script on an INT8 archive of the same model and the difference is plain:

```text
contract: yolo26m_raw_supported_einsum_mpk.json | Model SDK 2.1.0
elf: 1 | so: 0
input: images 4915200 bytes, range [0.0, 1.0]
  1  EV74  quantization_transform  int8  [1, 640, 640, 3]
  2  EV74  tessellation_transform  [1, 1228800]
  3  MLA   yolo26m_raw_supported_einsum_stage1_mla.elf
  4  EV74  unpack_transform
  5  EV74  detessellation_transform  [1, 80, 80, 4]
  6  EV74  dequantization_transform  [1, 80, 80, 4]
  ...
 16  EV74  dequantization_transform  [1, 20, 20, 80]
 17  EV74  pass_through
```

### How to tell which one you have

Do not trust the file name. Check the files:

| Look at | INT8 archive | BF16 archive |
|---|---|---|
| First step in `*_mpk.json` | `quantization_transform`, `"output_data_type": "int8"` | `cast_transform`, `"out_dtype": "bfloat16"` |
| Steps after the MLA | `dequantization_transform` per output | `cast_transform` to `"float32"` per output, no `dequantization_transform` |
| `0_preproc.json` → `output_dtype` | `EVXX_INT8` | `EVXX_BFLOAT16` |
| `0_preproc.json` → `q_scale`, `q_zp` | A real scale, e.g. `255.12…` and `-128` | `1.0` and `0` |
| `0_quanttess.json` | Present | Absent |
| `0_process_mla.json` → `data_type` | `INT8` per output | `EVXX_BFLOAT16` per output, or absent |
| `pipeline_sequence.json` → `is_bf16` | `false` | `true`, or the key is missing |
| Size of the `.elf` (yolo26m, Model Zoo) | 61 MB | 128 MB — roughly twice the INT8 size |

The first row is the reliable one. The `is_bf16` flag is missing from some archives, including the
Model Zoo BF16 archive above, so do not rely on it alone.

The quick test on the board:

```bash
sima@modalix:/media/nvme/example/models$ tar xzf yolo26m-det-bf16-mla_tess-b1.tar.gz -O \
    yolo26m_raw_supported_einsum_mpk.json | grep -o '"kernel": "[a-z_]*"' | sort | uniq -c
```

```text
      7 "kernel": "cast_transform"
      1 "kernel": "pass_through"
      3 "kernel": "slice_transform"
      1 "kernel": "unpack_transform"
```

`quantization_transform` in that list means INT8. `cast_transform` with no quantisation step means
BF16.

### Reading the INT8 numbers

The INT8 archive's first step has `"channel_params": [[255.12458585747225, -128]]` — a scale and a
zero point. With the input range of 0 to 1, that maps 0.0 to −128 and 1.0 to about 127: the whole
range of an 8-bit signed integer. The same pair appears as `q_scale` / `q_zp` in `0_preproc.json`
and `0_quanttess.json`. Each output has its own pair on its `dequantization_transform` step, and in
`0_postproc.json` as `dq_scale` / `dq_zp`.

---

## The one-check health test

A good archive has **exactly one `.elf` and zero `.so` files**. Count by file extension:

```bash
sima@modalix:/media/nvme/example/models$ tar tzf yolo26m-det-bf16-mla_tess-b1.tar.gz | grep -c '\.elf$'
1
sima@modalix:/media/nvme/example/models$ tar tzf yolo26m-det-bf16-mla_tess-b1.tar.gz | grep -c '\.so$'
0
```

| Archive contains | Meaning |
|---|---|
| 1 `.elf`, 0 `.so` | The whole network runs on the MLA. This is what you want |
| any `.so` | Part of the network could not be placed on the MLA and was compiled for the host CPU (A65) instead |
| several `.elf` | The network was split into pieces around something the MLA could not place |

Two cautions. A `.so` is also an ELF file inside, so a check that looks at file contents rather than
names counts it as an `.elf` — count by extension. And this test only proves **where** the model
runs, not that it gives correct answers. An INT8 model calibrated on the wrong images passes it
cleanly. Only running it on real images tells you that.

---

## Where an archive comes from

The Model Compiler builds it, in the Neat SDK on your host ([Chapter 15](15-install-neat-sdk.md) —
the optional Model Compiler extension). The chain is always the same three steps:

```text
 model.onnx
     │  load_model(...)      import the ONNX graph for Modalix
     ▼
 loaded model
     │  .quantize(...)       INT8 (with calibration images) or BF16
     ▼
 quantized model
     │  .compile(...)        place the graph on MLA + EV74, write the archive
     ▼
 <name>_mpk.tar.gz
```

You do not have to guess how a particular archive was made: `archived_compile_script.*.py` inside
it is the script that made it. Open it to see which compiler calls and settings were used:

```bash
sima@modalix:/media/nvme/example/models$ tar xzf yolo26m-det-bf16-mla_tess-b1.tar.gz -O \
    archived_compile_script.compile_yolo26_modelsdk.py | head -3
```

```text
#!/usr/bin/env python3
"""Compile a YOLO26 raw-head ONNX with Model SDK BF16 + MLA tessellation."""

```

In archives built by this repository's scripts, the precision is chosen by the `--bf16-weights` /
`--bf16-activations` command-line flags, so the script contains both `bfloat16_scheme()` and
`quantization_scheme(...)` and does not record which one ran. For the precision, check the first
step of `*_mpk.json` (see [How to tell which one you have](#how-to-tell-which-one-you-have)).

To compile a model yourself — export to ONNX, graph surgery, INT8 calibration on real images,
compile and test — follow [`model-compilation/`](../../model-compilation/README.md) in this
repository. Its [`MODEL-COMPILATION.md`](../../model-compilation/MODEL-COMPILATION.md) explains why
a stock YOLO export produces `.so` files and how to get to one `.elf`.

---

## What happens when your app loads it

`pyneat.Model(path)` in Python and `simaai::neat::Model(path)` in C++ do the same five things:

1. **Check the archive.** The name must end in exactly `.tar.gz` — `.tgz`, `.tar` and `.mpk` are
   rejected. Links, absolute paths and `..` in member names are rejected. The archive must contain
   a `*_mpk.json` (or `mpk.json`) and at least one `.elf` or `.so`.
2. **Extract it** into a private runtime folder, sorted by type:

   ```text
   <extract root>/proc_<pid>/pkg_<hash>/<archive name without .tar.gz>/
   ├── etc/     every .json, plus the _mla_stats.yaml
   ├── lib/     any .so
   └── share/   the .elf
   ```

   The compile script is not extracted. The extract root is the first writable one of: a mounted
   NVMe (`/media/nvme/simaai/coprocessing/models` on a board set up as in
   [Chapter 8](08-mount-nvme.md)), `/data/simaai/coprocessing/models`, then
   `simaai/coprocessing/models` under `$TMPDIR` (or `/tmp`). Set `SIMA_MPK_EXTRACT_ROOT` to choose
   it yourself.
3. **Parse the contract** (`etc/*_mpk.json`) — input, steps, outputs, dtypes and quantisation
   values.
4. **Plan the route.** The planner compares what the MLA needs with what your input is, then
   picks the stages: preprocess (resize, colour convert, normalise, and quantise-and-tessellate or
   cast as the contract requires), the MLA, and postprocess (detessellate, dequantise or cast, and
   box decode if you set `decode_type`).
5. **Build the pipeline** from Neat's own GStreamer elements: `neatprocesscvu` for the EV74 steps,
   `neatprocessmla` for the MLA — given the `.elf` from `share/` — and a box-decode element
   (`neatboxdecode` / `neatobjectdecode`) for detection output.

When the process exits, the runtime folder is deleted. To keep it for a look, set
`opt.cleanup_extracted_model_data = False` in `ModelOptions`, or run with
`SIMA_MPK_CLEANUP_EXTRACTED=0`, and remove it by hand afterwards.

> **Pass the `.tar.gz`, not a folder you unpacked.** A flat directory made by `tar xzf` is not
> accepted as a model. `Model` does its own extraction.

### Ask Neat what it planned

`pyneat` can tell you how it read the contract:

```bash
sima@modalix:~$ source ~/pyneat/bin/activate
(pyneat) sima@modalix:~$ python3 - << 'EOF'
import pyneat

model = pyneat.Model("/media/nvme/example/models/yolo26m-det-bf16-mla_tess-b1.tar.gz")
print(model.summary())

info = model.info()
print("MLA wants INT8 input:   ", info.needs.pre_quantization)
print("MLA wants a dtype cast: ", info.needs.pre_cast)
print("pre-MLA kernels:  ", info.pre_kernels)
print("post-MLA kernels: ", info.post_kernels)
EOF
```

```text
Model loaded: yolo26m-det-bf16-mla_tess-b1 (package storage: NVMe, runtime package path: /media/nvme/simaai/coprocessing/models/proc_5754/pkg_824be9b5e6b355dd/yolo26m-det-bf16-mla_tess-b1)
Model: yolo26m_raw_supported_einsum
  mpk: /media/nvme/simaai/coprocessing/models/proc_5754/pkg_824be9b5e6b355dd/yolo26m-det-bf16-mla_tess-b1/etc/yolo26m_raw_supported_einsum_mpk.json
  preprocess: disabled
  postprocess: cast
  outputs: 1 logical / 1 physical
MLA wants INT8 input:    False
MLA wants a dtype cast:  True
pre-MLA kernels:   ['cast']
post-MLA kernels:  ['unknown', 'cast']
```

`summary()` prints the model name, the path of the extracted `*_mpk.json`, the chosen preprocess
and postprocess, and the number of outputs. Here preprocess is `disabled` because no
`ModelOptions` were given; the planner adds resize and colour conversion when you ask for them, as
Chapter 12 does. The first line also shows where the archive was extracted. `info().needs` shows
what the planner read from the contract's dtypes: `pre_quantization` means the MLA expects INT8
input, `pre_cast` means a floating-point conversion such as `float32` to `bfloat16` is needed.

---

## Quick answers

| Question | Answer |
|---|---|
| Is this archive INT8 or BF16? | First step in `*_mpk.json`: `quantization_transform` = INT8, `cast_transform` to `bfloat16` = BF16 |
| Does it all run on the MLA? | One `.elf`, zero `.so` |
| What input does it expect? | `input_nodes` in `*_mpk.json`: name, byte size, value range. The shape is in the first step's `input_shapes` |
| What does it output? | The `output_shapes` of the last steps (before `pass_through`, if there is one) |
| Which `decode_type` do I use? | Not stored in the archive in a form Neat uses. Match it to the model family in your code — for YOLO26, `BoxDecodeType.YoloV26` |
| Can I change a setting by editing a `0_*.json`? | Not for a `Model` route. The planner works from `*_mpk.json` |
| Which compiler built it? | `model_sdk_version` in `*_mpk.json`, and the `archived_compile_script.*.py` |

---

## Where to go next

| Next step | Where |
|---|---|
| Run this archive on images | [Chapter 12](12-object-detection.md) |
| Use it in a C++ video app | [Chapter 18](18-cpp-video-app.md) |
| Compile your own model into an archive | [`model-compilation/`](../../model-compilation/README.md) in this repository |
| `Model`, `ModelOptions` and route inspection | [developer.sima.ai](https://developer.sima.ai/software/develop-apps/) |

---

| ← Previous | Contents | Next → |
|:---|:---:|---:|
| [Chapter 19 · Agentic development](19-agentic-development.md) | [All chapters](../README.md) | [Troubleshooting](troubleshooting.md) |
