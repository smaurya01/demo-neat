# Neat Tutorial Notebooks

Learn Neat application concepts through runnable notebooks. Start at `I-easy/01`, work down. Each
notebook is a markdown concept cell, a short runnable code cell, then a brief interpretation.

## Table of Contents

- [Running On The DevKit](#running-on-the-devkit)
- [I — Easy](#i--easy)
- [II — Medium](#ii--medium)
- [III — Advanced](#iii--advanced)
- [Next](#next)
- [Assets](#assets)
- [Getting the models](#getting-the-models)
  - [ResNet-50 — for notebook `I-easy/04`](#resnet-50--for-notebook-i-easy04)
  - [YOLOv8s — for notebooks `I-easy/05` and `I-easy/06`](#yolov8s--for-notebooks-i-easy05-and-i-easy06)
- [References](#references)

---

## Running On The DevKit

Run these **on the DevKit board**, so the notebook kernel can import and execute `pyneat`.
`/workspace` is NFS-mounted on the board at the same path, so you can edit host-side and run
board-side with no copying.

Activate the pyneat environment, install Jupyter if needed, and start the notebook server:

```bash
ssh sima@<devkit-ip>

source $HOME/pyneat/bin/activate

python -m pip install notebook (only once)

cd /workspace/demo-neat
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

Then open the DevKit notebook URL from your machine (host system):

```bash
http://modalix:8888/tree?token=************************
```

Replace the `modalix` name in above URL with devkit ip.

```bash
http://192.168.135.203:8888/tree?token=**************
```

## I — Easy

The core objects. Nothing here needs a camera.

| # | Notebook | What you learn |
| --- | --- | --- |
| 1 | [`01_neat_tensor.ipynb`](I-easy/01_neat_tensor.ipynb) | `Tensor` — shape, layout, memory, planes |
| 2 | [`02_node_and_graph.ipynb`](I-easy/02_node_and_graph.ipynb) | `Node` and `Graph` — how a pipeline is assembled |
| 3 | [`03_interpret_model_output_samples.ipynb`](I-easy/03_interpret_model_output_samples.ipynb) | `Sample` — reading what a model actually returned |
| 4 | [`04_image_classification_resnet.ipynb`](I-easy/04_image_classification_resnet.ipynb) | A first end-to-end model: ResNet-50 classification |
| 5 | [`05_yolo_cpu_decode.ipynb`](I-easy/05_yolo_cpu_decode.ipynb) | Decoding raw YOLO heads on the CPU — what box decode does for you |
| 6 | [`06_yolov8_image_detection_pipeline.ipynb`](I-easy/06_yolov8_image_detection_pipeline.ipynb) | A full image detection pipeline with Neat box decode |

## II — Medium

Options and I/O. These are the knobs you will actually turn in an app.

| # | Notebook | What you learn |
| --- | --- | --- |
| 1 | [`01_model_options.ipynb`](II-medium/01_model_options.ipynb) | `ModelOptions` — preprocess, resize/letterbox, normalize preset, `BoxDecodeType` |
| 2 | [`02_rtsp_input_and_decode_options.ipynb`](II-medium/02_rtsp_input_and_decode_options.ipynb) | `RtspDecodedInputOptions` — live H.264 in, NV12 out |
| 3 | [`03_run_options.ipynb`](II-medium/03_run_options.ipynb) | `RunOptions` — preset, `queue_depth`, overflow policy, output memory |
| 4 | [`04_input_output_options.ipynb`](II-medium/04_input_output_options.ipynb) | `InputOptions` / `OutputOptions` — graph boundaries and caps |
| 5 | [`05_video_sender_options.ipynb`](II-medium/05_video_sender_options.ipynb) | `VideoSenderOptions` — H.264 encode → RTP → UDP |
| 6 | [`06_metadata_sender_options.ipynb`](II-medium/06_metadata_sender_options.ipynb) | `MetadataSender` — ship detections as JSON alongside the video |
| 7 | [`07_rtsp_decode_encode_metadata_to_insight.ipynb`](II-medium/07_rtsp_decode_encode_metadata_to_insight.ipynb) | Put it together: RTSP → decode → infer → encode → video + metadata into Neat Insight |

Notebook 7 pairs with [`installation/neat_insight.md`](../installation/neat_insight.md).

## III — Advanced

Composition patterns for stages that sit beside the rest of a graph.

| # | Notebook | What you learn |
| --- | --- | --- |
| 1 | [`01_genai_model_in_graph.ipynb`](III-advance/01_genai_model_in_graph.ipynb) | `neat.genai.graphs.vision_language` — put a VLM/LLM in a `Graph` as a stage with named `prompt`/`image` inputs and streamed `tokens`/`done` outputs |

Notebook 1 needs a deployed LLiMa VLM on the DevKit (`llima pull <model>`), and mirrors the core
tutorial `022_compose_genai_into_graph` and the public
[*Compose GenAI into a Graph*](https://developer.sima.ai/software/tutorials/compose-genai-into-graph)
page. For a worked example of the direct-model-handle alternative, see the
[`detection-vlm-assistant`](../apps/detection-vlm-assistant/README.md) app.

## Next

Once the concepts land, move to a complete application in [`apps/`](../README.md#apps) — start with
[`single-stream-yolo-yolo11`](../apps/single-stream-yolo-yolo11/README.md).

## Assets

| Path | What |
| --- | --- |
| `assets/images/` | Sample images used by the notebooks |
| `assets/imagenet_labels.txt` | ImageNet class names (notebook `I-easy/04`) |
| `assets/coco_labels.txt` | COCO class names (notebooks `I-easy/05`, `06`) |
| `assets/models/` | Where you put model archives. **Git-ignored** — you download or build these. |

Model paths are variables near the top of each notebook. Update them to match what is on your DevKit.

## Getting the models

The notebooks need two archives from the SiMa Model Zoo. Run these once, from the `tutorial/` folder
**on the DevKit**:

### ResNet-50 — for notebook `I-easy/04`

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get resnet_50
cd ../..
```

→ `assets/models/resnet_50_mpk.tar.gz`

### YOLOv8s — for notebooks `I-easy/05` and `I-easy/06`

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_v8s
cd ../..
```

→ `assets/models/yolo_v8s_mpk.tar.gz`

If your DevKit uses a different platform version, replace `2.1.2` with that release.

> **YOLO11 is in the model zoo** — `sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_11n`
> works, as do `yolo_11s/m/l/x`. No notebook here needs it, but the YOLO11 apps do; see
> [`single-stream-yolo-yolo11`](../apps/single-stream-yolo-yolo11/README.md). A zoo YOLO11 archive
> decodes with `BoxDecodeType.YoloV8` (raw 64-channel DFL bbox heads), which is why `II-medium/01`
> describes that family as YOLOv8/YOLO11-style. Compile YOLO11 yourself only for a variant the zoo
> does not publish — the graph-surgery flow in
> [`model-compilation/`](../model-compilation/README.md) produces 4-channel l/t/r/b heads instead,
> which need `BoxDecodeType.YoloV26`. The zoo carries a large catalog of models — always check it
> before compiling something from scratch.

## References

- Core tutorials: <https://github.com/sima-neat/core/tree/main/tutorials>
- Apps examples: <https://github.com/sima-neat/apps/tree/main/examples>
- Public docs: <https://developer.sima.ai/software/tutorials>
