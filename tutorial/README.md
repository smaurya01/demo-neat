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
- [References](#references)

---

## Running On The DevKit

Run these **on the DevKit board**, so the notebook kernel can import and execute `pyneat`.
When the SDK is paired with the board, `/workspace` is NFS-mounted on the board at the same path
(check with `dk status` in the SDK: its `NFS mount path` line), so you can edit host-side and run
board-side with no copying. Without the mount, copy this folder to the board first.

Activate the pyneat environment, install Jupyter if needed, and start the notebook server:

```bash
ssh sima@<devkit-ip>

source $HOME/pyneat/bin/activate

python -m pip install notebook   # only once, if Jupyter is missing

cd /workspace/demo-neat   # your clone of this repo under /workspace
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

Then open the notebook URL the server prints, from your machine:

```text
http://modalix:8888/tree?token=<token>
```

If `modalix` does not resolve on your network, use the DevKit's IP instead:
`http://<devkit-ip>:8888/tree?token=<token>`.

## I — Easy

The core objects. Nothing here needs a camera.

| # | Notebook | What you learn |
| --- | --- | --- |
| 1 | [`01_neat_tensor.ipynb`](I-easy/01_neat_tensor.ipynb) | `Tensor` — shape, layout, memory, planes |
| 2 | [`02_node_and_graph.ipynb`](I-easy/02_node_and_graph.ipynb) | `Node` and `Graph` — how a pipeline is assembled |
| 3 | [`03_interpret_model_output_samples.ipynb`](I-easy/03_interpret_model_output_samples.ipynb) | `Sample` — reading what a model actually returned |
| 4 | [`04_image_classification_resnet.ipynb`](I-easy/04_image_classification_resnet.ipynb) | A first end-to-end model: ResNet-50 classification |
| 5 | [`05_yolo_cpu_decode.ipynb`](I-easy/05_yolo_cpu_decode.ipynb) | Parsing Neat's decoded BBOX payload on the CPU (count + 24-byte records) |
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

Notebooks 2, 5, 6 and 7 need **Neat Insight**: 2 and 7 read an Insight RTSP source, and 5, 6 and 7
send video or metadata to it. Before running them, replace the `<sdk-host-ip>` placeholder in the
notebook's settings cell with the Insight host's address as seen from the DevKit
(the IP on the **Insight (remote)** line of `neat`, or `curl -sk https://127.0.0.1:9900/api/server-ip`
on the SDK host; never `127.0.0.1`). See
[`appendix.md`](../appendix.md#1-neat-insight-sources-in-video-out) for serving an RTSP source.
Notebook 7 also pairs with [`installation/neat_insight.md`](../installation/neat_insight.md).

## III — Advanced

Composition patterns for stages that sit beside the rest of a graph.

| # | Notebook | What you learn |
| --- | --- | --- |
| 1 | [`01_genai_model_in_graph.ipynb`](III-advance/01_genai_model_in_graph.ipynb) | `neat.genai.graphs.vision_language` — put a VLM/LLM in a `Graph` as a stage with named `prompt`/`image` inputs and streamed `tokens`/`done` outputs |

Notebook 1 needs a deployed LLiMa VLM on the DevKit. Its `MODEL_DIR` defaults to
`/media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4`; deploy it with
`llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4`, or point `MODEL_DIR` at another deployed VLM. It mirrors the core
tutorial `022_compose_genai_into_graph` and the public
[*Compose GenAI into a Graph*](https://developer.sima.ai/software/tutorials/compose-genai-into-graph)
page. For a worked example of the direct-model-handle alternative, see the
[`detection-vlm-assistant`](../apps/detection-vlm-assistant/README.md) app.

## Next

Once the concepts land, move to a complete application in [`apps/`](../apps/README.md) — start with
[`single-stream-yolo-yolo11`](../apps/single-stream-yolo-yolo11/README.md).

## Assets

| Path | What |
| --- | --- |
| `assets/images/` | Sample images used by the notebooks |
| `assets/imagenet_labels.txt` | ImageNet class names (notebook `I-easy/04`) |
| `assets/coco_labels.txt` | COCO class names (notebooks `I-easy/05`, `06`, `II-medium/07`) |
| `assets/models/` | Where you put model archives. **Git-ignored** — you download or build these. |

Model paths are variables in each notebook's settings cell (in `II-medium/01`, the cell that loads the
model, near the end). Update them to match what is on your DevKit.

## Getting the models

The notebooks need three archives from the SiMa Model Zoo. Run this once, from the `tutorial/`
folder **on the DevKit**:

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v 2.1.3 --boardtype modalix get resnet_50   # I-easy/04
sima-cli modelzoo -v 2.1.3 --boardtype modalix get yolo_v8s    # I-easy/05, I-easy/06
sima-cli modelzoo -v 2.1.3 --boardtype modalix get yolo_v8n    # II-medium/01, II-medium/07
cd ../..
```

That gives `assets/models/resnet_50_mpk.tar.gz`, `yolo_v8s_mpk.tar.gz` and `yolo_v8n_mpk.tar.gz`.
If a `sima-cli` update prompt aborts the download, set `SIMA_CLI_CHECK_FOR_UPDATE=0` first.

> **Before compiling a model yourself, check the zoo**: it carries a large catalog, including YOLO11
> (`yolo_11n/s/m/l/x`). Which Neat box decoder each YOLO family needs:
> [`model-compilation/BOX-DECODE.md`](../model-compilation/BOX-DECODE.md).

## References

- Core tutorials: <https://github.com/sima-neat/core/tree/main/tutorials>
- Apps examples: <https://github.com/sima-neat/apps/tree/main/examples>
- Public docs: <https://developer.sima.ai/software/tutorials>
