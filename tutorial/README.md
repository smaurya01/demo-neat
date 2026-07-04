# Neat Tutorial Notebooks

This folder contains a tutorial sequence for learning Neat application concepts with runnable Python notebook examples.

## Running On The DevKit

Run these notebooks from the SiMa DevKit board so the notebook kernel can import and execute `pyneat`.

Activate the pyneat environment, install Jupyter if needed, and start the notebook server:

```bash
source $HOME/pyneat/bin/activate
python -m pip install notebook
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

Then open the DevKit notebook URL from your local system, for example:

```text
http://<devkit-ip>:8888/tree
```

## I - Easy

1. Neat Tensor - `I-easy/01_neat_tensor.ipynb`
2. Node and Graph - `I-easy/02_node_and_graph.ipynb`
3. Interpret Model Output Samples - `I-easy/03_interpret_model_output_samples.ipynb`
4. Image Classification with ResNet-50 - `I-easy/04_image_classification_resnet.ipynb`
5. YOLO CPU Decode - `I-easy/05_yolo_cpu_decode.ipynb`
6. YOLOv8 Image Detection Pipeline - `I-easy/06_yolov8_image_detection_pipeline.ipynb`

## Assets

- Images: `assets/images/`
- ImageNet labels: `assets/imagenet_labels.txt`
- COCO labels: `assets/coco_labels.txt`
- Optional model packages: `assets/models/`

Model paths are variables near the top of notebooks. Update them to match model packages available on the DevKit.

### Download ResNet-50 For Notebook 04

Notebook `I-easy/04_image_classification_resnet.ipynb` uses the Model Zoo classification model `resnet_50`.

Run this once from the `tutorial` folder on the DevKit:

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get resnet_50
cd ../..
```

Expected model path:

```text
assets/models/resnet_50_mpk.tar.gz
```

If your DevKit uses a different platform version, replace `2.1.2` with that release version.

### Download YOLOv8s For Notebooks 05 and 06

Notebooks `I-easy/05_yolo_cpu_decode.ipynb` and `I-easy/06_yolov8_image_detection_pipeline.ipynb` use the Model Zoo detection model `yolo_v8s`.

Run this once from the `tutorial` folder on the DevKit:

```bash
mkdir -p assets/models
cd assets/models
sima-cli modelzoo -v 2.1.2 --boardtype modalix get yolo_v8s
cd ../..
```

Expected model path:

```text
assets/models/yolo_v8s_mpk.tar.gz
```

If your DevKit uses a different platform version, replace `2.1.2` with that release version.

## References

- Core tutorials: [https://github.com/sima-neat/core/tree/main/tutorials](https://github.com/sima-neat/core/tree/main/tutorials)
- Apps examples: [https://github.com/sima-neat/apps/tree/main/examples](https://github.com/sima-neat/apps/tree/main/examples)
- Public docs: [https://developer.sima.ai/software/tutorials](https://developer.sima.ai/software/tutorials)
