#!/usr/bin/env python3
"""T5 phase 1: compile_ready graph surgery for the T5 model set.

Copied and generalized from 09_yolo_compile_ready_surgery.py (frozen 01-12).
Handles three head families, all reduced to a stable set of raw-head ONNX
outputs so the MLA compiler never sees the CPU-side decode/NMS/TopK tail:

  * detection  (yolo11s)      -> 6 outputs:  bbox_{0,1,2}, class_logit_{0,1,2}
  * segmentation (yolo11s-seg) -> 10 outputs: the 6 detection tensors
        + mask_coeff_{0,1,2} (32 mask coefficients per scale)
        + proto (32x160x160 prototype masks, already graph output1)
  * pose (yolo26s-pose)       -> 9 outputs:  bbox_{0,1,2}, class_logit_{0,1,2}
        + kpt_{0,1,2} (51 = 17 keypoints x (x,y,visibility) per scale)

Shared surgery, identical to the proven yolo11n/yolo26n flow:
  1. attention MatMul -> Einsum rewrite (MLA-friendly batched attention),
  2. YOLO11 DFL bbox heads (16 bins) -> Split/Softmax/1x1-Conv/Concat = 4
     distance channels; YOLO26 heads already emit 4 distance channels,
  3. expose the raw head Conv outputs and DELETE the decode/postprocess tail
     (Concat/Reshape/Transpose/TopK/NMS) by replacing graph.output.

Node names were pinned from the actual exported ONNX (see reports/head_map.json
written by scripts/13 inspection); yolo11s head names are identical to yolo11n
and yolo26s-pose bbox/class names match yolo26n (one2one_* NMS-free head).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify


ROOT = Path(__file__).resolve().parents[1]
INPUT_NAME = "images"
INPUT_SHAPE = [1, 3, 640, 640]
SCALES = [("0", 80, 80), ("1", 40, 40), ("2", 20, 20)]

# dfl_bins=16 -> YOLO11 DFL bbox heads (cv2.*.2 emit 64 = 4*16 channels).
# dfl_bins=0  -> YOLO26 one2one heads already emit 4 distance channels.
YOLO_SPECS = {
    "yolo11s": {
        "attention_blocks": ["/model.10/m/m.0/attn"],
        "bbox_sources": [
            "/model.23/cv2.0/cv2.0.2/Conv_output_0",
            "/model.23/cv2.1/cv2.1.2/Conv_output_0",
            "/model.23/cv2.2/cv2.2.2/Conv_output_0",
        ],
        "class_sources": [
            "/model.23/cv3.0/cv3.0.2/Conv_output_0",
            "/model.23/cv3.1/cv3.1.2/Conv_output_0",
            "/model.23/cv3.2/cv3.2.2/Conv_output_0",
        ],
        "dfl_bins": 16,
        "extra_scale_heads": [],
        "passthrough_outputs": [],
    },
    "yolo11s-seg": {
        "attention_blocks": ["/model.10/m/m.0/attn"],
        "bbox_sources": [
            "/model.23/cv2.0/cv2.0.2/Conv_output_0",
            "/model.23/cv2.1/cv2.1.2/Conv_output_0",
            "/model.23/cv2.2/cv2.2.2/Conv_output_0",
        ],
        "class_sources": [
            "/model.23/cv3.0/cv3.0.2/Conv_output_0",
            "/model.23/cv3.1/cv3.1.2/Conv_output_0",
            "/model.23/cv3.2/cv3.2.2/Conv_output_0",
        ],
        "dfl_bins": 16,
        # mask coefficient head: cv4.*.2 -> 32 coeffs per scale.
        "extra_scale_heads": [
            {
                "name": "mask_coeff",
                "channels": 32,
                "sources": [
                    "/model.23/cv4.0/cv4.0.2/Conv_output_0",
                    "/model.23/cv4.1/cv4.1.2/Conv_output_0",
                    "/model.23/cv4.2/cv4.2.2/Conv_output_0",
                ],
            }
        ],
        # prototype masks: already graph output1 = [1,32,160,160].
        "passthrough_outputs": [
            {"name": "proto", "source": "output1", "shape": [1, 32, 160, 160]}
        ],
    },
    "yolo26s-pose": {
        "attention_blocks": ["/model.10/m/m.0/attn", "/model.22/m.0/m.0.1/attn"],
        "bbox_sources": [
            "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
            "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
            "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
        ],
        "class_sources": [
            "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
            "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
            "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
        ],
        "dfl_bins": 0,
        # keypoint head: 51 = 17 kpts * (x, y, visibility) per scale.
        "extra_scale_heads": [
            {
                "name": "kpt",
                "channels": 51,
                "sources": [
                    "/model.23/one2one_cv4_kpts.0/Conv_output_0",
                    "/model.23/one2one_cv4_kpts.1/Conv_output_0",
                    "/model.23/one2one_cv4_kpts.2/Conv_output_0",
                ],
            }
        ],
        "passthrough_outputs": [],
    },
}


def node_by_name(model, name):
    for node in model.graph.node:
        if node.name == name:
            return node
    return None


def all_node_outputs(model):
    return {output for node in model.graph.node for output in node.output}


def replace_node(model, old_name, new_node):
    for index, node in enumerate(model.graph.node):
        if node.name == old_name:
            model.graph.node.remove(node)
            model.graph.node.insert(index, new_node)
            return
    raise KeyError(old_name)


def replace_attention_matmuls(model, blocks):
    replaced = []
    for prefix in blocks:
        matmul0 = node_by_name(model, f"{prefix}/MatMul")
        matmul1 = node_by_name(model, f"{prefix}/MatMul_1")
        if matmul0 is None or matmul1 is None:
            continue
        replace_node(
            model,
            matmul0.name,
            helper.make_node(
                "Einsum",
                inputs=list(matmul0.input),
                outputs=list(matmul0.output),
                name=f"{prefix}/Einsum",
                equation="bhnc,bhck->bhnk",
            ),
        )
        replace_node(
            model,
            matmul1.name,
            helper.make_node(
                "Einsum",
                inputs=list(matmul1.input),
                outputs=list(matmul1.output),
                name=f"{prefix}/Einsum_1",
                equation="bhcn,bhnm->bhcm",
            ),
        )
        replaced.append(prefix)
    return replaced


def make_value_info(name, channels, height, width):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, channels, height, width])


def add_identity(nodes, source, output):
    nodes.append(helper.make_node("Identity", [source], [output], name=f"/sima_t5_heads/{output}/Identity"))


def add_yolo11_dfl(nodes, initializers, source, output, bins):
    split_outputs = [f"{output}_split_{idx}" for idx in range(4)]
    split_sizes = f"{output}_split_sizes"
    initializers.append(numpy_helper.from_array(np.asarray([bins, bins, bins, bins], dtype=np.int64), split_sizes))
    nodes.append(helper.make_node("Split", [source, split_sizes], split_outputs, name=f"{output}/Split", axis=1))

    conv_outputs = []
    weight_name = f"{output}_dfl_weight"
    weights = np.arange(bins, dtype=np.float32).reshape(1, bins, 1, 1)
    initializers.append(numpy_helper.from_array(weights, weight_name))
    for idx, split_output in enumerate(split_outputs):
        softmax = f"{output}_softmax_{idx}"
        distance = f"{output}_distance_{idx}"
        nodes.append(helper.make_node("Softmax", [split_output], [softmax], name=f"{output}/Softmax_{idx}", axis=1))
        nodes.append(helper.make_node("Conv", [softmax, weight_name], [distance], name=f"{output}/DflConv_{idx}"))
        conv_outputs.append(distance)
    nodes.append(helper.make_node("Concat", conv_outputs, [output], name=f"{output}/Concat", axis=1))


def expose_compile_ready_outputs(model, spec):
    available = all_node_outputs(model)
    required = [*spec["bbox_sources"], *spec["class_sources"]]
    for head in spec["extra_scale_heads"]:
        required += head["sources"]
    for pt in spec["passthrough_outputs"]:
        required.append(pt["source"])
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"missing head tensors: {missing}")

    new_nodes = []
    new_initializers = []
    outputs = []
    output_names = []

    # bbox heads
    for index, (suffix, height, width) in enumerate(SCALES):
        bbox_name = f"bbox_{suffix}"
        bbox_source = spec["bbox_sources"][index]
        if spec["dfl_bins"]:
            add_yolo11_dfl(new_nodes, new_initializers, bbox_source, bbox_name, spec["dfl_bins"])
        else:
            add_identity(new_nodes, bbox_source, bbox_name)
        outputs.append(make_value_info(bbox_name, 4, height, width))
        output_names.append(bbox_name)

    # class heads (channels inferred from graph shapes)
    inferred = onnx.shape_inference.infer_shapes(model)
    shape_by_name = {}
    for value in [*inferred.graph.value_info, *inferred.graph.output]:
        dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if dims:
            shape_by_name[value.name] = dims

    class_channels = None
    for index, (suffix, height, width) in enumerate(SCALES):
        class_name = f"class_logit_{suffix}"
        class_source = spec["class_sources"][index]
        shape = shape_by_name.get(class_source)
        if not shape or len(shape) != 4:
            raise ValueError(f"could not infer 4D class source shape for {class_source}: {shape}")
        class_channels = class_channels or int(shape[1])
        if int(shape[1]) != class_channels:
            raise ValueError(f"class channel mismatch for {class_source}: {shape[1]} != {class_channels}")
        add_identity(new_nodes, class_source, class_name)
        outputs.append(make_value_info(class_name, class_channels, height, width))
        output_names.append(class_name)

    # extra per-scale heads (mask coeffs, keypoints)
    for head in spec["extra_scale_heads"]:
        for index, (suffix, height, width) in enumerate(SCALES):
            name = f"{head['name']}_{suffix}"
            source = head["sources"][index]
            add_identity(new_nodes, source, name)
            outputs.append(make_value_info(name, head["channels"], height, width))
            output_names.append(name)

    # passthrough outputs (proto masks kept intact)
    for pt in spec["passthrough_outputs"]:
        add_identity(new_nodes, pt["source"], pt["name"])
        n, c, h, w = pt["shape"]
        outputs.append(make_value_info(pt["name"], c, h, w))
        output_names.append(pt["name"])

    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    del model.graph.output[:]
    model.graph.output.extend(outputs)
    return output_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=sorted(YOLO_SPECS), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    spec = YOLO_SPECS[args.model_id]
    base = ROOT / "work" / args.model_id
    source = base / "onnx" / f"{args.model_id}.onnx"
    output = base / "surgery" / f"{args.model_id}.compile_ready.onnx"
    report_dir = base / "reports"
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        status = "exists"
        replaced_attention = []
        output_names = [out.name for out in onnx.load(output).graph.output]
    else:
        graph = onnx.load(source)
        onnx.checker.check_model(graph)
        replaced_attention = replace_attention_matmuls(graph, spec["attention_blocks"])
        output_names = expose_compile_ready_outputs(graph, spec)
        simplified, ok = simplify(
            graph,
            overwrite_input_shapes={INPUT_NAME: INPUT_SHAPE},
            dynamic_input_shape=False,
        )
        if not ok:
            raise ValueError("ONNX simplification check failed")
        simplified = onnx.shape_inference.infer_shapes(simplified)
        onnx.checker.check_model(simplified)
        onnx.save(simplified, output)
        status = "exported"

    report = {
        "model_id": args.model_id,
        "status": status,
        "source": str(source),
        "output": str(output),
        "attention_rewrites": replaced_attention,
        "outputs": output_names,
        "num_outputs": len(output_names),
        "contract": "raw YOLO head outputs; CPU decode/NMS removed",
    }
    (report_dir / "compile_ready_surgery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
