#!/usr/bin/env python3
"""Prepare YOLO11n/YOLO26n ONNX for Neat YoloV26 decode and MLA compile.

The output graph keeps image input preprocessing outside ONNX and exposes six
model outputs in the order expected by Neat BoxDecodeType.YoloV26:

    bbox_0, bbox_1, bbox_2, class_logit_0, class_logit_1, class_logit_2

YOLO26 already emits four bbox distance channels per scale. YOLO11 emits DFL
bins, so this script converts each DFL bbox head to four distance channels with
Split -> Softmax -> 1x1 Conv -> Concat before exposing it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify
import yaml


ROOT = Path(__file__).resolve().parents[1]

YOLO_SPECS = {
    "yolo11n": {
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
    },
    "yolo26n": {
        "attention_blocks": [
            "/model.10/m/m.0/attn",
            "/model.22/m.0/m.0.1/attn",
        ],
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
    },
}

SCALES = [("0", 80, 80), ("1", 40, 40), ("2", 20, 20)]


def model_cfg(model_id: str) -> dict:
    reg = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    for model in reg["models"]:
        if model["id"] == model_id:
            return model
    raise KeyError(model_id)


def node_by_name(model: onnx.ModelProto, name: str) -> onnx.NodeProto | None:
    for node in model.graph.node:
        if node.name == name:
            return node
    return None


def all_node_outputs(model: onnx.ModelProto) -> set[str]:
    return {output for node in model.graph.node for output in node.output}


def replace_node(model: onnx.ModelProto, old_name: str, new_node: onnx.NodeProto) -> None:
    for index, node in enumerate(model.graph.node):
        if node.name == old_name:
            model.graph.node.remove(node)
            model.graph.node.insert(index, new_node)
            return
    raise KeyError(old_name)


def replace_attention_matmuls(model: onnx.ModelProto, blocks: list[str]) -> list[str]:
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


def make_value_info(name: str, channels: int, height: int, width: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, channels, height, width])


def add_identity(nodes: list[onnx.NodeProto], source: str, output: str) -> None:
    nodes.append(helper.make_node("Identity", [source], [output], name=f"/sima_yolo_heads/{output}/Identity"))


def add_yolo11_dfl(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    source: str,
    output: str,
    bins: int,
) -> None:
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


def expose_compile_ready_outputs(model: onnx.ModelProto, spec: dict) -> list[str]:
    available = all_node_outputs(model)
    required = [*spec["bbox_sources"], *spec["class_sources"]]
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"missing YOLO head tensors: {missing}")

    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    outputs: list[onnx.ValueInfoProto] = []
    output_names: list[str] = []

    for index, (suffix, height, width) in enumerate(SCALES):
        bbox_name = f"bbox_{suffix}"
        bbox_source = spec["bbox_sources"][index]
        if spec["dfl_bins"]:
            add_yolo11_dfl(new_nodes, new_initializers, bbox_source, bbox_name, spec["dfl_bins"])
        else:
            add_identity(new_nodes, bbox_source, bbox_name)
        outputs.append(make_value_info(bbox_name, 4, height, width))
        output_names.append(bbox_name)

    class_channels = None
    inferred = onnx.shape_inference.infer_shapes(model)
    shape_by_name = {}
    for value in [*inferred.graph.value_info, *inferred.graph.output]:
        dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if dims:
            shape_by_name[value.name] = dims
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

    model = model_cfg(args.model_id)
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
            overwrite_input_shapes={model["input_name"]: model["input_shape"]},
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
        "contract": "Neat BoxDecodeType.YoloV26 grouped bbox/class-logit outputs",
    }
    (report_dir / "compile_ready_surgery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
