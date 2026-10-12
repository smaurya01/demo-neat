#!/usr/bin/env python3
"""Prepare YOLO26n ONNX for BF16 MLA compile without external tool repos.

This script performs the two local graph edits needed by this app:

1. Replace C2PSA attention MatMul nodes with equivalent 4D Einsum nodes.
2. Expose raw YOLO head tensors for Neat BoxDecodeType.YoloV26.

It intentionally depends only on standard Model SDK Python packages:
`onnx`, `onnxsim`, and `numpy`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify


HEAD = "/model.23"
ATTN_BLOCKS = ["/model.10/m/m.0/attn", "/model.22/m.0/m.0.1/attn"]


def load_model(path: str | Path) -> onnx.ModelProto:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return model


def save_model(model: onnx.ModelProto, path: str | Path) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def node_by_name(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if node.name == name:
            return node
    raise KeyError(f"node not found: {name}")


def initializer_array(model: onnx.ModelProto, name: str) -> np.ndarray:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(f"initializer not found: {name}")


def input_shape(model: onnx.ModelProto) -> tuple[int, int]:
    for value in model.graph.input:
        if any(init.name == value.name for init in model.graph.initializer):
            continue
        dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if len(dims) != 4 or any(dim <= 0 for dim in dims):
            raise ValueError(f"expected static NCHW image input, got {value.name}: {dims}")
        return int(dims[2]), int(dims[3])
    raise ValueError("no non-initializer graph input found")


def make_value_info(name: str, shape: tuple[int, ...]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def replace_node(model: onnx.ModelProto, old_name: str, new_node: onnx.NodeProto) -> None:
    nodes = model.graph.node
    for index, node in enumerate(nodes):
        if node.name == old_name:
            nodes.remove(node)
            nodes.insert(index, new_node)
            return
    raise KeyError(f"node not found: {old_name}")


def remove_graph_outputs(model: onnx.ModelProto) -> None:
    del model.graph.output[:]


def append_identity_output(
    model: onnx.ModelProto,
    source_tensor: str,
    output_name: str,
    shape: tuple[int, ...],
) -> None:
    identity = helper.make_node(
        "Identity",
        inputs=[source_tensor],
        outputs=[output_name],
        name=f"{HEAD}/raw/{output_name}/Identity",
    )
    model.graph.node.append(identity)
    model.graph.output.append(make_value_info(output_name, shape))


def remove_inferred_value_info(model: onnx.ModelProto) -> None:
    del model.graph.value_info[:]


def replace_attention_matmuls(model: onnx.ModelProto) -> None:
    for prefix in ATTN_BLOCKS:
        matmul0 = node_by_name(model, f"{prefix}/MatMul")
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

        matmul1 = node_by_name(model, f"{prefix}/MatMul_1")
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


def head_prefix(model: onnx.ModelProto) -> str:
    """Pick the YOLO26 head branch to expose.

    `one2one_*` is the end-to-end, NMS-free head: Neat's YoloV26 decoder is built for it and
    (since Neat 0.5.0) runs no NMS. An ONNX exported with end2end=False contains only the
    one-to-many `cv2/cv3` head; that still decodes as YoloV26 but returns several overlapping
    boxes per object, and the app runs no NMS. Re-export the .pt with the end-to-end head.
    """
    names = {node.name for node in model.graph.node}
    if f"{HEAD}/one2one_cv2.0/one2one_cv2.0.2/Conv" in names:
        return "one2one_"
    print("[surgery] WARNING: no one2one head in this ONNX (exported with end2end=False); "
          "exposing the one-to-many head. The app runs no NMS, so each defect will show several "
          "overlapping boxes. Re-export the .pt with the end-to-end head (see README).")
    return ""


def expose_raw_heads(model: onnx.ModelProto, image_h: int, image_w: int) -> int:
    prefix = head_prefix(model)
    class_weight = initializer_array(model, f"model.23.{prefix}cv3.0.2.weight")
    num_classes = int(class_weight.shape[0])

    remove_graph_outputs(model)
    box_outputs = []
    class_outputs = []

    for index in range(3):
        stride = 2 ** (index + 3)
        grid_h = image_h // stride
        grid_w = image_w // stride

        box = node_by_name(model, f"{HEAD}/{prefix}cv2.{index}/{prefix}cv2.{index}.2/Conv")
        cls = node_by_name(model, f"{HEAD}/{prefix}cv3.{index}/{prefix}cv3.{index}.2/Conv")

        box_name = f"bbox_{index}"
        cls_name = f"class_logit_{index}"
        append_identity_output(model, box.output[0], box_name, (1, 4, grid_h, grid_w))
        append_identity_output(model, cls.output[0], cls_name, (1, num_classes, grid_h, grid_w))
        box_outputs.append(box_name)
        class_outputs.append(cls_name)

    outputs_by_name = {output.name: output for output in model.graph.output}
    remove_graph_outputs(model)
    for name in [*box_outputs, *class_outputs]:
        model.graph.output.append(outputs_by_name[name])

    prune_unused_nodes(model)
    return num_classes


def prune_unused_nodes(model: onnx.ModelProto) -> None:
    """Drop every node the exposed raw heads do not depend on (the CPU decode/NMS tail).

    Works for both export variants: the end-to-end export and the end2end=False export have
    different decode tails, so a fixed node list would leave dangling nodes in one of them.
    """
    needed = {output.name for output in model.graph.output}
    keep = []
    for node in reversed(list(model.graph.node)):
        if any(out in needed for out in node.output):
            keep.append(node)
            needed.update(name for name in node.input if name)
    keep.reverse()
    del model.graph.node[:]
    model.graph.node.extend(keep)


def simplify_and_infer(model: onnx.ModelProto) -> onnx.ModelProto:
    simplified, ok = simplify(model)
    if ok:
        model = simplified
    remove_inferred_value_info(model)
    return onnx.shape_inference.infer_shapes(model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO26n ONNX for BF16 MLA compile.")
    parser.add_argument("--model_path", default="yolo26n.onnx")
    parser.add_argument("--out", default="yolo26n_einsum_raw.onnx")
    args = parser.parse_args()

    model = load_model(args.model_path)
    image_h, image_w = input_shape(model)
    print(f"input: {image_h}x{image_w}")

    replace_attention_matmuls(model)
    print("attention: MatMul -> Einsum")

    model = simplify_and_infer(model)
    num_classes = expose_raw_heads(model, image_h, image_w)
    model = simplify_and_infer(model)

    save_model(model, args.out)
    print(f"saved: {args.out} (num_classes={num_classes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
