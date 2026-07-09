#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxsim import simplify
import yaml


ROOT = Path(__file__).resolve().parents[1]


def model_cfg(model_id):
    reg = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    for model in reg["models"]:
        if model["id"] == model_id:
            return model
    raise KeyError(model_id)


def constant_array(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
    return None


def make_i64_const(name, values):
    tensor = helper.make_tensor(name, TensorProto.INT64, [len(values)], [int(v) for v in values])
    return helper.make_node("Constant", inputs=[], outputs=[name], value=tensor, name=f"{name}_const")


def tensor_shapes(model):
    shapes = {}
    value_infos = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in value_infos:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") else None)
        shapes[value.name] = dims
    return shapes


def replace_constant_gather_with_slice(model):
    changed = 0
    shapes = tensor_shapes(model)
    new_nodes = []
    for node in model.graph.node:
        if node.op_type != "Gather" or len(node.input) < 2:
            new_nodes.append(node)
            continue

        indices = constant_array(model, node.input[1])
        if indices is None or indices.size != 1:
            new_nodes.append(node)
            continue

        idx = int(indices.reshape(-1)[0])

        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(helper.get_attribute_value(attr))

        if idx < 0:
            input_shape = shapes.get(node.input[0], [])
            rank = len(input_shape)
            normalized_axis = axis if axis >= 0 else axis + rank
            dim = input_shape[normalized_axis] if 0 <= normalized_axis < rank else None
            if dim is None:
                new_nodes.append(node)
                continue
            idx += dim

        prefix = (node.name or node.output[0]).replace("/", "_").strip("_")
        starts = f"{prefix}_slice_starts"
        ends = f"{prefix}_slice_ends"
        axes = f"{prefix}_slice_axes"
        steps = f"{prefix}_slice_steps"
        slice_out = f"{prefix}_slice_output"

        new_nodes.extend([
            make_i64_const(starts, [idx]),
            make_i64_const(ends, [idx + 1]),
            make_i64_const(axes, [axis]),
            make_i64_const(steps, [1]),
        ])

        indices_is_scalar = indices.shape == ()
        if indices_is_scalar:
            squeeze_axes = f"{prefix}_squeeze_axes"
            new_nodes.append(
                helper.make_node(
                    "Slice",
                    inputs=[node.input[0], starts, ends, axes, steps],
                    outputs=[slice_out],
                    name=f"{prefix}_slice",
                )
            )
            new_nodes.append(make_i64_const(squeeze_axes, [axis]))
            new_nodes.append(
                helper.make_node(
                    "Squeeze",
                    inputs=[slice_out, squeeze_axes],
                    outputs=list(node.output),
                    name=f"{prefix}_squeeze",
                )
            )
        else:
            new_nodes.append(
                helper.make_node(
                    "Slice",
                    inputs=[node.input[0], starts, ends, axes, steps],
                    outputs=list(node.output),
                    name=f"{prefix}_slice",
                )
            )
        changed += 1

    if changed:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    model = model_cfg(args.model_id)
    base = ROOT / "work" / model["id"]
    src = base / "onnx" / f"{model['id']}.onnx"
    dst = base / "surgery" / f"{model['id']}.surgery.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    report_dir = base / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    notes = []
    try:
        simplified, ok = simplify(
            str(src),
            overwrite_input_shapes={model["input_name"]: model["input_shape"]},
            dynamic_input_shape=False,
        )
        if ok:
            candidate = simplified
            notes.append("Applied fixed-shape ONNX simplification.")
        else:
            candidate = onnx.load(src)
            notes.append("ONNX simplification returned check=False; used original ONNX.")
    except Exception as exc:
        candidate = onnx.load(src)
        notes.append(f"ONNX simplification failed; used original ONNX. Error: {exc}")

    onnx.checker.check_model(candidate)
    replaced = replace_constant_gather_with_slice(candidate)
    if replaced:
        notes.append(f"Replaced {replaced} constant-index Gather node(s) with Slice/Squeeze.")
    else:
        notes.append("No constant-index Gather nodes were replaced.")

    onnx.checker.check_model(candidate)
    onnx.save(candidate, dst)

    report = {
        "model_id": model["id"],
        "source": str(src),
        "output": str(dst),
        "surgery_applied": True,
        "constant_gather_replacements": replaced,
        "notes": notes,
    }
    (report_dir / "surgery_notes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
