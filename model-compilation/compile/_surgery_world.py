#!/usr/bin/env python3
"""compile_ready graph surgery for YOLO-World (yolov8-worldv2, fixed vocabulary).

YOLO-World's WorldDetect head differs from a plain YOLO detect head only in how the
CLASS logits are produced. After set_classes() bakes the vocabulary, each scale's
head is:

    bbox_i  = cv2.i.2 Conv                                 # 64ch raw DFL (4 x 16 bins)
    class_i = Einsum(cv3.i.2 embed, text_norm) * exp(s) + b # contrastive -> nc logits

where `text_norm` (normalized text features, [1, nc, embed]), `exp(s)` (logit scale)
and `b` (bias) are all CONSTANTS once the vocabulary is fixed. That makes the whole
contrastive step an affine map over the channel dim, i.e. exactly a 1x1 convolution:

    class_i[k,h,w] = sum_c embed[c,h,w] * (text_norm[k,c] * exp(s)) + b

We FOLD it into a real Conv1x1 (weight = text_norm * exp(s), bias = b). This is the
single most important move for a single-ELF result: it replaces an Einsum whose
contraction layout (bchw,bkc->bkhw) is not a native MLA placement with a 1x1 Conv,
which the MLA places trivially. The math is identical.

bbox is the yolov8/yolo11 16-bin DFL head, exposed RAW (64 channels), same as the yolo11
detection path: Neat's YoloV8 decoder integrates the DFL and runs NMS. (Rebuilding it into
4 l/t/r/b channels would force the YoloV26 decoder, which skips NMS since Neat 0.5.0.)

Output: 6 raw heads  bbox_{0,1,2} (64ch) + class_logit_{0,1,2} (nc ch), decode/NMS tail
removed -> the whole graph stays on the MLA. Decode: BoxDecodeType.YoloV8 with
BoxDecodeTypeOption.GroupedByRoleLogit (class heads are logits).

    python compile/_surgery_world.py --model-id yolov8s-worldv2
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
HEAD = "/model.22"
DFL_BINS = 16


def get_init(inits, name):
    if name not in inits:
        raise KeyError(f"initializer not found: {name}")
    return numpy_helper.to_array(inits[name])


def build(model_id: str, force: bool) -> dict:
    base = ROOT / "work" / model_id
    source = base / "onnx" / f"{model_id}.onnx"
    output = base / "surgery" / f"{model_id}.compile_ready.onnx"
    report_dir = base / "reports"
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    graph = onnx.load(source)
    onnx.checker.check_model(graph)
    g = graph.graph
    inits = {i.name: i for i in g.initializer}

    new_nodes, new_inits, outputs, output_names = [], [], [], []

    # ---- bbox heads: raw 64-channel DFL, decoded by Neat's YoloV8 ---------------
    for suffix, h, w in SCALES:
        src = f"{HEAD}/cv2.{suffix}/cv2.{suffix}.2/Conv_output_0"
        name = f"bbox_{suffix}"
        new_nodes.append(helper.make_node("Identity", [src], [name], name=f"/sima_world_heads/{name}/Identity"))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4 * DFL_BINS, h, w]))
        output_names.append(name)

    # ---- class heads: fold contrastive (Einsum*scale+bias) -> Conv1x1 ---------
    nc = None
    for suffix, h, w in SCALES:
        embed_src = f"{HEAD}/cv3.{suffix}/cv3.{suffix}.2/Conv_output_0"
        text = get_init(inits, f"{HEAD}/cv4.{suffix}/Div_output_0")   # [1, nc, embed] normalized text
        scale = float(get_init(inits, f"{HEAD}/cv4.{suffix}/Exp_output_0"))  # scalar exp(logit_scale)
        bias = float(get_init(inits, f"model.22.cv4.{suffix}.bias"))         # scalar bias
        text = np.asarray(text, dtype=np.float32)
        if text.ndim == 3:
            text = text[0]                          # [nc, embed]
        nc_i, embed = text.shape
        nc = nc or nc_i
        if nc_i != nc:
            raise ValueError(f"class count mismatch scale {suffix}: {nc_i} != {nc}")
        conv_w = (text * scale).reshape(nc_i, embed, 1, 1).astype(np.float32)   # [nc, embed, 1, 1]
        conv_b = np.full((nc_i,), bias, dtype=np.float32)
        w_name, b_name = f"class_logit_{suffix}_w", f"class_logit_{suffix}_b"
        new_inits.append(numpy_helper.from_array(conv_w, w_name))
        new_inits.append(numpy_helper.from_array(conv_b, b_name))
        name = f"class_logit_{suffix}"
        new_nodes.append(helper.make_node(
            "Conv", [embed_src, w_name, b_name], [name],
            name=f"/sima_world_heads/{name}/Conv",
            kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], group=1))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, nc_i, h, w]))
        output_names.append(name)

    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)
    del g.output[:]
    g.output.extend(outputs)

    simplified, ok = simplify(graph, overwrite_input_shapes={INPUT_NAME: INPUT_SHAPE},
                              dynamic_input_shape=False)
    if not ok:
        raise ValueError("ONNX simplification check failed")
    simplified = onnx.shape_inference.infer_shapes(simplified)
    onnx.checker.check_model(simplified)
    onnx.save(simplified, output)

    report = {
        "model_id": model_id,
        "source": str(source),
        "output": str(output),
        "outputs": output_names,
        "num_outputs": len(output_names),
        "num_classes": nc,
        "contrastive": "folded Einsum*scale+bias -> Conv1x1 (weight=text_norm*exp(logit_scale))",
        "bbox": f"raw DFL {DFL_BINS}-bin heads ({4 * DFL_BINS} ch)",
        "contract": "raw YOLO-World head outputs; CPU decode/NMS removed",
    }
    (report_dir / "compile_ready_surgery.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="yolov8s-worldv2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    report = build(args.model_id, args.force)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
