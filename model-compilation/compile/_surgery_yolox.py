#!/usr/bin/env python3
"""T5 phase 1: compile_ready surgery for YOLOX-s (Megvii, non-Ultralytics).

YOLOX has a decoupled, anchor-free head that is structurally different from the
Ultralytics YOLO family, so it needs its own surgery rather than the shared
14_t5_compile_ready_surgery.py path. Key differences documented for teaching:

  * No transformer attention block -> no MatMul->Einsum rewrite needed.
  * Decoupled head per scale: a reg branch (4 = cx,cy,w,h, raw), an obj branch
    (1, already Sigmoid-activated in the exported ONNX) and a cls branch
    (80, already Sigmoid-activated). They are Concat'd to [1,85,H,W] per scale.
  * No DFL: box regression is 4 raw channels decoded with grid+stride offsets
    on the CPU (YOLOX decode), analogous to the Ultralytics decode we strip.
  * The exported ONNX tail flattens the 3 scales:
        [1,85,H,W] --Reshape--> [1,85,N] --Concat--> [1,85,8400]
        --Transpose--> [1,8400,85]
    That transpose/reshape flatten is postprocess layout only; we cut it and
    expose the three per-scale [1,85,H,W] head tensors so the compiler keeps a
    clean NCHW conv-head boundary and the decode stays on the host.

Node/tensor names are pinned from the official 0.1.1rc0 yolox_s.onnx (opset 11).
To rediscover them on another export: trace back from graph output through the
final Transpose -> Concat -> per-scale Reshape -> per-scale Concat; the Concat
inputs at [1,85,80,80]/[1,85,40,40]/[1,85,20,20] are the head tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference
from onnxsim import simplify


ROOT = Path(__file__).resolve().parents[1]
INPUT_NAME = "images"
INPUT_SHAPE = [1, 3, 640, 640]

# Per-scale decoupled-head Concat outputs in the official yolox_s.onnx.
HEAD_TENSORS = [
    ("yolox_head_0", "798", 85, 80, 80),
    ("yolox_head_1", "824", 85, 40, 40),
    ("yolox_head_2", "850", 85, 20, 20),
]


def all_node_outputs(model):
    return {o for n in model.graph.node for o in n.output}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = ROOT / "work" / "yolox_s"
    source = base / "source" / "yolox_s.onnx"
    out_onnx = base / "onnx" / "yolox_s.onnx"
    output = base / "surgery" / "yolox_s.compile_ready.onnx"
    report_dir = base / "reports"
    for p in (base / "onnx", output.parent, report_dir):
        p.mkdir(parents=True, exist_ok=True)

    graph = onnx.load(source)
    onnx.checker.check_model(graph)
    # keep a normalized copy of the exported onnx in the standard onnx/ slot
    onnx.save(graph, out_onnx)

    available = all_node_outputs(graph)
    missing = [t for _, t, *_ in HEAD_TENSORS if t not in available]
    if missing:
        raise SystemExit(f"expected head tensors not found: {missing}")

    new_nodes = []
    new_outputs = []
    output_names = []
    for name, src, c, h, w in HEAD_TENSORS:
        new_nodes.append(helper.make_node("Identity", [src], [name], name=f"/sima_t5_heads/{name}/Identity"))
        new_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, c, h, w]))
        output_names.append(name)

    graph.graph.node.extend(new_nodes)
    del graph.graph.output[:]
    graph.graph.output.extend(new_outputs)

    simplified, ok = simplify(graph, overwrite_input_shapes={INPUT_NAME: INPUT_SHAPE}, dynamic_input_shape=False)
    if not ok:
        raise SystemExit("ONNX simplification check failed")
    simplified = shape_inference.infer_shapes(simplified)
    onnx.checker.check_model(simplified)
    onnx.save(simplified, output)

    report = {
        "model_id": "yolox_s",
        "status": "exported",
        "source": str(source),
        "output": str(output),
        "attention_rewrites": [],
        "outputs": output_names,
        "num_outputs": len(output_names),
        "contract": "YOLOX per-scale decoupled heads [1,85,H,W]; flatten/transpose tail removed; grid+stride decode on host",
        "head_layout": "channels 0:4 = reg(cx,cy,w,h raw), 4 = obj(sigmoid), 5:85 = cls(sigmoid)",
    }
    (report_dir / "compile_ready_surgery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
