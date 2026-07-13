#!/usr/bin/env python3
"""T5 phase 1: compile_ready surgery for YOLOX-s (Megvii, non-Ultralytics).

YOLOX has a decoupled, anchor-free head that is structurally different from the
Ultralytics YOLO family, so it needs its own surgery rather than the shared
14_t5_compile_ready_surgery.py path. Key differences documented for teaching:

  * No transformer attention block -> no MatMul->Einsum rewrite needed.
  * Decoupled head per scale: a reg branch (4 = cx,cy,w,h, raw), an obj branch
    (1) and a cls branch (80). In the exported ONNX the obj/cls branches are
    Sigmoid-activated and all three are Concat'd to [1,85,H,W] per scale.
  * No DFL: box regression is 4 raw channels decoded with grid+stride offsets.
  * The exported ONNX tail flattens the 3 scales:
        [1,85,H,W] --Reshape--> [1,85,N] --Concat--> [1,85,8400]
        --Transpose--> [1,8400,85]
    That transpose/reshape flatten is postprocess layout only, and so is the
    per-scale Concat. We cut both.

WHY THE HEADS ARE EXPOSED SPLIT, NOT PACKED
-------------------------------------------
Neat's on-device decoder (`BoxDecodeType::YoloX`) does not accept a packed
[1,85,H,W] head. Its contract is three SEPARATE tensors per scale, interleaved
scale-major -- (bbox, obj, cls) x 3 -- with depths (4, 1, 80). See
`infer_yolox_interleaved_class_depth` in core/src/pipeline/internal/sima/
stagesemantics/BoxDecodeStageSemantics.cpp, which rejects the contract outright
unless the tensor count is a multiple of 3 and the depths are exactly 4/1/N.
An earlier version of this surgery exposed the packed Concat output, which meant
YOLOX could not use the Neat decoder at all and had to be decoded in NumPy on
the host -- the single biggest cost in the quad-stream app.

We expose the PRE-Sigmoid obj/cls logits, not the exported Sigmoid outputs,
because Neat forces `score_activation = Sigmoid` for the YoloX family
(`apply_raw_yolov6_yolox_compiled_payload_overrides`). Handing it already-
activated scores would apply sigmoid twice and silently corrupt every score.
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

# Split decoupled-head branches in the official yolox_s.onnx, pinned by tensor name.
#
# Per scale the exported graph is:
#     reg  Conv    -> 794 / 820 / 846        [1, 4,H,W]   raw cx,cy,w,h
#     obj  Conv    -> 795 / 821 / 847        [1, 1,H,W]   LOGIT (Sigmoid consumes it)
#     cls  Conv    -> 785 / 811 / 837        [1,80,H,W]   LOGIT (Sigmoid consumes it)
#     Concat(reg, Sigmoid(obj), Sigmoid(cls)) -> 798 / 824 / 850   [1,85,H,W]
#
# We take the three branch tensors and drop the Concat. The obj/cls names are the
# Sigmoid *inputs*, so the compiled archive carries logits and Neat applies the
# sigmoid itself (see module docstring).
#
# Emitted scale-major as (bbox, obj, cls) triplets -- the order YoloX decode expects.
#
# To rediscover on another export: walk back from each graph output through
# Transpose -> Concat -> per-scale Reshape -> per-scale Concat, then take that
# Concat's three inputs; follow the 2nd and 3rd back through their Sigmoid.
HEAD_SCALES = [
    # (suffix, H, W, reg_tensor, obj_logit_tensor, cls_logit_tensor)
    ("0", 80, 80, "794", "795", "785"),
    ("1", 40, 40, "820", "821", "811"),
    ("2", 20, 20, "846", "847", "837"),
]
REG_C, OBJ_C, CLS_C = 4, 1, 80


def all_node_outputs(model):
    return {o for n in model.graph.node for o in n.output}


def main() -> int:
    parser = argparse.ArgumentParser()
    # graph_surgery.py always passes --model-id (see its `cmd` construction), so this
    # MUST accept it. It previously did not, and yolox_s failed instantly with
    # "unrecognized arguments: --model-id yolox_s" — i.e. it could not be compiled at
    # all through the documented flow.
    parser.add_argument("--model-id", default="yolox_s", choices=["yolox_s"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = ROOT / "work" / args.model_id
    # Read the ONNX that step 1 produced. convert_to_onnx.py downloads Megvii's
    # pre-exported graph straight to work/<id>/onnx/<id>.onnx — there is no source/
    # directory in this flow. Reading from a non-existent source/ was the second half
    # of the bug above.
    source = base / "onnx" / f"{args.model_id}.onnx"
    output = base / "surgery" / f"{args.model_id}.compile_ready.onnx"
    report_dir = base / "reports"
    for p in (output.parent, report_dir):
        p.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise SystemExit(f"missing ONNX export: {source} — run convert_to_onnx.py first")

    graph = onnx.load(source)
    onnx.checker.check_model(graph)

    available = all_node_outputs(graph)
    wanted = [t for _, _, _, *tensors in HEAD_SCALES for t in tensors]
    missing = [t for t in wanted if t not in available]
    if missing:
        raise SystemExit(f"expected head tensors not found: {missing}")

    new_nodes = []
    new_outputs = []
    output_names = []
    # Scale-major (bbox, obj, cls) triplets: BoxDecodeType::YoloX / Split3Interleaved.
    for suffix, h, w, reg_t, obj_t, cls_t in HEAD_SCALES:
        for role, src, depth in (
            (f"bbox_{suffix}", reg_t, REG_C),
            (f"obj_logit_{suffix}", obj_t, OBJ_C),
            (f"class_logit_{suffix}", cls_t, CLS_C),
        ):
            new_nodes.append(
                helper.make_node("Identity", [src], [role], name=f"/sima_t5_heads/{role}/Identity")
            )
            new_outputs.append(
                helper.make_tensor_value_info(role, TensorProto.FLOAT, [1, depth, h, w])
            )
            output_names.append(role)

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
        "contract": (
            "YOLOX split decoupled heads, scale-major (bbox,obj,cls) triplets with depths "
            "(4,1,80); per-scale Concat and flatten/transpose tail removed. Matches "
            "BoxDecodeType::YoloX (Split3Interleaved) -> on-device Neat decode."
        ),
        "head_layout": "bbox_i = reg(cx,cy,w,h raw); obj_logit_i, class_logit_i = PRE-sigmoid logits (Neat applies sigmoid)",
    }
    (report_dir / "compile_ready_surgery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
