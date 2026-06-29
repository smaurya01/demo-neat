#!/usr/bin/env python
"""Apply SiMa's C2PSA einsum-attention surgery to YOLO26n, then expose raw heads.

The on-device pyneat 0.2.0 Model loader needs the whole network on a coherent MLA
stage. YOLO26n's C2PSA attention (model.10, model.22) uses MatMul/Reshape that the
MLA can't run as-is, fragmenting the pack and breaking the preprocess planner. The
fix (the model-zoo `*_supported_einsum` recipe) rewrites the attention MatMuls as
4D Einsum and replaces the flatten-reshapes with Split/Concat so it stays 4D and
compiles onto the MLA.

We reuse the *exact* transformation from the official tool's yolo11 surgeon
(`SurgeonYoloVX.do_surgery`) by extracting its attention-block source and applying
it per C2PSA prefix — avoiding a hand-retype of ~300 lines of graph edits. Then we
expose the raw detection heads (box_*/class_logit_*) for BoxDecodeType.YoloV26.
"""
import argparse
import re
import textwrap

import numpy as np
import onnx
from onnxsim import simplify
import model_to_pipeline.utils.onnx_helpers as oh

SURGEON = "/workspace/tool-model-to-pipeline/model_to_pipeline/surgeons/surgeon_yolo11.py"
HEAD = "/model.23"

# C2PSA attention prefixes in the YOLO26n export (both: 2 heads, qkv=256, 20x20).
ATTN_BLOCKS = ["/model.10/m/m.0/attn", "/model.22/m.0/m.0.1/attn"]


class _Yolo:
    version = 11
    flavor = "n"


def _extract_attention_body() -> str:
    """Pull the attention transformation statements out of the official surgeon."""
    src = open(SURGEON).read().splitlines()
    start = end = None
    for i, ln in enumerate(src):
        if start is None and re.search(r'matmul1\s*=\s*f"\{model_prefix\}/MatMul"', ln):
            start = i
        if start is not None and "conv.input[0]=conc.output[0]" in ln:
            end = i
            break
    if start is None or end is None:
        raise RuntimeError(f"could not locate attention block ({start},{end})")
    return textwrap.dedent("\n".join(src[start:end + 1]))


def apply_einsum_attention(model, body: str, model_prefix: str, block: int, H: int, W: int):
    splits = 128 * np.ones(2)          # yolo*n: 2 heads x 128 ch (from _infer_model_flavor)
    ns = dict(model=model, model_prefix=model_prefix, block=block, H=H, W=W,
              splits=splits, yolo=_Yolo(), onnx=onnx, np=np, oh=oh)
    exec(compile(body, "<einsum_attn>", "exec"), ns)
    return ns["model"]


def expose_raw_heads(model, H, W):
    nc = oh.find_initializer_value(
        model, oh.find_node(model, f"{HEAD}/cv3.0/cv3.0.2/Conv").input[1]).shape[0]
    oh.remove_output(model)
    for i in range(3):
        s = 2 ** (i + 3)
        box = oh.find_node(model, f"{HEAD}/cv2.{i}/cv2.{i}.2/Conv")
        oh.add_output(model, f"box_{i}", (1, 4, H // s, W // s))
        oh.insert_node(model, box, oh.make_node(
            name=f"{HEAD}/raw/box{i}/Identity", op_type="Identity",
            inputs=[box.output[0]], outputs=[f"box_{i}"]), insert_only=True)
    for i in range(3):
        s = 2 ** (i + 3)
        cls = oh.find_node(model, f"{HEAD}/cv3.{i}/cv3.{i}.2/Conv")
        oh.add_output(model, f"class_logit_{i}", (1, nc, H // s, W // s))
        oh.insert_node(model, cls, oh.make_node(
            name=f"{HEAD}/raw/cls{i}/Identity", op_type="Identity",
            inputs=[cls.output[0]], outputs=[f"class_logit_{i}"]), insert_only=True)
    for d in ["Concat", "Reshape", "Concat_1", "Reshape_1", "Concat_2", "Reshape_2",
              "Concat_3", "Split", "Sigmoid", "Slice", "Slice_1", "Sub", "Add_1",
              "Add_2", "Sub_1", "Div_1", "Concat_4", "Mul_2", "Concat_5"]:
        try:
            oh.remove_node(model, f"{HEAD}/{d}", True)
        except Exception as e:
            print(f"  skip {d}: {e}")
    return nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="yolo26n.onnx")
    ap.add_argument("--out", default="yolo26n_einsum_raw.onnx")
    a = ap.parse_args()

    model = oh.load_model(a.model_path)
    in_shape = next(tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
                    for i in model.graph.input if not oh.is_initializer(model, i.name))
    H, W = in_shape[2], in_shape[3]
    print(f"Input {H}x{W}")

    body = _extract_attention_body()
    print(f"extracted attention body: {len(body.splitlines())} lines")
    for blk, prefix in enumerate(ATTN_BLOCKS):
        print(f"einsum attention -> {prefix} (block {blk})")
        model = apply_einsum_attention(model, body, prefix, blk, H, W)

    # re-simplify + shape infer after the attention rewrite
    model_s, ok = simplify(model)
    if ok:
        model = model_s
    oh.remove_infer_shape(model)
    model = onnx.shape_inference.infer_shapes(model)

    nc = expose_raw_heads(model, H, W)
    oh.save_model(model, a.out)
    print(f"saved {a.out} (num_classes={nc})")


if __name__ == "__main__":
    main()
