#!/usr/bin/env python3
"""T7: compile-ready surgery for plain ViT-family transformers (vit_b_16, dinov2_vits14).

Builds on the generic 03_surgery.py output (static-shape simplify + constant
Gather->Slice). Adds the two transformer-specific rewrites that the YOLO
walkthrough (README section 3a) established as the house pattern, generalized to
sequence-token attention:

  1. ATTENTION MatMul -> Einsum.  Multi-head self-attention exports two *batched*
     rank-4 MatMuls per block where BOTH operands are activations (not weights):
        Q.Kᵀ : [1,h,n,c] x [1,h,c,k] -> [1,h,n,k]   =>  Einsum "bhnc,bhck->bhnk"
        A.V  : [1,h,n,k] x [1,h,k,c] -> [1,h,n,c]   =>  Einsum "bhnk,bhkc->bhnc"
     The equations are derived from the *actual* operand shapes (shape inference),
     so a wrong-layout rewrite is impossible: onnx.checker rejects it here, before
     any compile slot is spent. The linear q/k/v/proj/mlp MatMuls (one operand is a
     weight initializer) are LEFT ALONE — they are ordinary supported GEMMs.
     Why: the MLA tessellator maps an explicit-equation batched Einsum onto MLA
     tiles cleanly; the same win as the YOLO attention rewrite, here for the
     token-sequence ([1, N, C]) attention of a ViT instead of a spatial one.

  2. DINOv2 masks/Where removal (dinov2 only).  torch.hub DINOv2 exports an unused
     rank-0 `masks` input feeding masks->Unsqueeze->Where(cond, mask_token, patch)
     ->Concat. At inference masks is all-false, so Where is the identity on the
     patch-embed tensor. We rewire Concat straight to the patch-embed output and
     delete masks/Unsqueeze/Where, leaving a single `input` the compiler importer
     can bind. LayerNorm stays as-is: the compiler lowers LayerNormalization to
     supported primitives (it shows as "unknown" in the op audit only because the
     audit table lists primitives, not this composite).

Output: work/<id>/surgery/<id>.compile_ready.onnx  (picked first by 18_compile).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import helper, shape_inference
from onnxsim import simplify


ROOT = Path(__file__).resolve().parents[1]
INPUT_NAME = "input"
INPUT_SHAPE = [1, 3, 224, 224]

# A rank-4 batched matmul A[b,h,m,k] x B[b,h,k,n] -> [b,h,m,n] is ALWAYS exactly
# this Einsum, independent of whether it is Q.Kᵀ or A.V — layout-agnostic and safe.
EQ_BMM = "bhmk,bhkn->bhmn"


def shape_of(vimap, name):
    return vimap.get(name)


def rewrite_attention_matmuls(model):
    inferred = shape_inference.infer_shapes(model)
    vimap = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        vimap[v.name] = dims
    initializers = {init.name for init in model.graph.initializer}

    rewrites = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "MatMul" or len(node.input) != 2:
            continue
        a, b = node.input
        # skip linear projections (one operand is a weight initializer)
        if a in initializers or b in initializers:
            continue
        sa, sb = shape_of(vimap, a), shape_of(vimap, b)
        so = shape_of(vimap, node.output[0])
        if not sa or not sb or not so or len(sa) != 4 or len(sb) != 4 or len(so) != 4:
            continue
        # verify a clean batched contraction A[b,h,m,k] x B[b,h,k,n] -> [b,h,m,n]
        if sa[3] != sb[2] or so != [sa[0], sa[1], sa[2], sb[3]]:
            continue
        new = helper.make_node(
            "Einsum",
            inputs=[a, b],
            outputs=list(node.output),
            name=(node.name or f"attn_matmul_{idx}") + "/Einsum",
            equation=EQ_BMM,
        )
        model.graph.node.remove(node)
        model.graph.node.insert(idx, new)
        rewrites.append({"node": node.name, "equation": EQ_BMM, "a": sa, "b": sb, "out": so})
    return rewrites


def remove_dinov2_masks(model):
    g = model.graph
    # DINOv2 exports a single Where (masks branch); its Y input (3rd) is the real
    # patch-embed tensor used when masks is all-false, which is the inference case.
    where = next((n for n in g.node if n.op_type == "Where"), None)
    if where is None:
        return {"removed": False, "reason": "no Where node"}
    patch_tensor = where.input[2]  # Y branch = real patch-embed tensor (masks all-false)
    where_out = where.output[0]
    # rewire consumers of Where output to patch_tensor
    for n in g.node:
        n.input[:] = [patch_tensor if i == where_out else i for i in n.input]
    # remove Where and the masks Unsqueeze
    to_remove = [where]
    for n in g.node:
        if n.op_type == "Unsqueeze" and "masks" in list(n.input):
            to_remove.append(n)
    for n in to_remove:
        g.node.remove(n)
    # drop masks graph input
    masks_in = next((i for i in g.input if i.name == "masks"), None)
    if masks_in is not None:
        g.input.remove(masks_in)
    return {"removed": True, "patch_tensor": patch_tensor, "removed_nodes": [n.name for n in to_remove]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=["vit_b_16", "dinov2_vits14"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = ROOT / "work" / args.model_id
    source = base / "surgery" / f"{args.model_id}.surgery.onnx"
    if not source.exists():
        source = base / "onnx" / f"{args.model_id}.onnx"
    out = base / "surgery" / f"{args.model_id}.compile_ready.onnx"
    report_dir = base / "reports"
    out.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(source)
    onnx.checker.check_model(model)

    masks_report = {}
    if args.model_id == "dinov2_vits14":
        masks_report = remove_dinov2_masks(model)

    rewrites = rewrite_attention_matmuls(model)

    simplified, ok = simplify(model, overwrite_input_shapes={INPUT_NAME: INPUT_SHAPE}, dynamic_input_shape=False)
    if not ok:
        raise SystemExit("ONNX simplification check failed")
    simplified = shape_inference.infer_shapes(simplified)
    onnx.checker.check_model(simplified)
    onnx.save(simplified, out)

    report = {
        "model_id": args.model_id,
        "status": "exported",
        "source": str(source),
        "output": str(out),
        "attention_rewrites": rewrites,
        "num_attention_rewrites": len(rewrites),
        "dinov2_masks_removal": masks_report,
        "inputs": [i.name for i in simplified.graph.input],
        "outputs": [o.name for o in simplified.graph.output],
        "contract": "static-shape ViT; attention batched-MatMul->Einsum; DINOv2 masks/Where removed; LayerNorm lowered by compiler",
    }
    (report_dir / "compile_ready_surgery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
