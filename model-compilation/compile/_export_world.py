#!/usr/bin/env python3
"""Export a YOLO-World model to ONNX with a FIXED vocabulary baked in.

YOLO-World is open-vocabulary: at runtime it takes an image AND a set of text
prompts, and a CLIP text encoder turns the prompts into class embeddings. A SiMa
archive is a fixed graph, so we must pin the vocabulary first with set_classes(),
which precomputes the text embeddings and bakes them into the head as constants.
The exported ONNX then takes only the image and emits detections for those classes
-- the CLIP text encoder disappears from the graph entirely.

We use worldv2 on purpose: its head is BNContrastiveHead (BatchNorm2d + an einsum
against the FIXED text features), so the only per-frame op is a BatchNorm (folds
into conv) and a fixed-weight einsum (== a 1x1 conv). worldv1's ContrastiveHead
does a runtime L2-normalize over channels, which is the op that risks a host
(.so) fallback. worldv2 is the single-ELF-friendly choice.

Output: work/<id>/onnx/<id>.onnx  (raw heads; graph_surgery exposes them next)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO_LABELS = Path(
    "/workspace/apps/examples/object-detection/yolo26-object-detector/src/common/coco_label.txt"
)


def load_classes(path: Path) -> list[str]:
    names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not names:
        raise SystemExit(f"no class names in {path}")
    return names


def patch_maxsigmoid_attn_to_4d():
    """Rewrite MaxSigmoidAttnBlock.forward to use only 4D ops so it stays on the MLA.

    The stock forward reshapes features to 5D (bs, nh, hc, h, w) and runs a 5D einsum
    'bmchw,bnmc->bmhwn' plus a 5D broadcast multiply. SiMa's Einsum supports 2D spatial
    only (4D tensors), so each of the 4 neck attn blocks spills to the host (A65) and the
    graph fragments (measured: A65:8, 9 elf / 8 so).

    This replacement is mathematically identical but per-head and fully 4D:
      * the einsum-then-max over the N text classes becomes, per head m, a 1x1 Conv
        (weight = the FIXED guide features, constant after set_classes) producing N score
        maps, followed by a channel-wise ReduceMax  (Conv, ReduceMax: both MLA-supported);
      * the 5D broadcast multiply becomes per-head Slice * (1-channel) Mul + Concat.
    All ops (Conv, Slice->Conv2d, ReduceMax 3D, Concat, Mul) are MLA-placeable.
    """
    import torch
    import torch.nn.functional as F
    from ultralytics.nn.modules.block import MaxSigmoidAttnBlock

    def forward_4d(self, x, guide):
        bs, _, h, w = x.shape
        nh, hc = self.nh, self.hc
        g = self.gl(guide)                                   # [bs, N, ec]  (fixed -> constant)
        n = g.shape[1]
        g = g.view(bs, n, nh, hc)                            # [bs, N, nh, hc]
        embed = self.ec(x) if self.ec is not None else x     # [bs, nh*hc, h, w]
        aws = []
        for m in range(nh):
            em = embed[:, m * hc:(m + 1) * hc, :, :]          # [bs, hc, h, w]
            wm = g[:, :, m, :].reshape(n, hc, 1, 1)           # [N, hc, 1, 1]  (constant)
            scores = F.conv2d(em, wm)                         # [bs, N, h, w]
            aws.append(scores.max(dim=1, keepdim=True)[0])    # [bs, 1, h, w]  max over classes
        aw = torch.cat(aws, dim=1)                            # [bs, nh, h, w]
        aw = aw / (hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale                        # [bs, nh, h, w]
        x = self.proj_conv(x)                                 # [bs, nh*hc2, h, w]
        hc2 = x.shape[1] // nh
        outs = [x[:, m * hc2:(m + 1) * hc2, :, :] * aw[:, m:m + 1, :, :] for m in range(nh)]
        return torch.cat(outs, dim=1)                         # [bs, nh*hc2, h, w]

    MaxSigmoidAttnBlock.forward = forward_4d
    print("[export] patched MaxSigmoidAttnBlock.forward -> 4D-native (Conv1x1 + channel ReduceMax)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="yolov8s-worldv2")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt",
                    help="Ultralytics YOLO-World checkpoint (auto-downloads)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--labels", type=Path, default=COCO_LABELS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "work" / args.model_id / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_id}.onnx"
    if out_path.exists() and not args.force:
        print(f"[export] {out_path} exists; use --force to re-export")
        return 0

    classes = load_classes(args.labels)
    print(f"[export] {args.model_id}: {args.weights} with {len(classes)} fixed classes")

    from ultralytics import YOLO

    patch_maxsigmoid_attn_to_4d()       # 4D-native neck attention (single-ELF fix)
    model = YOLO(args.weights)          # downloads the .pt on first use
    model.set_classes(classes)          # bake the vocabulary -> drops the CLIP text encoder
    produced = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,                  # STATIC shapes: dynamic axes break the compiler
        simplify=True,
        nms=False,                      # raw heads; surgery exposes them next
    )
    shutil.move(str(produced), str(out_path))
    print(f"[export] {args.model_id}: OK -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[export] baked classes: {len(classes)} (COCO order)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
