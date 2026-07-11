#!/usr/bin/env python3
"""STEP 1 - export a model to ONNX.

    python compile/convert_to_onnx.py --model-id resnet50
    python compile/convert_to_onnx.py --all

Handles every family in models.yaml:
  torchvision / timm  -> torch.onnx.export from the pretrained checkpoint
  ultralytics         -> YOLO(...).export(format="onnx")   (static shapes, no NMS baked in)
  megvii              -> download the official pre-exported ONNX (no torch needed)

Output: work/<id>/onnx/<id>.onnx
"""
from __future__ import annotations

import argparse
import urllib.request

from common import ensure_dirs, model_cfg, all_model_ids, paths


def export_torch(cfg: dict, project: dict, out_path) -> None:
    import torch

    fam, arch = cfg["family"], cfg["arch"]
    if fam == "torchvision":
        import torchvision.models as tvm
        weights = cfg.get("weights")
        model = getattr(tvm, arch)(weights=weights) if weights else getattr(tvm, arch)(weights="DEFAULT")
    elif fam == "timm":
        import timm
        model = timm.create_model(arch, pretrained=True)
    else:
        raise SystemExit(f"unsupported torch family: {fam}")

    model.eval()
    dummy = torch.randn(*cfg["input_shape"])
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=[cfg["input_name"]],
        output_names=cfg["output_names"],
        opset_version=project.get("default_opset", 17),
        dynamic_axes=None,                      # STATIC shapes: dynamic axes break the compiler
    )


def export_ultralytics(cfg: dict, project: dict, out_path) -> None:
    from ultralytics import YOLO

    model = YOLO(cfg["arch"])                   # downloads the .pt on first use
    imgsz = cfg["input_shape"][2]
    produced = model.export(
        format="onnx", imgsz=imgsz, opset=project.get("default_opset", 17),
        dynamic=False, simplify=True, nms=False,   # raw heads; surgery exposes them next
    )
    import shutil
    shutil.move(str(produced), str(out_path))


def export_megvii(cfg: dict, project: dict, out_path) -> None:
    """YOLOX ships an official pre-exported ONNX -- no need to install the yolox package."""
    url = cfg["source"]
    print(f"  downloading pre-exported ONNX: {url}")
    urllib.request.urlretrieve(url, str(out_path))


EXPORTERS = {
    "torchvision": export_torch,
    "timm": export_torch,
    "ultralytics": export_ultralytics,
    "megvii": export_megvii,
}


def run(model_id: str, force: bool = False) -> int:
    cfg, project = model_cfg(model_id)
    p = ensure_dirs(model_id)
    out = p["onnx"]
    if out.exists() and not force:
        print(f"[convert] {model_id}: ONNX already exists ({out}); use --force to re-export")
        return 0

    fam = cfg["family"]
    exporter = EXPORTERS.get(fam)
    if exporter is None:
        raise SystemExit(f"no exporter for family '{fam}'")

    print(f"[convert] {model_id}: {fam}/{cfg['arch']} -> {out}")
    exporter(cfg, project, out)
    print(f"[convert] {model_id}: OK  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-export even if the ONNX exists")
    args = ap.parse_args()

    ids = all_model_ids() if args.all else [args.model_id]
    if not ids or ids == [None]:
        raise SystemExit("give --model-id <id> or --all")
    rc = 0
    for mid in ids:
        rc |= run(mid, args.force)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
