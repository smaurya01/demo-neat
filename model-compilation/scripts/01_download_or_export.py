#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_registry():
    with (ROOT / "models.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(registry, model_id):
    for model in registry["models"]:
        if model["id"] == model_id:
            return model
    raise KeyError(f"unknown model id: {model_id}")


def ensure_dirs(model_id):
    base = ROOT / "work" / model_id
    for sub in ["source", "onnx", "surgery", "compile", "package", "reports"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def write_sample_assets(model):
    sample_dir = ROOT / "assets" / "sample_images"
    calib_dir = ROOT / "assets" / "calibration"
    sample_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    _, _, h, w = model["input_shape"]
    sample_path = sample_dir / "synthetic_rgb_gradient.jpg"
    if not sample_path.exists():
        x = np.linspace(0, 255, w, dtype=np.uint8)
        y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        img = np.stack([
            np.broadcast_to(x, (h, w)),
            np.broadcast_to(y, (h, w)),
            np.full((h, w), 128, dtype=np.uint8),
        ], axis=-1)
        Image.fromarray(img, "RGB").save(sample_path)

    for idx in range(8):
        path = calib_dir / f"synthetic_calib_{idx:02d}.jpg"
        if not path.exists():
            rng = np.random.default_rng(idx)
            img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            Image.fromarray(img, "RGB").save(path)


def resolve_torchvision_weights(models_module, weights_spec):
    if not weights_spec:
        return None
    enum_name, member = weights_spec.split(".", 1)
    enum_obj = getattr(models_module, enum_name)
    return getattr(enum_obj, member)


def export_torchvision(model, out_path):
    import torchvision.models as models

    weights = resolve_torchvision_weights(models, model.get("weights"))
    ctor = getattr(models, model["arch"])
    net = ctor(weights=weights)
    net.eval()

    dummy = torch.randn(*model["input_shape"], dtype=torch.float32)
    input_name = model["input_name"]
    output_names = model.get("output_names") or ["output"]

    torch.onnx.export(
        net,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=None,
    )


def export_timm(model, out_path):
    import timm

    net = timm.create_model(model["arch"], pretrained=True)
    net.eval()
    dummy = torch.randn(*model["input_shape"], dtype=torch.float32)
    torch.onnx.export(
        net,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[model["input_name"]],
        output_names=model.get("output_names") or ["output"],
        dynamic_axes=None,
    )


def export_ultralytics(model, out_path):
    from ultralytics import YOLO

    yolo = YOLO(model["arch"])
    exported = yolo.export(format="onnx", imgsz=model["input_shape"][2], opset=17, simplify=False)
    exported_path = Path(exported)
    out_path.write_bytes(exported_path.read_bytes())


def export_torchhub(model, out_path):
    repo_by_arch = {
        "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14"),
        "detr_resnet50": ("facebookresearch/detr", "detr_resnet50"),
    }
    if model["arch"] not in repo_by_arch:
        raise NotImplementedError(f"torchhub export not implemented for {model['arch']}")
    repo, entrypoint = repo_by_arch[model["arch"]]
    net = torch.hub.load(repo, entrypoint, pretrained=True)
    net.eval()

    if model["arch"] == "detr_resnet50":
        class DetrExportWrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, images):
                outputs = self.wrapped(images)
                return outputs["pred_logits"], outputs["pred_boxes"]

        net = DetrExportWrapper(net)

    dummy = torch.randn(*model["input_shape"], dtype=torch.float32)
    torch.onnx.export(
        net,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[model["input_name"]],
        output_names=model.get("output_names") or ["output"],
        dynamic_axes=None,
    )


def onnx_metadata(path):
    import onnx

    proto = onnx.load(path)
    onnx.checker.check_model(proto)
    return {
        "ir_version": proto.ir_version,
        "opsets": {op.domain or "ai.onnx": op.version for op in proto.opset_import},
        "inputs": [i.name for i in proto.graph.input],
        "outputs": [o.name for o in proto.graph.output],
        "nodes": len(proto.graph.node),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry = load_registry()
    model = get_model(registry, args.model_id)
    base = ensure_dirs(model["id"])
    write_sample_assets(model)

    report = {
        "model_id": model["id"],
        "source": model.get("source"),
        "license_note": model.get("license_note"),
        "family": model["family"],
        "arch": model["arch"],
        "input_shape": model["input_shape"],
        "input_name": model["input_name"],
        "output_names": model.get("output_names", []),
    }

    onnx_path = base / "onnx" / f"{model['id']}.onnx"
    if onnx_path.exists() and not args.force:
        report["status"] = "exists"
    else:
        family = model["family"]
        if family == "torchvision":
            export_torchvision(model, onnx_path)
        elif family == "timm":
            export_timm(model, onnx_path)
        elif family == "ultralytics":
            export_ultralytics(model, onnx_path)
        elif family == "torchhub":
            export_torchhub(model, onnx_path)
        else:
            raise NotImplementedError(f"export family not implemented yet: {family}")
        report["status"] = "exported"

    report["onnx_path"] = str(onnx_path)
    report["onnx"] = onnx_metadata(onnx_path)
    (base / "reports" / "source.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
