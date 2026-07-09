#!/usr/bin/env python3
"""Download fresh Ultralytics YOLO weights and export static ONNX."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
YOLO_ARCHES = {
    "yolo11n": "yolo11n.pt",
    "yolo26n": "yolo26n.pt",
}


def onnx_metadata(path: Path) -> dict:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return {
        "ir_version": model.ir_version,
        "opsets": {op.domain or "ai.onnx": op.version for op in model.opset_import},
        "inputs": [node.name for node in model.graph.input],
        "outputs": [node.name for node in model.graph.output],
        "nodes": len(model.graph.node),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=sorted(YOLO_ARCHES), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    base = ROOT / "work" / args.model_id
    source_dir = base / "source"
    onnx_dir = base / "onnx"
    report_dir = base / "reports"
    for path in [source_dir, onnx_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    arch = YOLO_ARCHES[args.model_id]
    yolo = YOLO(arch)
    pt_source = Path(getattr(yolo, "ckpt_path", arch)).resolve()
    pt_dest = source_dir / arch
    if pt_source.is_file():
        shutil.copy2(pt_source, pt_dest)

    exported = Path(
        yolo.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=17,
            simplify=False,
            dynamic=False,
        )
    ).resolve()
    onnx_dest = onnx_dir / f"{args.model_id}.onnx"
    shutil.copy2(exported, onnx_dest)
    source_export = source_dir / f"ultralytics_export_{args.model_id}.onnx"
    shutil.copy2(exported, source_export)

    report = {
        "model_id": args.model_id,
        "arch": arch,
        "pt": str(pt_dest),
        "ultralytics_pt_source": str(pt_source),
        "ultralytics_export": str(exported),
        "onnx": str(onnx_dest),
        "source_export": str(source_export),
        "metadata": onnx_metadata(onnx_dest),
    }
    (report_dir / "source.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
