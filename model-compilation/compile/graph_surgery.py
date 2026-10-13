#!/usr/bin/env python3
"""STEP 2 - graph surgery: make an ONNX export compile-ready for the MLA.

    python compile/graph_surgery.py --model-id yolo11n
    python compile/graph_surgery.py --all

WHY surgery is needed (this is the whole trick behind "one .elf, no .so"):

  A stock YOLO export ends in a CPU-shaped decode/NMS tail (DFL softmax, anchor grids, concat,
  transpose, NMS). Those ops cannot be placed on the MLA, so the compiler splits the graph and
  spills them to the host as `.so` stages. Cut that tail off and expose the RAW per-scale heads
  instead, and the entire graph stays on the MLA -> a single ELF, zero .so.
  Neat then does the box decode itself (BoxDecodeType), or the app decodes the raw heads.

What each surgery kind does (see models.yaml `surgery:`):

  none              CNNs. Nothing to do -- they already compile to a single ELF.
  yolo_ultralytics  1) attention MatMul -> Einsum (MLA-friendly)
                    2) expose pre-decode head convs: cv2.* = bbox, cv3.* = class
                       (YOLO26 uses one2one_cv*), plus seg mask-coeff/proto and pose kpt heads
                    3) YOLOv8/YOLO11 keep their raw 64-channel DFL bbox heads (Neat's YoloV8
                       decoder integrates the DFL); YOLO26 one2one heads are already 4-channel.
  yolox             Megvii YOLOX: decoupled anchor-free head with numeric node names, no DFL/attn.
                    Trace back to each scale's Concat and expose its three PRE-sigmoid inputs,
                    (bbox 4, obj 1, cls 80) x 3 scales (YoloX / Split3Interleaved).
  world             YOLO-World v2: fold the fixed-vocabulary contrastive Einsum into a 1x1 Conv,
                    expose the 64-ch DFL bbox heads and the class logits (YoloV8).

Input : work/<id>/onnx/<id>.onnx
Output: work/<id>/surgery/<id>.compile_ready.onnx  (+ reports/)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import all_model_ids, model_cfg, paths

HERE = Path(__file__).resolve().parent


def run(model_id: str, force: bool = False) -> int:
    cfg, _ = model_cfg(model_id)
    kind = cfg.get("surgery", "none")
    p = paths(model_id)

    if kind == "none":
        print(f"[surgery] {model_id}: kind=none -- CNN compiles straight from the export, skipping")
        return 0

    if not p["onnx"].exists():
        raise SystemExit(f"[surgery] {model_id}: missing {p['onnx']} -- run convert_to_onnx.py first")

    if p["surgery"].exists() and not force:
        print(f"[surgery] {model_id}: {p['surgery'].name} already exists; use --force to redo")
        return 0

    impl = {"yolo_ultralytics": "_surgery_ultralytics.py", "yolox": "_surgery_yolox.py",
            "world": "_surgery_world.py"}.get(kind)
    if impl is None:
        raise SystemExit(f"[surgery] unknown surgery kind '{kind}' for {model_id}")

    cmd = [sys.executable, str(HERE / impl), "--model-id", model_id]
    if force:
        cmd.append("--force")
    print(f"[surgery] {model_id}: kind={kind} -> {p['surgery']}")
    proc = subprocess.run(cmd, cwd=str(HERE.parent))
    if proc.returncode != 0:
        return proc.returncode
    import onnx                                    # the compiler is told to expect these names
    got = [o.name for o in onnx.load(str(p["surgery"]), load_external_data=False).graph.output]
    if got != list(cfg["output_names"]):
        print(f"[surgery] {model_id}: FAIL  outputs {got} != models.yaml output_names "
              f"{cfg['output_names']}")
        return 1
    print(f"[surgery] {model_id}: OK  outputs={got}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ids = all_model_ids() if args.all else [args.model_id]
    if not ids or ids == [None]:
        raise SystemExit("give --model-id <id> or --all")
    rc = 0
    for mid in ids:
        try:
            rc |= run(mid, args.force)
        except Exception as e:            # with --all, report and go on to the next model
            if len(ids) == 1:
                raise
            print(f"[FAIL] {mid}: {e}"); rc |= 1
        except SystemExit as e:
            if len(ids) == 1:
                raise
            print(e); rc |= 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
