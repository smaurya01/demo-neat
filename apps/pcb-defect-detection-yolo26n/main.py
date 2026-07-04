"""PLC (PCB) defect detection on the SiMa Modalix DevKit — YOLO26n.

Runs the BF16 compiled MLA model pack on the DevKit via the public `pyneat` runtime
over a folder of PCB images, drawing class-colored boxes + labels at the original
resolution. Invoke on the board through `dk`.

Usage:
  python3 main.py [--config config/default.conf] [--score 0.30] [--nms 0.50]
                  [--log-level INFO]

CLI flags override config values (single-stream-yolo26n convention). Unknown
extra tokens are ignored so the `dk` remote-exec wrapper stays compatible.

Exit codes: 0 success (>=1 image processed) | 2 runtime/IO error | 3 no images.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.config import load_config, resolve_path
from src.inference import run_pipeline
from src.utils import init_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="PLC defect detection (YOLO26n) on SiMa Modalix")
    parser.add_argument("--config", default=str(HERE / "config" / "default.conf"),
                        help="Path to runtime config (key=value)")
    parser.add_argument("--model", help="Override compiled MLA pack path")
    parser.add_argument("--input-dir", help="Override input image directory")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--score", type=float, help="Override detection score threshold")
    parser.add_argument("--nms", type=float, help="Override NMS IoU threshold")
    parser.add_argument("--log-level", default="INFO")
    args, _unknown = parser.parse_known_args()

    logger = init_logging(args.log_level)
    cfg = load_config(args.config)

    if args.score is not None:
        cfg.score = args.score
    if args.nms is not None:
        cfg.nms = args.nms
    if args.model:
        cfg.model = args.model
    if args.input_dir:
        cfg.input_dir = args.input_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir

    model_pack = resolve_path(HERE, cfg.model)
    input_dir = resolve_path(HERE, cfg.input_dir)
    output_dir = resolve_path(HERE, cfg.output_dir)

    if not model_pack.exists():
        logger.error("compiled pack not found: %s (see README compile commands)", model_pack)
        return 2
    if not input_dir.is_dir():
        logger.error("input dir not found: %s", input_dir)
        return 2

    try:
        return run_pipeline(cfg, str(model_pack), input_dir, output_dir, logger)
    except Exception:
        logger.exception("pipeline failed")
        return 2


if __name__ == "__main__":
    sys.exit(main())
