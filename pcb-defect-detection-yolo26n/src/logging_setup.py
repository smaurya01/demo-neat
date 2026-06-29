"""Standard logging initializer for the PLC defect-detection pipeline."""

import logging
import sys


def init_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("plc")
