"""Wrap a BGR image as a pyneat input tensor.

With pyneat 0.2.0 the Model does its own preprocessing (color-convert, resize,
normalize) on-device — configured via ModelOptions.preprocess. So we hand the
model the raw BGR uint8 frame at its native size; the EV74 preprocess resizes to
640 and applies the COCO-YOLO normalization (RGB, /255) that the pack expects.
"""

import numpy as np


def to_bgr_tensor(bgr: np.ndarray):
    """Raw BGR uint8 image -> pyneat.Tensor (BGR, EV74 memory)."""
    import pyneat

    if bgr is None or bgr.size == 0:
        raise ValueError("empty input image")
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )
