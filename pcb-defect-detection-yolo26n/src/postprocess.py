"""Parse the on-device detection payload from pyneat's sima_box_decode node.

The graph surgery baked the box decode into the compiled pack, so the MLA emits
the SiMa box-decoder contract (`bbox_*` + `class_prob_*`). The on-device
`sima_box_decode` node consumes that and produces a BBOX payload whose
coordinates are already in original-image space (it un-maps the resize). The
host only unpacks the bytes and applies the score threshold.
"""

import struct
from typing import List


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int, min_score: float) -> List[dict]:
    """Parse the BBOX payload: 4-byte LE count header, then N x 24-byte records
    '<iiiifi' = (x, y, w, h, score, class_id), coords in original-image space.
    """
    if not payload or len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    out, off = [], 4
    for _ in range(count):
        x, y, w, h, score, cls = struct.unpack_from("<iiiifi", payload, off)
        off += 24
        if float(score) < min_score:
            continue
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score), class_id=int(cls)))
    return out
