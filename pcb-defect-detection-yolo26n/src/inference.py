"""On-device YOLO26n detector — runs the compiled MLA pack via pyneat 0.2.0.

This mirrors the shipped `yolo26-object-detector` example for this DevKit's
pyneat build: the Model owns the whole pipeline (preprocess -> MLA -> on-device
YOLO26 box decode), so we just feed a raw BGR frame and read back a BBOX payload
whose coordinates are already in original-image space.

    opt.preprocess: Image, BGR in, COCO-YOLO normalize (RGB /255) -> 640
    opt.decode_type = BoxDecodeType.YoloV26  (+ score / nms / top_k overrides)
    model = pyneat.Model(pack, opt)
    runner = model.build([seed_tensor], ...)
    out = runner.run([bgr_tensor]); payload = out[0].copy_payload_bytes()

`pyneat` is the on-device runtime (NEAT SDK), imported here so this module only
loads where the model actually runs.
"""

from typing import List

from .postprocess import parse_bbox_payload
from .preprocess import to_bgr_tensor


class PyNeatDetector:
    def __init__(self, pack_path: str, score: float, nms_iou: float, top_k: int,
                 timeout_ms: int, seed_bgr):
        import pyneat

        self.pyneat = pyneat
        self.score = score
        self.timeout_ms = timeout_ms

        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO   # RGB, /255 (matches compile)
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
        opt.score_threshold = score
        opt.nms_iou_threshold = nms_iou
        opt.top_k = top_k
        self.model = pyneat.Model(pack_path, opt)

        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = 8
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        run_opt.preset = pyneat.RunPreset.Balanced

        t_seed = to_bgr_tensor(seed_bgr)
        self.runner = self.model.build(
            [t_seed],
            route_options=pyneat.ModelRouteOptions(),
            run_options=run_opt,
        )
        self.runner.run([t_seed], timeout_ms=timeout_ms)   # warmup

    @staticmethod
    def _extract_bbox_payload(tensors):
        for tensor in tensors:
            try:
                payload = tensor.copy_payload_bytes()
            except Exception:
                continue
            if payload:
                return payload
        return None

    def infer(self, bgr) -> List[dict]:
        oh, ow = bgr.shape[:2]
        out = self.runner.run([to_bgr_tensor(bgr)], timeout_ms=self.timeout_ms)
        payload = self._extract_bbox_payload(out)
        if not payload:
            return []
        return parse_bbox_payload(payload, ow, oh, self.score)

    def describe(self) -> str:
        try:
            return self.model.summary()
        except Exception:
            return "(model summary unavailable)"

    def close(self):
        try:
            self.runner.close()
        except Exception:
            pass


def build_detector(cfg, model_pack: str, seed_bgr) -> PyNeatDetector:
    return PyNeatDetector(model_pack, cfg.score, cfg.nms, cfg.top_k,
                          cfg.timeout_ms, seed_bgr)
