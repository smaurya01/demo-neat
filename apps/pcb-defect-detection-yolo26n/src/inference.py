"""Inference and pipeline orchestration for the PCB defect detector.

The compiled model pack owns the Neat path:

    preprocess -> MLA inference -> on-device YOLO26 box decode

The application feeds raw BGR frames, receives the BBOX payload, parses it into
Python dictionaries, draws class-colored boxes, and writes annotated images.
"""

import struct
import time
from collections import Counter
from logging import Logger
from pathlib import Path
from typing import List

from .config import Config
from .utils import (
    class_name,
    discover_images,
    draw_detections,
    ensure_output_dir,
)


def to_bgr_tensor(bgr):
    """Raw BGR uint8 image -> pyneat.Tensor."""
    import numpy as np
    import pyneat

    if bgr is None or bgr.size == 0:
        raise ValueError("empty input image")
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int, min_score: float) -> List[dict]:
    """Parse pyneat BBOX payload records into original-image-space detections."""
    if not payload or len(payload) < 4:
        return []

    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    detections = []
    offset = 4
    for _ in range(count):
        x, y, w, h, score, cls = struct.unpack_from("<iiiifi", payload, offset)
        offset += 24
        if float(score) < min_score:
            continue

        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            dict(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score), class_id=int(cls))
        )
    return detections


class PyNeatDetector:
    def __init__(self, pack_path: str, score: float, nms_iou: float, top_k: int,
                 timeout_ms: int, num_classes: int, model_size: int, seed_bgr):
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
        opt.num_classes = num_classes
        # boxdecode_original_* is the frame size the model's coordinate inversion maps back to.
        # infer() resizes EVERY image to model_size x model_size before pushing, so that is the
        # frame the graph actually sees; infer() then scales the returned boxes back to the
        # original image. (Feeding full-size images while claiming model_size here was what put
        # every box in the top-left corner of any larger image.)
        opt.boxdecode_original_width = model_size
        opt.boxdecode_original_height = model_size
        self.model_size = model_size
        self.model = pyneat.Model(pack_path, opt)

        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = 8
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        run_opt.preset = pyneat.RunPreset.Balanced

        # Seed at the SAME size infer() will push, or the runner's fixed input caps would be
        # built for the first image's dimensions and reject every resized frame after it.
        import cv2
        if seed_bgr.shape[:2] != (model_size, model_size):
            seed_bgr = cv2.resize(seed_bgr, (model_size, model_size),
                                  interpolation=cv2.INTER_AREA)
        t_seed = to_bgr_tensor(seed_bgr)
        self.runner = self.model.build(
            [t_seed],
            route_options=pyneat.ModelRouteOptions(),
            run_options=run_opt,
        )
        self.runner.run([t_seed], timeout_ms=timeout_ms)   # warmup

    @staticmethod
    def _looks_like_bbox(payload: bytes) -> bool:
        """Structural check on the BBOX wire format before we reinterpret bytes as detections.

        Taking the FIRST non-empty payload and parsing it as BBOX is how you silently draw
        garbage: if the route ever emits an auxiliary tensor first (a raw head, a preprocess
        passthrough), its first 4 bytes become a detection count and its body becomes 24-byte
        records. `count` is capped by the length so nothing crashes -- you just get
        plausible-looking boxes on the board. Neat's own decoder refuses this
        (`bbox tensor format mismatch`, and `body % 24 != 0`); mirror that here.
        """
        if not payload or len(payload) < 4:
            return False
        body = len(payload) - 4
        if body % 24 != 0:                       # records are exactly 24 bytes (struct RawBox)
            return False
        # The buffer is fixed-capacity, so count <= capacity (parse_bbox_payload clamps the same
        # way). Requiring equality rejected real payloads.
        count = struct.unpack_from("<I", payload, 0)[0]
        return count <= body // 24

    @classmethod
    def _extract_bbox_payload(cls, tensors):
        for tensor in tensors:
            try:
                payload = tensor.copy_payload_bytes()
            except Exception:
                continue
            if payload and cls._looks_like_bbox(payload):
                return payload
        return None

    def infer(self, bgr) -> List[dict]:
        """Accept an image of ANY size; resize to the model input before inference.

        The runner's input caps are fixed by the seed image at build time, so pushing a
        differently-sized image later fails. Normalising every image to model_size x model_size
        here means a mixed-resolution folder works, and it is also what makes
        boxdecode_original_* correct: the graph always sees exactly that frame.
        """
        import cv2

        oh, ow = bgr.shape[:2]
        size = self.model_size
        if (oh, ow) != (size, size):
            # INTER_AREA downscales without aliasing; INTER_LINEAR is the right choice upscaling.
            interp = cv2.INTER_AREA if (ow > size or oh > size) else cv2.INTER_LINEAR
            model_in = cv2.resize(bgr, (size, size), interpolation=interp)
        else:
            model_in = bgr

        out = self.runner.run([to_bgr_tensor(model_in)], timeout_ms=self.timeout_ms)
        payload = self._extract_bbox_payload(out)
        if not payload:
            return []

        # Boxes come back in model_size space; map them back to the caller's image.
        dets = parse_bbox_payload(payload, size, size, self.score)
        if (oh, ow) != (size, size):
            sx = ow / float(size)
            sy = oh / float(size)
            for d in dets:
                d["x1"] *= sx
                d["x2"] *= sx
                d["y1"] *= sy
                d["y2"] *= sy
        return dets

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
    num_classes = len(cfg.labels) if cfg.labels else 6
    return PyNeatDetector(model_pack, cfg.score, cfg.nms, cfg.top_k,
                          cfg.timeout_ms, num_classes, cfg.infer_size, seed_bgr)


def run_pipeline(cfg: Config, model_pack: str, input_dir: Path, output_dir: Path,
                 logger: Logger) -> int:
    """Discover images, run MLA inference, draw detections, and write outputs."""
    import cv2

    nc = len(cfg.labels) if cfg.labels else 6
    logger.info(
        "starting | model=%s infer_size=%d score=%.2f nms=%.2f top_k=%d nc=%d",
        Path(model_pack).name,
        cfg.infer_size,
        cfg.score,
        cfg.nms,
        cfg.top_k,
        nc,
    )

    ensure_output_dir(output_dir)
    images = discover_images(input_dir, cfg.image_extensions)
    if not images:
        logger.error("no images found in %s", input_dir)
        return 3
    logger.info("found %d images in %s", len(images), input_dir)

    seed = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if seed is None:
        logger.error("failed to read seed image: %s", images[0].name)
        return 2

    detector = build_detector(cfg, model_pack, seed)
    logger.info("detector built on MLA (warmup done)")

    start = time.time()
    processed = 0
    total_dets = 0
    images_with_dets = 0
    per_class = Counter()
    written: set = set()
    try:
        for image_path in images:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("skip unreadable image: %s", image_path.name)
                continue

            try:
                dets = detector.infer(bgr)
            except Exception:
                logger.exception("inference failed for %s", image_path.name)
                continue

            overlay = bgr.copy()
            draw_detections(overlay, dets, cfg.labels)
            # Documented name: <stem><output_suffix>.jpg (see README). Only disambiguate when it
            # would actually collide -- board.jpg and board.png both map to board_detected.jpg,
            # and the second used to silently overwrite the first while the summary still counted
            # both as processed.
            out_file = output_dir / (image_path.stem + cfg.output_suffix + ".jpg")
            if out_file in written:
                out_file = output_dir / (image_path.stem + image_path.suffix.replace(".", "_")
                                         + cfg.output_suffix + ".jpg")
                logger.warning("output name collision for %s; writing %s",
                               image_path.name, out_file.name)
            if out_file.resolve() == image_path.resolve():
                logger.error("refusing to overwrite input %s", image_path)
                continue
            if not cv2.imwrite(str(out_file), overlay):
                logger.error("failed to write %s", out_file)
                continue
            written.add(out_file)

            counts = Counter(class_name(d["class_id"], cfg.labels) for d in dets)
            per_class.update(counts)
            total_dets += len(dets)
            if dets:
                images_with_dets += 1
            processed += 1

            logger.info(
                "img=%s dets=%d %s wrote=%s",
                image_path.name,
                len(dets),
                dict(counts),
                out_file.name,
            )
    finally:
        detector.close()

    elapsed = time.time() - start
    logger.info(
        "processed %d / %d images in %.2f s | images_with_detections=%d total_detections=%d",
        processed,
        len(images),
        elapsed,
        images_with_dets,
        total_dets,
    )
    if per_class:
        logger.info("per-class totals: %s", dict(per_class))
    return 0 if processed > 0 else 2
