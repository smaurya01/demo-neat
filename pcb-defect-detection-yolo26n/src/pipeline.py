"""Pipeline orchestration (on-device): discover -> MLA infer -> draw -> write."""

import time
from collections import Counter
from logging import Logger
from pathlib import Path

import cv2

from .config import Config
from .inference import build_detector
from .io_utils import discover_images, ensure_output_dir
from .labels import class_name
from .overlay import draw_detections


def run_pipeline(cfg: Config, model_pack: str, input_dir: Path, output_dir: Path,
                 logger: Logger) -> int:
    nc = len(cfg.labels) if cfg.labels else 6
    logger.info("starting | model=%s infer_size=%d score=%.2f nms=%.2f top_k=%d nc=%d",
                Path(model_pack).name, cfg.infer_size, cfg.score, cfg.nms, cfg.top_k, nc)

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

    t0 = time.time()
    processed = 0
    total_dets = 0
    images_with_dets = 0
    per_class = Counter()
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
            out_file = output_dir / (image_path.stem + cfg.output_suffix + ".jpg")
            cv2.imwrite(str(out_file), overlay)

            counts = Counter(class_name(d["class_id"], cfg.labels) for d in dets)
            per_class.update(counts)
            total_dets += len(dets)
            if dets:
                images_with_dets += 1
            processed += 1
            logger.info("img=%s dets=%d %s wrote=%s",
                        image_path.name, len(dets), dict(counts), out_file.name)
    finally:
        detector.close()

    dt = time.time() - t0
    logger.info("processed %d / %d images in %.2f s | images_with_detections=%d total_detections=%d",
                processed, len(images), dt, images_with_dets, total_dets)
    if per_class:
        logger.info("per-class totals: %s", dict(per_class))
    return 0 if processed > 0 else 2
