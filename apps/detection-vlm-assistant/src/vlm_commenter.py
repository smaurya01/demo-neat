"""Trigger-gated, de-duplicated, bounded VLM commenter.

This module is the whole VLM leg of the app, kept deliberately separate from the
detection loop so the detector can be validated live without the VLM being present or
called. Two modes:

* ``dry_run=True``  -- log the selected crop (bbox, shape, class, score) and the exact
  prompt that WOULD be sent to the VLM. Nothing is loaded, nothing is inferred. This is
  how the detection leg is validated on the DevKit without touching the VLM.
* ``dry_run=False`` -- lazily construct ``pyneat.genai.VisionLanguageModel`` and run a
  ``GenerationRequest`` per selected crop, from a bounded background worker so the
  detection loop never blocks on multi-second VLM latency.

Why trigger-based multimodal: the detector is cheap and always-on (~37 fps on the
2-stream YOLO11 app); the VLM is expensive (seconds per call). Running the VLM per frame
is impossible, so we (1) gate which detections are worth a call, (2) de-duplicate so the
same object does not re-fire every frame, (3) rate-limit, and (4) absorb latency in a
bounded queue.

VLM API verified against:
  * /workspace/core/include/genai/VisionLanguageModel.h
  * /workspace/core/include/genai/GenAITypes.h  (GenerationRequest / GenerationResult)
  * /workspace/core/python/src/module.cpp        (pyneat.genai bindings)
  * llima/02-run-llm-vlm/02_run_vlm.ipynb         (house VLM usage)
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
import time
from typing import Callable

# cv2 / numpy / pyneat are imported lazily inside methods that need them, so this module
# imports cleanly off-board (for py_compile / --help) where those packages are absent.


@dataclass
class _Trigger:
    """One crop selected for a VLM call."""
    crop_bgr: object            # uint8 HWC BGR numpy array (OpenCV-native)
    label: str
    score: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in frame pixels


class VlmCommenter:
    def __init__(self, cfg, dry_run: bool, label_fn: Callable[[int], str]):
        self.cfg = cfg
        self.dry_run = dry_run
        self.label_fn = label_fn
        self.queue: "Queue[_Trigger]" = Queue(maxsize=cfg.vlm_max_pending)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.started = False
        self.last_enqueue_at = 0.0
        # Dedup memory: for each recently-triggered object we remember its class,
        # normalized-center box, and when it last fired.
        self._recent: list[dict] = []
        self._model = None   # lazily-loaded VisionLanguageModel (real mode only)

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        if not self.started:
            self.worker.start()
            self.started = True

    def close(self) -> None:
        self.stop_event.set()
        if self.started:
            self.worker.join(timeout=2.0)

    # -- selection ---------------------------------------------------------- #
    def _passes_gate(self, box: dict, frame_area: float) -> bool:
        """Confidence / class / area gating: which detections deserve a VLM call."""
        if box["score"] < self.cfg.vlm_trigger_min_score:
            return False
        label = self.label_fn(box["class_id"]).lower()
        allowed = self.cfg.vlm_trigger_classes
        if allowed and label not in allowed:
            return False
        area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
        if frame_area > 0 and area / frame_area < self.cfg.vlm_trigger_min_area_frac:
            return False
        return True

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        ix1 = max(a["x1"], b["x1"])
        iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"])
        iy2 = min(a["y2"], b["y2"])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _is_duplicate(self, box: dict, now: float) -> bool:
        """True if this box is the same object we recently sent (still in cooldown)."""
        self._recent = [
            r for r in self._recent
            if now - r["at"] < self.cfg.vlm_dedup_cooldown_s
        ]
        for r in self._recent:
            if r["class_id"] == box["class_id"] and \
                    self._iou(r["box"], box) >= self.cfg.vlm_dedup_iou:
                return True
        return False

    def _remember(self, box: dict, now: float) -> None:
        self._recent.append({"class_id": box["class_id"], "box": box, "at": now})

    def select(self, boxes: list[dict], frame_shape) -> "_Trigger | None":
        """Pick at most one crop per frame: highest-score gated, non-duplicate box."""
        height, width = frame_shape[:2]
        frame_area = float(width * height)
        candidates = [b for b in boxes if self._passes_gate(b, frame_area)]
        if not candidates:
            return None
        candidates.sort(key=lambda b: b["score"], reverse=True)
        now = time.monotonic()
        for box in candidates:
            if self._is_duplicate(box, now):
                continue
            self._remember(box, now)
            return box
        return None

    # -- ingestion ---------------------------------------------------------- #
    def on_frame(self, frame_bgr, boxes: list[dict]) -> None:
        """Called every frame by the detection loop. Never blocks on the VLM."""
        now = time.monotonic()
        # Rate-limit: at most one enqueue per vlm_interval_seconds.
        if now - self.last_enqueue_at < self.cfg.vlm_interval_seconds:
            return
        box = self.select(boxes, frame_bgr.shape)
        if box is None:
            return
        crop = self._crop(frame_bgr, box)
        if crop is None:
            return
        trigger = _Trigger(
            crop_bgr=crop,
            label=self.label_fn(box["class_id"]),
            score=box["score"],
            bbox=(int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])),
        )
        try:
            self.queue.put_nowait(trigger)
            self.last_enqueue_at = now
        except Full:
            # Bounded queue is full: a VLM call is still in flight. Drop and move on;
            # the detector keeps running. This is the latency-absorbing backpressure.
            self.last_enqueue_at = now
            print("vlm: queue busy, dropping trigger", flush=True)

    @staticmethod
    def _crop(frame_bgr, box: dict):
        x1 = max(0, int(box["x1"]))
        y1 = max(0, int(box["y1"]))
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        h, w = frame_bgr.shape[:2]
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2].copy()

    def _prompt_for(self, label: str) -> str:
        try:
            return self.cfg.vlm_prompt.format(label=label.lower())
        except (KeyError, IndexError):
            # Prompt template without a {label} placeholder: use it verbatim.
            return self.cfg.vlm_prompt

    # -- worker ------------------------------------------------------------- #
    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                trigger = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                if self.dry_run:
                    self._log_dry_run(trigger)
                else:
                    self._call_vlm(trigger)
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                print(f"vlm: request failed: {exc}", flush=True)
            finally:
                self.queue.task_done()

    def _log_dry_run(self, trigger: _Trigger) -> None:
        prompt = self._prompt_for(trigger.label)
        shape = getattr(trigger.crop_bgr, "shape", "?")
        print(
            "vlm[dry-run] WOULD send crop -> VLM\n"
            f"  class   : {trigger.label} score={trigger.score:.2f}\n"
            f"  bbox    : {trigger.bbox}\n"
            f"  crop    : shape={shape} (BGR; converted to RGB before the request)\n"
            f"  model   : {self.cfg.vlm_model_dir}\n"
            f"  prompt  : {prompt!r}",
            flush=True,
        )

    def _call_vlm(self, trigger: _Trigger) -> None:
        # Lazy imports so dry-run / off-board use never needs these.
        import cv2
        import numpy as np
        import pyneat as neat

        if self._model is None:
            # One VLM handle for the process; holds LM weights + vision encoder resident.
            self._model = neat.genai.VisionLanguageModel(self.cfg.vlm_model_dir)
            print(f"vlm: loaded {self._model.model_id()} "
                  f"accepts_image={self._model.accepts_image()}", flush=True)

        # CRITICAL colour correctness: VLM images must be uint8 HWC *RGB*. Our crop is
        # OpenCV-native BGR, so convert here. Skipping this silently feeds the VLM
        # channel-swapped images and quietly degrades every answer.
        crop_rgb = np.ascontiguousarray(cv2.cvtColor(trigger.crop_bgr, cv2.COLOR_BGR2RGB))

        request = neat.genai.GenerationRequest()
        request.prompt = self._prompt_for(trigger.label)
        request.images = [crop_rgb]
        request.max_new_tokens = self.cfg.vlm_max_new_tokens

        result = self._model.run(request)
        metrics = result.metrics
        print(
            f"vlm[{trigger.label} score={trigger.score:.2f} bbox={trigger.bbox}]: "
            f"{result.text.strip()}  "
            f"({metrics.generated_tokens} tok, "
            f"{metrics.tokens_per_second:.1f} tok/s, "
            f"ttft={metrics.time_to_first_token_s:.2f}s)",
            flush=True,
        )
