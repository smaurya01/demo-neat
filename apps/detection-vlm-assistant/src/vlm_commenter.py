"""Trigger-gated, de-duplicated, bounded VLM commenter.

This module is the whole VLM leg of the app, kept deliberately separate from the
detection loop so the detector can be validated live without the VLM being present or
called. Two modes:

* ``dry_run=True``  -- log the selected crop (bbox, shape, class, score) and the exact
  prompt that WOULD be sent to the VLM. Nothing is loaded, nothing is inferred. This is
  how the detection leg is validated on the DevKit without touching the VLM.
* ``dry_run=False`` -- load ``pyneat.genai.VisionLanguageModel`` in a separate process and
  run a ``GenerationRequest`` per selected crop, fed by a bounded background worker so the
  detection loop never blocks on multi-second VLM latency. The VLM cannot share a
  process with the detector's Neat model graph (see ``_vlm_server``).

Why trigger-based multimodal: the detector is cheap and always-on (~37 fps on the
2-stream YOLO11 app); the VLM is expensive (seconds per call). Running the VLM per frame
is impossible, so we (1) gate which detections are worth a call, (2) de-duplicate so the
same object does not re-fire every frame, (3) rate-limit, and (4) absorb latency in a
bounded queue.

VLM API verified against:
  * https://github.com/sima-neat/core/blob/main/include/genai/VisionLanguageModel.h
  * https://github.com/sima-neat/core/blob/main/include/genai/GenAITypes.h  (GenerationRequest / GenerationResult)
  * https://github.com/sima-neat/core/blob/main/python/src/module.cpp        (pyneat.genai bindings)
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


def _vlm_server(conn, model_dir: str) -> None:
    """Child-process entry point: own the VLM and answer requests sent over ``conn``.

    The VLM must live in its own process. On Neat 0.6.0 a ``VisionLanguageModel`` and a
    Neat model graph cannot share one process: whichever starts second fails in the MLA
    runtime (``Failed to connect MLA-RT``, or a fatal ``cannot respecify argc/argv``).
    Requests are ``(crop_rgb, prompt, max_new_tokens)``; ``None`` stops the server.
    """
    import pyneat as neat

    try:
        started = time.monotonic()
        model = neat.genai.VisionLanguageModel(model_dir)
        conn.send(("ready", f"loaded {model.model_id()} accepts_image={model.accepts_image()} "
                            f"({time.monotonic() - started:.1f}s)"))
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            item = conn.recv()
        except EOFError:
            break
        if item is None:
            break
        crop_rgb, prompt, max_new_tokens = item
        try:
            request = neat.genai.GenerationRequest()
            request.prompt = prompt
            request.images = [crop_rgb]
            request.max_new_tokens = max_new_tokens
            result = model.run(request)
            m = result.metrics
            conn.send(("ok", (result.text, m.generated_tokens, m.tokens_per_second,
                              m.time_to_first_token_s)))
        except Exception as exc:  # noqa: BLE001 - keep the server alive
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


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
        self._proc = None    # child process that owns the VLM (real mode only); see preload()
        self._conn = None    # parent end of the pipe to that process
        self.completed = 0   # captions actually produced (for the run summary)

    # -- lifecycle ---------------------------------------------------------- #
    def preload(self) -> None:
        """Start the VLM process and wait for the model to load, before streaming starts.

        The VLM runs in a separate process (see ``_vlm_server``), started with ``spawn`` so
        it inherits nothing from this one. Waiting here pays the ~10 s weight load at
        startup, where blocking is free, instead of mid-stream.
        """
        if self.dry_run or self._proc is not None:
            return
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_vlm_server, args=(child, self.cfg.vlm_model_dir),
                                 name="vlm", daemon=True)
        self._proc.start()
        child.close()
        if not self._conn.poll(self._load_timeout_s):
            self._stop_process()
            raise RuntimeError(f"VLM did not load within {self._load_timeout_s:.0f} s: "
                               f"{self.cfg.vlm_model_dir}")
        try:
            status, detail = self._conn.recv()
        except EOFError:
            status, detail = "error", "the VLM process exited while loading"
        if status != "ready":
            self._stop_process()
            raise RuntimeError(f"VLM failed to load from {self.cfg.vlm_model_dir}: {detail}")
        print(f"vlm: {detail}", flush=True)

    _load_timeout_s = 180.0

    def _stop_process(self) -> None:
        if self._proc is None:
            return
        try:
            self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=10.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5.0)
        self._conn.close()
        self._proc = None
        self._conn = None

    def start(self) -> None:
        if not self.started:
            self.worker.start()
            self.started = True

    def close(self) -> None:
        self.stop_event.set()
        if self.started:
            # A generation in flight takes seconds and cannot be interrupted; joining
            # for less than that tears the interpreter down while a daemon thread is
            # still inside a C++ MLA call.
            self.worker.join(timeout=self._join_timeout_s())
            if self.worker.is_alive():
                print("vlm: worker still busy at shutdown; stopping the VLM process",
                      flush=True)
        self._stop_process()

    def _join_timeout_s(self) -> float:
        if self.dry_run:
            return 2.0
        # ~20 tok/s observed on this board, plus TTFT and slack.
        return max(5.0, self.cfg.vlm_max_new_tokens / 15.0 + 3.0)

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

    def gate_candidate(self, boxes: list[dict], frame_shape) -> "dict | None":
        """The box the VLM would treat as its subject this frame — for the overlay.

        Applies only the class/score/area gate (the same _passes_gate used by the
        real send path) and returns the highest-score gate-passer, or None. It is
        deliberately NON-mutating: no rate-limit, no dedup memory, no queue side
        effects. Used purely to highlight that one box in red in the video output,
        so the highlight can update every frame without disturbing the send logic.
        """
        height, width = frame_shape[:2]
        frame_area = float(width * height)
        candidates = [b for b in boxes if self._passes_gate(b, frame_area)]
        if not candidates:
            return None
        return max(candidates, key=lambda b: b["score"])

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
            # Deliberately not _remember()ing here: the caller records the object only once the trigger
            # is queued, else a box dropped by the bounded queue is suppressed for the full cooldown.
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
        except Full:
            # Bounded queue is full: a VLM call is still in flight. Drop and move on;
            # the detector keeps running. This is the latency-absorbing backpressure.
            # The object is NOT remembered, so it can trigger again once the VLM frees.
            self.last_enqueue_at = now
            print("vlm: queue busy, dropping trigger", flush=True)
            return
        self.last_enqueue_at = now
        self._remember(box, now)

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

        if self._proc is None:
            self.preload()
        if not self._proc.is_alive():
            raise RuntimeError("the VLM process has exited")

        # CRITICAL colour correctness: VLM images must be uint8 HWC *RGB*. Our crop is
        # OpenCV-native BGR, so convert here. Skipping this silently feeds the VLM
        # channel-swapped images and quietly degrades every answer.
        crop_rgb = np.ascontiguousarray(cv2.cvtColor(trigger.crop_bgr, cv2.COLOR_BGR2RGB))

        self._conn.send((crop_rgb, self._prompt_for(trigger.label), self.cfg.vlm_max_new_tokens))
        timeout = self._join_timeout_s() + 30.0
        if not self._conn.poll(timeout):
            raise RuntimeError(f"no VLM answer within {timeout:.0f} s")
        status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(payload)
        text, tokens, tok_per_s, ttft = payload
        self.completed += 1
        print(
            f"vlm[{trigger.label} score={trigger.score:.2f} bbox={trigger.bbox}]: "
            f"{text.strip()}  "
            f"({tokens} tok, "
            f"{tok_per_s:.1f} tok/s, "
            f"ttft={ttft:.2f}s)",
            flush=True,
        )
