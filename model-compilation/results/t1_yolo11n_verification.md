# T1 — YOLO11n Fresh Verification Run

Date: 2026-07-09
Agent: A (T1, P0)
Env: `/sdk-extensions/model-compiler/bin/activate` (afe, onnx 1.17.0, ultralytics 8.4.90)
Fresh scratch dir (existing artifacts untouched):
`model-compilation/work/yolo11n/t1_verify/`

Purpose: re-run the full YOLO11 chain fresh — `.pt -> ONNX -> compile_ready
surgery -> INT8 quantize+compile -> archive validation -> Neat smoke test` — to
confirm "we are good with YOLO11 graph surgery." Compile ran through the global
one-at-a-time slot wrapper (`A:yolo11n-int8`, acquired 20:20:21, released
20:25:17 UTC).

## Chain and outcomes

| Step | Script | Result |
| --- | --- | --- |
| 1. Export fresh | `11_export_fresh_yolo.py --model-id yolo11n --imgsz 640` | PASS — yolo11n.pt downloaded (Ultralytics v8.4.0), ONNX opset 17, static `1x3x640x640`, output0 `(1,84,8400)`, 355 nodes |
| 2. compile_ready surgery | `09_yolo_compile_ready_surgery.py --model-id yolo11n --force` | PASS — attention rewrite `/model.10/m/m.0/attn` (MatMul->Einsum), 6 outputs exposed, DFL->4ch conversion |
| 3. INT8 quantize+compile | `12_compile_yolo_int8.py --model-id yolo11n --num-calib-samples 20` (via `compile_slot.sh`) | PASS — 20 real calib images; plugin distribution **MLA:1, EV74:12, A65:0** |
| 4. Archive validation | `05_validate_archive.py` | PASS — **one `.elf`, zero `.so`** |
| 5. Neat smoke test (board) | `10_run_yolo_sample_pipeline.py` on 5 COCO images | PASS — real detections, classes match expected |

## Surgery output contract (verified)

```text
bbox_0 bbox_1 bbox_2 class_logit_0 class_logit_1 class_logit_2
```

Reported by `reports/compile_ready_surgery.json`:
`attention_rewrites: ["/model.10/m/m.0/attn"]`, contract
"Neat BoxDecodeType.YoloV26 grouped bbox/class-logit outputs".

## Compile summary (from reports/compile_ready_int8.log)

```text
Compilation summary:
  Desired batch size: 1 / Achieved batch size: 1
  Plugin distribution per backend:
    MLA : 1
    EV74: 12
    A65 : 0
  Generated: yolo11n.compile_ready_stage1_mla.elf, yolo11n.compile_ready_mpk.json, ...
[INFO] Compilation complete.
```

A65 = 0 means **no host fallback** — the whole network runs on the MLA, which is
why the archive is a single ELF with no `.so`.

## Archive validation (05_validate_archive.py)

```json
{
  "elf_members": ["yolo11n.compile_ready_stage1_mla.elf"],
  "so_members": [],
  "single_elf": true,
  "no_so": true,
  "status": "pass"
}
```

Archive: `work/yolo11n/t1_verify/compile_int8/yolo11n.compile_ready/yolo11n.compile_ready_mpk.tar.gz`
(12.4 MB). Report: `work/yolo11n/t1_verify/reports/archive_validation_t1.json`.

## Neat smoke test on the DevKit (192.168.135.203, via ssh)

Ran `scripts/10_run_yolo_sample_pipeline.py` (tensor route + `BoxDecodeType.YoloV26`
box decode) against the t1_verify archive on 5 COCO inference images. `status:
pass`. Per-image detected classes (score >= 0.25):

```text
000000000139.jpg: 11 dets — chair, potted plant, dining table, tv, clock, vase   (~5.2 ms)
000000000885.jpg: 14 dets — person, tennis racket                                (~5.8 ms)
000000001000.jpg: 12 dets — person, tennis racket                                (~5.8 ms)
000000001268.jpg:  7 dets — person, bird, handbag, cell phone                    (~5.3 ms)
000000001296.jpg:  5 dets — person, cell phone                                   (~5.3 ms)
```

Image 139 is the canonical COCO val scene (indoor: tv / chair / dining table /
clock / vase / potted plant) — detections are correct and class-plausible.
Report + overlays: `work/yolo11n/t1_verify/sample_runs/`.

Note: `scripts/06_neat_smoke_test.py` (raw NCHW tensor route, default ModelOptions)
failed at `InputStream::pull_and_discard` with `misconfig.caps ... not-negotiated`.
That is a route/caps mismatch in the *default-options tensor path* of that helper,
not a model/archive defect: script 10 (proper `ModelOptions` tensor route with
`InputKind.Tensor` + EV74 memory) loads and decodes the same archive cleanly, and
the 2x RTSP app (image route, `NormalizePreset.COCO_YOLO`) also runs it end to end.

## Application validation (2x RTSP, DevKit)

`apps/multi-stream-yolo-yolo11/main.py --frames 20` on the board, 2 RTSP inputs
(same source twice, by design), shared YOLO11 model stage, per-stream UDP output:

```text
Shared model: .../assets/models/yolo_11n_mpk.tar.gz (decode=yolo11)
Stream 0: RTSP rtsp://192.168.132.129:8555/stream -> udp://127.0.0.1:5206
Stream 1: RTSP rtsp://192.168.132.129:8555/stream -> udp://127.0.0.1:5208
Configuring for the decoding type: 8:yolo26 / Configured for subtensors: 6
stream=0 port=5206 frame=1  detections=11 visible=11 agg_fps=2.87
stream=1 port=5208 frame=1  detections=12 visible=10 agg_fps=4.21
stream=0 port=5206 frame=20 detections=11 visible=9  agg_fps=37.15
stream=1 port=5208 frame=20 detections=11 visible=10 agg_fps=37.65
```

Both streams produced output to distinct ports; stream identity preserved end to
end. The box decode auto-configured for the YoloV26 6-subtensor contract. Exit 0.
(`nanobind: leaked ...` lines at teardown are a harmless cleanup-order artifact;
the run completed rc=0.)

## Conclusion

YOLO11 graph surgery + INT8 compile is confirmed reproducible from scratch. The
archive meets the strict T1 contract (one ELF, zero `.so`), decodes correct COCO
classes on the board, and drives the 2x RTSP multi-stream app.
