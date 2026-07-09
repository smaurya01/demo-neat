# T5 phase 1 — model preparation status

Owner: Agent C. Date: 2026-07-09. Target: modalix, INT8, calibration = 20 images from
`assets/yolo_calibration`. Artifact policy: strict (exactly one `.elf`, zero `.so`).

All CPU prep (export + surgery + audit) is complete for all four models; every surgery graph
audits with **0 unsupported ops** for int8. Compiles run one at a time through the global
compile-slot wrapper, in cheapest-first order.

| Model | Task | Source | Exported | Surgery | Audit (unsupp.) | Compiled | Archive valid (1 elf / 0 so) | Smoke-tested | Blocked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11s | detection | Ultralytics yolo11s.pt | yes | yes (6 out) | 0 | **PASS** | **PASS** (1 elf / 0 so) | host-ONNX PASS; board pending | no |
| yolo11s-seg | segmentation | Ultralytics yolo11s-seg.pt | yes | yes (10 out) | 0 | **PASS** | **PASS** (1 elf / 0 so) | host-ONNX PASS; board pending | no |
| yolo26s-pose | pose | Ultralytics yolo26s-pose.pt | yes | yes (9 out) | 0 | **PASS** | **PASS** (1 elf / 0 so) | host-ONNX PASS; board pending | no |
| yolox_s | detection | Megvii yolox_s.onnx (0.1.1rc0) | yes | yes (3 out) | 0 | **PASS** | **PASS** (1 elf / 0 so) | host-ONNX PASS; board pending | no |

Compile order (cheapest / most-certain first): yolo11s -> yolo11s-seg -> yolo26s-pose -> yolox_s.

## Artifacts per model

- ONNX (exported): `work/<model>/onnx/<model>.onnx`
- compile_ready ONNX (surgery output): `work/<model>/surgery/<model>.compile_ready.onnx`
- audit: `work/<model>/reports/audit_compile_ready_int8.json`
- surgery writeup: `work/<model>/reports/SURGERY.md`
- head map: `work/<model>/reports/head_map.txt`
- compile log: `work/<model>/reports/compile_ready_int8.log`
- MPK archive: `work/<model>/compile_int8/<model>.compile_ready/<model>.compile_ready_mpk.tar.gz`
- archive validation: `work/<model>/reports/archive_validation_compile_ready_int8.json`

## Output contracts (raw heads exposed; host does task decode)

- **yolo11s** (6): `bbox_{0,1,2}` `[1,4,H,W]`, `class_logit_{0,1,2}` `[1,80,H,W]`.
- **yolo11s-seg** (10): the 6 detection tensors + `mask_coeff_{0,1,2}` `[1,32,H,W]` +
  `proto` `[1,32,160,160]`.
- **yolo26s-pose** (9): `bbox_{0,1,2}` `[1,4,H,W]` (no DFL) + `class_logit_{0,1,2}` `[1,1,H,W]`
  + `kpt_{0,1,2}` `[1,51,H,W]`.
- **yolox_s** (3): `yolox_head_{0,1,2}` `[1,85,H,W]` (0:4 reg, 4 obj, 5:85 cls).

_Status of the "Compiled / Archive valid" columns is updated below as each slot completes._

## Compile results

- **yolo11s** — PASS. ~6 min. Plugin distribution MLA:1, EV74:12, **A65:0** (no host
  fallback). Archive: single `.elf` (`yolo11s.compile_ready_stage1_mla.elf`), zero `.so`.
  `05_validate_archive.py` -> `status pass`.
- **yolo11s-seg** — PASS. ~6 min. Plugin distribution MLA:1, EV74:16, **A65:0**. The proto
  `ConvTranspose` compiled onto the MLA (no host fallback). Archive: single `.elf`, zero `.so`.
  `05_validate_archive.py` -> `status pass`.
- **yolo26s-pose** — PASS. ~6 min. Plugin distribution MLA:1, EV74:21, **A65:0**. Both
  attention Einsum blocks and the keypoint head compiled onto the MLA. Archive: single `.elf`,
  zero `.so`. `05_validate_archive.py` -> `status pass`.
- **yolox_s** — PASS. ~8 min. Plugin distribution MLA:1, EV74:9, **A65:0**. The non-Ultralytics
  decoupled head compiled cleanly onto the MLA with the flatten tail removed. Archive: single
  `.elf`, zero `.so`. `05_validate_archive.py` -> `status pass`. The genuinely-new YOLOX surgery
  landed — no blocker.

## Host-side ONNX smoke test (scripts/17_t5_onnx_smoke.py)

All four compile_ready ONNX graphs run under onnxruntime on a real calibration image and emit
every documented head with a finite tensor of the expected shape (`reports/onnx_smoke.json`
per model). This is a host sanity gate; an on-device (MLA) smoke test on the shared DevKit
board is the remaining validation step (board was left to Agent A during this wave).

## Summary

**4 / 4 models compiled to INT8 modalix archives that pass the strict one-ELF / zero-`.so`
policy. Zero blockers.** Every archive shows A65:0 (nothing fell back to the host CPU).
