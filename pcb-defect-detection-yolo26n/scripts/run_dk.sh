#!/usr/bin/env bash
# Run the PLC defect detector on the connected Modalix DevKit via `dk`.
# Targets must live under /workspace. Run from a NEAT SDK shell paired with the
# board (so `dk` is on PATH and /workspace maps to this host's modalix_workspace).
#
#   bash scripts/run_dk.sh                 # uses config/default.conf
#   bash scripts/run_dk.sh --score 0.30    # extra flags forwarded to main.py
#
# `dk` SSHes to the board (sima@<devkit-ip>), activates pyneat, runs main.py on
# the MLA, and streams logs back. Annotated images land in the shared
# output_images/ (this project is under the /workspace NFS mount).
set -euo pipefail

PROJ=/workspace/NEAT/demo-neat/plc-defect-detection-yolo26n
dk "$PROJ/main.py" --config "$PROJ/config/default.conf" "$@"
