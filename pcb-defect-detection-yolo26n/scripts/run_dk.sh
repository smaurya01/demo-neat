#!/usr/bin/env bash
# Run the PCB defect detector on the connected Modalix DevKit via `dk`.
#
# `dk` is a shell *function* defined by the NEAT SDK shell, not a binary, so this
# wrapper must be **sourced** from a paired SDK shell (running it as `bash
# scripts/run_dk.sh` spawns a sub-shell that cannot see `dk`):
#
#   source scripts/run_dk.sh                 # uses config/default.conf
#   source scripts/run_dk.sh --score 0.30    # extra flags forwarded to main.py
#
# `dk` SSHes to the board (sima@<devkit-ip>), activates pyneat, runs main.py on
# the MLA, and streams logs back. Annotated images land in the shared
# output_images/ (this project is under the /workspace NFS mount).

# Project root = the directory containing this script's parent (scripts/..).
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dk "$PROJ/main.py" --config "$PROJ/config/default.conf" "$@"
