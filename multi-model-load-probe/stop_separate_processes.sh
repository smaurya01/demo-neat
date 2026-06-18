#!/usr/bin/env bash
set -euo pipefail

for pidfile in /tmp/multi-model-load-probe/*.pid; do
  [[ -e "$pidfile" ]] || continue
  pid=$(cat "$pidfile")
  kill "$pid" 2>/dev/null || true
  rm -f "$pidfile"
done
