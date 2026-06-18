#!/usr/bin/env bash
set -euo pipefail

APP=/workspace/demo-neat/multi-model-load-probe/build/multi_model_load_probe

dk "$APP" --allow-missing --load-only "$@"
