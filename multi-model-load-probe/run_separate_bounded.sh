#!/usr/bin/env bash
set -euo pipefail

APP=${APP:-/workspace/demo-neat/multi-model-load-probe/build/multi_model_load_probe}
RTSP_URL=${RTSP_URL:-rtsp://192.168.131.12:8555/stream}
UDP_HOST=${UDP_HOST:-192.168.131.12}
FRAMES=${FRAMES:-30}
LOG_DIR=${LOG_DIR:-/tmp/multi-model-load-probe-separate}

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

run_one() {
  local name=$1
  local port=$2
  "$APP" --only "$name" --rtsp "$RTSP_URL" --udp-host "$UDP_HOST" \
    --udp-port-base "$port" --frames "$FRAMES" > "$LOG_DIR/${name}.log" 2>&1
  echo $? > "$LOG_DIR/${name}.exit"
}

run_one yolov8n 5301 &
run_one yolov8n-seg 5302 &
run_one yolo26n 5303 &
run_one open_pose 5304 &
wait

for f in "$LOG_DIR"/*.exit; do
  echo "$(basename "$f" .exit) exit=$(cat "$f")"
done

for f in "$LOG_DIR"/*.log; do
  echo "--- $(basename "$f") ---"
  grep -E "RTSP input|Loaded models:|\\[[^]]+\\] frame=${FRAMES}|\\[probe\\] source_frames=${FRAMES}|NEAT error|Error:|Not Found|timed out|terminate called" "$f" | tail -30 || true
done
