#!/usr/bin/env bash
set -euo pipefail

APP=/workspace/demo-neat/multi-model-load-probe/build/multi_model_load_probe
RTSP_URL=${RTSP_URL:-rtsp://192.168.131.12:8555/stream}
UDP_HOST=${UDP_HOST:-192.168.131.12}

mkdir -p /tmp/multi-model-load-probe

dk "$APP" --only yolov8n --rtsp "$RTSP_URL" --udp-host "$UDP_HOST" --udp-port-base 5201 \
  > /tmp/multi-model-load-probe/yolov8n.log 2>&1 &
echo $! > /tmp/multi-model-load-probe/yolov8n.pid

dk "$APP" --only yolov8n-seg --rtsp "$RTSP_URL" --udp-host "$UDP_HOST" --udp-port-base 5202 \
  > /tmp/multi-model-load-probe/yolov8n-seg.log 2>&1 &
echo $! > /tmp/multi-model-load-probe/yolov8n-seg.pid

dk "$APP" --only yolo26n --rtsp "$RTSP_URL" --udp-host "$UDP_HOST" --udp-port-base 5203 \
  > /tmp/multi-model-load-probe/yolo26n.log 2>&1 &
echo $! > /tmp/multi-model-load-probe/yolo26n.pid

dk "$APP" --only open_pose --rtsp "$RTSP_URL" --udp-host "$UDP_HOST" --udp-port-base 5204 \
  > /tmp/multi-model-load-probe/open_pose.log 2>&1 &
echo $! > /tmp/multi-model-load-probe/open_pose.pid

echo "Started separate-process probe. Logs and PIDs are in /tmp/multi-model-load-probe."
