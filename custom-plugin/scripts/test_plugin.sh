#!/usr/bin/env bash
# Smoke-test the framemarker plugin on the DevKit with plain GStreamer, no Neat involved.
#   1. GStreamer can find and load the plugin.
#   2. It processes frames in every mode without errors.
#   3. It writes one JPEG per mode to out/ so you can look at the result.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh >/dev/null
mkdir -p out

echo "== 1. inspect"
gst-inspect-1.0 framemarker | sed -n '1,/Pad Templates/p' | head -20

echo "== 2. 300 frames through each mode (NV12 and I420)"
for fmt in NV12 I420; do
  for mode in none grayscale invert; do
    gst-launch-1.0 -q videotestsrc num-buffers=300 \
      ! "video/x-raw,format=${fmt},width=1280,height=720,framerate=30/1" \
      ! framemarker mode=${mode} border-width=16 color=0xFF0000 log-interval=100 \
      ! fakesink
  done
done

echo "== 3. snapshots"
for mode in none grayscale invert; do
  gst-launch-1.0 -q videotestsrc num-buffers=60 pattern=smpte \
    ! "video/x-raw,format=NV12,width=640,height=360,framerate=30/1" \
    ! framemarker mode=${mode} border-width=16 color=0xFF0000 \
    ! videoconvert ! jpegenc snapshot=false ! multifilesink location="out/${mode}_%03d.jpg"
  ls out/${mode}_059.jpg
done
echo "PASS"
