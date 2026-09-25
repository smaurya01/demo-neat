#!/bin/bash
# Assign the 16 720p30 H.264 catalog clips below to Insight media sources src1..src16,
# start them, and verify every one delivers ~30 fps before the app is launched.
#
# Run on the SDK host (Insight on 127.0.0.1:9900), not on the board.
#   ./scripts/start_sources.sh            # stop all, assign, start, verify
#   ./scripts/start_sources.sh --check    # verify only
#   ./scripts/start_sources.sh --stop     # stop all sources
set -u
INSIGHT=${INSIGHT:-https://127.0.0.1:9900}
RTSP_HOST=${RTSP_HOST:-127.0.0.1}
N=16
FPS=30
# Insight source i+1 feeds stream i; streams 0-3 model 0, 4-7 model 1, 8-11 model 2, 12-15 model 3.
# Every clip is 1280x720, 30/1 fps, H.264 Constrained Baseline (no B-frames). Keep this list in
# step with the table in README.md.
CLIPS=(
  catalog/parking_garage_cars/parking_garage_cars_720p30_30fps_h264.mp4                  # src1
  catalog/sj_almaden_street/sj_almaden_street_720p30_30fps_h264.mp4                      # src2
  catalog/sj_highway/sj_highway_720p30_30fps_h264.mp4                                    # src3
  catalog/sj_intersection_san_carlos/sj_intersection_san_carlos_720p30_30fps_h264.mp4    # src4
  catalog/sj_highway2/sj_highway2_720p30_30fps_h264.mp4                                  # src5
  catalog/parking_garage_cars/parking_garage_cars_720p30_30fps_h264.mp4                  # src6
  catalog/sj_highway3/sj_highway3_720p30_30fps_h264.mp4                                  # src7
  catalog/sj_highway4/sj_highway4_720p30_30fps_h264.mp4                                  # src8
  catalog/sj_park/sj_park_720p30_30fps_h264.mp4                                          # src9
  catalog/sj_intersection_san_carlos/sj_intersection_san_carlos_720p30_30fps_h264.mp4    # src10
  catalog/sj_walking_street/sj_walking_street_720p30_30fps_h264.mp4                      # src11
  catalog/sj_almaden_street/sj_almaden_street_720p30_30fps_h264.mp4                      # src12
  catalog/person_gym_bike/person_gym_bike_720p30_30fps_h264.mp4                          # src13
  catalog/person_gym_jumping/person_gym_jumping_720p30_30fps_h264.mp4                    # src14
  catalog/person_gym_threadmill/person_gym_threadmill_720p30_30fps_h264.mp4              # src15
  catalog/person_gym_workout/person_gym_workout_720p30_30fps_h264.mp4                    # src16
)

api() {  # api <path> [json]; fails on HTTP errors and unreachable Insight
  if [ $# -gt 1 ]; then curl -skf --max-time 15 -H "Content-Type: application/json" -d "$2" "$INSIGHT/api/$1"
  else curl -skf --max-time 15 -X POST "$INSIGHT/api/$1"; fi
}

check() {
  echo "== delivery check: 20 s per source, all $N in parallel =="
  local tmp bad=0
  tmp=$(mktemp -d)
  # Expanded now: $tmp is local and out of scope by the time an EXIT trap runs.
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  for i in $(seq 1 $N); do
    ( timeout 26 ffmpeg -hide_banner -loglevel error -rtsp_transport tcp \
        -i "rtsp://$RTSP_HOST:8554/src$i" -t 20 -map 0:v -c copy -f framecrc - 2>/dev/null \
        | grep -vc '^#' > "$tmp/$i" ) &
  done
  wait
  for i in $(seq 1 $N); do
    n=$(cat "$tmp/$i"); n=${n:-0}; pct=$(( n * 100 / (FPS * 20) ))
    st=OK; [ "$pct" -lt 90 ] && { st=LOW; bad=$((bad+1)); }
    printf "  src%-2s model%d  %5s fps of %d (%3s%%)  %s\n" "$i" $(((i-1)/4)) \
      "$(awk -v n="$n" 'BEGIN{printf "%.1f", n/20}')" $FPS "$pct" "$st"
  done
  rm -rf "$tmp"
  [ $bad -eq 0 ] && echo "all $N sources delivering -- safe to launch the app" && return 0
  echo "$bad source(s) LOW -- re-run --check before launching"; return 1
}

case "${1:-}" in
  --stop)
    api mediasrc/stop-all >/dev/null || { echo "Insight not reachable at $INSIGHT"; exit 1; }
    echo "all sources stopped"; exit 0 ;;
  --check|"") ;;
  *) echo "usage: $0 [--check|--stop]"; exit 2 ;;
esac
command -v ffmpeg >/dev/null || { echo "ffmpeg not found (needed for the delivery check)"; exit 1; }
[ "${1:-}" = "--check" ] && { check; exit $?; }

have=$(curl -skf --max-time 15 "$INSIGHT/api/mediasrc/videos") \
  || { echo "Insight not reachable at $INSIGHT"; exit 1; }
for c in $(printf "%s\n" "${CLIPS[@]}" | sort -u); do
  echo "$have" | grep -qF "\"$c\"" \
    || { echo "Insight is missing $c (install it from the Insight catalog)"; exit 1; }
done

echo "== stopping all sources =="
api mediasrc/stop-all >/dev/null || { echo "stop-all failed"; exit 1; }
echo "== assigning src1..src$N =="
for i in $(seq 1 $N); do
  clip=${CLIPS[$((i-1))]}
  out=$(api mediasrc/assign "{\"index\":$i,\"file\":\"$clip\",\"transport\":\"rtsp\"}") \
    || { echo "assign src$i failed (HTTP error)"; exit 1; }
  echo "$out" | grep -q '"error"' && { echo "assign src$i failed: $out"; exit 1; }
  printf "  src%-2s model%d <- %s\n" "$i" $(((i-1)/4)) "$(basename "$clip")"
done
echo "== starting $N sources =="
out=$(api mediasrc/start-bulk "{\"count\":$N}") || { echo "start-bulk failed (HTTP error)"; exit 1; }
echo "$out" | grep -q '"success":true' || { echo "start-bulk failed: $out"; exit 1; }
echo "$out" | head -c 300; echo
sleep 8
check
