# Source this on the DevKit before running anything that uses the framemarker plugin:
#     source scripts/env.sh
# It changes the current shell only.
#
# Why three variables:
#  - GST_PLUGIN_PATH is enough for plain gst-launch-1.0 / gst-inspect-1.0.
#  - A Neat application ignores it. At startup the Neat runtime replaces GStreamer's plugin
#    path with its own plugin folder (SIMA_GST_NEAT_ONLY=1 by default) and unsets
#    GST_PLUGIN_PATH, so the app fails with build.plugin_missing.
#  - With SIMA_GST_NEAT_ONLY=0, Neat prepends its folder to GST_PLUGIN_PATH_1_0 and keeps what
#    is already there. GStreamer reads GST_PLUGIN_PATH_1_0 in preference to GST_PLUGIN_PATH,
#    so the plugin folder has to be in GST_PLUGIN_PATH_1_0.
_cp_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/install/lib/gstreamer-1.0"
export GST_PLUGIN_PATH="${_cp_dir}${GST_PLUGIN_PATH:+:${GST_PLUGIN_PATH}}"
export GST_PLUGIN_PATH_1_0="${_cp_dir}${GST_PLUGIN_PATH_1_0:+:${GST_PLUGIN_PATH_1_0}}"
export SIMA_GST_NEAT_ONLY=0
echo "GST_PLUGIN_PATH_1_0=${GST_PLUGIN_PATH_1_0}  SIMA_GST_NEAT_ONLY=${SIMA_GST_NEAT_ONLY}"
unset _cp_dir
