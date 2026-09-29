# Builds libgstframemarker.so, a GStreamer plugin that provides the `framemarker` element.
#
# In the Neat Development Environment (SDK) the environment already points CC, CFLAGS,
# LDFLAGS and pkg-config at the Modalix sysroot, so a plain configure cross-compiles for the
# DevKit. On the DevKit itself the same commands build natively.
cmake_minimum_required(VERSION 3.16)
project(gst_framemarker VERSION 1.0.0 LANGUAGES C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-base-1.0
  gstreamer-video-1.0)

# MODULE = a shared object that is dlopen()ed, never linked against. GStreamer loads it.
add_library(gstframemarker MODULE gstframemarker.c)
target_link_libraries(gstframemarker PRIVATE PkgConfig::GST)
target_compile_options(gstframemarker PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(gstframemarker PRIVATE PACKAGE="framemarker")

# Installs to <prefix>/lib/gstreamer-1.0/libgstframemarker.so. scripts/env.sh points GStreamer and
# Neat there (GST_PLUGIN_PATH_1_0 + SIMA_GST_NEAT_ONLY=0).
install(TARGETS gstframemarker LIBRARY DESTINATION lib/gstreamer-1.0)
