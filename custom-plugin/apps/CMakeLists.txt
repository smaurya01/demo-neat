# Sample Neat apps that use the `framemarker` plugin. Build from the SDK:
#
#   cmake -S apps -B apps/build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr
#   cmake --build apps/build -j2
#
# The plugin itself is not linked into these apps. GStreamer loads it at run time from
# GST_PLUGIN_PATH, so the apps only need the Neat library.
cmake_minimum_required(VERSION 3.16)
project(custom_plugin_apps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)
# Only videoio is needed (to probe the RTSP stream size and rate). It is located directly
# because the sysroot's OpenCVConfig.cmake references modules that are not installed.
find_path(OPENCV_INCLUDE_DIR opencv2/videoio.hpp PATH_SUFFIXES include/opencv4 REQUIRED)
find_library(OPENCV_CORE opencv_core PATH_SUFFIXES lib/aarch64-linux-gnu REQUIRED)
find_library(OPENCV_VIDEOIO opencv_videoio PATH_SUFFIXES lib/aarch64-linux-gnu REQUIRED)

foreach(app stream_to_insight detect_to_insight)
  add_executable(${app} ${app}/main.cpp)
  target_link_libraries(${app} PRIVATE SimaNeat::sima_neat ${OPENCV_VIDEOIO} ${OPENCV_CORE})
  target_include_directories(${app} PRIVATE ${OPENCV_INCLUDE_DIR})
endforeach()

# Keep the labels next to the detector binary so the default --labels path works.
configure_file(detect_to_insight/coco_labels.txt ${CMAKE_BINARY_DIR}/coco_labels.txt COPYONLY)
