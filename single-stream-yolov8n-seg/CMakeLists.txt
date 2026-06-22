cmake_minimum_required(VERSION 3.16)
project(single_stream_yolov8n_seg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(single_stream_yolov8n_seg main.cpp)
target_link_libraries(single_stream_yolov8n_seg PRIVATE SimaNeat::sima_neat)
