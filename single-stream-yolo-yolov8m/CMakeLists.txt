cmake_minimum_required(VERSION 3.16)
project(single_stream_yolo_yolov8m LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(single_stream_yolo_yolov8m main.cpp)
target_link_libraries(single_stream_yolo_yolov8m PRIVATE SimaNeat::sima_neat)
