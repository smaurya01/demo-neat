cmake_minimum_required(VERSION 3.16)
project(multi_model_load_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(multi_model_load_probe main.cpp)
target_link_libraries(multi_model_load_probe PRIVATE SimaNeat::sima_neat)
