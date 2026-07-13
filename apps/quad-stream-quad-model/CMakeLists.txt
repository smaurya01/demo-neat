cmake_minimum_required(VERSION 3.16)
project(quad_stream_quad_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)
find_package(Threads REQUIRED)

add_executable(quad_stream_quad_model main.cpp)
target_link_libraries(quad_stream_quad_model
  PRIVATE
    SimaNeat::sima_neat
    Threads::Threads)
