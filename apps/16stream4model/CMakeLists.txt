cmake_minimum_required(VERSION 3.16)
project(stream16_model4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)
find_package(Threads REQUIRED)

add_executable(16stream4model main.cpp)
# support/detection_egress.h needs nlohmann/json from the sysroot.
target_include_directories(16stream4model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(16stream4model PRIVATE SimaNeat::sima_neat Threads::Threads)
