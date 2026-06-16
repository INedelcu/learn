# Optional CMake build. From this folder:
#   cmake -S . -B build
#   cmake --build build --config Release
# Each NN_*.cpp becomes its own executable target named after the file.
cmake_minimum_required(VERSION 3.15)
project(MonteCarloTutorials CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB TUTORIALS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/[0-9][0-9]_*.cpp")
foreach(src ${TUTORIALS})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
