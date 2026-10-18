cmake_minimum_required(VERSION 3.10)
project(RmVisionWeek2)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.31.0")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)
add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS} onnxruntime)
