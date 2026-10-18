cmake_minimum_required(VERSION 3.10)
project(RmVisionWeek2)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.31.0")

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# ===== 目标 1：传统视觉方案 =====
add_executable(main 
    src/main.cpp
    src/detector.cpp
    src/solver.cpp
)
target_link_libraries(main ${OpenCV_LIBS} onnxruntime)

# ===== 目标 2：纯血 YOLO 方案 =====
add_executable(main_yolo 
    src/main_yolo.cpp
    src/solver.cpp    # <--- 加上这行，把解算模块连进来
)

target_link_libraries(main_yolo ${OpenCV_LIBS} onnxruntime)