cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)

project(yolovTrtPose LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

find_package(CUDA 11.8 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        include
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        )

include_directories(${INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
        main.cpp
        src/yolo_v8_pose.cc
        include/common.hpp
        )

link_directories(${ALL_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin ${CUDA_LIBRARIES} ${OpenCV_LIBS})
