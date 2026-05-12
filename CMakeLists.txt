cmake_minimum_required(VERSION 3.20)

project(adas LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Which GPU architecture to compile for
# 86 = RTX 3000 series, 89 = RTX 4000 series
# Find yours at: https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_ARCHITECTURES 86)

set(TENSORRT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/TensorRT-10.16.1.11" CACHE PATH "Path to TensorRT extraction directory")


list(APPEND CMAKE_PREFIX_PATH ${TENSORRT_DIR})

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)


# Build a library from all our source files
add_library(inference_lib
    src/yolo_engine.cpp
    src/video_source.cpp
    src/pipeline.cpp
    kernels/preprocess.cu # compiled by nvcc due to the .cu extension
)

find_library(TENSORRT_LIB nvinfer HINTS ${TENSORRT_DIR}/lib)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_DIR}/lib)

# Tell the compiler where to find .hpp files
target_include_directories(inference_lib PUBLIC
    include                          # our own headers
    ${OpenCV_INCLUDE_DIRS}           # OpenCV headers
    ${CUDAToolkit_INCLUDE_DIRS}      # CUDA headers
    ${TENSORRT_DIR}/include
)

# Link the libraries
target_link_libraries(inference_lib PUBLIC
    ${OpenCV_LIBS}                   # OpenCV
    CUDA::cudart
    CUDA::cuda_driver
    ${TENSORRT_LIB}                          # TensorRT core
    ${TENSORRT_PLUGIN_LIB}                   # TensorRT built-in layers
    nlohmann_json::nlohmann_json
)

# Build the final executable
add_executable(yolo_run src/main.cpp)
target_link_libraries(yolo_run PRIVATE inference_lib)
