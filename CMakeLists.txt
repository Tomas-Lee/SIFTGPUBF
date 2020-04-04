cmake_minimum_required(VERSION 3.5)
project(siftgpu)

add_compile_options(-std=c++11)

#CUDA
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS -std=c++11)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#glog
find_package(Glog REQUIRED)

cuda_add_executable(siftgpu
        main.cu
        CUDASiftConstant.cu
        SiftGPU.cpp
        ProgramCU.cu
        SiftPyramid.cpp
        GlobalUtil.cpp
        CuTexImage.cpp
        SiftMatch.cpp
        SIFTImageManager.cpp
        )
target_link_libraries(siftgpu
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        ${GLOG_LIBRARIES}
        )

