cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
 
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_32,code=sm_32)
 
cuda_add_executable(
  jetsonSIFT
  sift.h
  utils.h
  gpuSift.h
  gpuSift.cpp
  main.cpp
  sift.cu
)

target_link_libraries(
  jetsonSIFT
  -lopencv_core
  -lopencv_highgui
  -lopencv_imgproc
  -lopencv_features2d
) 
