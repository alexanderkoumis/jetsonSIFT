#ifndef SIFT_H_
#define SIFT_H_

#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define ROWS_HALO_STEPS 1
#define	COLUMNS_BLOCKDIM_X 16
#define	COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define	COLUMNS_HALO_STEPS 1
#define KERNEL_RADIUS 2

#define MAXEXTREMAS 10000
#define CONTRASTTHRESHOLD 72 // Base value of 64 + 8
#define EDGETHRESHOLD 10 // r

#define SIFT_MAX_INTERP_STEPS 5

#include <cuda.h> // 'exit'?
#include <cuda_runtime_api.h> // IntelliSense info for cudaCreateTextureObject, cudaDestroyTextureObjectm, cudaFree
#include <device_launch_parameters.h> // IntelliSense info for cudaError_t, and blockIdx.x/y/z
#include "helper_math.h" // make_int4
#include "utils.h"
#include "opencv2/gpu/device/common.hpp"

#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__




#include <math.h> // abs


using namespace cv::gpu;

void createDoGSpace(unsigned char* inImage, float** deviceDoGData, int scales, int scaleRows, int scaleCols);
void findExtremas(float* deviceDoGData, int4** extremaBuffer, unsigned int** maxCounter, int octave, int scales, int rows, int cols);
void localization(float* deviceDoGData, int rows, int cols, int scales, int octave, int octaves, int4* extremaBuffer, unsigned int* maxCounter, float* xRow, float* yRow, float* octaveRow, float* sizeRow, float* angleRow, float* responseRow);

#endif