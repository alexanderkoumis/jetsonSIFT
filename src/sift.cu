#include "sift.h"

#include <cstdio>

#include <opencv2/gpu/device/utility.hpp>

////////////////////////////////////////////////////////////////////////
// Creating Difference-of-Gaussian Space                              //
__constant__ float gaussKernel1D[6][5] = {	{ 0.010333864010783912, 0.20756120714779008, 0.564209857682852,   0.20756120714779008, 0.010333864010783912 },
											{ 0.05448868454964433,  0.24420134200323346, 0.40261994689424435, 0.24420134200323346, 0.05448868454964433  },
											{ 0.11170336406408216,  0.23647602357935057, 0.30364122471313454, 0.23647602357935057, 0.11170336406408216  },
											{ 0.15246914402033807,  0.2218412955437766,  0.25137912087177056, 0.2218412955437766,  0.15246914402033807  },
											{ 0.17554682216870093,  0.21174988708961204, 0.22540658148337414, 0.21174988708961204, 0.17554682216870093  },
											{ 0.1876271619513903,   0.2060681238893415,  0.21260942831853635, 0.2060681238893415,  0.1876271619513903   }  };

__global__ void rowConvolve (float* outImg, unsigned char* inImg, int scales, int rows, int cols, int pitch)
{
	__shared__ unsigned char neighborhood[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

	inImg += baseY * pitch + baseX;
	outImg += baseY * pitch + baseX;

	//Load main data
	#pragma unroll
	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		neighborhood[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = inImg[i * ROWS_BLOCKDIM_X];
	}

	//Load left halo
	#pragma unroll
	for (int i = 0; i < ROWS_HALO_STEPS; i++)
	{
		neighborhood[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? inImg[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Load right halo
	#pragma unroll
	for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
	{
		neighborhood[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 
			(cols - baseX > i * ROWS_BLOCKDIM_X) 
			?
			inImg[i * ROWS_BLOCKDIM_X] 
			:
			0;
	}

	__syncthreads();

	//Compute and store results
	#pragma unroll
    for (int scl = 0; scl < scales; scl++)
	{
		#pragma unroll
		for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
		{
			float sum = 0;
			#pragma unroll
			for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
			{
				sum += gaussKernel1D[scl][KERNEL_RADIUS - j] * neighborhood[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
			}
			outImg[(i * ROWS_BLOCKDIM_X) + (rows * cols * scl * sizeof(unsigned char))] = sum;
		}
		__syncthreads();
	}
}

__global__ void colConvolve (float* outImg, float* inImg, int scales, int rows, int cols, int pitch)
{
	__shared__ float neighborhood[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	inImg += baseY * pitch + baseX;
	outImg += baseY * pitch + baseX;

	//Load main data
	#pragma unroll
	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		neighborhood[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = inImg[i * COLUMNS_BLOCKDIM_Y * pitch];
	}

	//Load upper halo
	#pragma unroll
	for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
	{
		neighborhood[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? inImg[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Load lower halo
	#pragma unroll
	for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
	{
		neighborhood[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (rows - baseY > i * COLUMNS_BLOCKDIM_Y) ? inImg[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Compute and store results
	__syncthreads();

	#pragma unroll
    for (int scl = 0; scl < scales; scl++)
	{
		#pragma unroll
		for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
		{
			float sum = 0;
			#pragma unroll
			for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
			{
				sum += gaussKernel1D[scl][KERNEL_RADIUS - j] * neighborhood[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
			}
			outImg[i * COLUMNS_BLOCKDIM_Y * pitch + (rows * cols * scl * sizeof(unsigned char))] = sum;
		}
	}
}

__global__ void difference (float* difImage, float* inImage, int scales, int rows, int cols)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	for (int scl = 0; scl < (scales - 1); scl++)
	{
		int location = (y * cols + x) + (scl * rows * cols);
		int nextLocation = location + (rows * cols);
		float difference = inImage[location] - inImage[nextLocation];
		difImage[location] = 64 +  10 * difference;
	}
}

__global__ void convertToUnsignedChar (unsigned char* uCharData, float* deviceDoGData, int scales, int rows, int cols)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	for (int scl = 0; scl < (scales - 1); scl++)
	{
		int location = (y * cols + x) + (scl * rows * cols);
		uCharData[location] = (unsigned char)deviceDoGData[location];
	}
}


void createDoGSpace(unsigned char* inImage, float** deviceDoGData, int scales, int rows, int cols)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

 	unsigned char* deviceInputData;
	float* deviceConvolveBuffer;
	float* deviceDifferenceData;
	
	checkCudaErrors(cudaMalloc((void**)&deviceInputData, rows * cols * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void**)&deviceConvolveBuffer, rows * cols * scales * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&deviceDifferenceData, rows * cols * scales * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)deviceDoGData, rows * cols * (scales - 1) * sizeof(float)));
	cudaMemset(deviceInputData, 0, rows * cols * sizeof(unsigned char));
	cudaMemset(deviceConvolveBuffer, 0, rows * cols * scales * sizeof(float));
	cudaMemset(deviceDifferenceData, 0, rows * cols * scales * sizeof(float));
	cudaMemset(deviceDoGData, 0, rows * cols * (scales - 1) * sizeof(float));

	checkCudaErrors(cudaMemcpy(deviceInputData, inImage, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
	assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
	assert(cols % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
	assert(rows % ROWS_BLOCKDIM_Y == 0);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the allocation: %f ms\n", time);
	time = 0;
	cudaEventRecord(start, 0);

	dim3 rowThreads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
	dim3 rowBlocks(cols / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), rows / ROWS_BLOCKDIM_Y);

	rowConvolve <<< rowBlocks, rowThreads >>> (deviceConvolveBuffer, deviceInputData, scales, cols, rows, cols);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for row kernel: %f ms\n", time);
	time = 0;
	cudaEventRecord(start, 0);

	assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
	assert(cols % COLUMNS_BLOCKDIM_X == 0);
	assert(rows % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

	dim3 colThreads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
	dim3 colBlocks(cols / COLUMNS_BLOCKDIM_X, rows / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
  	
	colConvolve <<< colBlocks, colThreads >>> (deviceDifferenceData, deviceConvolveBuffer, scales, cols, rows, cols);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the column kernel: %f ms\n", time);
	time = 0;
	cudaEventRecord(start, 0);

	dim3 difThreads(16, 16);
	dim3 difBlocks(cols / difThreads.x, rows / difThreads.y );

	difference <<< difBlocks, difThreads >>> (*deviceDoGData, deviceDifferenceData, scales, rows, cols);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the difference kernel: %f ms\n", time);


	unsigned char* uCharData;

	checkCudaErrors(cudaMalloc((void**)&uCharData, rows * cols * scales * sizeof(unsigned char)));
	cudaMemset(uCharData, 0, rows * cols * sizeof(unsigned char));

	convertToUnsignedChar <<< difBlocks, difThreads >>> (uCharData, *deviceDoGData, scales, rows, cols);
	cudaDeviceSynchronize();
	uCharData += (rows * cols * 0); // Adjust the integer to see respective DoG scale


	cudaMemcpy(inImage, uCharData, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(deviceInputData);
	cudaFree(deviceConvolveBuffer);
	cudaFree(deviceDifferenceData);
}
//                                                                    //
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// Extracting extremas                                                //
__global__ void findExtremasGPU(float* deviceDoGData, int4* extremaBuffer, unsigned int* maxCounter, int octave, int scales, int rows, int cols)
{
	extern __shared__ float neighborhood[];

	const int gridDim_Y = gridDim.y / scales; // Number of vertical blocks in a scale
	const int blockIdx_Y = blockIdx.y % gridDim_Y; // Scale-relative blockIndex.y
	const int blockIdx_Z = blockIdx.y / gridDim_Y; // Scale-relative blockIndex.z

	const int scale = blockIdx_Z + 1; // Scale that will hold the center of the 3x3x3 box

	const int x = threadIdx.x + blockIdx.x * (blockDim.x - 2); // Image-relative x coordinate
	const int y = threadIdx.y + blockIdx_Y * (blockDim.y - 2); // Image-relative y coordinate

	const int zOff = blockDim.x * blockDim.y; // Area of the block
	const int center = threadIdx.x + threadIdx.y * blockDim.x + zOff; // The shared memory-relative center of the 3x3x3 box

	neighborhood[center - zOff] = deviceDoGData[(::min(::max(y, 0), rows - 1)) * cols + (::min(::max(x, 0), cols - 1)) + (scale - 1) * (rows * cols)];
	neighborhood[center       ] = deviceDoGData[(::min(::max(y, 0), rows - 1)) * cols + (::min(::max(x, 0), cols - 1)) + (scale    ) * (rows * cols)];
	neighborhood[center + zOff] = deviceDoGData[(::min(::max(y, 0), rows - 1)) * cols + (::min(::max(x, 0), cols - 1)) + (scale + 1) * (rows * cols)];

	__syncthreads();

	float candidate = neighborhood[center];
	if ( (y < rows - 1) && (x < cols - 1) && (threadIdx.x > 0) && (threadIdx.x < blockDim.x - 1) && (threadIdx.y > 0) && (threadIdx.y < blockDim.y - 1))
	{
		if (candidate > CONTRASTTHRESHOLD)
		{
			const bool maxima = candidate > neighborhood[center - 1 - blockDim.x - zOff]
							&&  candidate > neighborhood[center     - blockDim.x - zOff]
							&&  candidate > neighborhood[center + 1 - blockDim.x - zOff]
							&&  candidate > neighborhood[center - 1              - zOff]
							&&  candidate > neighborhood[center                  - zOff]
							&&  candidate > neighborhood[center + 1              - zOff]
							&&  candidate > neighborhood[center - 1 + blockDim.x - zOff]
							&&  candidate > neighborhood[center     + blockDim.x - zOff]
							&&  candidate > neighborhood[center + 1 + blockDim.x - zOff] 
			
							&&	candidate > neighborhood[center - 1 - blockDim.x       ] 
							&&  candidate > neighborhood[center     - blockDim.x       ]
							&&  candidate > neighborhood[center + 1 - blockDim.x       ]
							&&  candidate > neighborhood[center - 1                    ]
							&&  candidate > neighborhood[center + 1                    ]
							&&  candidate > neighborhood[center - 1 + blockDim.x       ]
							&&  candidate > neighborhood[center     + blockDim.x       ]
							&&  candidate > neighborhood[center + 1 + blockDim.x       ]

							&&  candidate > neighborhood[center - 1 - blockDim.x + zOff]
							&&  candidate > neighborhood[center     - blockDim.x + zOff]
							&&  candidate > neighborhood[center + 1 - blockDim.x + zOff]
							&&  candidate > neighborhood[center - 1              + zOff]
							&&  candidate > neighborhood[center                  + zOff]
							&&  candidate > neighborhood[center + 1              + zOff]
							&&  candidate > neighborhood[center - 1 + blockDim.x + zOff]
							&&  candidate > neighborhood[center     + blockDim.x + zOff]
							&&  candidate > neighborhood[center + 1 + blockDim.x + zOff] ;

			if (maxima)
			{
				unsigned int index = atomicInc(maxCounter, (unsigned int) - 1);
				if (index < MAXEXTREMAS)
				{
					extremaBuffer[index] = make_int4(x, y, scale, octave);				
				}
			}
		}
	}
}

void findExtremas(float* deviceDoGData, int4** extremaBuffer, unsigned int** maxCounter, int octave, int scales, int rows, int cols)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMallocManaged(maxCounter, sizeof(unsigned int));
	cudaMallocManaged(extremaBuffer, MAXEXTREMAS * sizeof(int4));
	cudaMemset(*maxCounter, 0, sizeof(unsigned int));
	cudaMemset(*extremaBuffer, 0, MAXEXTREMAS * sizeof(int4));
	
	dim3 threads(16, 16);
	dim3 blocks(divUp(cols - 2, threads.x - 2), divUp(rows - 2, threads.y - 2) * (scales-3));
	const size_t sharedSize = threads.x * threads.y * 3 * sizeof(float);

	findExtremasGPU <<< blocks, threads, sharedSize >>> (deviceDoGData, *extremaBuffer, *maxCounter, octave, scales-3, rows, cols);
	cudaDeviceSynchronize();


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the extrema search kernel: %f ms\n", time);
}
//                                                                    //
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// Localization                                                       //
__global__ void interpolate(float* deviceDoGData, int rows, int cols, int scales, int octave, int octaves, int4* extremaBuffer, unsigned int* maxCounter, float* xRow, float* yRow, float* octaveRow, float* sizeRow, float* angleRow, float* responseRow)
{
	const int4 maxPos = extremaBuffer[blockIdx.x];

	const int x = maxPos.x - 1 + threadIdx.x;
	const int y = maxPos.y - 1 + threadIdx.y;
	const int scl = maxPos.z - 1 + threadIdx.z;

	__shared__ float neighbors[3][3][3];
	__shared__ float firstPartials[3]; // dx, dy, ds
	__shared__ float secondPartials[3]; // dxx, dyy, dxy
	__shared__ bool doAgain;

	doAgain = false;

	float xOff = 0;
	float yOff = 0;
	float sclOff = 0;

	int newX = x;
	int newY = y;
	int newScl = scl;

	neighbors[threadIdx.z][threadIdx.y][threadIdx.x] = deviceDoGData[(y * cols + x) +  (rows * cols * scl)];

	__syncthreads();


	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{

		__shared__ float dD[3];

		//dx
		dD[0] = -0.5f * (neighbors[1][1][2] - neighbors[1][1][0]);
		//dy
		dD[1] = -0.5f * (neighbors[1][2][1] - neighbors[1][0][1]);
		//ds
		dD[2] = -0.5f * (neighbors[2][1][1] - neighbors[0][1][1]);

		__shared__ float H[3][3];

		//dxx
		H[0][0] = neighbors[1][1][0] - 2.0f * neighbors[1][1][1] + neighbors[1][1][2];
		//dxy
		H[0][1]= 0.25f * (neighbors[1][2][2] - neighbors[1][2][0] - neighbors[1][0][2] + neighbors[1][0][0]);
		//dxs
		H[0][2]= 0.25f * (neighbors[2][1][2] - neighbors[2][1][0] - neighbors[0][1][2] + neighbors[0][1][0]);
		//dyx = dxy
		H[1][0] = H[0][1];
		//dyy
		H[1][1] = neighbors[1][0][1] - 2.0f * neighbors[1][1][1] + neighbors[1][2][1];
		//dys
		H[1][2]= 0.25f * (neighbors[2][2][1] - neighbors[2][0][1] - neighbors[0][2][1] + neighbors[0][0][1]);
		//dsx = dxs
		H[2][0] = H[0][2];
		//dsy = dys
		H[2][1] = H[1][2];
		//dss
		H[2][2] = neighbors[0][1][1] - 2.0f * neighbors[1][1][1] + neighbors[2][1][1];

		firstPartials[0] = -dD[0]; // dx
		firstPartials[1] = -dD[1]; // dy
		firstPartials[2] = -dD[2]; // ds

		secondPartials[0] = H[0][0]; // dxx
		secondPartials[1] = H[1][1]; // dyy
		secondPartials[2] = H[0][1]; // dxy

		__shared__ float X[3];

		if (device::solve3x3(H, dD, X))
		{
			if (fabs(X[0]) > 0.5f && fabs(X[1]) > 0.5f && fabs(X[2]) > 0.5f)
			{
				doAgain = true;
				newX += X[0];
				newY += X[1];
				newScl = min(max(0,scl + (int)lrintf(X[2])),scales-1); // This will result in some bullshit coordinates which will probably not pass filtering
			}
			xOff = X[0];
			yOff = X[1];
			sclOff = X[2];
		}
	}

	__syncthreads();

	if (doAgain)
	{
		neighbors[threadIdx.z][threadIdx.y][threadIdx.x] = deviceDoGData[(newY * cols + newX) + (rows * cols * newScl)];
	
		__syncthreads();

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		{		
			__shared__ float dD[3];

			//dx
			dD[0] = -0.5f * (neighbors[1][1][2] - neighbors[1][1][0]);
			//dy
			dD[1] = -0.5f * (neighbors[1][2][1] - neighbors[1][0][1]);
			//ds
			dD[2] = -0.5f * (neighbors[2][1][1] - neighbors[0][1][1]);

			__shared__ float H[3][3];

			//dxx
			H[0][0] = neighbors[1][1][0] - 2.0f * neighbors[1][1][1] + neighbors[1][1][2];
			//dxy
			H[0][1]= 0.25f * (neighbors[1][2][2] - neighbors[1][2][0] - neighbors[1][0][2] + neighbors[1][0][0]);
			//dxs
			H[0][2]= 0.25f * (neighbors[2][1][2] - neighbors[2][1][0] - neighbors[0][1][2] + neighbors[0][1][0]);
			//dyx = dxy
			H[1][0] = H[0][1];
			//dyy
			H[1][1] = neighbors[1][0][1] - 2.0f * neighbors[1][1][1] + neighbors[1][2][1];
			//dys
			H[1][2]= 0.25f * (neighbors[2][2][1] - neighbors[2][0][1] - neighbors[0][2][1] + neighbors[0][0][1]);
			//dsx = dxs
			H[2][0] = H[0][2];
			//dsy = dys
			H[2][1] = H[1][2];
			//dss
			H[2][2] = neighbors[0][1][1] - 2.0f * neighbors[1][1][1] + neighbors[2][1][1];

			firstPartials[0] = -dD[0]; // dx
			firstPartials[1] = -dD[1]; // dy
			firstPartials[2] = -dD[2]; // ds

			secondPartials[0] = H[0][0]; // dxx
			secondPartials[1] = H[1][1]; // dyy
			secondPartials[2] = H[0][1]; // dxy

			__shared__ float X[3];

			if (device::solve3x3(H, dD, X))
			{
				xOff = X[0];
				yOff = X[1];
				sclOff = X[2];
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
		float contrast = neighbors[1][1][1] + 0.5f * ( (xOff*firstPartials[0]) + (yOff*firstPartials[1]) + (sclOff*firstPartials[2]) );
		if ( contrast > CONTRASTTHRESHOLD )
		{
			float trace = secondPartials[0] + secondPartials[1];
			float det = secondPartials[0] * secondPartials[1] - secondPartials[2] * secondPartials[2];

			if (trace*trace*EDGETHRESHOLD < (EDGETHRESHOLD + 1) * (EDGETHRESHOLD + 1) * det)
			{
				unsigned int index = atomicInc(maxCounter, (unsigned int)-1);

				xRow[index] = (x + xOff) * (1 << octave);
				yRow[index] = (y + yOff) * (1 << octave);
				octaveRow[index] = octave;
				sizeRow[index] = 2.5 * fmax(2.5f, sqrtf( scalbnf((neighbors[1][1][2] - neighbors[1][1][0]),2) + scalbnf((neighbors[1][2][1] - neighbors[1][0][1]),2))) ;
				angleRow[index] = 180 + (57.2958 * atan2f( (neighbors[1][2][1] - neighbors[scl][0][1]) , (neighbors[scl][1][2] - neighbors[1][0][1])  ));
				responseRow[index] = contrast;
			}
		}
	}
}

void localization(float* deviceDoGData, int rows, int cols, int scales, int octave, int octaves, int4* extremaBuffer, unsigned int* maxCounter, float* xRow, float* yRow, float* octaveRow, float* sizeRow, float* angleRow, float* responseRow)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);




	dim3 threads(3,3,3);
	dim3 blocks(maxCounter[0], 1, 1);

	cudaMemset(maxCounter, 0, sizeof(unsigned int));


	interpolate <<< blocks, threads >>> (deviceDoGData, rows, cols, scales, octave, octaves, extremaBuffer, maxCounter, xRow, yRow, octaveRow, sizeRow, angleRow, responseRow);
	cudaDeviceSynchronize();



	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for localization: %f ms\n", time);
}
//                                                                    //
////////////////////////////////////////////////////////////////////////
