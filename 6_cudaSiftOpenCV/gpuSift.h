#ifndef GPUSIFT_H_
#define GPUSIFT_H_

#include <cuda_runtime.h>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "fakeGpuMat.h"

class SIFT_GPU
{
public:

	enum KeypointLayout
	{
		XROW = 0,
		Y_ROW,
		OCTAVE_ROW,
		MAGNITUDE_ROW,
		ANGLE_ROW
	};

	SIFT_GPU();
	explicit SIFT_GPU(int _nOctaves, int _nScales, double _localizeThreshold, double _hessianThreshold, float keypointsRatio); 

	void operator()(fakeGpuMat& image, fakeGpuMat& keypoints);
	void releaseMemory();

	cudaTextureObject_t imageTexture;
	cudaTextureObject_t DoGSpaceTexture;

	int nOctaves;
	int nScales;
	double localizeThreshold;
	double hessianThreshold;
	float keypointsRatio;

};




#endif