#ifndef GPUSIFT_H_
#define GPUSIFT_H_

#include <iostream>
#include <vector>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpumat.hpp>


#include "math_constants.h"

#include "sift.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace gpu;

class SIFT_GPU
{
public:

	enum KeypointLayout
	{
		X_ROW = 0,
		Y_ROW,
		OCTAVE_ROW,
		SIZE_ROW,
		ANGLE_ROW,
		RESPONSE_ROW,
		ROWS_COUNT
	};

	SIFT_GPU(); 

	void operator()(const Mat& image, GpuMat& keypoints);
	void downloadKeypoints(GpuMat& keypoints_GPU, vector<KeyPoint>& keypoints_CPU);
	void releaseMemory();

	int nOctaves;
	int nScales;
	float localizeThreshold;
	float hessianThreshold;
	float keypointsRatio;

	int4* extremaBuffer;
};




#endif