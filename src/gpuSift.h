#ifndef GPUSIFT_H_
#define GPUSIFT_H_

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "math_constants.h"
#include "helper_math.h"

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

	void operator()(const cv::Mat& image, cv::gpu::GpuMat& keypoints);
	void downloadKeypoints(cv::gpu::GpuMat& keypoints_GPU, std::vector<cv::KeyPoint>& keypoints_CPU);
	void releaseMemory();

	int nOctaves;
	int nScales;
	float localizeThreshold;
	float hessianThreshold;
	float keypointsRatio;

	int4* extremaBuffer;
};




#endif
