#include <iostream>

#include <cuda_runtime.h>
#include "gpuSift.h"


using namespace cv;



class SIFT_GPU_Invoker
{
public:
	SIFT_GPU_Invoker(SIFT_GPU& sift, const Mat& image) : sift_(sift), inImage(image), cols(image.cols), rows(image.rows)
	{
		CV_Assert(!image.empty() && image.type() == CV_8UC1);
		CV_Assert(sift_.nOctaves > 0 && sift_.nScales > 0);

		cols = image.cols;
		rows = image.rows;
	}

	void detectKeypoints(GpuMat& keypoints, int scales)
	{

		ensureSizeIsEnough(SIFT_GPU::ROWS_COUNT, MAXEXTREMAS, CV_32FC1, keypoints);

		keypoints.setTo(Scalar::all(0));
		for (int octave = 0; octave < 1; ++octave)
		{
			const int scaleCols = cols >> octave;
			const int scaleRows = rows >> octave;

			createDoGSpace(inImage.data, &deviceDoGData, scales, scaleRows, scaleCols);
			findExtremas(deviceDoGData, &sift_.extremaBuffer, &maxCounter, octave, scales, scaleRows, scaleCols);
			localization(deviceDoGData, scaleRows, scaleCols, scales, octave, sift_.nOctaves, sift_.extremaBuffer, maxCounter,
						keypoints.ptr<float>(SIFT_GPU::X_ROW), keypoints.ptr<float>(SIFT_GPU::Y_ROW),
						keypoints.ptr<float>(SIFT_GPU::OCTAVE_ROW),	keypoints.ptr<float>(SIFT_GPU::SIZE_ROW),
						keypoints.ptr<float>(SIFT_GPU::ANGLE_ROW), keypoints.ptr<float>(SIFT_GPU::RESPONSE_ROW)); 
		}

		cout << "Number of keypoints: " << maxCounter[0] << endl;
	}

private:
	SIFT_GPU& sift_;
	
	const Mat& inImage;
	float* deviceDoGData;

	unsigned int* maxCounter;
	int cols, rows;

	int counter;
};

SIFT_GPU::SIFT_GPU()
{
	nOctaves = 4;
	nScales = 5;
	keypointsRatio = 0.01f;
	keypointsRatio = 2;
}

void SIFT_GPU::operator()(const Mat& image, GpuMat& keypoints)
{
	if (!image.empty())
	{
		SIFT_GPU_Invoker sift(*this, image);
		sift.detectKeypoints(keypoints, nScales);
	}
	else
	{
		std::cout << "Error: The image is empty." << std::endl;
	}
}

void SIFT_GPU::downloadKeypoints(GpuMat& keypoints_GPU, vector<KeyPoint>& keypoints_CPU)
{
	const int nFeatures = keypoints_GPU.cols;

	if (nFeatures == 0)
	{
		keypoints_CPU.clear();
	}
	else
	{
        Mat tempKeypoints_CPU(keypoints_GPU);

        keypoints_CPU.resize(nFeatures);

		float* kp_x = tempKeypoints_CPU.ptr<float>(SIFT_GPU::X_ROW);
		float* kp_y = tempKeypoints_CPU.ptr<float>(SIFT_GPU::Y_ROW);
		int* kp_octave = tempKeypoints_CPU.ptr<int>(SIFT_GPU::OCTAVE_ROW);
		float* kp_size = tempKeypoints_CPU.ptr<float>(SIFT_GPU::SIZE_ROW);
		float* kp_dir = tempKeypoints_CPU.ptr<float>(SIFT_GPU::ANGLE_ROW);
		float* kp_res = tempKeypoints_CPU.ptr<float>(SIFT_GPU::RESPONSE_ROW);


        for (int i = 0; i < nFeatures; ++i)
        {
            KeyPoint& kp = keypoints_CPU[i];
            kp.pt.x = kp_x[i];
            kp.pt.y = kp_y[i];
            kp.class_id = 0;
            kp.octave = kp_octave[i];
            kp.size = kp_size[i];
            kp.angle = kp_dir[i];
            kp.response = kp_res[i];
        }
	}
}

void SIFT_GPU::releaseMemory()
{

}