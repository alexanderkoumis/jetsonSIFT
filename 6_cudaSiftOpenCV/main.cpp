#include <iostream>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "gpuSift.h"
#include "fakeGpuMat.h"

#define SIFT 1

using namespace std;
using namespace cv;
using namespace gpu;

int main()
{
	VideoCapture webcam(0);
	Mat resultImage;

#if(SIFT)
	SIFT_GPU sift;
#else
	SURF_GPU surf;
#endif
//	BruteForceMatcher_GPU<L2<float>> matcher;


	Mat trainImage_CPU;
	vector<KeyPoint> trainKeypoints_CPU;
	vector<float> trainDescriptors_CPU;
	
	fakeGpuMat trainImage_GPU;
	fakeGpuMat trainKeypoints_GPU;
//	GpuMat trainDescriptors_GPU;

	trainImage_CPU = imread("image.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	trainImage_GPU.upload(trainImage_CPU);

#if(SIFT)
	sift(trainImage_GPU, trainKeypoints_GPU);
#else
	surf(trainImage_GPU, GpuMat(), trainKeypoints_GPU, trainDescriptors_GPU, false);
	surf.downloadKeypoints(trainKeypoints_GPU, trainKeypoints_CPU);
	surf.downloadDescriptors(trainDescriptors_GPU, trainDescriptors_CPU); // Not used
#endif

	#if(!SIFT)
	while(true)
	{
		Mat queryImage_CPU, webcamImage_CPU;
		vector<KeyPoint> queryKeypoints_CPU;
		vector<float> queryDescriptors_CPU;

		GpuMat queryImage_GPU;
		GpuMat queryKeypoints_GPU, queryDescriptors_GPU;

		vector< vector<DMatch>> badMatches;
		vector<DMatch> goodMatches;
		
		webcam >> webcamImage_CPU;
		cvtColor(webcamImage_CPU, queryImage_CPU, CV_BGR2GRAY);
		queryImage_GPU.upload(queryImage_CPU);

//		surf(queryImage_GPU, GpuMat(), queryKeypoints_GPU, queryDescriptors_GPU, false);
// 		surf.downloadKeypoints(queryKeypoints_GPU, queryKeypoints_CPU);
		surf.downloadDescriptors(queryDescriptors_GPU, queryDescriptors_CPU); // Not used

		matcher.knnMatch(trainDescriptors_GPU, queryDescriptors_GPU, badMatches, 3);
		double tresholdDist = 0.25 * sqrt(double(webcamImage_CPU.size().height*webcamImage_CPU.size().height + webcamImage_CPU.size().width*webcamImage_CPU.size().width));

		goodMatches.reserve(badMatches.size());
		for (size_t i = 0; i < badMatches.size(); ++i)
		{
			for (int j = 0; j < badMatches[i].size(); j++)
			{
				Point2f from = trainKeypoints_CPU[badMatches[i][j].queryIdx].pt;
				Point2f to = queryKeypoints_CPU[badMatches[i][j].trainIdx].pt;        
				double dist = sqrt((from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y));
				
				if (dist < tresholdDist)
				{
					goodMatches.push_back(badMatches[i][j]);
					j = badMatches[i].size();
				}
			}
		}

 		drawMatches(trainImage_CPU, trainKeypoints_CPU, queryImage_CPU, queryKeypoints_CPU, goodMatches, resultImage);
 
		imshow("debug_img", resultImage);
		waitKey(1);
	}
	#endif
	return 0;
}