#include "gpuSift.h"
#include "opencv2\imgproc\imgproc.hpp"


int main()
{
	Mat inImage, outImage;

	SIFT_GPU sift;

	vector<KeyPoint> keypoints_CPU;
	vector<float> descriptors_CPU;
	GpuMat keypoints_GPU;

	inImage = imread("lenna.jpg");
	cvtColor(inImage, outImage, CV_BGR2GRAY);
	GaussianBlur( outImage, outImage, Size( 5, 5 ), 1, 1 );
	sift(outImage, keypoints_GPU);
	sift.downloadKeypoints(keypoints_GPU, keypoints_CPU);


	namedWindow("hi", CV_WINDOW_AUTOSIZE);
	drawKeypoints(inImage, keypoints_CPU, inImage, Scalar(0, 255, 0), 4);

	imshow("hi", inImage);
	waitKey(0);

	return 0;
}