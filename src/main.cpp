#include <cstdio>

#include <opencv2/imgproc/imgproc.hpp> // cvtColor, GaussianBlur
#include <opencv2/highgui/highgui.hpp> // imread, imshow, waitKey, namedWindow, CV_WINDOW_AUTOSIZE

#include "gpuSift.h"

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: ./jetsonSift yourimage.jpg\n");
		exit(0);
	}

	cv::Mat inImage;
	cv::Mat outImage;

	SIFT_GPU sift;

	std::vector<cv::KeyPoint> keypoints_CPU;
	std::vector<float> descriptors_CPU;
	cv::gpu::GpuMat keypoints_GPU;

	inImage = cv::imread(argv[1]);

	if (inImage.data == NULL)
	{
		printf("\"%s\" is not a valid image\n", argv[1]);
		exit(0);
	}

	cv::cvtColor(inImage, outImage, CV_BGR2GRAY);
	cv::GaussianBlur( outImage, outImage, cv::Size( 5, 5 ), 1, 1 );
	sift(outImage, keypoints_GPU);
	sift.downloadKeypoints(keypoints_GPU, keypoints_CPU);

	cv::namedWindow("hi", CV_WINDOW_AUTOSIZE);
	cv::drawKeypoints(inImage, keypoints_CPU, inImage, cv::Scalar(0, 255, 0), 4);

	cv::imshow("hi", inImage);
	cv::waitKey(0);

	return 0;
}
