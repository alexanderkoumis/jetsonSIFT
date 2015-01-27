#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>

#include "gpuSift.h"

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: ./jetsonSift yourimage.jpg\n");
		exit(0);
	}

	Mat inImage, outImage;

	SIFT_GPU sift;

	vector<KeyPoint> keypoints_CPU;
	vector<float> descriptors_CPU;
	GpuMat keypoints_GPU;

	inImage = imread(argv[1]);

	if (inImage.data == NULL)
	{
		printf("\"%s\" is not a valid image\n", argv[1]);
		exit(0);
	}

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
