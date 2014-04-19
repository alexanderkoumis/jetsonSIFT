#ifndef FAKEGPUMAT_H_
#define FAKEGPUMAT_H_

#include "opencv2/core/core.hpp"
#include <cuda_runtime.h>

using namespace cv;

class fakeGpuMat
{
public:
	fakeGpuMat();
	void upload(const Mat& m);
	void fakeGpuMatCreate(int rows, int cols, size_t elemSize);
	void fakeGpuMatCopy(const Mat& src, fakeGpuMat& dst);
	int size();
	bool empty();

	uchar* data;
	int cols, rows;
	size_t pitch;
	bool empt;
};

#endif

