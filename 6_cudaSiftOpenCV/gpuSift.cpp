#include <iostream>
#include <cuda_runtime.h>
#include "sift.h"
#include "gpuSift.h"

class SIFT_GPU_Invoker
{
public:
	SIFT_GPU_Invoker(SIFT_GPU& sift, fakeGpuMat& image, cudaTextureObject_t& imageTexture) : sift_(sift), width(image.cols), height(image.rows)
	{
		CV_Assert(!image.empty());
		CV_Assert(sift_.nOctaves > 0 && sift_.nScales > 0);
		
		maxCandidates = min(static_cast<int>(1.5 * image.size() * sift.keypointsRatio), 65535); // Hmm, what is the rationale?

		CV_Assert (maxCandidates > 0);

		counters.fakeGpuMatCreate(1, sift_.nOctaves + 1, sizeof(CV_32SC1)); // Why is it nOctaves + 1?

		loadGlobalConstants(maxCandidates, width, height, sift_.nScales, static_cast<float>(sift_.localizeThreshold), static_cast<float>(sift_.hessianThreshold));
		
		createImgTexObj(image, imageTexture);
	}

	void detectKeypoints(cudaTextureObject_t& imageTexture, cudaTextureObject_t& DoGSpaceTexture, fakeGpuMat& keypoints, int nScales)
	{
		for (int octave = 0; octave < sift_.nOctaves; ++octave)
		{
			const int scaleWidth = width >> octave;
			const int scaleHeight = height >> octave;

			loadOctaveConstants(octave, scaleWidth, scaleHeight);
			createDoGSpaceTexObj(DoGSpaceTexture, imageTexture, scaleWidth, scaleHeight, nScales);
				
			// call function that creates DoG Space
			// call function that calls extrema extraction kernel

			cudaDestroyTextureObject(DoGSpaceTexture);
		}
	}

private:
	SIFT_GPU& sift_;

	int width, height, scales;
	int maxFeatures, maxCandidates;

	fakeGpuMat counters;
};


SIFT_GPU::SIFT_GPU()
{
	nOctaves = 4;
	nScales = 5;
	keypointsRatio = 0.01f;
	hessianThreshold = 5;
	keypointsRatio = 2;
}

SIFT_GPU::SIFT_GPU(int _nOctaves, int _nScales, double _localizeThreshold, double _hessianThreshold, float _keypointsRatio)
{
	nOctaves = _nOctaves;
	nScales = _nScales;
	localizeThreshold = _localizeThreshold;
	hessianThreshold = _hessianThreshold;
	keypointsRatio = _keypointsRatio;
}

void SIFT_GPU::operator()(fakeGpuMat& image, fakeGpuMat& keypoints)
{

	if (!image.empty())
	{
		SIFT_GPU_Invoker sift(*this, image, imageTexture);
		sift.detectKeypoints(imageTexture, DoGSpaceTexture, keypoints, nScales);
	}
	else
	{
		std::cout << "Error: The GpuImage is empty." << std::endl;
		system("pause");
	}
}

void SIFT_GPU::releaseMemory()
{
	cudaDestroyTextureObject(imageTexture);

}