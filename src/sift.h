#ifndef SIFT_H_
#define SIFT_H_

const int MAXEXTREMAS = 10000;

void createDoGSpace(unsigned char* inImage, float** deviceDoGData, int scales, int scaleRows, int scaleCols);
void findExtremas(float* deviceDoGData, int4** extremaBuffer, unsigned int** maxCounter, int octave, int scales, int rows, int cols);
void localization(float* deviceDoGData, int rows, int cols, int scales, int octave, int octaves, int4* extremaBuffer, unsigned int* maxCounter, float* xRow, float* yRow, float* octaveRow, float* sizeRow, float* angleRow, float* responseRow);

#endif