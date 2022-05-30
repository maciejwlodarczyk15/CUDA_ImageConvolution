#ifndef _Inversion_CUDA
#define _Inversion_CUDA
void ImageInversionCUDA(unsigned char* InputImage, int Height, int Width, int Channels);
void ImageBlur(unsigned char* InputImage, int Height, int Width, int Channels);
#endif
