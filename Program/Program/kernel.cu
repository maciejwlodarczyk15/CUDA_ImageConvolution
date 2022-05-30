#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define MAX_KERNEL_WIDTH 441
__constant__ double K[MAX_KERNEL_WIDTH];

__global__ void Inversion_CUDA(unsigned char* Image, int Channels);
__global__ void Gaussian(double*, double*, int, int, int, int);
__host__ void generateGaussian(std::vector<double>&, int, int);
template<typename T> size_t vBytes(const typename std::vector<T>&);

void ImageInversionCUDA(unsigned char* InputImage, int Height, int Width, int Channels)
{
	unsigned char* DevInputImage = NULL;
	cudaMalloc((void**)&DevInputImage, Height * Width * Channels);
	cudaMemcpy(DevInputImage, InputImage, Height * Width * Channels, cudaMemcpyHostToDevice);

	dim3 GridImage(Width, Height);
	Inversion_CUDA << <GridImage, 1 >> > (DevInputImage, Channels);

	cudaMemcpy(InputImage, DevInputImage, Height * Width * Channels, cudaMemcpyDeviceToHost);

	cudaFree(DevInputImage);
}

__global__ void Inversion_CUDA(unsigned char* Image, int Channels)
{
	int idx = (blockIdx.x + blockIdx.y * gridDim.x) * Channels;
	for (int i = 0; i < Channels; i++)
	{
		Image[idx + i] = 255 - Image[idx + i];
	}
}

void ImageBlur(unsigned char* Image, int Height, int Width, int Channels)
{
	std::vector<double> hIn, hKernel, hOut;
	double* dIn, * dOut;
	int inputCollumns, inputRows;
	int kernelDim, kRadius;
	int outputCollumns, outputRows;
	int max = 0;
	double bw = 8;

	Mat image = imread("TestImage.png", IMREAD_GRAYSCALE);
	hIn.assign(image.data, image.data + image.total());
	inputCollumns = image.cols;
	inputRows = image.rows;
	hOut.resize(inputCollumns * inputRows, 0);

	kernelDim = 5;
	kRadius = floor(kernelDim / 2.0);
	hKernel.resize(pow(kernelDim, 2), 0);
	generateGaussian(hKernel, kernelDim, kRadius);

	outputCollumns = inputCollumns - (kernelDim - 1);
	outputRows = inputRows - (kernelDim - 1);

	cudaMalloc((void**)&dIn, vBytes(hIn));
	cudaMemcpy(dIn, hIn.data(), vBytes(hIn), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dOut, vBytes(hOut));
	cudaMemcpy(dOut, hOut.data(), vBytes(hOut), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(K, hKernel.data(), vBytes(hKernel));

	int bwHalo = bw + (kernelDim - 1);
	dim3 dimBlock(bwHalo, bwHalo);
	dim3 dimGrid(ceil(inputCollumns / bw), ceil(inputRows / bw));
	Gaussian << <dimGrid, dimBlock, bwHalo* bwHalo * sizeof(double) >> > (dIn, dOut, kernelDim, inputCollumns, outputCollumns, outputRows);
	cudaMemcpy(hOut.data(), dOut, vBytes(hOut), cudaMemcpyDeviceToHost);

	std::vector<int> toInt(hOut.begin(), hOut.end());
	Mat blurImg = Mat(toInt).reshape(0, inputRows);

	imwrite("GrayImage.png", image);
	imwrite("BlurredImage.png", blurImg);

	image.release();
	cudaFree(dIn);
	cudaFree(dOut);

	exit(EXIT_SUCCESS);
}

__global__ void Gaussian(double* In, double* Out, int kernelDim, int inWidth, int outWidth, int outHeight) {
	extern __shared__ double loadIn[];

	int trueDimX = blockDim.x - (kernelDim - 1);
	int trueDimY = blockDim.y - (kernelDim - 1);

	int col = (blockIdx.x * trueDimX) + threadIdx.x;
	int row = (blockIdx.y * trueDimY) + threadIdx.y;

	if (col < outWidth && row < outHeight) 
	{
		loadIn[threadIdx.y * blockDim.x + threadIdx.x] = In[row * inWidth + col];
		__syncthreads();

		if (threadIdx.y < trueDimY && threadIdx.x < trueDimX) 
		{
			double acc = 0;
			for (int i = 0; i < kernelDim; ++i)
			{
				for (int j = 0; j < kernelDim; ++j)
				{
				acc += loadIn[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] * K[(i * kernelDim) + j];
				}
			}
			Out[row * inWidth + col] = acc;
		}
	}
	else
		loadIn[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
}

__host__ void generateGaussian(std::vector<double>& K, int dim, int radius) {
	double stdev = 1.0;
	double pi = 355.0 / 113.0;
	double constant = 1.0 / (2.0 * pi * pow(stdev, 2));

	for (int i = -radius; i < radius + 1; ++i)
	{
		for (int j = -radius; j < radius + 1; ++j) 
		{
			K[(i + radius) * dim + (j + radius)] = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stdev, 2))));
		}
	}
}

template<typename T> size_t vBytes(const typename std::vector<T>& v)
{
	return sizeof(T) * v.size();
}
