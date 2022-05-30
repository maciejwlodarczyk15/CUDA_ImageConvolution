#include "Program.h"

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main()
{
	Mat InputImage = imread("TestImage.png");

	ImageInversionCUDA(InputImage.data, InputImage.rows, InputImage.cols, InputImage.channels());
	imwrite("InvertedImage.png", InputImage);

	ImageBlur(InputImage.data, InputImage.rows, InputImage.cols, InputImage.channels());

	return 0;
}