// TestCPPLibrary.cpp : Defines the exported functions for the DLL application.
//
#include "stdafx.h"
#include "TestCppLibrary.h"
#include <string>

extern "C" {
    float TestMultiply(float a, float b)
    {
        return a * b;
    }

    float TestDivide(float a, float b)
    {
        if (b == 0) {
            return 0;
        }

        return a / b;
    }

	bool TestString(TCHAR* c)
	{
		char* example = "example";
		c = (TCHAR*)example;
		return true;
	}
	
	double* LinearRegression(int* xCollection, int* yCollection, int dataSize)
	{
		if (xCollection == NULL || yCollection == NULL || dataSize == 0)
		{
			printf("Empty data set!\n");
			return NULL;
		}

		double SUMx = 0;     //sum of x values
		double SUMy = 0;     //sum of y values
		double SUMxy = 0;    //sum of x * y
		double SUMxx = 0;    //sum of x^2
		double slope = 0;    //slope of regression line
		double y_intercept = 0; //y intercept of regression line
		double AVGy = 0;     //mean of y
		double AVGx = 0;     //mean of x

		for (int i = 0; i < dataSize; i++)
		{
			//sum of x
			SUMx += *(xCollection + i);
			//sum of y
			SUMy += *(yCollection + i);
			//sum of x*y
			SUMxy += *(xCollection + i) * (*(yCollection + i));
			//sum of squared x
			SUMxx += *(xCollection + i) * (*(xCollection + i));
		}

		//avg of x and y
		AVGy = SUMy / dataSize;
		AVGx = SUMx / dataSize;

		//slope
		slope = (dataSize * SUMxy - SUMx * SUMy) / (dataSize * SUMxx - SUMx*SUMx);

		//y itercept
		y_intercept = AVGy - slope * AVGx;

		// slope * x + y_intercept = y

		double result[2] = { slope, y_intercept };

		return result;
	}

	char* __stdcall StringReturnAPI01()
	{
		return "Hello";
	}


}