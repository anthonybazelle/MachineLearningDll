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
	
	float VerifLinearRegression(float* xCollection, float* yCollection, int dataSize)
	{
		if (xCollection == NULL || yCollection == NULL || dataSize == 0)
		{
			printf("Empty data set!\n");
			return NULL;
		}

		float SUMx = 0;     //sum of x values
		float SUMy = 0;     //sum of y values
		float SUMxy = 0;    //sum of x * y
		float SUMxx = 0;    //sum of x^2
		float slope = 0;    //slope of regression line
		float y_intercept = 0; //y intercept of regression line
		float AVGy = 0;     //mean of y
		float AVGx = 0;     //mean of x

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

		float result[2] = { slope, y_intercept };

		return result[0];
	}

	// Initialiser chaque poids pour chaque valeur
	// Passer la matrice Entree / matrice Resultat / Le pas pour maj du poids
	// Transformer les matrices en matrice pseudo inverse avec la lib Eigen
	// Algo Rosenblatt (On compare à chaque iteration la matrice resultat attendue et la matrice entrée modifiée, 
	// si les deux matrices sont différentes, on mets à jour les poids)

	
	
	float* PerceptronRosenblatt(float* inputs, float* outputs, float* weights,int sizeInput, float step)
	{

	}
}