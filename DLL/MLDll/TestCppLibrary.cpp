// TestCPPLibrary.cpp : Defines the exported functions for the DLL application.
//
#include "stdafx.h"
#include "TestCppLibrary.h"
#include <string>
#include <Eigen/Dense>
#include <vector>

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

	int makeRandomWeight(float* weights)
	{
		int i = 0;
		while(weights[i] != NULL)
		{
			weights[i] =  -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1-(-1))));
			++i;
		}

		return i;
	}

	float LinearRegressionWithEigen(float* xCollection, float* yCollection, int nbXCollection, int nbYCollection)
	{
		//Eigen::Matrix2d mat(nbXCollection, 

		return 0.f;
	}
	
	float LinearRegression(float* xCollection, float* yCollection, int dataSize)
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

		// TODO : Return the whole array and notonly the first row
		return result[0];
	}

	// Initialiser chaque poids pour chaque valeur
	// Passer la matrice Entree / matrice Resultat / Le pas pour maj du poids
	// Transformer les matrices en matrice pseudo inverse avec la lib Eigen
	// Algo Rosenblatt (On compare à chaque iteration la matrice resultat attendue et la matrice entrée modifiée, 
	// si les deux matrices sont différentes, on mets à jour les poids)
	

	/// Parameters : 
	// inputs : corresponding to input parameters 
	// expected : corresponding to the expected value of perceptron's output
	// weights : corresponding of the weight of parameters
	// nbParameters : with this parameter we can parse the array of input. With this, we know that each nbParameter, we have an other sample
	// nbSample : Same reason
	// stepLearning : We need the learning's step for the formula
	// nbIteration : we need this because, if inputs can't be "lineary resolvable", we'll have an infinite loop. Need to be parametizable in Unity

	// With this function, the number of parameter doesn't matter! 
	float* PerceptronRosenblatt(float* inputs, float* expected, float* weights, int nbParameters,int nbSample, float stepLearning, int nbIteration)
	{
		// Initialize weight
		int countWeight = makeRandomWeight(weights);

		// Initialize array of sample given in input with the expected value in output for each sample in third parameter of sample's constructor
		std::vector<Sample*> nativeInputs;
		for (int i = 0; i < nbSample; ++i)
		{
			std::vector<float> parameters;
			for(int j = 0; j < nbParameters; ++j)
			{
				parameters.push_back(inputs[j + i]);
			}

			Sample* sample = new Sample(parameters, expected[i]);

			nativeInputs.push_back(sample);
		}

		// Here we'll update weights of parameters
		// Loop sample
		for (int i = 0; i < nativeInputs.size(); ++i)
		{
			// Loop parameter of sample and update weight like this formula: (w1 * x1) + (w2 * x2) - w0 (-w0 because we chose x0 = - 1)
			// Logicaly the number of weight correspond to the number of parameter in a sample so we stop the loop when we are out of the number of weight
			for(int j = 0; j < countWeight; ++j)
			{

			}
		}

	}
}