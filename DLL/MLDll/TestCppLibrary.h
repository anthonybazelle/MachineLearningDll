// TestCPPLibrary.h

#ifdef TESTFUNCDLL_EXPORT
#define TESTFUNCDLL_API __declspec(dllexport) 
#else
#define TESTFUNCDLL_API __declspec(dllimport) 
#endif

#include <vector>

extern "C" {
	//std::vector<std::vector<int>> layers;
	//float error = 0.05f;
	//std::vector<std::vector<std::vector<float>>> weights; // Correspond to all weight of all neuron with the next layer, of all layer
	//std::vector<std::vector<float>> values;
	//bool executed;
	//float(*ActivateFunc)(float);
	//int verboseMode = 0;

    TESTFUNCDLL_API float TestMultiply(float a, float b);
    TESTFUNCDLL_API float TestDivide(float a, float b);
	TESTFUNCDLL_API float* LinearRegression(float* xCollection, float* yCollection, int dataSize);
	TESTFUNCDLL_API float* LinearRegressionWithEigen(float* inputs, float* zBuffer, const int nbParameter, const int nbSample);
	TESTFUNCDLL_API float* TestRefArrayOfInts(int** ppArray, int* pSize);  
	TESTFUNCDLL_API float* PerceptronRosenblatt(float* inputs, float* expected, float* weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance);
	TESTFUNCDLL_API float* PLA(float* inputs, float* expected, float* weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance);
	TESTFUNCDLL_API float* RBFNaiveTraining(float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs);
	TESTFUNCDLL_API float* RBFRegression(float gamma, float* inputs, float* data, float* weights, int nbParameters, int nbSamples, int nbOutputs);
	TESTFUNCDLL_API float* RBFClassification(float gamma, float* inputs, float* data, float* weights, int nbParameters, int nbSamples, int nbOutputs);
	TESTFUNCDLL_API float* RBFkMeansTraining(float epsilon, int cluster, float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs);
	TESTFUNCDLL_API float* LearnMLP(int nbSample, float* inputs, const int nbInputParam, float* outputs, const int nbOutputParam, int nbIteration, float initWeight, float error, int* neuronsPerLayerArray, int nbLayer, int activateFunc, float biasValue, int verboseMode = 0);
}