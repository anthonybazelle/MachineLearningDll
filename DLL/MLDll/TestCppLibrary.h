// TestCPPLibrary.h

#ifdef TESTFUNCDLL_EXPORT
#define TESTFUNCDLL_API __declspec(dllexport) 
#else
#define TESTFUNCDLL_API __declspec(dllimport) 
#endif

extern "C" {
    TESTFUNCDLL_API float TestMultiply(float a, float b);
    TESTFUNCDLL_API float TestDivide(float a, float b);
	TESTFUNCDLL_API float* LinearRegression(float* xCollection, float* yCollection, int dataSize);
	TESTFUNCDLL_API float* TestRefArrayOfInts(int** ppArray, int* pSize);  
	TESTFUNCDLL_API float* PerceptronRosenblatt(float* inputs, float* expected, float* weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance);
}
