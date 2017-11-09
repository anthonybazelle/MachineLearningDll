// TestCPPLibrary.h

#ifdef TESTFUNCDLL_EXPORT
#define TESTFUNCDLL_API __declspec(dllexport) 
#else
#define TESTFUNCDLL_API __declspec(dllimport) 
#endif

extern "C" {
    TESTFUNCDLL_API float TestMultiply(float a, float b);
    TESTFUNCDLL_API float TestDivide(float a, float b);
	TESTFUNCDLL_API float VerifLinearRegression(float* xCollection, float* yCollection, int dataSize);
}


struct Sample{

private:
	std::vector<float> parameters;

	float expected;

public :
	Sample(std::vector<float> parameters, float expected)
	{
		this->parameters = parameters;
		this->expected = expected;
	}

	std::vector<float> getParameters() { return this->parameters; }
	float getExpected() { return this->expected; }

	void setExpected(float expected) { this->expected = expected; }
};