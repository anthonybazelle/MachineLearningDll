// TestCPPLibrary.h

#ifdef TESTFUNCDLL_EXPORT
#define TESTFUNCDLL_API __declspec(dllexport) 
#else
#define TESTFUNCDLL_API __declspec(dllimport) 
#endif

extern "C" {
    TESTFUNCDLL_API float TestMultiply(float a, float b);
    TESTFUNCDLL_API float TestDivide(float a, float b);
	TESTFUNCDLL_API bool TestString(TCHAR* c);
	TESTFUNCDLL_API char* __stdcall StringReturnAPI01();
	TESTFUNCDLL_API double* LinearRegression(int* xCollection, int* yCollection, int dataSize);
}
