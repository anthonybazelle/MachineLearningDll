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

	char* TestString(char* c)
	{
		return c;
	}
}