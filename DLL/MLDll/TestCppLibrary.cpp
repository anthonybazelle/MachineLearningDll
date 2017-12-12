// TestCPPLibrary.cpp : Defines the exported functions for the DLL application.
//
#include "stdafx.h"
#include "TestCppLibrary.h"
#include <string>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <ctime>
#include <map>
#include <math.h>
#include <fstream>

extern "C" {


	struct Sample {

	private:
		std::vector<float> parameters;

		float expected;

	public:
		Sample(std::vector<float> parameters, float expected)
		{
			this->parameters = parameters;
			this->expected = expected;
		}

		std::vector<float> getParameters() { return this->parameters; }
		float getExpected() { return this->expected; }

		void setExpected(float expected) { this->expected = expected; }
	};

	float TestMultiply(float a, float b)
	{
		return a * b;
	}

	float TestDivide(float a, float b)
	{
		if (b == 0) 
		{
			return 0;
		}

		return a / b;
	}

	int makeRandomWeight(float* weights)
	{
		int i = 0;
		while (weights[i] != NULL)
		{
			weights[i] = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));
			++i;
		}

		return i;
	}

	float* LinearRegressionWithEigen(float* inputs, float* zBuffer, int nbParameter, int nbSample)
	{
		//std::ofstream logFile;
		//logFile.open("logFileDll.log");

		//for(int i = 0; i < nbSample; i+=2)
		//{
		//	logFile << "Sample " << i << " : x =" << inputs[i] << " y =" << inputs[i+1] << std::endl;
		//}

		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > matInputs(inputs, nbSample, nbParameter);
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > matZBuffer(zBuffer, nbSample, 1);

		// W = ((X^T X)^-1 X^T)Y

		//logFile << "Matrix inputs : " << std::endl << matInputs << std::endl << std::endl;
		//logFile << "Matrix inputs transpose: " << std::endl << matInputs.transpose() << std::endl << std::endl;
		//logFile << "Product Matrix inputs transpose and matInputs: " << std::endl << (matInputs.transpose() * matInputs) << std::endl << std::endl;
		//logFile << "Pseudo Inverse product Matrix inputs transpose and matInputs: " << std::endl << ((matInputs.transpose() * matInputs).completeOrthogonalDecomposition().pseudoInverse()) << std::endl << std::endl;
		//logFile << "Transpose product Matrix inputs transpose and matInputs: " << std::endl << ((matInputs.transpose() * matInputs).completeOrthogonalDecomposition().pseudoInverse() * matInputs.transpose()) << std::endl << std::endl;
		//logFile << "ZBuffer : " << std::endl << matZBuffer << std::endl << std::endl;

		Eigen::MatrixXf result = ((matInputs.transpose() * matInputs).completeOrthogonalDecomposition().pseudoInverse() * matInputs.transpose()) * matZBuffer;

		//logFile << "Result : " << std::endl << result << std::endl;
		//logFile << "DATA : " << std::endl << result.data()[0] << std::endl << result.data()[1]<< std::endl;
		return (float*)result.data();
	}

	float* LinearRegression(float* xCollection, float* yCollection, int dataSize)
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

		//std::cout << "Slope in DLL : " << slope << std::endl << std::endl;
		//y itercept
		y_intercept = AVGy - slope * AVGx;

		//std::ofstream logFile;
		//logFile.open("logFileDll.log");
		//logFile << "In DLL : " << std::endl << std::endl << "DataSize in DLL : " << dataSize << std::endl <<"AVGy : " << AVGy << std::endl << "AVGx : " << AVGx << std::endl<< std::endl << "y_intercept : " << y_intercept << std::endl << "Slope : " << slope << std::endl;
		//logFile.close();

		// slope * x + y_intercept = y

		float* result = new float[2];
		result[0] = slope;
		result[1] = y_intercept;

		return result;
	}

	/// Parameters : 
	// inputs : corresponding to input parameters 
	// expected : corresponding to the expected value of perceptron's output
	// weights : corresponding of the weight of parameters
	// nbParameters : with this parameter we can parse the array of input. With this, we know that each nbParameter, we have an other sample
	// nbSample : Same reason
	// stepLearning : We need the learning's step for the formula
	// nbIteration : we need this because, if inputs can't be "lineary resolvable", we'll have an infinite loop. Need to be parametizable in Unity

	// With this function, the number of parameter doesn't matter! 

	// Weigths in parameters useless, need to be modify later
	float* PerceptronRosenblatt(float* inputs, float* expected, float* weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance)
	{

		//std::ofstream logFile;
		//logFile.open("logFileDll.log");
		//logFile << "Start RosenBlatt Perceptron : " << std::endl << std::endl << std::endl;
		// Initialize weight with random between -1 and 1, or just initialize 0 maybe ?
		int countWeight = makeRandomWeight(weights);

		// bias
		float x0 = -1;

		// Initialize array of sample given in input with the expected value in output for each sample in third parameter of sample's constructor
		std::vector<Sample*> nativeInputs;
		for (int i = 0; i < nbSample; ++i)
		{
			std::vector<float> parameters;
			for (int j = 0; j < nbParameters; ++j)
			{
				parameters.push_back(inputs[j + i]);
			}

			Sample* sample = new Sample(parameters, expected[i]);

			nativeInputs.push_back(sample);
		}
		
		//logFile << "Initialize sample : DONE - NbSample : " << nbSample <<  std::endl << std::endl;

		float w0 = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));

		bool different = true;
		int iteration = 0;
		std::vector<float> realResult;


		while (iteration < nbIteration && different)
		{
			//logFile << "Iteration " << iteration << " : " << std::endl << std::endl << std::endl;
			// Here we'll update weights of parameters
			// Loop sample

			bool needToContinue = false;

			for (int i = 0; i < nbSample; ++i)
			{
				// Logicaly the number of weight correspond to the number of parameter in a sample so we stop the loop when we are out of the number of weight
				// formula: (w1 * x1) + (w2 * x2) - w0 (-w0 because we chose x0 = - 1) when the two vector are different
				float result = 0.f;
				float y;

				for (int j = 0; j < countWeight; ++j)
				{
					result += nativeInputs[i]->getParameters()[j] * weights[j];
				}
				result -= w0;

				if(result < 0)
				{
					y = -1.f;
				}
				else
				{
					y = 1.f;
				}

				//logFile << "Result for sample " << i << " : " << y << "  Expected : " << expected[i] << std::endl;
				
				if (std::abs(y - nativeInputs[i]->getExpected()) > tolerance)
				{
					needToContinue = true;
					for (int j = 0; j < countWeight; ++j)
					{
						weights[j] = weights[j] + stepLearning * (expected[i] - y) * inputs[j + i * nbParameters];
					}

					w0 = w0 + stepLearning * (expected[i] - y) * x0;

					//for(int i = 0; i < nbParameters; ++i)
					//{
					//	logFile << "Weight " << i << " : " << weights[i] << std::endl;
					//}

					//logFile << std::endl;
				}
			}

			if (!needToContinue)
			{
				break;
			}

			++iteration;
		}

		//logFile << "Whole necessary iteration (" << iteration <<") : DONE" << std::endl;

		//for(int i = 0; i < nbParameters; ++i)
		//{
		//	logFile << "Weight " << i << " : " << weights[i] << std::endl;
		//}

		float* resultWeight = new float[nbParameters + 1];
		
		for(int i = 0; i < nbParameters; ++i)
		{
			resultWeight[i] = weights[i];
		}

		// Release memory
		for(int i = 0; i < nativeInputs.size(); ++i)
		{
			Sample* s = nativeInputs[i];
			nativeInputs.erase(nativeInputs.begin() + i);
			delete s;
		}

		resultWeight[nbParameters] = w0;
		//logFile.close();
		return resultWeight;
	}

	float* PLA(float* inputs, float* expected, float* weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance)
	{

		//std::ofstream logFile;
		//logFile.open("logFileDll.log");
		//logFile << "Start PLA Perceptron : " << std::endl << std::endl << std::endl;
		// Initialize weight with random between -1 and 1, or just initialize 0 maybe ?
		int countWeight = makeRandomWeight(weights);

		// bias
		float x0 = -1;

		// Initialize array of sample given in input with the expected value in output for each sample in third parameter of sample's constructor
		std::vector<Sample*> nativeInputs;
		for (int i = 0; i < nbSample; ++i)
		{
			std::vector<float> parameters;
			for (int j = 0; j < nbParameters; ++j)
			{
				parameters.push_back(inputs[j + i]);
			}

			Sample* sample = new Sample(parameters, expected[i]);

			nativeInputs.push_back(sample);
		}
		
		//logFile << "Initialize sample : DONE - NbSample : " << nbSample <<  std::endl << std::endl;

		float w0 = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));

		bool different = true;
		int iteration = 0;
		std::vector<float> realResult;


		while (iteration < nbIteration && different)
		{
			//logFile << "Iteration " << iteration << " : " << std::endl << std::endl << std::endl;
			// Here we'll update weights of parameters
			// Loop sample

			bool needToContinue = false;

			for (int i = 0; i < nbSample; ++i)
			{
				// Logicaly the number of weight correspond to the number of parameter in a sample so we stop the loop when we are out of the number of weight
				// formula: (w1 * x1) + (w2 * x2) - w0 (-w0 because we chose x0 = - 1) when the two vector are different
				float result = 0.f;
				float y;

				for (int j = 0; j < countWeight; ++j)
				{
					result += nativeInputs[i]->getParameters()[j] * weights[j];
				}
				result -= w0;

				if(result < 0)
				{
					y = -1.f;
				}
				else
				{
					y = 1.f;
				}

				//logFile << "Result for sample " << i << " : " << y << "  Expected : " << expected[i] << std::endl;
				
				if (std::abs(y - nativeInputs[i]->getExpected()) > tolerance)
				{
					needToContinue = true;
					for (int j = 0; j < countWeight; ++j)
					{
						weights[j] = weights[j] + stepLearning * expected[i] * inputs[j + i * nbParameters];
					}

					w0 = w0 + stepLearning * expected[i] * x0;

					/*
					for(int i = 0; i < nbParameters; ++i)
					{
						logFile << "Weight " << i << " : " << weights[i] << std::endl;
					}

					logFile << std::endl;*/
				}
			}

			if (!needToContinue)
			{
				break;
			}

			++iteration;
		}

		//logFile << "Whole necessary iteration (" << iteration <<") : DONE" << std::endl;

		/*for(int i = 0; i < nbParameters; ++i)
		{
			logFile << "Weight " << i << " : " << weights[i] << std::endl;
		}*/

		float* resultWeight = new float[nbParameters + 1];
		
		for(int i = 0; i < nbParameters; ++i)
		{
			resultWeight[i] = weights[i];
		}

		// Release memory
		for(int i = 0; i < nativeInputs.size(); ++i)
		{
			Sample* s = nativeInputs[i];
			nativeInputs.erase(nativeInputs.begin() + i);
			delete s;
		}

		resultWeight[nbParameters] = w0;
		//logFile.close();
		return resultWeight;
	}

	Eigen::MatrixXf pointerToMatrix(float* m, int rows, int cols)
	{
		Eigen::MatrixXf mat(rows, cols);

		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				mat(i, j) = m[i * cols + j];
			}
		}

		return mat;
	}

	float* MLPerceptronClassification(float* inputs, float* expected, float* weights, int* layersNeurones, int nbParameters, int nbSample, int nbLayers, float stepLearning, int nbIteration, float tolerance)
	{
		int countNeurones = nbParameters;
		int maxNeurone = 0;
		for (int i = 0; i < nbLayers; ++i)
		{
			countNeurones += layersNeurones[i];
			if (layersNeurones[i] > maxNeurone)
			{
				maxNeurone = layersNeurones[i];
			}
		}

		// Initialize weight with random between -1 and 1
		int countWeight = makeRandomWeight(weights);

		Eigen::MatrixXf samples = pointerToMatrix(inputs, nbSample, nbParameters);
		Eigen::MatrixXf W = pointerToMatrix(weights, nbLayers - 1, maxNeurone * maxNeurone);
		Eigen::MatrixXf x_li(nbLayers, maxNeurone);
		Eigen::MatrixXf d_li(nbLayers - 1, maxNeurone);
		Eigen::MatrixXf y(nbSample, 1);

		float w0 = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));
				
		int iteration = 0;
		bool needToContinue;
		float result;

		while (iteration < nbIteration)
		{
			needToContinue = false;

			//pour chaque sample d'entraînement
			for (int i = 0; i < nbSample; ++i)
			{
				//pour chaque paramètre (plus neurone de biais) on ajoute les entrées dans le vecteur des x_li
				for (int j = 0; j < nbParameters; ++j)
				{
					x_li(0, j) = samples(i, j);		//chaque entrée
				}

				//pour chaque couche de notre perceptron
				for (int j = 1; j < nbLayers - 1; ++j)
				{
					//pour chaque neurone de la couche en cours
					for (int k = 0; k < layersNeurones[j]; ++k)
					{
						result = 0.f;

						//pour chaque x de la couche d'avant, on ajoute le x * w
						for (int l = 0; l < layersNeurones[j - 1]; ++l)
						{
							result += x_li(j - 1, l) * W(j - 1, k + l * layersNeurones[j]);
						}
						result = tanh(result);								//sigmoïde de la somme pondérée
						x_li(j, k) = result;								//on ajoute le x au vecteur des x_li						
					}
				}

				//calcule final à partir des x de la dernière couche
				result = 0.f;
				for (int l = 0; l = layersNeurones[nbLayers - 2]; ++l)
				{
					result += x_li(nbLayers - 2, l) * W(nbLayers - 2, l);
				}

				//pour classification, on applique la sigmoïde à la sortie, pas pour la régression
				result -= w0;
				result = tanh(result);
				x_li(nbLayers - 1, 0) = result;

				//si pas satisfait du résultat, rétropropagation du gradient
				if (std::abs(result - y(i)) > tolerance)
				{
					needToContinue = true;									//on doit continuer l'apprentissage

					//calcul du delta de la dernière couche (calcul différent pour la régression)
					float delta = (1 - pow(x_li(nbLayers - 1, 0), 2)) * (x_li(nbLayers - 1, 0) - y(i));
					d_li(nbLayers - 1, 0) = delta;

					//pour chaque couche en partant de la dernière
					for (int j = nbLayers - 1; j > 0; --j)
					{
						//pour chaque neurone de la couche en partant du dernier
						for (int k = 0; k < layersNeurones[j - 1]; ++k)
						{
							delta = 0;
							//pour chaque neurone vers lequel pointe de neurone en cours
							for (int l = 0; l < layersNeurones[j]; ++l)
							{
								delta += W(j - 1, k + l * layersNeurones[j]) * d_li(j, l);			//on décrémente counter pour être au bon poids
							}
							delta *= (1 - pow(x_li(j - 1, k), 2));

							d_li(j - 1, k) = delta;
						}
					}

					//lorsqu'on a tous les deltas, on ajuste les w
					//pour chaque couche
					for (int j = 0; j < nbLayers - 1; ++j)
					{
						//pour chaque neurone de la couche en cours
						for (int k = 0; k < layersNeurones[j + 1]; ++k)
						{
							for (int l = 0; l < layersNeurones[j]; ++l)
							{
								W(j, k + l * layersNeurones[j + 1]) -= stepLearning * x_li(j, l) * d_li(j + 1, l);
							}
						}
					}

					//pour le neurone de biais
					w0 -= stepLearning * d_li(nbLayers - 1, 0);
				}
			}

			if (!needToContinue)
			{
				break;
			}

			++iteration;
		}

		return W.data();
	}

	Eigen::MatrixXf initializeClusterRepresentative(Eigen::MatrixXf inputs, int cluster, int nbParameters, int nbSamples)
	{
		Eigen::MatrixXf rpz(cluster, nbParameters);
		Eigen::MatrixXf extrema(nbParameters, 2);

		for (int i = 0; i < nbParameters; ++i)
		{
			extrema(i, 0) = inputs(0, i);
			extrema(i, 1) = inputs(0, i);
		}

		for (int i = 1; i < nbSamples; ++i)
		{
			for (int j = 0; j < nbParameters; ++j)
			{
				if (inputs(i, j) < extrema(j, 0))
				{
					extrema(j, 0) = inputs(i, j);
				}
				if (inputs(i, j) > extrema(j, 1))
				{
					extrema(j, 1) = inputs(i, j);
				}
			}
		}

		srand(static_cast <unsigned> (time(0)));

		for (int i = 0; i < cluster; ++i)
		{
			for (int j = 0; j < nbParameters; ++j)
			{
				rpz(i, j) = extrema(j, 0) + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (extrema(j, 1) - extrema(j, 0))));
			}
			//rpz(i, nbParameters) = i;
		}

		return rpz;
	}

	float* RBFNaiveTraining(float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs)
	{
		Eigen::MatrixXf phi(nbSamples, nbSamples);
		Eigen::MatrixXf weights(nbSamples, nbOutputs);
		Eigen::MatrixXf outputs(nbSamples, nbOutputs);
		Eigen::MatrixXf samples(nbSamples, nbParameters);

		float dist = 0.f;

		samples = pointerToMatrix(inputs, nbSamples, nbParameters);
		outputs = pointerToMatrix(expected, nbSamples, nbOutputs);

		for (int i = 0; i < nbSamples; ++i)
		{
			for (int j = 0; j < nbSamples; ++j)
			{
				dist = 0.f;

				for (int k = 0; k < nbParameters; ++k)
				{
					dist += pow(samples(i, k) - samples(j, k), 2);
				}

				phi(i, j) = exp(-gamma * dist);
			}
		}

		weights = phi.inverse() * outputs;

		return weights.data();
	}

	float* RBFRegression(float gamma, float* inputs, float* data, float* weights, int nbParameters, int nbSamples, int nbOutputs)
	{
		Eigen::MatrixXf dat(nbParameters, 1);
		Eigen::MatrixXf X(nbSamples, nbParameters);
		Eigen::MatrixXf W(nbSamples, nbOutputs);
		Eigen::MatrixXf result(nbOutputs, 1);
		float res, dist;

		X = pointerToMatrix(inputs, nbSamples, nbParameters);
		W = pointerToMatrix(weights, nbSamples, nbOutputs);
		dat = pointerToMatrix(data, nbParameters, 1);

		for (int i = 0; i < nbOutputs; ++i)
		{
			res = 0.f;
			for (int j = 0; j < nbSamples; ++j)
			{
				dist = 0.f;
				for (int k = 0; k < nbParameters; ++k)
				{
					dist += pow(X(j, k) - dat(k, 1), 2);
				}
				res += W(j, i) * exp(-gamma * dist);
			}
			result(i, 1) = res;
		}

		return result.data();
	}

	float* RBFClassification(float gamma, float* inputs, float* data, float* weights, int nbParameters, int nbSamples, int nbOutputs)
	{
		Eigen::MatrixXf dat(nbParameters, 1);
		Eigen::MatrixXf X(nbSamples, nbParameters);
		Eigen::MatrixXf W(nbSamples, nbOutputs);
		Eigen::MatrixXf result(nbOutputs, 1);
		float res, dist;

		X = pointerToMatrix(inputs, nbSamples, nbParameters);
		W = pointerToMatrix(weights, nbSamples, nbOutputs);
		dat = pointerToMatrix(data, nbParameters, 1);

		for (int i = 0; i < nbOutputs; ++i)
		{
			res = 0.f;
			for (int j = 0; j < nbSamples; ++j)
			{
				dist = 0.f;
				for (int k = 0; k < nbParameters; ++k)
				{
					dist += pow(X(j, k) - dat(k, 1), 2);
				}
				res += W(j, i) * exp(-gamma * dist);
			}
			result(i, 1) = res < 0.f ? -1.f : res > 0.f ? 1.f : 0.f;
		}

		return result.data();
	}

	float* RBFkMeansTraining(float epsilon, int cluster, float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs)
	{
		Eigen::MatrixXf phi(nbSamples, cluster);
		Eigen::MatrixXf outputs(nbSamples, nbOutputs);
		Eigen::MatrixXf samples(nbSamples, nbParameters);
		Eigen::MatrixXf µ(cluster, nbParameters);
		Eigen::MatrixXf clust(nbSamples, 2);
		Eigen::MatrixXf sampleCluster(cluster, 1);
		Eigen::MatrixXf barycentres(cluster, nbParameters);
	
		float dist;
		bool converge = false;
		bool emptyCluster = false;
		int nbC;

		samples = pointerToMatrix(inputs, nbSamples, nbParameters);
		outputs = pointerToMatrix(expected, nbSamples, nbOutputs);

		Eigen::MatrixXf rpz = initializeClusterRepresentative(samples, cluster, nbParameters, nbSamples);

		while (!converge)
		{
			for (int i = 0; i < cluster; ++i)
			{
				sampleCluster(i, 0) = 0;
			}

			for (int i = 0; i < nbSamples; ++i)
			{
				clust(i, 1) = FLT_MAX;
				for (int j = 0; j < cluster; ++j)
				{
					dist = 0.f;
					for (int k = 0; k < nbParameters; ++k)
					{
						dist += pow(samples(i, k) - rpz(j, k), 2);
					}
					if (dist < clust(i, 1))
					{
						clust(i, 0) = j;
						clust(i, 1) = dist;
						sampleCluster(j, 0)++;
					}
				}
			}

			for (int i = 0; i < cluster; ++i)
			{
				if (sampleCluster(i, 0) == 0)
				{
					Eigen::MatrixXf tmpRpz = initializeClusterRepresentative(samples, 1, nbParameters, nbSamples);
					for (int j = 0; j < nbParameters; ++j)
					{
						rpz(i, j) = tmpRpz(1, j);
					}
					emptyCluster = true;
				}
			}

			if (emptyCluster)
			{
				emptyCluster = false;
				continue;
			}

			converge = true;
			for (int i = 0; i < cluster; ++i)
			{
				nbC = 0;
				for (int j = 0; j < nbParameters; ++j)
				{
					barycentres(i, j) = 0.f;
				}

				for (int j = 0; j < nbSamples; ++j)
				{
					if (clust(j, 0) == i)
					{
						for (int k = 0; k < nbParameters; ++k)
						{
							barycentres(i, k) += samples(j, k);
							++nbC;
						}
					}
				}

				dist = 0.f;
				for (int j = 0; j < nbParameters; ++j)
				{
					barycentres(i, j) /= nbC;
					dist += pow(barycentres(i, j) - rpz(i, j), 2);
				}
				dist = sqrt(dist);

				for (int j = 0; j < nbParameters; ++j)
				{
					rpz(i, j) = barycentres(i, j);
				}

				if (dist > epsilon)
				{
					converge = false;
				}
			}
		}

		for (int i = 0; i < cluster; ++i)
		{
			for (int j = 0; j < nbSamples; ++j)
			{
				if (clust(j, 0) == i)
				{
					for (int k = 0; k < nbParameters; ++k)
					{
						µ(i, k) += samples(i, k);
					}
				}
			}

			for (int k = 0; k < nbParameters; ++k)
			{
				µ(i, k) /= sampleCluster(i, 0);
			}
		}

		for (int i = 0; i < nbSamples; ++i)
		{
			for (int j = 0; j < cluster; ++j)
			{
				dist = 0.f;

				for (int k = 0; k < nbParameters; ++k)
				{
					dist += pow(samples(i, k) - µ(j, k), 2);
				}

				phi(i, j) = exp(-gamma * dist);
			}
		}

		Eigen::MatrixXf weights = (phi.transpose() * phi).inverse() * phi.transpose() * outputs;

		return weights.data();
	}

	float* MLPerceptronRegression(float* inputs, float* expected, float* weights, int* layersNeurones, int nbParameters, int nbSample, int nbLayers, float stepLearning, int nbIteration, float tolerance)
	{
		int countNeurones = nbParameters;
		int maxNeurone = 0;
		for (int i = 0; i < nbLayers; ++i)
		{
			countNeurones += layersNeurones[i];
			if (layersNeurones[i] > maxNeurone)
			{
				maxNeurone = layersNeurones[i];
			}
		}

		// Initialize weight with random between -1 and 1
		int countWeight = makeRandomWeight(weights);

		Eigen::MatrixXf samples = pointerToMatrix(inputs, nbSample, nbParameters);
		Eigen::MatrixXf W = pointerToMatrix(weights, nbLayers - 1, maxNeurone * maxNeurone);
		Eigen::MatrixXf x_li(nbLayers, maxNeurone);
		Eigen::MatrixXf d_li(nbLayers - 1, maxNeurone);
		Eigen::MatrixXf y(nbSample, 1);

		float w0 = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));

		int iteration = 0;
		bool needToContinue;
		float result;

		while (iteration < nbIteration)
		{
			needToContinue = false;

			//pour chaque sample d'entraînement
			for (int i = 0; i < nbSample; ++i)
			{
				//pour chaque paramètre (plus neurone de biais) on ajoute les entrées dans le vecteur des x_li
				for (int j = 0; j < nbParameters; ++j)
				{
					x_li(0, j) = samples(i, j);		//chaque entrée
				}

				//pour chaque couche de notre perceptron
				for (int j = 1; j < nbLayers - 1; ++j)
				{
					//pour chaque neurone de la couche en cours
					for (int k = 0; k < layersNeurones[j]; ++k)
					{
						result = 0.f;

						//pour chaque x de la couche d'avant, on ajoute le x * w
						for (int l = 0; l < layersNeurones[j - 1]; ++l)
						{
							result += x_li(j - 1, l) * W(j - 1, k + l * layersNeurones[j]);
						}
						result = tanh(result);								//sigmoïde de la somme pondérée
						x_li(j, k) = result;								//on ajoute le x au vecteur des x_li						
					}
				}

				//calcule final à partir des x de la dernière couche
				result = 0.f;
				for (int l = 0; l = layersNeurones[nbLayers - 2]; ++l)
				{
					result += x_li(nbLayers - 2, l) * W(nbLayers - 2, l);
				}

				//pour classification, on applique la sigmoïde à la sortie, pas pour la régression
				result -= w0;
				x_li(nbLayers - 1, 0) = result;

				//si pas satisfait du résultat, rétropropagation du gradient
				if (std::abs(result - y(i)) > tolerance)
				{
					needToContinue = true;									//on doit continuer l'apprentissage

					float delta = x_li(nbLayers - 1, 0) - y(i);
					d_li(nbLayers - 1, 0) = delta;

					//pour chaque couche en partant de la dernière
					for (int j = nbLayers - 1; j > 0; --j)
					{
						//pour chaque neurone de la couche en partant du dernier
						for (int k = 0; k < layersNeurones[j - 1]; ++k)
						{
							delta = 0;
							//pour chaque neurone vers lequel pointe de neurone en cours
							for (int l = 0; l < layersNeurones[j]; ++l)
							{
								delta += W(j - 1, k + l * layersNeurones[j]) * d_li(j, l);			//on décrémente counter pour être au bon poids
							}
							delta *= (1 - pow(x_li(j - 1, k), 2));

							d_li(j - 1, k) = delta;
						}
					}

					//lorsqu'on a tous les deltas, on ajuste les w
					//pour chaque couche
					for (int j = 0; j < nbLayers - 1; ++j)
					{
						//pour chaque neurone de la couche en cours
						for (int k = 0; k < layersNeurones[j + 1]; ++k)
						{
							for (int l = 0; l < layersNeurones[j]; ++l)
							{
								W(j, k + l * layersNeurones[j + 1]) -= stepLearning * x_li(j, l) * d_li(j + 1, l);
							}
						}
					}

					//pour le neurone de biais
					w0 -= stepLearning * d_li(nbLayers - 1, 0);
				}
			}

			if (!needToContinue)
			{
				break;
			}

			++iteration;
		}

		return W.data();
	}
}