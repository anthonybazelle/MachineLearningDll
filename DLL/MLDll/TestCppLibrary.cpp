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

		float* res = new float[nbParameter];

		res[0] = result.data()[0];
		res[1] = result.data()[1];
		res[2] = result.data()[2];

		return res;
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

				if (result < 0)
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

		for (int i = 0; i < nbParameters; ++i)
		{
			resultWeight[i] = weights[i];
		}

		// Release memory
		for (int i = 0; i < nativeInputs.size(); ++i)
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

				if (result < 0)
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

		for (int i = 0; i < nbParameters; ++i)
		{
			resultWeight[i] = weights[i];
		}

		// Release memory
		for (int i = 0; i < nativeInputs.size(); ++i)
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
		Eigen::MatrixXf rpz(cluster, nbParameters);			//matrice des repréentants
		Eigen::MatrixXf extrema(nbParameters, 2);			//extremums (minimum : 0, maximum : 1) des paramètres dans les exemples fournis

		for (int i = 0; i < nbParameters; ++i)				//initialisation des extremums
		{
			extrema(i, 0) = inputs(0, i);
			extrema(i, 1) = inputs(0, i);
		}

		for (int i = 1; i < nbSamples; ++i)					//attribution des extremums
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

		srand(static_cast <unsigned> (time(0)));			//random des paramètres des représentants

		for (int i = 0; i < cluster; ++i)					//pour chaque cluster
		{
			for (int j = 0; j < nbParameters; ++j)				//pour chaque paramètre
			{
				rpz(i, j) = extrema(j, 0) + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (extrema(j, 1) - extrema(j, 0))));	//attribution d'une coordonnée aléatoire dans le rectangle englobant
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

	float* RBFkMeansTraining(float epsilon, int cluster, float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs, int nbIterations)
	{
		Eigen::MatrixXf phi(nbSamples, cluster);				//matrice Phi intermédiaire au calcul de W
		Eigen::MatrixXf outputs(nbSamples, nbOutputs);			//matrice des sorties attendues du training
		Eigen::MatrixXf samples(nbSamples, nbParameters);		//matrice des exemples de training
		Eigen::MatrixXf µ(cluster, nbParameters);				//moyenne des exemples dans un cluster
		Eigen::MatrixXf clust(nbSamples, 2);					//numéro du cluster d'appartenance de chaque exemple clust(i, 0) et distance à son représentant clust(i, 1)
		Eigen::MatrixXf sampleCluster(cluster, 1);				//population de chaque cluster
		Eigen::MatrixXf barycentres(cluster, nbParameters);		//barycentre de chaque cluster

		int it = 0;
		float dist;												//distance euclidienne d'un exemple au représentant d'un cluster (locale)
		bool converge = false;									//condition de convergence du modèle satisfaite ? (globale)
		bool emptyCluster = false;								//un des clusters vide ? (locale)
		bool sampleClustered = false;

		samples = pointerToMatrix(inputs, nbSamples, nbParameters);
		outputs = pointerToMatrix(expected, nbSamples, nbOutputs);

		//initialisation des représentants des clusters
		Eigen::MatrixXf rpz = initializeClusterRepresentative(samples, cluster, nbParameters, nbSamples);

		while (!converge && it < nbIterations)
		{
			++it;

			for (int i = 0; i < cluster; ++i)
			{
				sampleCluster(i, 0) = 0;							//réinitialisation de la population de chaque cluster
			}

			for (int i = 0; i < nbSamples; ++i)						//répartition de tous les exemples dans les clusters
			{
				sampleClustered = false;
				clust(i, 1) = FLT_MAX;
				for (int j = 0; j < cluster; ++j)
				{
					dist = 0.f;
					for (int k = 0; k < nbParameters; ++k)				//calcul de la distance euclidienne de l'exemple avec le représentant du cluster parcouru
					{
						dist += pow(samples(i, k) - rpz(j, k), 2);
					}

					if (dist < clust(i, 1))								//si plus proche qu'avant : 
					{
						if (sampleClustered)								//décrémentation de la population de l'ancien cluster
						{
							sampleCluster(clust(i, 0), 0)--;				
						}
						sampleClustered = true;
						sampleCluster(j, 0)++;								//incrémentation de la population du nouveau cluster
						clust(i, 0) = j;									//attribution du nouveau cluster
						clust(i, 1) = dist;									//attribution de la nouvelle distance
					}
				}
			}

			for (int i = 0; i < cluster; ++i)						//test de cluster vide (condition de non-convergence)						
			{
				if (sampleCluster(i, 0) == 0)							//si un cluster vide
				{
					Eigen::MatrixXf tmpRpz = initializeClusterRepresentative(samples, 1, nbParameters, nbSamples);		//calcul d'un nouveau représentant 
					for (int j = 0; j < nbParameters; ++j)																//attribution du nouveau représentant
					{
						rpz(i, j) = tmpRpz(1, j);
					}
					emptyCluster = true;
				}
			}

			if (emptyCluster)										//si cluster vide, alors arrêt de l'itération et réattribution avec le(s) nouveau(x) représentant(s) calculé(s)
			{
				emptyCluster = false;
				continue;
			}

			converge = true;
			for (int i = 0; i < cluster; ++i)						//calcul des barycentres de chaque cluster et remplacement des représentants
			{
				for (int j = 0; j < nbParameters; ++j)					//calcul du barycentre
				{
					barycentres(i, j) = 0.f;

					for (int k = 0; k < nbSamples; ++k)
					{
						if (clust(k, 0) == i)
						{
							barycentres(i, j) += samples(k, j);
						}
					}

					barycentres(i, j) /= sampleCluster(i, 0);
				}

				dist = 0.f;												//calcul de la distance euclidienne entre le représentant et le barycentre
				for (int j = 0; j < nbParameters; ++j)
				{
					dist += pow(barycentres(i, j) - rpz(i, j), 2);
				}
				dist = sqrt(dist);

				if (dist > epsilon)										//si la distance est supérieure à l'objectif de convergence => on ne sort pas du while 
				{
					converge = false;
				}

				for (int j = 0; j < nbParameters; ++j)					//remplacement du représentant par le barycentre
				{
					rpz(i, j) = barycentres(i, j);
				}
			}
		}

		//CONVERGENCE => CALCULS FINAUX DES POIDS

		for (int i = 0; i < cluster; ++i)								//calcul de la moyenne de la population de chaque cluster
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

		for (int i = 0; i < nbSamples; ++i)								//calcul de la matrice Phi servant au calcul des poids
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

		Eigen::MatrixXf weights = (phi.transpose() * phi).inverse() * phi.transpose() * outputs;		//Calcul final de W = (tPhi * Phi)-1 * tPhi * Y

		Eigen::MatrixXf rpzVector(cluster * nbParameters, 1);

		for (int i = 0; i < nbParameters; ++i)
		{
			for (int j = 0; j < cluster; ++j)
			{
				rpzVector(i * cluster + j, 0) = rpz(j, i);
			}
		}

		Eigen::MatrixXf result(weights.rows() + rpzVector.rows(), weights.cols());

		return result.data();
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



	///////////////////
	/////// MLP ///////
	///////////////////

	// Constructor neural network
	void ConstructorNeuralNetwork(float stepLearningParam, int nbLayers)
	{
		executed = false;
		error = stepLearningParam;

		if (nbLayers > 1)
		{
			for (int i = 0; i < nbLayers; ++i)
			{
				std::vector<int> vec;
				layers.push_back(vec);
			}
		}
	}

	float TanH(float value)
	{
		return tanh(value);
	}

	float Sigmoide(float value)
	{
		return 1 / (1 + exp(-1 * value));
	}

	int AddNeuron(int indexCouche, int nbNeurone)
	{
		if (!executed)
		{
			if (indexCouche >= 0 && indexCouche < layers.size() && nbNeurone > 0)
			{
				for (int i = 0; i < nbNeurone; ++i)
				{
					layers[indexCouche].push_back(0);
				}
			}

			return 0;
		}
		else
		{
			std::cout << "Network already instanciated." << std::endl;
			return -1;
		}
	}

	int AddAllNeuron(std::vector<int> neuronePerLayers)
	{
		if (!executed)
		{
			if (neuronePerLayers.size() == layers.size())
			{
				for (int i = 0; i < neuronePerLayers.size(); ++i)
				{
					int result = AddNeuron(i, neuronePerLayers[i]); // +1 for bias to all hidden layers and input layer

					if (result == -1)
					{
						return -1;
					}
				}

				return 0;
			}
			else
			{
				std::cout << "ADDALLNEURON : Network architecture doesn't correspond with parameters." << std::endl;
				return -1;
			}
		}
	}

	int InitNeuralNetwork(float initWeight, int* weightsPerLayerArray, int nbLayer, int activateFunc, float biasValue)
	{
		int res = 0;

		std::vector<int> weightsPerLayer;
		for (int i = 0; i < nbLayer; ++i)
		{
			weightsPerLayer.push_back(weightsPerLayerArray[i]);
		}

		ConstructorNeuralNetwork(0.05f, weightsPerLayer.size());
		// A ajouter dans des LOG ?
		//PrintValues();

		res = AddAllNeuron(weightsPerLayer);
		if (res == -1)
		{
			std::cout << "ERROR AddAllNeuron" << std::endl;
		}

		switch (activateFunc)
		{
		case 0:
			ActivateFunc = &Sigmoide;
			break;
		case 1:
			ActivateFunc = &TanH;
			break;
		default:
			ActivateFunc = &Sigmoide;
		}

		for (int i = 0; i < layers.size(); ++i)
		{
			if (layers[i].size() <= 0)
			{
				std::cout << "Layer requires at least one neuron. Layer " << i << std::endl;
				return -1;
			}
		}

		try
		{
			if (!executed)
			{
				executed = true;
				// Loop on each layer
				for (int i = 0; i < layers.size(); ++i)
				{
					std::vector<float> allWeightPerNeuron; // Correspond to all weight of one neuron between the current layer and the next layer
					std::vector<std::vector<float>> allWeightPerNeuronPerLayer; // Correspond to all weight of all neuron between the current layer and the next layer
					std::vector<float> addValues; // Correspond to neuron's values of the current layer

												  // Loop on each neuron of current layer
					for (int j = 0; j < layers[i].size(); ++j)
					{
						// Check if we are at the last layer, because last layer doesn't have link with a next layer
						if (i != layers.size() - 1)
						{
							// Add weight for each neuron of the current layer to each neuron of the next layer
							for (int k = 0; k < layers[i + 1].size(); ++k)
							{
								if (i < layers.size() - 2 && k == layers[i + 1].size() - 1)
									continue;
								// Initialize weight's value to 0.5
								allWeightPerNeuron.push_back(0.5f);
							}

							// Add all weight of the current neuron to the current layer
							allWeightPerNeuronPerLayer.push_back(allWeightPerNeuron);
							allWeightPerNeuron.clear();
						}

						if (layers[i].size() - 1 != j || i == layers.size() - 1)
						{
							// Initialize value of current neuron to 0
							addValues.push_back(0.f);
						}
						else//(layers[i].size() - 1 == j && i != layers.size() - 1)
						{
							// If it's bias
							addValues.push_back(biasValue);
						}

						if (j == layers[i].size() - 1)
						{
							if (i != layers.size() - 1)
							{
								weights.push_back(allWeightPerNeuronPerLayer);
							}

							// Add values of all neuron of this layer
							values.push_back(addValues);
							addValues.clear();
						}
					}
				}
			}

			return 0;
		}
		catch (std::exception &e)
		{
			std::cout << "InitNeuralNetwork ERROR : " << e.what() << std::endl;
			return -1;
		}
	}

	// Pass the value of each neuron of the first layer, and all values of each neuron of each layer will be calculated (here we use Sigmoide as activation function)
	int Propagation(std::vector<float> inputLayer)
	{
		if (executed)
		{
			// Check if the layer pass in parameter has the same number of neuron
			if (inputLayer.size() == layers[0].size() - 1)
			{
				// Initialize the neuron's values of the first layer with values pass in parameter
				for (int i = 0; i < inputLayer.size(); ++i)
				{
					values[0][i] = inputLayer[i];
				}

				// Loop on each layers, we start at hidden layer 1 because we already initialize the first layer
				for (int i = 1; i < values.size(); ++i)
				{
					// Loop on each neurons value
					for (int j = 0; j < values[i].size(); ++j)
					{
						if (values.size() - 1 != i && values[i].size() - 1 == j)
						{
							continue;
						}

						float value = 0.f;

						// Loop on previous layer because we need the value of each neuron of the previous layer to calculate each neuron of the actual layer
						for (int k = 0; k < values[i - 1].size(); ++k)
						{
							value += values[i - 1][k] * weights[i - 1][k][j];
						}

						// Apply Sigmoid function to the weighted sum (somme ponderee = weighted sum ??)
						values[i][j] = ActivateFunc(value);
					}
				}

				return 0;
			}
			else
			{
				std::cout << " PROPAGATION : Network architecture doesn't correspond with parameters." << std::endl;
				return -1;
			}
		}
		else
		{
			std::cout << "Neural network not initialized" << std::endl;
			return -1;
		}
	}

	int Retropropagation(std::vector<float> outputLayer)
	{
		try
		{
			if (outputLayer.size() == values[values.size() - 1].size())
			{
				for (int i = 0; i < outputLayer.size(); ++i)
				{
					values[values.size() - 1][i] = outputLayer[i] - values[values.size() - 1][i];
				}

				for (int i = values.size() - 1; i > 0; --i)
				{
					for (int j = 0; j < values[i - 1].size(); ++j)
					{
						for (int k = 0; k < weights[i - 1][j].size(); ++k)
						{
							float sum = 0.f;

							for (int l = 0; l < values[i - 1].size(); ++l)
							{
								sum += values[i - 1][l] * weights[i - 1][l][k];
							}

							sum = ActivateFunc(sum);

							weights[i - 1][j][k] -= error * (-1 * values[i][k] * sum * (1 - sum) * values[i - 1][j]);
						}
					}

					for (int j = 0; j < values[i - 1].size(); ++j)
					{
						float sum = 0.f;
						for (int k = 0; k < values[i].size(); ++k)
						{
							if (i != values.size() - 1 && k == values[i].size() - 1)
								continue;

							sum += values[i][k] * weights[i - 1][j][k];
						}
						values[i - 1][j] = sum;
					}
				}

				return 0;
			}
			else
			{
				std::cout << "RETROPOPAGATION : Network architecture doesn't correspond with parameters." << std::endl;
				return -1;
			}
		}
		catch (std::exception &e)
		{
			std::cout << e.what() << std::endl;
			return -1;
		}
	}



	int Learn(std::vector<float> inputLayer, std::vector<float> outputLayer)
	{
		if (executed)
		{
			if (inputLayer.size() == layers[0].size() - 1 && outputLayer.size() == layers[layers.size() - 1].size())
			{
				Propagation(inputLayer);
				Retropropagation(outputLayer);

				return 0;
			}
			else
			{
				std::cout << "LEARN : Network architecture doesn't correspond with parameters." << std::endl;
				return -1;
			}
		}
		else
		{
			std::cout << "Neural network not initialized" << std::endl;
			return -1;
		}
	}

	float* LearnMLP(int nbSample, float* inputs, const int nbInputParam, float* outputs, const int nbOutputParam, int nbIteration)
	{
		int it = 0;
		int min = 0;
		int max = nbSample - 1;
		while (it < nbIteration)
		{
			int i = min + (rand() % static_cast<int>(max - min + 1));

			std::vector<float> inputLayer;
			for (int j = 0; j < nbInputParam; ++j)
			{
				inputLayer.push_back(inputs[i + j]);
			}

			std::vector<float> outputLayer;
			for (int j = 0; j < nbOutputParam; ++j)
			{
				outputLayer.push_back(outputs[i + j]);
			}

			Learn(inputLayer, outputLayer);


			++it;
		}

		std::vector<float> resultWeights;

		for (int i = 0; i < weights.size(); ++i)
		{
			for (int j = 0; j < weights[i].size(); j++)
			{
				for (int k = 0; k < weights[i][j].size(); k++)
				{
					resultWeights.push_back(weights[i][j][k]);
				}
			}
		}

		return resultWeights.data();
	}
}