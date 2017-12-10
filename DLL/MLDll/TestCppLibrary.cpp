// TestCPPLibrary.cpp : Defines the exported functions for the DLL application.
//
#include "stdafx.h"
#include "TestCppLibrary.h"
#include <string>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
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
		if (b == 0) {
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

	float LinearRegressionWithEigen(float* xCollection, float* yCollection, int nbXCollection, int nbYCollection)
	{
		//Eigen::Matrix2d mat(nbXCollection, 

		return 0.f;
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

		std::ofstream logFile;
		logFile.open("logFileDll.log");
		logFile << "In DLL : " << std::endl << std::endl << "DataSize in DLL : " << dataSize << std::endl <<"AVGy : " << AVGy << std::endl << "AVGx : " << AVGx << std::endl<< std::endl << "y_intercept : " << y_intercept << std::endl << "Slope : " << slope << std::endl;
		logFile.close();

		// slope * x + y_intercept = y

		float* result = new float[2];
		result[0] = slope;
		result[1] = y_intercept;

		return result;
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

	// Weigths in parameters useless, need to be modify later
	float* PerceptronRosenblatt(float* inputs, float* expected, float* weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance)
	{

		std::ofstream logFile;
		logFile.open("logFileDll.log");
		logFile << "Start RosenBlatt Perceptron : " << std::endl << std::endl << std::endl;
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

		logFile << "Initialize sample : DONE" << std::endl << std::endl;

		// bias ? not sure
		float w0 = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));

		bool different = true;
		int iteration = 0;
		std::vector<float> realResult;

		while (iteration < nbIteration && different)
		{
			// Here we'll update weights of parameters
			// Loop sample
			//std::vector<float> realResultTmp;

			bool needToContinue = false;

			for (int i = 0; i < nbSample; ++i)
			{
				// Logicaly the number of weight correspond to the number of parameter in a sample so we stop the loop when we are out of the number of weight
				// formula: (w1 * x1) + (w2 * x2) - w0 (-w0 because we chose x0 = - 1) when the two vector are different
				float result = 0.f;

				for (int j = 0; j < countWeight; ++j)
				{
					result += nativeInputs[i]->getParameters()[j] * weights[j];
				}
				result -= w0;

				if (std::abs(result - nativeInputs[i]->getExpected()) > tolerance)
				{
					needToContinue = true;
					for (int j = 0; j < countWeight; ++j)
					{
						weights[j] = weights[j] + stepLearning * (expected[i] - result) * inputs[j + i * nbParameters];
					}

					// We forgot to update bias weight before
					w0 = w0 + stepLearning * (expected[i] - result) * x0;

					for(int i = 0; i < nbParameters; ++i)
					{
						logFile << "Weight " << i << " : " << weights[i] << std::endl;
					}

					logFile << std::endl;
				}
			}

			if (!needToContinue)
			{
				break;
			}

			++iteration;


			/*
			// Push float* in a vector to make the comparison between real result and expecte result easier
			std::vector<float> expectedResult;
			for(int i = 0; i < nativeInputs.size(); ++i)
			{
			expectedResult.push_back(expected[i]);
			}

			// Comparision between two vector we'll first, check the size of the two vector, and if they have the same size, compare value by value
			if(expectedResult == realResultTmp)
			{
			different = false;
			realResult = realResultTmp;
			}
			else
			{
			for(int indexSample = 0; indexSample < nativeInputs.size(); ++indexSample)
			{
			for (int i = 0; i < countWeight; ++i)
			{
			weights[i] = weights[i] + stepLearning * (expected[nbSample] - realResultTmp[nbSample]) * inputs[i + indexSample * nbParameters];
			}
			}
			}
			++iteration;*/
		}

		logFile << "Whole necessary iteration (" << iteration <<") : DONE" << std::endl;

		for(int i = 0; i < nbParameters; ++i)
		{
			logFile << "Weight " << i << " : " << weights[i] << std::endl;
		}

		//weights
		//float* result = &(realResult[0]);

		float* resultWeight = new float[nbParameters + 1];
		
		for(int i = 0; i < nbParameters; ++i)
		{
			resultWeight[i] = weights[i];
		}

		resultWeight[nbParameters] = w0;

		return resultWeight;
	}


	float* MLPerceptronClassification(float* inputs, float* expected, float* weights, int* layersNeurones, int nbParameters, int nbSample, int nbLayers, float stepLearning, int nbIteration, float tolerance)
	{
		//weights should contain even the weights for the bias neurone in the first layer

		// Initialize weight with random between -1 and 1
		int countWeight = makeRandomWeight(weights);

		int countNeurones = nbParameters + 1;
		for (int i = 0; i < nbLayers; ++i)
		{
			countNeurones += layersNeurones[i];
		}

		std::vector<float> x_li;
		x_li.reserve(countNeurones + 1);			//"+1" pour le dernier neurone résultat

		std::vector<float> d_li;
		d_li.reserve(countNeurones + 1);

		float w0 = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));

		// Initialize array of sample given in input with the expected value in output for each sample in third parameter of sample's constructor
		std::vector<Sample*> nativeInputs;
		for (int i = 0; i < nbSample; ++i)
		{
			std::vector<float> parameters;
			std::vector<float> exp;
			for (int j = 0; j < nbParameters; ++j)
			{
				parameters.push_back(inputs[j + i * nbParameters]);
			}

			Sample* sample = new Sample(parameters, expected[i]);
			nativeInputs.push_back(sample);
		}

		// Initializing final bias (the one not going through layers)
		//float w0 = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 2));

		bool different = true;
		int iteration = 0;
		std::vector<float> realResult;
		bool needToContinue;
		int counter, nbx, nox, nbd;
		float result;
		int lastLayerNb;

		while (iteration < nbIteration && different)
		{
			// Here we'll update weights of parameters
			// Loop sample
			//std::vector<float> realResultTmp;
			needToContinue = false;

			//pour chaque sample d'entraînement
			for (int i = 0; i < nbSample; ++i)
			{
				//on reset tout
				nbx = 0;
				counter = 0;
				x_li.clear();

				//pour chaque paramètre (plus neurone de biais) on ajoute les entrées dans le vecteur des x_li
				for (int j = 0; j < nbParameters + 1; ++j)
				{
					x_li.push_back(nativeInputs[i]->getParameters()[j]);		//chaque entrée
				}
				x_li.push_back(-1);										//neurone de biais

				//pour chaque couche de notre perceptron
				for (int j = 0; j < nbLayers; ++j)
				{
					//pour chaque neurone de la couche en cours
					for (int k = 0; k < layersNeurones[j]; ++k)
					{
						result = 0.f;										//reset du résultat

						//pour chaque x de la couche d'avant, on ajoute le x * w
						for (int l = nbx; l < x_li.size(); ++l)
						{
							result += x_li[l] * weights[counter++];		//on incrémente counter pour chercher le bon poids facilement
						}
						result = tanh(result);								//sigmoïde de la somme pondérée
						x_li.push_back(result);							//on ajoute le x au vecteur des x_li
					}

					//on incrémente nbx pour savoir à quels x commencer pour la boucle du dessus
					if (j == 0)
					{
						//si première couche, on incrémente du nombre de paramètres + 1 (neurone de biais)
						nbx += nbParameters + 1;							
					}
					else
					{
						//sinon on incrémente du nombre de neurones dans la couche en cours
						nbx += layersNeurones[j - 1];
					}
				}

				//calcule final à partir des x de la dernière couche
				result = 0.f;
				for (int l = nbx; l = x_li.size(); ++l)
				{
					result += x_li[l] * weights[counter++];
				}

				//pour classification, on applique la sigmoïde à la sortie, pas pour la régression
				result -= w0;
				result = tanh(result);
				x_li.push_back(result);
				nox = x_li.size() - 1;

				//si pas satisfait du résultat, rétropropagation du gradient
				if (std::abs(result - nativeInputs[i]->getExpected()) > tolerance)
				{
					needToContinue = true;									//on doit continuer l'apprentissage

					//calcul du delta de la dernière couche (calcul différent pour la régression)
					float delta = (1 - pow(x_li[nox], 2)) * (x_li[nox] - nativeInputs[i]->getExpected());
					d_li.push_back(delta);

					--nox;
					nbd = 0;

					//pour chaque couche en partant de la dernière
					for (int j = nbLayers - 1; j >= 0; --j)
					{
						//pour chaque neurone de la couche en partant du dernier
						for (int k = layersNeurones[j] - 1; k >= 0; --k)
						{
							delta = 0;
							//pour chaque neurone vers lequel pointe de neurone en cours
							for (int l = d_li.size() - 1; l >= nbd; --l)
							{
								delta += weights[--counter] * d_li[l];			//on décrémente counter pour être au bon poids
							}
							delta *= (1 - pow(x_li[nox--], 2));

							d_li.push_back(delta);
						}

						//on incrémente nbd pour parcourir dans la boucle du neurone uniquement les d_li de la couche d'après
						if (j == nbLayers - 1)
						{
							//si c'est la première itération, il n'y a qu'un seul neurone, celui de sortie
							++nbd;
						}
						else
						{
							//sinon on décale du nombre de neurones de la couche parcourue
							nbd += layersNeurones[j];
						}
					}

					//normalement counter est à 0, mais on s'en assure
					counter = 0;
					nox = 0;

					//lorsqu'on a tous les deltas, on ajuste les w
					//pour chaque couche
					for (int i = 0; i < nbLayers; ++i)
					{
						//pour chaque neurone de la couche en cours
						for (int j = 0; j < layersNeurones[i]; ++j)
						{
							if (i == 0)
							{
								//si première, on fait en fonction du nombre de paramètres (+ neurone de biais)
								for (int k = 0; k < nbParameters + 1; ++k)
								{
									weights[counter] -= stepLearning * x_li[nox + k] * d_li[d_li.size() - (nox + 1)];
									counter++;
								}
								nox += nbParameters + 1;
							}
							else
							{
								//sinon on fait selon le nombre de neurones dans la couche d'avant
								for (int k = 0; k < layersNeurones[i - 1]; ++i)
								{
									weights[counter] -= stepLearning * x_li[nox + k] * d_li[d_li.size() - (nox + 1 + j)];
									counter++;
								}
								nox += layersNeurones[i - 1];
							}
						}
						lastLayerNb = layersNeurones[i];
					}

					//pour la dernière couche
					for (int i = 0; i < lastLayerNb; ++i)
					{
						weights[counter] -= stepLearning * x_li[nox + i] * d_li[d_li.size() - 1];
						counter++;
					}
					w0 -= stepLearning * d_li[d_li.size() - 1];
				}
			}

			if (!needToContinue)
			{
				break;
			}

			++iteration;
		}

		//je sais pas quoi return ici du coup j'ai laissé comme c'était
		return &(realResult[0]);

		/*
		C'est éminemment le bordel niveau index des vecteurs, mais ce qu'il faut comprendre c'est que : 
		- weights doit être rangé de la sorte : 
		_ [w_000, w_001, w_002, ..., w_010, w_011, ..., w_100, w_101, ...]
		_ avec w_lij où l est le numéro de la couche, i est le numéro du neurone vers lequel pointe le poids et j est le numéro du neurone pointant vers i
		_ pour chaque neurone donc se suivront les poids de tous les neurones de la couche d'avant allant vers lui
		- x_li correspond au résultat de chaque neurone vu par les autres, c'est-à-dire :
		- pour l = 0, ce sera juste les paramètres tels quels
		_ après, ce sera la sigmoïde (tanh) de la somme des w * x de la couche d'avant
		_ le dernier x_li correspond au résultat final du perceptron
		- x_li est rangé dans l'ordre des neurones c'est-à-dire : 
		_ [x_00, x_01, x_02, ..., x_10, x_11, ...]
		_ avec x_li où l est le numéro de la couche (démarrant à la couche des paramètres + 1 avec le biais)
		_ et i et le numéro du neurone dans la couche l
		- d_li est le delta de chacun des neurones, mais pris à l'envers car parcouru depuis la fin lors du calcul
		_ [d, d_(L)(N1), d_(L)(N1-1), ..., d_(L)(0), d_(L-1)(N2), ..., d_00]
		_ avec d_li où l est le numéro de la couche parcourue jusqu'à la première car les entrées n'ont pas de delta (pas d'erreur sur les entrées logique)
		_ et i est le numéro du neurone de la couche parcourue
		- d_li est donc parcouru dans le sens inverse de x_li et à chaque x_li (hormis pour les entrées) correspond un d_li

		- Concernant "counter" : 
		_ counter permet de parcourir les poids dans l'ordre pour le calcul des x_li
		_ à la fin du calcul des x_li, counter est normalement au tout dernier poids
		_ on reparcoure les poids dans l'autre sens pour calculer les d_li de chaque neurone
		_ enfin on reparcoure les poids dans l'ordre pour les mettre à jour

		- Concernent "nbx" :
		_ il permet lors du calcul des x_li de ne parcourir que les neurones de la couche précédente
		_ à la fin du calcul des x_li de chaque couche, on ajoute à nbx le nombre de neurones de la couche d'avant

		- Concernant "nbd" : 
		_ il fonctionne exactement de la même manière que nbx mais pour les d_li

		- Concernant "nox" :
		_ il permet de reparcourir le vecteur des x_li dans l'autre sens pour le calcul des d_li
		_ à la fin du calcul des d_li, nox est normalement à 0 puisqu'on à reparcouru tous les x_li pour calculer les d_li correspondants
		_ pour la mise à jour des poids, il sert à parcourir les x_li et d_li :
		~ pour les x_li, il nous faut les x_li de la couche précédente participant à un neurone de la couche en cours
		~ on parcoure donc x_li[nox + k] avec k allant de 0 au nombre de neurones de la couche précédente
		~ et nox étant égal pour une couche donnée à la somme du nombre de neurones sur toutes les couches précédentes
		~ pour démarrer au bon neurone

		~ pour les d_li c'est plus simple, il nous faut juste le neurone parcouru sur la couche en cours
		~ donc on a d_li[d_li.size() - (nox + 1 + j)] avec j allant de 0 au nombre de neurones sur la couche en cours
		*/
	}

	// Test marshalling
	float* TestRefArrayOfInts(int** ppArray, int* pSize)
	{
		float* result = new float[2];
		result[0] = 0;
		for (int i = 0; i < 10; ++i)
		{
			std::cout << (*ppArray)[i];
			result[0] = result[0] + 2;
			std::cout << "      SUM : " << result[0] << std::endl;
		}

		std::cout << result[0] << std::endl;
		result[1] = 32.5f;

		return result;
	}
}