#include <iostream>
#include <vector>
#include <string>



std::vector<std::vector<int>> layers;
float error = 0.05f;
std::vector<std::vector<std::vector<float>>> weights; // Correspond to all weight of all neuron with the next layer, of all layer
std::vector<std::vector<float>> values; // Correspond to neuron's value of each layer
bool executed;

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
				int result = AddNeuron(i, neuronePerLayers[i]);

				if (result == -1)
				{
					return -1;
				}
			}

			return 0;
		}
		else
		{
			std::cout << "Network architecture doesn't correspond with parameters." << std::endl;
			return -1;
		}
	}
}

int InitNeuralNetwork()
{
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
							// Initialize weight's value to 0.5
							allWeightPerNeuron.push_back(0.5f);
						}

						// Add all weight of the current neuron to the current layer
						allWeightPerNeuronPerLayer.push_back(allWeightPerNeuron);
						allWeightPerNeuron.clear();
					}

					// Initialize value of current neuron to 0
					addValues.push_back(0.f);

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
		if (inputLayer.size() == layers[0].size())
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
					float value = 0.f;

					// Loop on previous layer because we need the value of each neuron of the previous layer to calculate each neuron of the actual layer
					for (int k = 0; k < values[i - 1].size(); ++k)
					{
						value += values[i - 1][k] * weights[i - 1][k][j];
					}

					// Apply Sigmoid function to the weighted sum (somme ponderee = weighted sum ??)
					values[i][j] = Sigmoide(value);
				}
			}

			return 0;
		}
		else
		{
			std::cout << "Network architecture doesn't correspond with parameters." << std::endl;
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

						sum = Sigmoide(sum);

						weights[i - 1][j][k] -= error * (-1 * values[i][k] * sum * (1 - sum) * values[i - 1][j]);
					}
				}

				for (int j = 0; j < values[i - 1].size(); ++j)
				{
					float sum = 0.f;
					for (int k = 0; k < values[i].size(); ++k)
					{
						sum += values[i][k] * weights[i - 1][j][k];
					}
					values[i - 1][j] = sum;
				}
			}

			return 0;
		}
		else
		{
			std::cout << "Network architecture doesn't correspond with parameters." << std::endl;
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
		if (inputLayer.size() == layers[0].size() && outputLayer.size() == layers[layers.size() - 1].size())
		{
			Propagation(inputLayer);
			Retropropagation(outputLayer);

			return 0;
		}
		else
		{
			std::cout << "Network architecture doesn't correspond with parameters." << std::endl;
			return -1;
		}
	}
	else
	{
		std::cout << "Neural network not initialized" << std::endl;
		return -1;
	}
}

void PrintValues()
{
	for (int i = 0, indexLayer = 1; i < values.size(); ++i, ++indexLayer)
	{
		std::cout << "Layer " << indexLayer << " :" << std::endl;
		for (int j = 0; j < values[i].size(); ++j)
		{
			std::cout << "\tNeuron " << j << " value : " << values[i][j] << std::endl;
		}
		std::cout << std::endl;
	}
}

void PrintWeights()
{
	for (int i = 0, indexLayer = 1; i < weights.size(); ++i, ++indexLayer)
	{
		std::cout << "Layer " << indexLayer << " :" << std::endl;
		for (int j = 0; j < weights[i].size(); ++j)
		{
			std::cout << "\tNeuron " << j << " :" << std::endl;
			for (int k = 0; k < weights[i][j].size(); ++k)
			{
				std::cout << "\t\tWeight " << k << " :" << weights[i][j][k] << std::endl;
			}
		}
		std::cout << std::endl;
	}
}

void PrintAll()
{
	std::cout << "All values :" << std::endl;
	PrintValues();

	std::cout << std::endl;

	std::cout << "All weight :" << std::endl;
	PrintWeights();
}


int main()
{
	int res = 0;
	ConstructorNeuralNetwork(0.05f, 4);

	PrintValues();

	std::vector<int> nbNeuronPerLayer;
	nbNeuronPerLayer.push_back(3);
	nbNeuronPerLayer.push_back(5);
	nbNeuronPerLayer.push_back(7);
	nbNeuronPerLayer.push_back(3);

	res = AddAllNeuron(nbNeuronPerLayer);
	if (res == -1)
	{
		std::cout << "ERROR AddAllNeuron" << std::endl;
	}

	res = InitNeuralNetwork();
	if (res == -1)
	{
		std::cout << "ERROR InitNeuralNetwork" << std::endl;
	}

	PrintAll();

	// 1
	std::vector<float> v1;
	v1.push_back(1);
	v1.push_back(0);
	v1.push_back(1);
	std::vector<float> v2;
	v2.push_back(1);
	v2.push_back(1);
	v2.push_back(0);

	Learn(v1, v2);
	v1.clear();
	v2.clear();

	// 2
	v1.push_back(1);
	v1.push_back(0);
	v1.push_back(0);

	v2.push_back(0);
	v2.push_back(0);
	v2.push_back(0);

	Learn(v1, v2);
	v1.clear();
	v2.clear();

	// 3
	v1.push_back(0);
	v1.push_back(0);
	v1.push_back(0);

	v2.push_back(1);
	v2.push_back(0);
	v2.push_back(1);

	Learn(v1, v2);
	v1.clear();
	v2.clear();

	// 4
	v1.push_back(1);
	v1.push_back(1);
	v1.push_back(1);

	v2.push_back(1);
	v2.push_back(1);
	v2.push_back(1);

	Learn(v1, v2);
	v1.clear();
	v2.clear();

	PrintAll();

	return 0;
}