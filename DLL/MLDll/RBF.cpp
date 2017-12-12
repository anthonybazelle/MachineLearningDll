
#include "RBF.h"

float* RBF::RBFNaiveTraining(float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs)
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

float* RBF::RBFRegression(float gamma, float* inputs, float* data, float* weights, int nbParameters, int nbSamples, int nbOutputs)
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

float* RBF::RBFClassification(float gamma, float* inputs, float* data, float* weights, int nbParameters, int nbSamples, int nbOutputs)
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

float* RBF::RBFkMeansTraining(float epsilon, int cluster, float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs)
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

Eigen::MatrixXf RBF::pointerToMatrix(float* m, int rows, int cols)
{
	Eigen::MatrixXf mat(rows, cols);

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			mat(i, j) = m[i * cols + j];
		}
	}
}

Eigen::MatrixXf RBF::initializeClusterRepresentative(Eigen::MatrixXf inputs, int cluster, int nbParameters, int nbSamples)
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