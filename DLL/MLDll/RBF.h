// RBF.h

#include <Eigen/Dense>
#include <ctime>

class RBF
{
	float* RBFNaiveTraining(float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs);
	float* RBFRegression(float gamma, float* inputs, float* data, float* weights, int nbParameters, int nbSamples, int nbOutputs);
	float* RBFClassification(float gamma, float* inputs, float* data, float* weights, int nbParameters, int nbSamples, int nbOutputs);
	float* RBFkMeansTraining(float epsilon, int cluster, float gamma, float* inputs, float* expected, int nbParameters, int nbSamples, int nbOutputs);
	public static Eigen::MatrixXf pointerToMatrix(float* m, int rows, int cols);
	Eigen::MatrixXf initializeClusterRepresentative(Eigen::MatrixXf inputs, int cluster, int nbParameters, int nbSamples);
};
