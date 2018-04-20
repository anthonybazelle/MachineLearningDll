using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using System;

public class UseDll : MonoBehaviour
{
    #region DLL Import
    [DllImport("MLDll", EntryPoint = "LinearRegression")]
    public static extern IntPtr GetLinearRegressionFunction(IntPtr xCollection, IntPtr yCollection, int dataSize);

    [DllImport("MLDll", EntryPoint = "PerceptronRosenblatt")]
    public static extern IntPtr GetLinearClassifierRosenblatt(IntPtr inputs, IntPtr expected, IntPtr weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance);

    [DllImport("MLDll", EntryPoint = "PLA")]
    public static extern IntPtr GetLinearClassifierPLA(IntPtr inputs, IntPtr expected, IntPtr weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance);

    [DllImport("MLDll", EntryPoint = "LinearRegressionWithEigen")]
    public static extern IntPtr GetLinearRegressionWithEigen(IntPtr inputs, IntPtr zBuffer, int nbParameter, int nbSample);

    [DllImport("MLDll", EntryPoint = "RBFNaiveTraining")]
    public static extern IntPtr RunRBFNaiveTraining(float gamma, IntPtr inputs, IntPtr expected, int nbParameters, int nbSamples, int nbOutputs);

    [DllImport("MLDll", EntryPoint = "RBFRegression")]
    public static extern IntPtr RunRBFRegression(float gamma, IntPtr inputs, IntPtr data, IntPtr weights, int nbParameters, int nbSamples, int nbOutputs);

    [DllImport("MLDll", EntryPoint = "RBFClassification")]
    public static extern IntPtr RunRBFClassification(float gamma, IntPtr inputs, IntPtr data, IntPtr weights, int nbParameters, int nbSamples, int nbOutputs);

    [DllImport("MLDll", EntryPoint = "RBFkMeansTraining")]
    public static extern IntPtr RunRBFkMeansTraining(float epsilon, int cluster, float gamma, IntPtr inputs, IntPtr expected, int nbParameters, int nbSamples, int nbOutputs);

    [DllImport("MLDll", EntryPoint = "LearnMLP")]
    public static extern IntPtr LearnMLP(int nbSample, IntPtr inputs, int nbInputParam, IntPtr outputs, int nbOutputParam, int nbIteration, float initWeight, float error, IntPtr neuronsPerLayer, int nbLayer, int activateFunc, float biasValue, int verboseMode = 0);

    [DllImport("MLDll", EntryPoint = "PredictMLP")]
    public static extern IntPtr PredictMLP(IntPtr inputs, IntPtr W, int nbInputParam, IntPtr neuronsPerLayerArray, int nbLayer, int activateFunc);
    #endregion


    #region Data
    [SerializeField]
    bool verboseMode = false;

    // Regression
    [SerializeField]
    private Transform[] cubeRegression;


    // Regression Pseudo Inverse
    [SerializeField]
    private GameObject[] cubeRegressionPI;

    [SerializeField]
    private int nbParameterRegressionPI = 2;

    // Rosenblat
    [SerializeField]
    private GameObject[] cubeClassifierRosenblatt;

    [SerializeField]
    private int nbParameterClassifierRosenblatt = 2;

    [SerializeField]
    private float stepLearningClassifierRosenblatt = 0.001f;

    [SerializeField]
    private int nbIterationClassifierRosenblatt = 1000;

    [SerializeField]
    private float toleranceClassifierRosenblatt = 0.001f;

    // Training MLP
    [SerializeField]
    private GameObject[] cubeMLP;

    [SerializeField]
    private int nbInputParameterMLP;

    [SerializeField]
    private int nbOutputParameterMLP;

    [SerializeField]
    private float errorMLP = 0.05f;

    [SerializeField]
    private int nbIterationMLP = 100;

    [SerializeField]
    private float initWeightMLP = 0.5f;

    [SerializeField]
    private float biasValueMLP = 1.0f;

    [SerializeField]
    private int activateFuncMLP = 0;

    [SerializeField]
    private int[] nbNeuronPerLayerMLP;

    [SerializeField]
    private int verboseModeMLP = 0;

    [SerializeField]
    private int nbClasses = 3;

    [SerializeField]
    private GameObject goToTest;

    [SerializeField]
    private Material matGoToTest;

    // Training RBF
    [SerializeField]
    int nbCluster = 2;

    [SerializeField]
    float epsilon = 0.0f;


    // RBF Classifier
    [SerializeField]
    private GameObject[] cubeClassifierRBF;

    [SerializeField]
    bool TrainRBFClassifer = true;

    [SerializeField]
    private GameObject newPointRBFClassification;

    [SerializeField]
    private int nbParameterClassifierRBF = 2;

    [SerializeField]
    private float gammaClassifierRBF = 0.0f;

    [SerializeField]
    private int nbOutputClassificationRBF = 0;

    [SerializeField]
    private bool regression = false;

    // RBF Regression
    [SerializeField]
    private GameObject[] cubeRegressionRBF;

    [SerializeField]
    bool TrainRBFRegression = true;

    [SerializeField]
    private GameObject newPointRBFRegression;

    [SerializeField]
    private int nbParameterRegressionRBF = 2;

    [SerializeField]
    private float gammaRegressionRBF = 0.0f;

    [SerializeField]
    private int nbOutputRegressionRBF = 0;


    // PLA
    [SerializeField]
    private GameObject[] cubeClassifierPLA;

    [SerializeField]
    private int nbParameterClassifierPLA = 2;

    [SerializeField]
    private float stepLearningClassifierPLA = 0.001f;

    [SerializeField]
    private int nbIterationClassifierPLA = 1000;

    [SerializeField]
    private float toleranceClassifierPLA = 0.001f;


    private float slopeRegression = 0.0f;
    private float y_interceptRegression = 0.0f;

    private float slopeRegressionPI = 0.0f;
    private float y_interceptRegressionPI = 0.0f;

    private float weightXRosenblatt = 0.0f;
    private float weightYRosenblatt = 0.0f;
    private float weightBiasRosenblatt = 0.0f;

    private float weightXPLA = 0.0f;
    private float weightYPLA = 0.0f;
    private float weightBiasPLA = 0.0f;

    private IntPtr bufferWeightRBFClass;
    private IntPtr bufferWeightRBFRegr;

    private bool alreadyTestedMLP;
    private float[] allMLPWeight;
    #endregion


    // Use this for initialization
    void Start()
    {
        // Init Regression
        if (cubeRegression.Length > 2)
        {
            float[] allXPosition = new float[cubeRegression.Length];
            float[] allYPosition = new float[cubeRegression.Length];

            for (int i = 0; i < cubeRegression.Length; ++i)
            {
                allXPosition[i] = cubeRegression[i].position.x;
                allYPosition[i] = cubeRegression[i].position.y;
            }

            int size = cubeRegression.Length;
            IntPtr bufferX = Marshal.AllocCoTaskMem(Marshal.SizeOf(size) * allXPosition.Length);
            IntPtr bufferY = Marshal.AllocCoTaskMem(Marshal.SizeOf(size) * allYPosition.Length);

            Marshal.Copy(allXPosition, 0, bufferX, allXPosition.Length);
            Marshal.Copy(allYPosition, 0, bufferY, allYPosition.Length);

            IntPtr result = GetLinearRegressionFunction(bufferX, bufferY, cubeRegression.Length);
            float[] resultLinearRegression = new float[2];
            Marshal.Copy(result, resultLinearRegression, 0, 2);

            this.slopeRegression = resultLinearRegression[0];
            this.y_interceptRegression = resultLinearRegression[1];

            // Print it out to the console
            Debug.Log("Function found after Linear Regression : " + resultLinearRegression[0] + "x + " + resultLinearRegression[1]);
        }
        // Init Classifier Rosenblatt
        else if (cubeClassifierRosenblatt.Length > 2)
        {
            int nbSample = cubeClassifierRosenblatt.Length; // count sample
            /*
            if (stepLearningClassifier == null || nbIterationClassifier == null || toleranceClassifier == null)
                Debug.Log("Need some parameters in the classifier algorithm.");
                return;
            */
            float[] inputs = new float[nbSample * nbParameterClassifierRosenblatt];
            float[] expected = new float[nbSample];
            float[] weights = new float[nbSample]; // Will be erased, cause initialized in DLL

            for (int i = 0, j = 0; i < cubeClassifierRosenblatt.Length; ++i, j += 2)
            {
                inputs[j] = cubeClassifierRosenblatt[i].transform.position.x;
                inputs[j + 1] = cubeClassifierRosenblatt[i].transform.position.y;

                expected[i] = Convert.ToInt32(cubeClassifierRosenblatt[i].tag);
            }

            for (int i = 0; i < this.nbParameterClassifierRosenblatt; ++i)
            {
                weights[i] = 1.0f;
            }

            IntPtr bufferInputs = Marshal.AllocCoTaskMem(Marshal.SizeOf(inputs.Length) * inputs.Length);
            IntPtr bufferExpected = Marshal.AllocCoTaskMem(Marshal.SizeOf(expected.Length) * expected.Length);
            IntPtr bufferWeights = Marshal.AllocCoTaskMem(Marshal.SizeOf(weights.Length) * weights.Length);

            Marshal.Copy(inputs, 0, bufferInputs, inputs.Length);
            Marshal.Copy(expected, 0, bufferExpected, expected.Length);
            Marshal.Copy(weights, 0, bufferWeights, weights.Length);

            IntPtr result = GetLinearClassifierRosenblatt(bufferInputs, bufferExpected, bufferWeights, this.nbParameterClassifierRosenblatt, this.cubeClassifierRosenblatt.Length, this.stepLearningClassifierRosenblatt, this.nbIterationClassifierRosenblatt, this.toleranceClassifierRosenblatt);
            float[] resultLinearClassifier = new float[this.nbParameterClassifierRosenblatt + 1]; // +1 for bias
            Marshal.Copy(result, resultLinearClassifier, 0, this.nbParameterClassifierRosenblatt + 1);

            this.weightXRosenblatt = resultLinearClassifier[0];
            this.weightYRosenblatt = resultLinearClassifier[1];
            this.weightBiasRosenblatt = resultLinearClassifier[2];
        }
        // Init LinearRegression Eigen
        else if (cubeRegressionPI.Length > 0)
        {
            // (float* inputs, float* zBuffer, const int nbParameter, const int nbSample)
            float[] inputs = new float[cubeRegressionPI.Length * (nbParameterRegressionPI + 1)];

            for (int i = 0, j = 0; i < cubeRegressionPI.Length; ++i, j += 3)
            {
                inputs[j] = 1;
                inputs[j + 1] = cubeRegressionPI[i].transform.position.x;
                inputs[j + 2] = cubeRegressionPI[i].transform.position.y;
            }

            float[] zBuffer = new float[cubeRegressionPI.Length];

            for (int i = 0; i < cubeRegressionPI.Length; ++i)
            {
                zBuffer[i] = cubeRegressionPI[i].transform.position.z;
            }

            //LinearRegressionWithEigen(float* inputs, float* zBuffer, const int nbParameter, const int nbSample);
            IntPtr bufferInputs = Marshal.AllocCoTaskMem(Marshal.SizeOf(inputs.Length) * inputs.Length);
            IntPtr bufferZBuffer = Marshal.AllocCoTaskMem(Marshal.SizeOf(zBuffer.Length) * zBuffer.Length);

            Marshal.Copy(inputs, 0, bufferInputs, inputs.Length);
            Marshal.Copy(zBuffer, 0, bufferZBuffer, zBuffer.Length);

            IntPtr result = GetLinearRegressionWithEigen(bufferInputs, bufferZBuffer, this.nbParameterRegressionPI + 1, this.cubeRegressionPI.Length);
            float[] resultRegressionPI = new float[nbParameterRegressionPI + 1];
            Marshal.Copy(result, resultRegressionPI, 0, nbParameterRegressionPI + 1);

            float w0 = resultRegressionPI[0];
            float w1 = resultRegressionPI[1];
            float w2 = resultRegressionPI[2];

            for (int i = 0; i < cubeRegressionPI.Length; ++i)
            {
                float z = w0 * 1 + w1 * cubeRegressionPI[i].transform.position.x + w2 * cubeRegressionPI[i].transform.position.y;
                cubeRegressionPI[i].transform.position = new Vector3(cubeRegressionPI[i].transform.position.x, cubeRegressionPI[i].transform.position.y, z);
            }
        }
        // Init Classifier PLA
        else if (cubeClassifierPLA.Length > 0)
        {
            int nbSample = cubeClassifierPLA.Length; // count sample
            /*
            if (stepLearningClassifier == null || nbIterationClassifier == null || toleranceClassifier == null)
                Debug.Log("Need some parameters in the classifier algorithm.");
                return;
            */
            float[] inputs = new float[nbSample * nbParameterClassifierPLA];
            float[] expected = new float[nbSample];
            float[] weights = new float[nbSample]; // Will be erased, cause initialized in DLL

            for (int i = 0, j = 0; i < cubeClassifierPLA.Length; ++i, j += 2)
            {
                inputs[j] = cubeClassifierPLA[i].transform.position.x;
                inputs[j + 1] = cubeClassifierPLA[i].transform.position.y;

                expected[i] = Convert.ToInt32(cubeClassifierPLA[i].tag);
            }

            for (int i = 0; i < this.nbParameterClassifierPLA; ++i)
            {
                weights[i] = 1.0f;
            }

            IntPtr bufferInputs = Marshal.AllocCoTaskMem(Marshal.SizeOf(inputs.Length) * inputs.Length);
            IntPtr bufferExpected = Marshal.AllocCoTaskMem(Marshal.SizeOf(expected.Length) * expected.Length);
            IntPtr bufferWeights = Marshal.AllocCoTaskMem(Marshal.SizeOf(weights.Length) * weights.Length);

            Marshal.Copy(inputs, 0, bufferInputs, inputs.Length);
            Marshal.Copy(expected, 0, bufferExpected, expected.Length);
            Marshal.Copy(weights, 0, bufferWeights, weights.Length);

            IntPtr result = GetLinearClassifierPLA(bufferInputs, bufferExpected, bufferWeights, this.nbParameterClassifierPLA, this.cubeClassifierPLA.Length, this.stepLearningClassifierPLA, this.nbIterationClassifierPLA, this.toleranceClassifierPLA);
            float[] resultLinearClassifier = new float[this.nbParameterClassifierPLA + 1]; // +1 for bias
            Marshal.Copy(result, resultLinearClassifier, 0, this.nbParameterClassifierPLA + 1);

            this.weightXPLA = resultLinearClassifier[0];
            this.weightYPLA = resultLinearClassifier[1];
            this.weightBiasPLA = resultLinearClassifier[2];
        }
        // Init Classifier RBF
        else if (this.cubeClassifierRBF.Length > 0)
        {
            int nbSample = cubeClassifierRBF.Length;
            float[] inputs = new float[nbSample * nbParameterClassifierRBF];
            float[] expected = new float[nbSample];

            for (int i = 0, j = 0; i < nbSample; ++i, j += 2)
            {
                inputs[j] = cubeClassifierRBF[i].transform.position.x;
                inputs[j + 1] = cubeClassifierRBF[i].transform.position.y;

                expected[i] = Convert.ToInt32(cubeClassifierRBF[i].tag);
            }

            IntPtr bufferInputs = Marshal.AllocCoTaskMem(Marshal.SizeOf(inputs.Length) * inputs.Length);
            IntPtr bufferExpected = Marshal.AllocCoTaskMem(Marshal.SizeOf(expected.Length) * expected.Length);

            Marshal.Copy(inputs, 0, bufferInputs, inputs.Length);
            Marshal.Copy(expected, 0, bufferExpected, expected.Length);

            if (this.TrainRBFClassifer || this.bufferWeightRBFClass == null)
            {
                //IntPtr result = RunRBFNaiveTraining(gammaClassifierRBF, bufferInputs, bufferExpected, this.nbParameterClassifierRBF, nbSample, this.nbOutputClassificationRBF);
                IntPtr result = RunRBFNaiveTraining(gammaClassifierRBF, bufferInputs, bufferExpected, this.nbParameterClassifierRBF, nbSample, this.nbOutputClassificationRBF);
                float[] resultLinearClassifierTraining = new float[this.nbParameterClassifierRBF];
                Marshal.Copy(result, resultLinearClassifierTraining, 0, this.nbParameterClassifierRBF * nbOutputClassificationRBF);
                bufferWeightRBFClass = Marshal.AllocCoTaskMem(Marshal.SizeOf(resultLinearClassifierTraining.Length) * resultLinearClassifierTraining.Length);
                float[] weights = new float[nbParameterClassifierRBF];
                Marshal.Copy(weights, 0, bufferWeightRBFClass, weights.Length);
            }

            float[] newPoint = new float[2];
            newPoint[0] = this.newPointRBFClassification.transform.position.x;
            newPoint[1] = this.newPointRBFClassification.transform.position.y;

            IntPtr bufferNewPoint = Marshal.AllocCoTaskMem(Marshal.SizeOf(newPoint.Length) * newPoint.Length);
            Marshal.Copy(newPoint, 0, bufferNewPoint, newPoint.Length);

            IntPtr resultClass = RunRBFClassification(gammaClassifierRBF, bufferInputs, bufferNewPoint, bufferWeightRBFClass, nbParameterRegressionRBF, 1, nbOutputRegressionRBF);
            float[] resultClassif = new float[nbOutputRegressionRBF];
            Marshal.Copy(resultClass, resultClassif, 0, nbOutputRegressionRBF);

            if (resultClassif[0] > 0.5f)
            {
                Debug.Log("This is a dog !");
            }
            else
            {
                Debug.Log("This is a cat !");
            }
        }
        // Init Regression RBF
        else if (this.cubeRegressionRBF.Length > 0)
        {
            int nbSample = cubeRegressionRBF.Length;
            float[] inputs = new float[nbSample * nbParameterRegressionRBF];
            float[] expected = new float[nbSample];

            for (int i = 0, j = 0; i < nbSample; ++i, j += 2)
            {
                inputs[j] = cubeRegressionRBF[i].transform.position.x;
                inputs[j + 1] = cubeRegressionRBF[i].transform.position.y;

                expected[i] = Convert.ToInt32(cubeRegressionRBF[i].tag);
            }

            IntPtr bufferInputs = Marshal.AllocCoTaskMem(Marshal.SizeOf(inputs.Length) * inputs.Length);
            IntPtr bufferExpected = Marshal.AllocCoTaskMem(Marshal.SizeOf(expected.Length) * expected.Length);

            Marshal.Copy(inputs, 0, bufferInputs, inputs.Length);
            Marshal.Copy(expected, 0, bufferExpected, expected.Length);

            if (this.TrainRBFRegression || this.bufferWeightRBFRegr == null)
            {
                IntPtr result = RunRBFNaiveTraining(gammaRegressionRBF, bufferInputs, bufferExpected, this.nbParameterRegressionRBF, nbSample, this.nbOutputRegressionRBF);
                float[] resultLinearRegressionTraining = new float[nbSample];
                Marshal.Copy(result, resultLinearRegressionTraining, 0, nbSample);

                bufferWeightRBFRegr = Marshal.AllocCoTaskMem(Marshal.SizeOf(resultLinearRegressionTraining.Length) * resultLinearRegressionTraining.Length);
                Marshal.Copy(resultLinearRegressionTraining, 0, bufferWeightRBFRegr, resultLinearRegressionTraining.Length);
            }

            float[] newPoint = new float[2];
            newPoint[0] = this.newPointRBFRegression.transform.position.x;
            newPoint[1] = this.newPointRBFRegression.transform.position.y;

            IntPtr bufferNewPoint = Marshal.AllocCoTaskMem(Marshal.SizeOf(newPoint.Length) * newPoint.Length);
            Marshal.Copy(newPoint, 0, bufferNewPoint, newPoint.Length);

            if (regression)
            {
                IntPtr resultRegr = RunRBFRegression(gammaClassifierRBF, bufferInputs, bufferNewPoint, bufferWeightRBFRegr, nbParameterRegressionRBF, 1, nbOutputRegressionRBF);
                float[] resultRegress = new float[nbOutputRegressionRBF];
                Marshal.Copy(resultRegr, resultRegress, 0, nbOutputRegressionRBF);

                if (resultRegress[0] < 0.5f)
                {
                    Debug.Log("This is a dog !");
                }
                else
                {
                    Debug.Log("This is a cat !");
                }
            }
            else
            {
                IntPtr resultClass = RunRBFClassification(gammaClassifierRBF, bufferInputs, bufferNewPoint, bufferWeightRBFClass, nbParameterRegressionRBF, 1, nbOutputRegressionRBF);
                float[] resultClassif = new float[nbOutputRegressionRBF];
                Marshal.Copy(resultClass, resultClassif, 0, nbOutputRegressionRBF);

                if (resultClassif[0] > 0.5f)
                {
                    Debug.Log("This is a dog !");
                }
                else
                {
                    Debug.Log("This is a cat !");
                }
            }
        }
        // MLP
        else if (this.cubeMLP.Length > 0)
        {
            int nbSample = this.cubeMLP.Length;
            float[] inputs = new float[nbSample * nbInputParameterMLP];
            float[] outputs = new float[nbSample * nbClasses];

            for (int i = 0, j = 0, k = 0; i < nbSample; ++i, j += nbInputParameterMLP, k += nbClasses)
            {
                inputs[j] = cubeMLP[i].transform.position.x;
                inputs[j + 1] = cubeMLP[i].transform.position.y;

                if (cubeMLP[i].tag == "red")
                {
                    outputs[k] = 1.0f;
                    outputs[k + 1] = 0.0f;
                    outputs[k + 2] = 0.0f;
                }
                else if (cubeMLP[i].tag == "green")
                {
                    outputs[k] = 0.0f;
                    outputs[k + 1] = 1.0f;
                    outputs[k + 2] = 0.0f;
                }
                else if (cubeMLP[i].tag == "blue")
                {
                    outputs[k] = 0.0f;
                    outputs[k + 1] = 0.0f;
                    outputs[k + 2] = 1.0f;
                }
            }

            IntPtr bufferNeuronsPerLayer = Marshal.AllocCoTaskMem(Marshal.SizeOf(nbNeuronPerLayerMLP.Length) * nbNeuronPerLayerMLP.Length);
            IntPtr bufferInputs = Marshal.AllocCoTaskMem(Marshal.SizeOf(inputs.Length) * inputs.Length);
            IntPtr bufferOutput = Marshal.AllocCoTaskMem(Marshal.SizeOf(outputs.Length) * outputs.Length);

            Marshal.Copy(nbNeuronPerLayerMLP, 0, bufferNeuronsPerLayer, nbNeuronPerLayerMLP.Length);
            Marshal.Copy(inputs, 0, bufferInputs, inputs.Length);
            Marshal.Copy(outputs, 0, bufferOutput, outputs.Length);

            // InitNeuralNetwork(initWeightMLP, bufferNeuronsPerLayer, nbNeuronPerLayerMLP.Length, activateFuncMLP, biasValueMLP, verboseModeMLP);
            IntPtr resultWeights = LearnMLP(nbSample, bufferInputs, nbInputParameterMLP, bufferOutput, nbOutputParameterMLP, nbIterationMLP, initWeightMLP, errorMLP, bufferNeuronsPerLayer, nbNeuronPerLayerMLP.Length,
                activateFuncMLP, biasValueMLP, verboseModeMLP);

            int nbWeightResult = 0;
            for (int i = 0; i < nbNeuronPerLayerMLP.Length; ++i)
            {
                if (i != nbNeuronPerLayerMLP.Length - 1)
                {
                    nbWeightResult += (nbNeuronPerLayerMLP[i] * nbNeuronPerLayerMLP[i + 1]);
                }
            }

            //float[] resultMLP = new float[nbSample * nbOutputParameterMLP]; // +1 for bias
            //Marshal.Copy(resultWeights, resultMLP, 0, nbSample * nbOutputParameterMLP);

            float[] resultWeightMLPPPP = new float[63];
            Marshal.Copy(resultWeights, resultWeightMLPPPP, 0, 63);

            allMLPWeight = resultWeightMLPPPP;
            int u = 0;
        }
    }


    void Update()
    {
        if (this.cubeRegression.Length > 2)
        {
            // for x = -50 and x = 50 for example and see the line
            float y1 = this.slopeRegression * (-50) + y_interceptRegression;
            float y2 = this.slopeRegression * 50 + y_interceptRegression;
            Debug.DrawLine(new Vector3(-50, y1, 0), new Vector3(50, y2, 0), Color.green);
        }
        else if (this.cubeClassifierRosenblatt.Length > 2)
        {
            // for x = -10 and x = -10 
            float x1 = -10;
            float y1 = -(x1 * this.weightXRosenblatt / this.weightYRosenblatt) - ((-1 * this.weightBiasRosenblatt) / this.weightYRosenblatt);

            float x2 = 10;
            float y2 = -(x2 * this.weightXRosenblatt / this.weightYRosenblatt) + (this.weightBiasRosenblatt / this.weightYRosenblatt);

            Vector3 p1 = new Vector3(-10, y1);
            Vector3 p2 = new Vector3(10, y2);

            Debug.DrawLine(p1, p2, Color.green);
        }
        else if (this.cubeClassifierPLA.Length > 2)
        {
            // for x = -10 and x = -10 
            float x1 = -10;
            float y1 = -(x1 * this.weightXPLA / this.weightYPLA) - ((-1 * this.weightBiasPLA) / this.weightYPLA);

            float x2 = 10;
            float y2 = -(x2 * this.weightXPLA / this.weightYPLA) + (this.weightBiasPLA / this.weightYPLA);

            Vector3 p1 = new Vector3(-10, y1);
            Vector3 p2 = new Vector3(10, y2);

            Debug.DrawLine(p1, p2, Color.green);
        }
        else if (!alreadyTestedMLP && goToTest != null)
        {
            alreadyTestedMLP = true;
            IntPtr bufferNeuronsPerLayer = Marshal.AllocCoTaskMem(Marshal.SizeOf(nbNeuronPerLayerMLP.Length) * nbNeuronPerLayerMLP.Length);
            Marshal.Copy(nbNeuronPerLayerMLP, 0, bufferNeuronsPerLayer, nbNeuronPerLayerMLP.Length);

            IntPtr bufferResultWeightMLP = Marshal.AllocCoTaskMem(Marshal.SizeOf(allMLPWeight.Length) * allMLPWeight.Length);
            Marshal.Copy(allMLPWeight, 0, bufferResultWeightMLP, allMLPWeight.Length);

            float[] inputToTest = new float[nbInputParameterMLP];
            IntPtr bufferInputToTestMLP = Marshal.AllocCoTaskMem(Marshal.SizeOf(inputToTest.Length) * inputToTest.Length);
            Marshal.Copy(inputToTest, 0, bufferInputToTestMLP, inputToTest.Length);

            IntPtr superResult = PredictMLP(bufferInputToTestMLP, bufferResultWeightMLP, nbInputParameterMLP, bufferNeuronsPerLayer, nbNeuronPerLayerMLP.Length, activateFuncMLP);
            //Marshal.Copy(superResult, inputToTest, 0, nbClasses);
            float[] resultPredict = new float[nbClasses]; // +1 for bias
            Marshal.Copy(superResult, resultPredict, 0, nbClasses);


            float max = 0.0f;
            int indiceMax = 0;

            for (int i = 0; i < resultPredict.Length; ++i)
            {
                if (max < resultPredict[i])
                {
                    max = resultPredict[i];
                    indiceMax = i;
                }
            }

            switch (indiceMax)
            {
                case 0:
                    //matGoToTest.color = Color.red;
                    Debug.Log("I'm RED !");
                    break;
                case 1:
                    //matGoToTest.color = Color.green;
                    Debug.Log("I'm GREEN !");
                    break;
                case 2:
                    //matGoToTest.color = Color.blue;
                    Debug.Log("I'm BLUE !");
                    break;
                default:
                    //matGoToTest.color = Color.white;
                    Debug.Log("I'm WHITE !");
                    break;
            }
        }
    }
}