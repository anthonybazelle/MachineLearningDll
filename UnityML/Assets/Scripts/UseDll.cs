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
    public static extern IntPtr GetLinearClassifierFunction(IntPtr inputs, IntPtr expected, IntPtr weights, int nbParameters, int nbSample, float stepLearning, int nbIteration, float tolerance);

    [DllImport("MLDll", EntryPoint = "TestRefArrayOfInts")]
    public static extern IntPtr GetTestRefArrayOfInts(ref IntPtr array, ref int size);
    #endregion


    #region Data
    [SerializeField]
    private Transform[] cubeRegression;

    [SerializeField]
    private GameObject[] cubeClassifier;

    [SerializeField]
    int nbParameterClassifier = 2;

    [SerializeField]
    float stepLearningClassifier = 0.001f;

    [SerializeField]
    int nbIterationClassifier = 1000;

    [SerializeField]
    float toleranceClassifier = 0.001f;

    private float slopeRegression = 0.0f;
    private float y_interceptRegression = 0.0f;

    private float weightX = 0.0f;
    private float weightY = 0.0f;
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
        // Init Classifier
        else if (cubeClassifier.Length > 2)
        {
            int nbSample = cubeClassifier.Length; // count sample
            /*
            if (stepLearningClassifier == null || nbIterationClassifier == null || toleranceClassifier == null)
                Debug.Log("Need some parameters in the classifier algorithm.");
                return;
            */
            float[] inputs = new float[nbSample * nbParameterClassifier];
            float[] expected = new float[nbSample];
            float[] weights = new float[nbSample]; // Will be erased, cause initialized in DLL

            for (int i = 0, j = 0; i < cubeClassifier.Length; ++i, j+=2)
            {
                inputs[j] = cubeClassifier[i].transform.position.x;
                inputs[j + 1] = cubeClassifier[i].transform.position.y;

                expected[i] = Convert.ToInt32(cubeClassifier[i].tag);
            }

            for (int i = 0; i < this.nbParameterClassifier; ++i)
            {
                weights[i] = 1.0f;
            }

            IntPtr bufferInputs = Marshal.AllocCoTaskMem(Marshal.SizeOf(inputs.Length) * inputs.Length);
            IntPtr bufferExpected = Marshal.AllocCoTaskMem(Marshal.SizeOf(expected.Length) * expected.Length);
            IntPtr bufferWeights = Marshal.AllocCoTaskMem(Marshal.SizeOf(weights.Length) * weights.Length);

            Marshal.Copy(inputs, 0, bufferInputs, inputs.Length);
            Marshal.Copy(expected, 0, bufferExpected, expected.Length);
            Marshal.Copy(weights, 0, bufferWeights, weights.Length);

            IntPtr result = GetLinearClassifierFunction(bufferInputs, bufferExpected, bufferWeights, this.nbParameterClassifier, this.cubeClassifier.Length, this.stepLearningClassifier, this.nbIterationClassifier, this.toleranceClassifier);
            float[] resultLinearClassifier = new float[this.nbParameterClassifier];
            Marshal.Copy(result, resultLinearClassifier, 0, this.nbParameterClassifier);

            this.weightX = resultLinearClassifier[0];
            this.weightY = resultLinearClassifier[1];

            Debug.Log("Formula : " + resultLinearClassifier[0] + "x + " + resultLinearClassifier[1] + "y  ????");
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
        else if (this.cubeClassifier.Length > 2)
        {
            // for x = 50 and y = 50 and x = -50 and y = -50
            Debug.DrawLine(new Vector3(-5 * this.weightX, -5 * this.weightY, 0), new Vector3(5 * this.weightX, 5 * this.weightY, 0), Color.green);
        }
    }
}