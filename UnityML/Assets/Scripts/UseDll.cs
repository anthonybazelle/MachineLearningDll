using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using System;

public class UseDll : MonoBehaviour
{

    [SerializeField]
    private Transform[] allCubes;

    private float slope = 0.0f;
    private float y_intercept = 0.0f;

    // Straight From the c++ Dll (unmanaged)
    [DllImport("MLDll", EntryPoint = "TestDivide")]
    public static extern float StraightFromDllTestDivide(float a, float b);

    [DllImport("MLDll", EntryPoint = "TestMultiply")]
    public static extern float StraightFromDllTestMultiply(float a, float b);

    /*[DllImport("MLDll", EntryPoint = "LinearRegression")]
    public static extern IntPtr GetLinearRegressionFunction([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 6)] float[] xCollection,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 6)] float[] yCollection, int dataSize);*/

    [DllImport("MLDll", EntryPoint = "LinearRegression")]
    public static extern IntPtr GetLinearRegressionFunction(IntPtr xCollection, IntPtr yCollection, int dataSize);


    [DllImport("MLDll", EntryPoint = "TestRefArrayOfInts")]
    public static extern IntPtr GetTestRefArrayOfInts(ref IntPtr array, ref int size);
    /*
    [DllImport("MLDll", EntryPoint = "TestString")]
    [return: System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.Bool)]
    public static extern bool StraightFromDllTestString(StringBuilder c);*/
    
    /*
    [DllImport("MLDll", EntryPoint = "StringReturnAPI01", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
    [return: MarshalAs(UnmanagedType.LPStr)]
    public static extern string StringReturnAPI01();
    */


    #region Données membres
    float slopeRegression;
    float y_interceptRegression;
    #endregion


    // Use this for initialization
    void Start()
    {


        int[] array2 = new int[10];
        int size2 = array2.Length;
        Console.WriteLine("\n\nInteger array passed ByRef before call:");

        for (int i = 0; i < array2.Length; i++)
        {
            array2[i] = i;
            Console.Write(" " + array2[i]);
        }

        IntPtr buffer = Marshal.AllocCoTaskMem(Marshal.SizeOf(size2) * array2.Length);
        Marshal.Copy(array2, 0, buffer, array2.Length);

        IntPtr r = GetTestRefArrayOfInts(ref buffer, ref size2);
        float[] result2 = new float[2];
        Marshal.Copy(r, result2, 0, 2);


        Console.ReadLine();



        float straightFromDllDivideResult = StraightFromDllTestDivide(20, 5);
        float straightFromDllMultiplyResult = StraightFromDllTestMultiply(20, 5);

        float[] allXPosition = new float[allCubes.Length];
        float[] allYPosition = new float[allCubes.Length];

        for (int i = 0; i < allCubes.Length; ++i)
        {
            allXPosition[i] = allCubes[i].position.x;
            allYPosition[i] = allCubes[i].position.y;
        }

        //float[] tab1 = new float[] { 0.0f, 3.0f, 5.0f, 1.0f, 2.0f, 3.0f };
        //float[] tab2 = new float[] { 0.0f, 1.0f, 5.0f, 3.0f, 3.0f, 3.0f };

        int size = allCubes.Length; 
        IntPtr bufferX = Marshal.AllocCoTaskMem(Marshal.SizeOf(size) * allXPosition.Length);
        IntPtr bufferY = Marshal.AllocCoTaskMem(Marshal.SizeOf(size) * allYPosition.Length);

        Marshal.Copy(allXPosition, 0, bufferX, allXPosition.Length);
        Marshal.Copy(allYPosition, 0, bufferY, allYPosition.Length);

        IntPtr result = GetLinearRegressionFunction(bufferX, bufferY, allCubes.Length);
        float[] resultLinearRegression = new float[2];
        Marshal.Copy(result, resultLinearRegression, 0, 2);

        this.slope = resultLinearRegression[0];
        this.y_intercept = resultLinearRegression[1];

        //StringBuilder sb = new StringBuilder(250);
        //string s = StringReturnAPI01();
        //bool straightFromDllStringResult = StraightFromDllTestString(sb);

        // Print it out to the console
        Debug.Log(straightFromDllDivideResult);
        Debug.Log(straightFromDllMultiplyResult);
        //Debug.Log("Slope : " + resultRegression[0]);
        Debug.Log("Function found after Linear Regression : " + resultLinearRegression[0] + "x + " + resultLinearRegression[1]);
        //Debug.Log("Y_Intercept : " + resultRegression[1]);

        //this.slopeRegression = resultRegression[0];
        this.slopeRegression = (float)Math.Round(Convert.ToDecimal(resultLinearRegression[0]), 2);
        this.y_interceptRegression = resultLinearRegression[1];
        //Debug.Log(straightFromDllStringResult);
        //Debug.Log(s);
        //Debug.Log(sb);
        // Write the result into a file, so we can even see it working in a build
        using (StreamWriter writer = new StreamWriter("debug.txt", true))
        {
            writer.WriteLine(straightFromDllDivideResult);
            writer.WriteLine(straightFromDllMultiplyResult);
            //writer.WriteLine(straightFromDllStringResult);
        }
    }


    void Update()
    {
        // for x = -50 and x = 50 for example and see the line
        float y1 = this.slope * (-50) + y_intercept;
        float y2 = this.slope * 50 + y_intercept;
        Debug.DrawLine(new Vector3(-50, y1, 0), new Vector3(50, y2,0), Color.green);
    }
}