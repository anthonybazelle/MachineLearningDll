using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;

public class UseDll : MonoBehaviour
{

    // Straight From the c++ Dll (unmanaged)
    [DllImport("MLDll", EntryPoint = "TestDivide")]
    public static extern float StraightFromDllTestDivide(float a, float b);

    [DllImport("MLDll", EntryPoint = "TestMultiply")]
    public static extern float StraightFromDllTestMultiply(float a, float b);

    /*
    [DllImport("MLDll", EntryPoint = "TestString")]
    [return: System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.Bool)]
    public static extern bool StraightFromDllTestString(StringBuilder c);*/

    [DllImport("MLDll", EntryPoint = "LinearRegression")]
    public static extern float GetLinearRegressionFunction([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 6)] float[] xCollection, 
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 6)] float[] yCollection, 
        int dataSize);

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

        float straightFromDllDivideResult = StraightFromDllTestDivide(20, 5);
        float straightFromDllMultiplyResult = StraightFromDllTestMultiply(20, 5);


        float[] tab1 = new float[] { 0.0f, 3.0f, 5.0f, 1.0f, 2.0f, 3.0f };
        float[] tab2 = new float[] { 0.0f, 1.0f, 5.0f, 3.0f, 3.0f, 3.0f };

        float resultRegression = GetLinearRegressionFunction(tab1, tab2, 6);


        //StringBuilder sb = new StringBuilder(250);
        //string s = StringReturnAPI01();
        //bool straightFromDllStringResult = StraightFromDllTestString(sb);

        // Print it out to the console
        Debug.Log(straightFromDllDivideResult);
        Debug.Log(straightFromDllMultiplyResult);
        //Debug.Log("Slope : " + resultRegression[0]);
        Debug.Log("Slope : " + resultRegression);
        //Debug.Log("Y_Intercept : " + resultRegression[1]);

        //this.slopeRegression = resultRegression[0];
        this.slopeRegression = resultRegression;
        //this.y_interceptRegression = resultRegression[1];
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

    void OnDrawGizmosSelected()
    {
        if (this.slopeRegression > -500 && this.y_interceptRegression > -500)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawLine(new Vector3(0, 0, 0), new Vector3(50, 50, 0));
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}