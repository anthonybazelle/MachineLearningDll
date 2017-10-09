using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;

public class UseDll : MonoBehaviour {

    // Straight From the c++ Dll (unmanaged)
    [DllImport("MLDll", EntryPoint="TestDivide")]
    public static extern float StraightFromDllTestDivide(float a, float b);

    [DllImport("MLDll", EntryPoint = "TestMultiply")]
    public static extern float StraightFromDllTestMultiply(float a, float b);

    [DllImport("MLDll", EntryPoint = "TestString")]
    public static extern StringBuilder StraightFromDllTestString(StringBuilder c);

    // Use this for initialization
    void Start () {

        float straightFromDllDivideResult = StraightFromDllTestDivide(20, 5);
        float straightFromDllMultiplyResult = StraightFromDllTestMultiply(20, 5);
        StringBuilder sb = new StringBuilder("Anthony");
        StringBuilder straightFromDllStringResult = StraightFromDllTestString(sb);

        // Print it out to the console
        Debug.Log(straightFromDllDivideResult);
        Debug.Log(straightFromDllMultiplyResult);
        Debug.Log(straightFromDllStringResult);

        // Write the result into a file, so we can even see it working in a build
        using (StreamWriter writer = new StreamWriter("debug.txt", true))
        {
            writer.WriteLine(straightFromDllDivideResult);
            writer.WriteLine(straightFromDllMultiplyResult);
            writer.WriteLine(straightFromDllStringResult);
        }
    }

    // Update is called once per frame
    void Update () {

    }
}