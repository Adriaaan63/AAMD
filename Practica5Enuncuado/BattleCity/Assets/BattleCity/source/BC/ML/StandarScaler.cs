using System;
using UnityEngine;
using UnityEngine.Windows;

public class StandarScaler
{
    private float[] mean;
    private float[] std;
    public StandarScaler(string serieliced)
    {
        string[] lines = serieliced.Split("\n");
        string[] meanStr = lines[0].Split(",");
        string[] stdStr = lines[1].Split(",");
        mean = new float[meanStr.Length];
        std = new float[stdStr.Length];
        for (int i = 0; i < meanStr.Length; i++)
        {
            mean[i] = float.Parse(meanStr[i], System.Globalization.CultureInfo.InvariantCulture);
        }

        for (int i = 0; i < stdStr.Length; i++)
        {
            std[i] = float.Parse(stdStr[i], System.Globalization.CultureInfo.InvariantCulture);
            //std[i] = Mathf.Sqrt(std[i]);
        }
    }

    /// <summary>
    /// Aplica una normalización - media entre desviación tipica.
    /// </summary>
    /// <param name="a_input"></param>
    /// <returns></returns>
    public float[] Transform(float[] a_input)
    {
        if(mean == null && std == null) throw new Exception("Scaler no inicializado.");
        if(a_input.Length != mean.Length) throw new Exception($"Scaler espera {mean.Length} features, pero recibe {a_input.Length}.");

        float[] scaled = new float[a_input.Length];
        for(int i = 0; i < a_input.Length; i++)
        {
            float stdValue = std[i] == 0f ? 1e-6f : std[i];
            scaled[i] = (a_input[i] - mean[i]) / stdValue;
        }
        return scaled;
    }
}