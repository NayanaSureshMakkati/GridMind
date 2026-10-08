using UnityEngine;

public static class CoordinateConverter
{
    public const float CellSize = 1.0f;
    public const float AgentHeight = 0.5f;

    /// <summary>
    /// Converts discrete logical (row, column) coordinates to a 3D Unity Vector3.
    /// </summary>
    public static Vector3 GridToWorld(int row, int col)
    {
        float x = col * CellSize;
        float y = AgentHeight;
        float z = -row * CellSize;
        return new Vector3(x, y, z);
    }
}