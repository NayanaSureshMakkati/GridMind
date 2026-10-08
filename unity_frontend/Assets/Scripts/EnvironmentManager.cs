using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Builds and manages the visual 3D grid world from logical environment data.
/// Prefabs are optional: if none are assigned, primitive fallbacks are generated
/// so the demo scene works out of the box (Developer 1 Phase 15).
///
/// 3D presentation (no hand-built assets, project convention):
///   * floor     - one continuous opaque base slab under the whole grid, plus
///                 solid (double-sided) cube tiles checkerboarded on top of it.
///                 A slab means the ground is always visible even if a tile is
///                 missing or a supplied tile prefab is single-sided geometry,
///   * obstacles - full-height dark slate blocks that visibly wall agents in,
///   * agents    - capsules in a vivid per-agent hue, with a dark heading
///                 "nose" so the facing direction of the last movement reads,
///   * goals     - flat pads in the owning agent's colour with a bright glowing
///                 cyan core, so a goal is never mistaken for an agent.
/// Collisions reported by Python flash the agent red (NotifyCollision).
///
/// Colour contract (deliberately four distinct families so any frame of the
/// demo is readable): floor = light grey, obstacle = dark slate, agent = vivid
/// per-agent hue (red-orange / blue / violet / amber / green),
/// goal = glowing cyan.
/// </summary>
public class EnvironmentManager : MonoBehaviour
{
    [Header("Prefabs (optional - primitives used if null)")]
    [SerializeField] private GameObject agentPrefab;
    [SerializeField] private GameObject obstaclePrefab;
    [SerializeField] private GameObject goalPrefab;
    [SerializeField] private GameObject groundTilePrefab;

    [Header("Fallback Materials (optional)")]
    [SerializeField] private Material agentMaterial;
    [SerializeField] private Material obstacleMaterial;
    [SerializeField] private Material goalMaterial;
    [SerializeField] private Material groundMaterial;

    [Header("3D World (fallback primitives)")]
    [SerializeField] private float tileHeight = 0.12f;
    [SerializeField] private float tileGap = 0.06f;
    [SerializeField] private float baseSlabThickness = 0.5f;
    [SerializeField] private float obstacleHeight = 1.0f;

    [Header("Palette (keep the four kinds visually distinct)")]
    [SerializeField] private Color tileLightColor = new Color(0.88f, 0.90f, 0.94f);
    [SerializeField] private Color tileDarkColor = new Color(0.62f, 0.67f, 0.75f);
    [SerializeField] private Color baseSlabColor = new Color(0.18f, 0.21f, 0.27f);
    [SerializeField] private Color obstacleColor = new Color(0.16f, 0.18f, 0.22f);
    [SerializeField] private Color goalBeaconColor = new Color(0.10f, 1.00f, 0.85f);
    [SerializeField] private float goalEmission = 1.4f;
    [SerializeField] private Color agentNoseColor = new Color(0.04f, 0.05f, 0.07f);
    [SerializeField] private List<Color> agentPalette = new List<Color>
    {
        new Color(0.96f, 0.30f, 0.22f),   // agent 0: red-orange
        new Color(0.16f, 0.52f, 1.00f),   // agent 1: blue
        new Color(0.88f, 0.42f, 0.96f),   // agent 2: violet
        new Color(1.00f, 0.76f, 0.10f),   // agent 3: amber
        new Color(0.36f, 0.86f, 0.42f),   // agent 4: green
    };

    [Header("Deterministic Demo World (10x10, 3 agents)")]
    [SerializeField] private bool buildDemoWorldOnStart = true;
    [SerializeField] private int demoHeight = 10;
    [SerializeField] private int demoWidth = 10;

    private readonly Dictionary<int, AgentController> spawnedAgents = new();
    private Transform worldRoot;

    private void Start()
    {
        if (buildDemoWorldOnStart)
        {
            BuildDemoWorld();
        }
    }

    /// <summary>
    /// Deterministic demo scenario mirroring main.py: 10x10 grid, three agents,
    /// fixed obstacles and goals. No randomness.
    /// </summary>
    public void BuildDemoWorld()
    {
        ClearWorld();

        worldRoot = new GameObject("WorldRoot").transform;
        worldRoot.SetParent(transform, false);

        var obstacles = new List<Vector2Int>
        {
            new Vector2Int(4, 1), new Vector2Int(4, 2),
            new Vector2Int(4, 7), new Vector2Int(4, 8),
            new Vector2Int(6, 3), new Vector2Int(6, 4), new Vector2Int(6, 5),
        };
        var goals = new List<Vector2Int>
        {
            new Vector2Int(9, 9),  // Agent 0
            new Vector2Int(0, 9),  // Agent 1
            new Vector2Int(4, 5),  // Agent 2
        };
        var starts = new List<Vector2Int>
        {
            new Vector2Int(0, 0),  // Agent 0
            new Vector2Int(9, 0),  // Agent 1
            new Vector2Int(0, 5),  // Agent 2
        };

        InitializeGrid(demoHeight, demoWidth, obstacles);

        for (int i = 0; i < starts.Count; i++)
        {
            SpawnAgent(i, starts[i].x, starts[i].y, goals[i].x, goals[i].y);
        }
    }

    /// <summary>
    /// Builds the ground (base slab + grid tiles) and the obstacle blocks.
    /// </summary>
    public void InitializeGrid(int height, int width, List<Vector2Int> obstacles)
    {
        // ClearWorld() nulls worldRoot; recreate it so a scenario rebuild can
        // never orphan tiles/obstacles/goals at scene root (they would then
        // duplicate on the next rebuild, e.g. after an auto-reconnect).
        if (worldRoot == null)
        {
            worldRoot = new GameObject("WorldRoot").transform;
            worldRoot.SetParent(transform, false);
        }

        height = Mathf.Max(1, height);
        width = Mathf.Max(1, width);

        // The slab first: it is the reason the ground can never vanish, since
        // it is a single solid block that does not depend on per-tile logic.
        BuildBaseSlab(height, width);

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                SpawnTile(r, c);
            }
        }

        if (obstacles == null) return;
        foreach (var obs in obstacles)
        {
            SpawnObstacle(obs.x, obs.y);
        }
    }

    /// <summary>
    /// Spawns one visual agent and its goal marker.
    /// </summary>
    public void SpawnAgent(int id, int startRow, int startCol, int goalRow, int goalCol)
    {
        if (worldRoot == null)
        {
            worldRoot = new GameObject("WorldRoot").transform;
            worldRoot.SetParent(transform, false);
        }

        Color agentColor = ColorForAgent(id);

        Vector3 spawnPos = CoordinateConverter.GridToWorld(startRow, startCol);
        GameObject agentObj = InstantiateAgent(spawnPos);
        agentObj.name = $"Agent_{id}";
        EnsureRenderer(agentObj);

        AgentController controller = agentObj.GetComponent<AgentController>();
        controller.agentId = id;
        if (agentMaterial != null)
        {
            SetMaterial(agentObj, agentMaterial);
        }
        else
        {
            Tint(agentObj, agentColor);
        }
        controller.SetBaseColor(agentColor);
        spawnedAgents[id] = controller;

        SpawnGoal(id, goalRow, goalCol, agentColor);
    }

    /// <summary>
    /// Visual feedback for a collision Python reported for this agent
    /// (the simulation itself is never changed here - only the look).
    /// </summary>
    public void NotifyCollision(int id)
    {
        if (spawnedAgents.TryGetValue(id, out AgentController controller))
        {
            controller.Flash();
        }
    }

    /// <summary>Deterministic colour per agent id (cycles after the palette).</summary>
    public Color ColorForAgent(int id)
    {
        if (agentPalette == null || agentPalette.Count == 0) return Color.white;
        return agentPalette[((id % agentPalette.Count) + agentPalette.Count) % agentPalette.Count];
    }

    /// <summary>
    /// Pushes a logical position update from Python into the visual agent.
    /// </summary>
    public void UpdateAgentPosition(int id, int row, int col)
    {
        if (spawnedAgents.TryGetValue(id, out AgentController controller))
        {
            controller.SetLogicalPosition(row, col);
        }
    }

    /// <summary>
    /// Removes all world objects so a new episode/layout can be built.
    /// </summary>
    public void ClearWorld()
    {
        spawnedAgents.Clear();
        if (worldRoot != null)
        {
            Destroy(worldRoot.gameObject);
            worldRoot = null;
        }
    }

    // ------------------------------------------------------------------
    // World pieces
    // ------------------------------------------------------------------

    /// <summary>
    /// One opaque slab spanning the whole grid, sitting just under the tiles.
    /// It guarantees a visible, solid ground even if a tile prefab is
    /// single-sided geometry or a tile is ever skipped.
    /// </summary>
    private void BuildBaseSlab(int height, int width)
    {
        float spanX = width * CoordinateConverter.CellSize + 0.4f;
        float spanZ = height * CoordinateConverter.CellSize + 0.4f;
        float topY = -Mathf.Max(0.02f, tileHeight) * 0.5f;

        Vector3 centre = new Vector3(
            (width - 1) * 0.5f * CoordinateConverter.CellSize,
            topY - baseSlabThickness * 0.5f,
            -(height - 1) * 0.5f * CoordinateConverter.CellSize);

        GameObject slab = Spawn(PrimitiveType.Cube, centre, worldRoot, template =>
        {
            template.transform.localScale = new Vector3(
                spanX, Mathf.Max(0.05f, baseSlabThickness), spanZ);
            DisableCollider(template);
        });
        slab.name = "FloorSlab";
        EnsureRenderer(slab);
        Tint(slab, baseSlabColor);
    }

    /// <summary>
    /// One solid, double-sided floor tile with its top surface at y = 0.
    /// Cubes are used on purpose: a Quad lies flat with its normal facing away
    /// from an overhead camera and is backface-culled, which is how a floor
    /// silently disappears.
    /// </summary>
    private void SpawnTile(int row, int col)
    {
        Vector3 position = CoordinateConverter.GridToWorld(row, col);
        position.y = -Mathf.Max(0.02f, tileHeight) * 0.5f; // solid tile, top at y = 0

        float fill = Mathf.Clamp01(1f - Mathf.Max(0f, tileGap));
        GameObject tile = Spawn(PrimitiveType.Cube, position, worldRoot, template =>
        {
            template.transform.localScale = new Vector3(
                fill, Mathf.Max(0.02f, tileHeight), fill);
            DisableCollider(template);
        });
        tile.name = $"Tile_{row}_{col}";
        EnsureRenderer(tile);

        if (groundMaterial != null)
        {
            SetMaterial(tile, groundMaterial);
        }
        else
        {
            Tint(tile, ((row + col) % 2 == 0) ? tileLightColor : tileDarkColor);
        }
    }

    /// <summary>One full-height obstacle block standing on the floor.</summary>
    private void SpawnObstacle(int row, int col)
    {
        Vector3 position = CoordinateConverter.GridToWorld(row, col);
        position.y = obstacleHeight * 0.5f;

        GameObject block = Spawn(PrimitiveType.Cube, position, worldRoot, template =>
        {
            template.transform.localScale = new Vector3(
                0.98f, Mathf.Max(0.1f, obstacleHeight), 0.98f);
        });
        block.name = $"Obstacle_{row}_{col}";
        EnsureRenderer(block);

        if (obstacleMaterial != null)
        {
            SetMaterial(block, obstacleMaterial);
        }
        else
        {
            Tint(block, obstacleColor);
        }
    }

    /// <summary>
    /// Goal marker: a flat pad in the owning agent's colour with a smaller
    /// glowing beacon on top. The pad keeps the agent/goal pairing readable
    /// while the shared cyan core makes "goal" its own colour family.
    /// </summary>
    private void SpawnGoal(int id, int goalRow, int goalCol, Color agentColor)
    {
        Vector3 padPosition = CoordinateConverter.GridToWorld(goalRow, goalCol);
        padPosition.y = 0.03f; // flat pad resting on the floor

        GameObject goal = new GameObject($"Goal_{id}");
        goal.transform.SetParent(worldRoot, false);
        goal.transform.position = padPosition;

        GameObject pad = Spawn(PrimitiveType.Cylinder, padPosition, goal.transform, template =>
        {
            template.transform.localScale = new Vector3(0.86f, 0.02f, 0.86f);
            DisableCollider(template);
        });
        pad.name = "Pad";
        EnsureRenderer(pad);
        if (goalMaterial != null)
        {
            SetMaterial(pad, goalMaterial);
        }
        else
        {
            Tint(pad, agentColor);
        }

        GameObject core = Spawn(PrimitiveType.Cylinder,
            padPosition + new Vector3(0f, 0.05f, 0f), goal.transform, template =>
        {
            template.transform.localScale = new Vector3(0.42f, 0.05f, 0.42f);
            DisableCollider(template);
        });
        core.name = "Beacon";
        EnsureRenderer(core);
        if (goalMaterial != null)
        {
            SetMaterial(core, goalMaterial);
        }
        else
        {
            TintEmissive(core, goalBeaconColor, goalEmission);
        }
    }

    // ---- Primitive fallbacks (templates are removed, never left behind) ----

    /// <summary>
    /// Instantiates a configured primitive without leaking the template object.
    /// <see cref="GameObject.CreatePrimitive"/> creates a real scene object that
    /// must be cloned and then removed; forgetting that leaves stray active
    /// primitives stacked at the origin on every rebuild.
    /// </summary>
    private static GameObject Spawn(PrimitiveType type, Vector3 position, Transform parent, Action<GameObject> configure)
    {
        GameObject template = GameObject.CreatePrimitive(type);
        template.name = "GridMindTemplate";
        if (configure != null) configure(template);

        GameObject instance = Instantiate(template, position, Quaternion.identity, parent);

        template.SetActive(false); // hide before the deferred Destroy so it never renders
        Destroy(template);
        return instance;
    }

    private GameObject InstantiateAgent(Vector3 position)
    {
        if (agentPrefab != null)
        {
            GameObject prefabInstance = Instantiate(agentPrefab, position, Quaternion.identity, worldRoot);
            if (prefabInstance.GetComponent<AgentController>() == null)
            {
                prefabInstance.AddComponent<AgentController>();
            }
            return prefabInstance;
        }

        return Spawn(PrimitiveType.Capsule, position, worldRoot, template =>
        {
            template.transform.localScale = new Vector3(0.62f, 0.5f, 0.62f); // 1.0 tall, feet at y = 0
            DisableCollider(template);

            // Heading "nose": rotates with the capsule so the last movement
            // direction stays readable in 3D. A child of the template, so the
            // clone inherits it and the template's copy dies with the template.
            GameObject nose = GameObject.CreatePrimitive(PrimitiveType.Cube);
            nose.name = "Nose";
            nose.transform.localScale = new Vector3(0.16f, 0.16f, 0.34f);
            nose.transform.localPosition = new Vector3(0f, 0.1f, 0.34f);
            DisableCollider(nose);
            Tint(nose, agentNoseColor);
            nose.transform.SetParent(template.transform, false);

            template.AddComponent<AgentController>();
        });
    }

    // ---- Rendering helpers ----

    /// <summary>
    /// Guarantees the object actually draws. A prefab can arrive from the
    /// Inspector with its Renderer disabled, which is one way a floor silently
    /// disappears while everything standing on it still shows.
    /// </summary>
    private static void EnsureRenderer(GameObject obj)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer == null)
        {
            Debug.LogWarning($"[GridMind] '{obj.name}' has no Renderer and will be invisible.");
            return;
        }
        renderer.enabled = true;
    }

    private static void DisableCollider(GameObject obj)
    {
        Collider collider = obj.GetComponent<Collider>();
        if (collider != null) collider.enabled = false;
    }

    /// <summary>Assigns a shared material without creating a per-object instance.</summary>
    private static void SetMaterial(GameObject obj, Material material)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer != null) renderer.sharedMaterial = material;
    }

    /// <summary>Tints the renderer of a fallback primitive (instance material).</summary>
    private static void Tint(GameObject obj, Color color)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer != null) renderer.material.color = color;
    }

    /// <summary>Tints a fallback primitive and makes it glow (Standard shader).</summary>
    private static void TintEmissive(GameObject obj, Color color, float emission)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer == null) return;

        Material material = renderer.material;
        material.color = color;

        if (material.HasProperty("_EmissionColor"))
        {
            material.EnableKeyword("_EMISSION");
            material.SetColor("_EmissionColor", color * Mathf.Max(0f, emission));
            material.globalIlluminationFlags = MaterialGlobalIlluminationFlags.RealtimeEmissive;
        }
    }
}
