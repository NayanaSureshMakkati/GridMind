using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Scrollable event log panel (IMGUI, no prefab wiring - project convention).
///
/// Reads everything Python streams about the running simulation, one line per
/// event, so all agent movements, obstacle/agent collisions, goal reaches,
/// deadlocks and episode summaries can be inspected live:
///
///     s 12  A0 RIGHT (2,3)-(2,4)  r=-1.0
///     s 13  A1 OBSTACLE target(4,4) r=-20.0
///     s 14  A0 vs A2 SWAP at (4,5)
///     s 15  A2 GOAL REACHED
///     s 16  DEADLOCK detected
///     EP1 SUCCESS steps=54 collisions=2 deadlocks=0
///
/// Toggle with L, clear with the button, auto-scroll pins to the newest line.
/// The log only renders data received from Python; it never infers state.
/// </summary>
public class EventLogUI : MonoBehaviour
{
    private struct LogEntry
    {
        public string text;
        public Color color;

        public LogEntry(string text, Color color)
        {
            this.text = text;
            this.color = color;
        }
    }

    [Header("Layout")]
    [SerializeField] private bool showPanel = true;
    [SerializeField] private KeyCode toggleKey = KeyCode.L;
    [SerializeField] private int panelWidth = 480;
    [SerializeField] private float panelHeightFraction = 0.62f;
    [SerializeField] private float uiScale = 1.0f;
    [SerializeField] private int maxEntries = 500;
    [SerializeField] private bool autoScroll = true;

    private readonly List<LogEntry> entries = new List<LogEntry>();
    private Vector2 scrollPosition;
    private GUIStyle entryStyle;
    private GUIStyle headerStyle;
    private Texture2D panelTexture;

    // Event colours (readable on the dark panel)
    private static readonly Color MoveColor = new Color(0.88f, 0.90f, 0.94f);
    private static readonly Color StayColor = new Color(0.62f, 0.65f, 0.70f);
    private static readonly Color StaticCollisionColor = new Color(1f, 0.78f, 0.25f); // OBSTACLE / BOUNDARY
    private static readonly Color AgentCollisionColor = new Color(1f, 0.42f, 0.35f);  // SAME_CELL / SWAP
    private static readonly Color GoalColor = new Color(0.45f, 1f, 0.55f);
    private static readonly Color DeadlockColor = new Color(1f, 0.58f, 0.10f);
    private static readonly Color EpisodeColor = new Color(0.55f, 0.88f, 1f);

    private void Awake()
    {
        panelTexture = new Texture2D(1, 1);
        panelTexture.SetPixel(0, 0, new Color(0.04f, 0.05f, 0.08f, 0.84f));
        panelTexture.Apply();
    }

    private void Update()
    {
        if (Input.GetKeyDown(toggleKey)) showPanel = !showPanel;
    }

    // ------------------------------------------------------------------
    // Data pushed by CommunicationManager
    // ------------------------------------------------------------------

    /// <summary>Removes every line (new scenario = fresh world).</summary>
    public void Clear()
    {
        entries.Clear();
    }

    /// <summary>Neutral information line (scenario, connection, errors).</summary>
    public void LogInfo(string message)
    {
        Add(new LogEntry(message, MoveColor));
    }

    /// <summary>One line per agent movement, collision, goal and deadlock.</summary>
    public void LogStep(StepResultMessage result)
    {
        if (result == null) return;
        int step = result.step;

        if (result.movements != null)
        {
            foreach (MovementEntry move in result.movements)
            {
                if (move == null || move.from == null || move.to == null) continue;
                string actionName = move.action_name ?? "?";
                string line = "s" + step.ToString().PadLeft(3) + "  A" + move.agent_id + " "
                    + actionName.PadRight(6)
                    + " (" + move.from[0] + "," + move.from[1] + ")-(" + move.to[0] + "," + move.to[1] + ")";
                if (!move.moved && actionName != "STAY" && move.target != null)
                {
                    line += " blocked, target(" + move.target[0] + "," + move.target[1] + ")";
                }
                line += "  r=" + move.reward.ToString("0.0");

                Add(new LogEntry(line, move.moved ? MoveColor : StayColor));
            }
        }

        if (result.collisions != null)
        {
            foreach (CollisionEntryData collision in result.collisions)
            {
                if (collision == null) continue;
                bool agentVsAgent = collision.type == "SAME_CELL" || collision.type == "SWAP";
                string line = "s" + step.ToString().PadLeft(3) + "  A" + collision.agent_id
                    + " COLLISION " + collision.type;
                if (collision.position != null)
                {
                    line += " at (" + collision.position[0] + "," + collision.position[1] + ")";
                }
                if (collision.target != null && collision.target[0] != (collision.position != null ? collision.position[0] : -1))
                {
                    line += " target(" + collision.target[0] + "," + collision.target[1] + ")";
                }
                if (collision.other >= 0)
                {
                    line += " vs A" + collision.other;
                }
                Add(new LogEntry(line, agentVsAgent ? AgentCollisionColor : StaticCollisionColor));
            }
        }

        if (result.goals_reached != null)
        {
            foreach (int agentId in result.goals_reached)
            {
                Add(new LogEntry("s" + step.ToString().PadLeft(3) + "  A" + agentId + " GOAL REACHED", GoalColor));
            }
        }

        if (result.deadlocks > 0)
        {
            Add(new LogEntry("s" + step.ToString().PadLeft(3) + "  DEADLOCK detected", DeadlockColor));
        }
    }

    /// <summary>End-of-episode summary line.</summary>
    public void LogEpisodeEnd(EpisodeEndMessage end)
    {
        if (end == null) return;
        string outcome = end.success ? "SUCCESS" : (end.truncated ? "TIME LIMIT" : "INCOMPLETE");
        Add(new LogEntry("EP" + end.episode + " " + outcome
            + " steps=" + end.steps
            + " collisions=" + end.collisions
            + " deadlocks=" + end.deadlocks
            + " goals=" + end.goal_completion_rate.ToString("0.00"), EpisodeColor));
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    private void Add(LogEntry entry)
    {
        entries.Add(entry);
        if (entries.Count > maxEntries)
        {
            entries.RemoveRange(0, entries.Count - maxEntries);
        }
    }

    private void EnsureStyles()
    {
        if (entryStyle != null) return;
        entryStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(13 * uiScale),
            normal = { textColor = Color.white }
        };
        headerStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(14 * uiScale),
            fontStyle = FontStyle.Bold,
            normal = { textColor = new Color(0.70f, 0.85f, 1f) }
        };
    }

    private void OnGUI()
    {
        if (!showPanel) return;
        EnsureStyles();

        float panelHeight = Screen.height * panelHeightFraction;
        Rect area = new Rect(Screen.width - panelWidth - 12, 12, panelWidth, panelHeight);
        GUILayout.BeginArea(area, GUI.skin.box);
        GUI.DrawTexture(new Rect(0, 0, panelWidth, panelHeight), panelTexture, ScaleMode.StretchToFill);

        GUILayout.BeginHorizontal();
        GUILayout.Label("EVENT LOG  (toggle: " + toggleKey + ")", headerStyle);
        autoScroll = GUILayout.Toggle(autoScroll, " auto-scroll", entryStyle);
        if (GUILayout.Button("Clear", GUILayout.Width(60))) entries.Clear();
        GUILayout.EndHorizontal();

        scrollPosition = GUILayout.BeginScrollView(scrollPosition);
        for (int i = 0; i < entries.Count; i++)
        {
            Color previous = GUI.color;
            GUI.color = entries[i].color;
            GUILayout.Label(entries[i].text, entryStyle);
            GUI.color = previous;
        }
        GUILayout.EndScrollView();

        if (autoScroll)
        {
            scrollPosition = new Vector2(0f, float.MaxValue);
        }

        GUILayout.EndArea();
    }
}
