using System.Text;
using UnityEngine;

/// <summary>
/// Final demonstration UI (Developer 4, Phase 7/14).
///
/// Shows, from live Python messages only:
///     GRIDMIND
///     Algorithm: Coordinated DQN      Agents: 3
///     Episode: 1 / 3                  Step: 54 / 60
///     Agent 0: GOAL REACHED
///     Agent 1: NAVIGATING
///     Agent 2: NAVIGATING
///     Collisions: 0     Deadlocks: 0     Reward: -14.00
/// plus the run controls: Start, Pause, Resume, Step, Reset, Exit.
///
/// Implemented with IMGUI (OnGUI) so the scene needs no prefab wiring and the
/// panel exists in the moment you press Play — consistent with the project's
/// "primitives, no hand-built assets" convention.
/// </summary>
public class UIManager : MonoBehaviour
{
    [Header("References (auto-found when left empty)")]
    [SerializeField] private CommunicationManager communicationManager;

    [Header("Layout")]
    [SerializeField] private bool showPanel = true;
    [SerializeField] private int panelWidth = 380;
    [SerializeField] private float uiScale = 1.15f;
    [SerializeField] private bool showObservationToggle = true;
    [SerializeField] private bool showObservations = false;

    private bool connected;
    private string host = "127.0.0.1";
    private int port = 8765;

    private string algorithm = "-";
    private string scenarioName = "-";
    private int scenarioAgents;
    private int gridHeight;
    private int gridWidth;
    private int maxSteps;
    private bool trained;
    private int episodeLimit;

    private StateMessage lastState;
    private StepResultMessage lastStepResult;
    private int speedMs = 250;
    private string episodeSummary = string.Empty;
    private string errorText = string.Empty;
    private GUIStyle titleStyle;
    private GUIStyle headerStyle;
    private GUIStyle labelStyle;
    private Texture2D panelTexture;

    private void Awake()
    {
        if (communicationManager == null) communicationManager = FindObjectOfType<CommunicationManager>();
        panelTexture = new Texture2D(1, 1);
        panelTexture.SetPixel(0, 0, new Color(0.05f, 0.06f, 0.09f, 0.86f));
        panelTexture.Apply();
    }

    // ------------------------------------------------------------------
    // Data pushed by CommunicationManager
    // ------------------------------------------------------------------

    /// <summary>Updates the connection indicator.</summary>
    public void SetConnectionStatus(bool isConnected, string serverHost, int serverPort)
    {
        connected = isConnected;
        host = serverHost;
        port = serverPort;
        if (isConnected) errorText = string.Empty;
    }

    /// <summary>Records session identity sent in the welcome message.</summary>
    public void SetSessionInfo(string algorithmLabel, string scenario, int numAgents, bool isTrained, int episodes)
    {
        algorithm = string.IsNullOrEmpty(algorithmLabel) ? "-" : algorithmLabel;
        scenarioName = string.IsNullOrEmpty(scenario) ? "-" : scenario;
        scenarioAgents = numAgents;
        trained = isTrained;
        episodeLimit = episodes;
    }

    /// <summary>Syncs the panel's speed slider with the server's pace.</summary>
    public void SetSpeedMs(int millisecondsPerStep)
    {
        speedMs = Mathf.Max(0, millisecondsPerStep);
    }

    /// <summary>Records the logical grid description from the scenario message.</summary>
    public void SetScenarioInfo(string name, int height, int width, int steps)
    {
        scenarioName = string.IsNullOrEmpty(name) ? scenarioName : name;
        gridHeight = height;
        gridWidth = width;
        maxSteps = steps;
    }

    /// <summary>Stores the newest authoritative state (rendered every frame).</summary>
    public void UpdateState(StateMessage state)
    {
        lastState = state;
        if (state.stats != null)
        {
            if (state.stats.algorithm_label != null && state.stats.algorithm_label.Length > 0)
            {
                algorithm = state.stats.algorithm_label;
            }
            scenarioAgents = state.stats.agents;
            maxSteps = state.stats.max_steps;
            episodeLimit = state.stats.episode_limit;
        }
    }

    /// <summary>Stores the newest step outcome (per-step rewards/collisions).</summary>
    public void ShowStepResult(StepResultMessage result)
    {
        lastStepResult = result;
    }

    /// <summary>Stores the end-of-episode summary line.</summary>
    public void ShowEpisodeEnd(EpisodeEndMessage end)
    {
        string outcome = end.success ? "SUCCESS" : (end.truncated ? "TIME LIMIT" : "INCOMPLETE");
        episodeSummary = "Episode " + end.episode + " " + outcome +
                         " | steps=" + end.steps +
                         " | goals=" + end.goal_completion_rate.ToString("0.00") +
                         " | collisions=" + end.collisions +
                         " | deadlocks=" + end.deadlocks +
                         " | messages=" + end.messages;
    }

    /// <summary>Displays a server error until the next successful connection.</summary>
    public void ShowError(string message)
    {
        errorText = message;
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    private void EnsureStyles()
    {
        if (titleStyle != null) return;
        float scale = uiScale;
        titleStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(22 * scale),
            fontStyle = FontStyle.Bold,
            normal = { textColor = new Color(0.85f, 0.92f, 1f) }
        };
        headerStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(15 * scale),
            fontStyle = FontStyle.Bold,
            normal = { textColor = new Color(0.70f, 0.85f, 1f) }
        };
        labelStyle = new GUIStyle(GUI.skin.label)
        {
            fontSize = Mathf.RoundToInt(14 * scale),
            normal = { textColor = Color.white },
            richText = false
        };
    }

    private void OnGUI()
    {
        if (!showPanel) return;
        EnsureStyles();

        GUILayout.BeginArea(new Rect(12, 12, panelWidth, Screen.height - 24), GUI.skin.box);
        GUI.DrawTexture(new Rect(0, 0, panelWidth, Screen.height - 24), panelTexture, ScaleMode.StretchToFill);

        GUILayout.Label("GRIDMIND", titleStyle);
        GUILayout.Label(connected ? ("connected to " + host + ":" + port)
                                  : ("OFFLINE - start the Python server (" + host + ":" + port + ")"),
                        headerStyle);
        GUILayout.Space(6);

        GUILayout.Label("Algorithm: " + algorithm + "   |   Agents: " + scenarioAgents, labelStyle);
        GUILayout.Label("Scenario: " + scenarioName + "   |   Grid: " + gridWidth + "x" + gridHeight +
                        (trained ? "" : "   (untrained)"), labelStyle);

        if (lastState != null && lastState.stats != null)
        {
            SessionStats stats = lastState.stats;
            GUILayout.Label("Episode: " + stats.episode + " / " + stats.episode_limit +
                            "    Step: " + stats.step + " / " + stats.max_steps, labelStyle);
            GUILayout.Space(4);

            if (lastState.agents != null)
            {
                foreach (AgentState agent in lastState.agents)
                {
                    string status = agent.status == "GOAL_REACHED" ? "GOAL REACHED" : "NAVIGATING";
                    GUILayout.Label("Agent " + agent.id + ": " + status +
                                    "   reward=" + agent.reward.ToString("0.0"), labelStyle);
                    if (showObservations && agent.observation != null)
                    {
                        GUILayout.Label("    obs: " + Join(agent.observation), labelStyle);
                    }
                }
            }

            GUILayout.Space(6);
            GUILayout.Label("Collisions: " + stats.collisions + "   Deadlocks: " + stats.deadlocks +
                            "   Messages: " + stats.messages, labelStyle);
            GUILayout.Label("Reward (episode): " + stats.total_reward.ToString("0.00") +
                            "   Success rate: " + (stats.success_rate * 100f).ToString("0.0") + "%" +
                            "   Episodes done: " + stats.episodes_completed, labelStyle);
            GUILayout.Label("Action source: " + stats.action_source, labelStyle);
        }
        else
        {
            GUILayout.Label("Waiting for state from Python...", labelStyle);
        }

        if (lastStepResult != null)
        {
            string goals = (lastStepResult.goals_reached == null || lastStepResult.goals_reached.Length == 0)
                ? "-"
                : string.Join(",", System.Array.ConvertAll(lastStepResult.goals_reached, element => element.ToString()));
            GUILayout.Label("Last step: " + lastStepResult.step +
                            " | goals reached: " + goals, labelStyle);
        }

        if (!string.IsNullOrEmpty(episodeSummary))
        {
            GUILayout.Space(4);
            GUILayout.Label(episodeSummary, headerStyle);
        }
        if (!string.IsNullOrEmpty(errorText))
        {
            GUILayout.Space(4);
            GUILayout.Label("SERVER ERROR: " + errorText, headerStyle);
        }

        GUILayout.Space(10);
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Start")) Send("start");
        if (GUILayout.Button("Pause")) Send("pause");
        if (GUILayout.Button("Resume")) Send("resume");
        GUILayout.EndHorizontal();

        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Step")) Send("step");
        if (GUILayout.Button("Reset")) Send("reset");
        if (GUILayout.Button("Exit")) ExitApplication();
        GUILayout.EndHorizontal();

        GUILayout.Space(6);
        GUILayout.Label("Speed: " + speedMs + " ms per step", labelStyle);
        float requested = GUILayout.HorizontalSlider(speedMs, 0f, 1000f);
        // Quantised so a slider drag sends one command per meaningful change.
        int quantised = Mathf.RoundToInt(requested / 10f) * 10;
        if (quantised != speedMs)
        {
            speedMs = quantised;
            if (communicationManager != null) communicationManager.SetSpeed(speedMs);
        }

        if (showObservationToggle)
        {
            showObservations = GUILayout.Toggle(showObservations, " show agent observations (12-v)");
        }

        GUILayout.Space(6);
        GUILayout.Label("Camera: drag = orbit, wheel = zoom, F = follow selected agent, " +
                        "1-5 = select agent, R = reframe", labelStyle);
        GUILayout.EndArea();
    }

    private void Send(string command)
    {
        if (communicationManager != null)
        {
            communicationManager.SendControl(command);
        }
        else
        {
            Debug.LogWarning("[GridMind] No CommunicationManager: cannot send '" + command + "'.");
        }
    }

    private void ExitApplication()
    {
        Send("quit");
        Debug.Log("[GridMind] Exit requested: stopping the Unity client.");
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }

    private static string Join(int[] values)
    {
        StringBuilder builder = new StringBuilder();
        for (int index = 0; index < values.Length; index++)
        {
            if (index > 0) builder.Append(',');
            builder.Append(values[index]);
        }
        return builder.ToString();
    }
}
