using System;
using System.Collections.Generic;
using UnityEngine;

// ---------------------------------------------------------------------------
// Wire-format data classes (JsonUtility-compatible: public fields, 1-D arrays)
// Field names match the JSON keys exactly (see docs/communication_protocol.md).
// ---------------------------------------------------------------------------

[Serializable]
public class ObstacleEntry
{
    public int row;
    public int col;
}

[Serializable]
public class ScenarioAgent
{
    public int id;
    public int[] start;
    public int[] goal;
}

[Serializable]
public class ScenarioMessage
{
    public string type;
    public string name;
    public int height;
    public int width;
    public int max_steps;
    public ObstacleEntry[] obstacles;
    public ScenarioAgent[] agents;
}

[Serializable]
public class AgentState
{
    public int id;
    public int[] position;
    public int[] goal;
    public int[] observation;
    public string status;
    public bool done;
    public float reward;
}

[Serializable]
public class SessionStats
{
    public string algorithm_label;
    public int episode;
    public int episode_limit;
    public int step;
    public int max_steps;
    public int agents;
    public int collisions;
    public int episode_collisions;
    public int deadlocks;
    public int episode_deadlocks;
    public int messages;
    public int episode_messages;
    public int distance;
    public int episodes_completed;
    public int successes;
    public float success_rate;
    public float total_reward;
    public bool trained;
    public string action_source;
}

[Serializable]
public class StateMessage
{
    public string type;
    public int episode;
    public int step;
    public string algorithm;
    public string action_source;
    public int max_steps;
    public AgentState[] agents;
    public SessionStats stats;
}

[Serializable]
public class WelcomeMessage
{
    public string type;
    public int protocol;
    public string algorithm;
    public string algorithm_label;
    public string scenario;
    public int num_agents;
    public int speed_ms;
    public bool trained;
    public int episode_limit;
    public string action_source;
}

[Serializable]
public class MovementEntry
{
    public int agent_id;
    public int action;
    public string action_name;
    public int[] from;
    public int[] to;
    public int[] target;
    public bool moved;
    public float reward;
}

[Serializable]
public class CollisionEntryData
{
    public int agent_id;
    public string type;
    public int[] position;
    public int[] target;
    public int other;
}

[Serializable]
public class StepResultMessage
{
    public string type;
    public int step;
    public int[] goals_reached;
    public int deadlocks;
    public bool terminated;
    public bool truncated;
    public MovementEntry[] movements;
    public CollisionEntryData[] collisions;
}

[Serializable]
public class EpisodeEndMessage
{
    public string type;
    public int episode;
    public int steps;
    public bool terminated;
    public bool truncated;
    public bool success;
    public int collisions;
    public int deadlocks;
    public int messages;
    public float goal_completion_rate;
}

[Serializable]
public class ErrorMessage
{
    public string type;
    public string code;
    public string detail;
}

[Serializable]
public class AckMessage
{
    public string type;
    public string command;
    public bool ok;
}

/// <summary>
/// Turns received protocol messages into scene and UI updates
/// (Developer 4, Phases 3-7).
///
/// This class is the ONLY place that interprets the protocol on the Unity side.
/// It never simulates: positions, rewards, collisions and termination all come
/// from Python, and the environment's own conflict resolution is never
/// re-implemented here.
/// </summary>
public class CommunicationManager : MonoBehaviour
{
    [Header("References (auto-found when left empty)")]
    [SerializeField] private NetworkManager networkManager;
    [SerializeField] private EnvironmentManager environmentManager;
    [SerializeField] private UIManager uiManager;
    [SerializeField] private EventLogUI eventLog;

    [Header("Behaviour")]
    [SerializeField] private bool autoStartOnConnect = false;
    [SerializeField] private int speedMs = 250;

    /// <summary>Latest scenario received from Python (null before the first one).</summary>
    public ScenarioMessage CurrentScenario { get; private set; }

    /// <summary>Latest state received from Python.</summary>
    public StateMessage CurrentState { get; private set; }

    /// <summary>Number of messages received so far (diagnostics).</summary>
    public int MessagesReceived { get; private set; }

    /// <summary>Last server error, shown in the UI when present.</summary>
    public string LastError { get; private set; }

    private void Awake()
    {
        if (networkManager == null) networkManager = FindObjectOfType<NetworkManager>();
        if (environmentManager == null) environmentManager = FindObjectOfType<EnvironmentManager>();
        if (uiManager == null) uiManager = FindObjectOfType<UIManager>();
        if (eventLog == null) eventLog = FindObjectOfType<EventLogUI>();
        if (eventLog == null)
        {
            // Self-bootstrapping panel: the scene needs no manual wiring and the
            // log exists the moment Play is pressed (project convention).
            eventLog = new GameObject("EventLogUI").AddComponent<EventLogUI>();
        }
    }

    private void Start()
    {
        if (networkManager == null)
        {
            Debug.LogError("[GridMind] CommunicationManager needs a NetworkManager in the scene.");
            return;
        }
        networkManager.OnRawMessage += HandleRawMessage;
        networkManager.OnConnected += HandleConnected;
        networkManager.OnDisconnected += HandleDisconnected;
    }

    private void OnDestroy()
    {
        if (networkManager == null) return;
        networkManager.OnRawMessage -= HandleRawMessage;
        networkManager.OnConnected -= HandleConnected;
        networkManager.OnDisconnected -= HandleDisconnected;
    }

    // ------------------------------------------------------------------
    // Connection events
    // ------------------------------------------------------------------

    private void HandleConnected()
    {
        if (uiManager != null) uiManager.SetConnectionStatus(true, networkManager.Host, networkManager.Port);
        if (autoStartOnConnect)
        {
            networkManager.SendSpeed(speedMs);
            networkManager.SendControl("start");
        }
    }

    private void HandleDisconnected()
    {
        if (uiManager != null) uiManager.SetConnectionStatus(false, networkManager.Host, networkManager.Port);
    }

    // ------------------------------------------------------------------
    // Message dispatch
    // ------------------------------------------------------------------

    private void HandleRawMessage(string line)
    {
        string type = ExtractType(line);
        MessagesReceived++;

        switch (type)
        {
            case "welcome":
                HandleWelcome(JsonUtility.FromJson<WelcomeMessage>(line));
                break;
            case "scenario":
                HandleScenario(JsonUtility.FromJson<ScenarioMessage>(line));
                break;
            case "state":
                HandleState(JsonUtility.FromJson<StateMessage>(line));
                break;
            case "step_result":
                HandleStepResult(JsonUtility.FromJson<StepResultMessage>(line));
                break;
            case "episode_end":
                HandleEpisodeEnd(JsonUtility.FromJson<EpisodeEndMessage>(line));
                break;
            case "error":
                HandleError(JsonUtility.FromJson<ErrorMessage>(line));
                break;
            case "ack":
                // acknowledgements need no visual reaction (logged for debugging)
                break;
            case "pong":
                break;
            default:
                Debug.LogWarning("[GridMind] Unknown message type '" + type + "': " + line);
                break;
        }
    }

    /// <summary>
    /// Reads "type" without a full parse. JsonUtility cannot deserialize into a
    /// dynamic object, so a tiny scan is used to route messages.
    /// </summary>
    private static string ExtractType(string json)
    {
        const string key = "\"type\":\"";
        int start = json.IndexOf(key, StringComparison.Ordinal);
        if (start < 0) return string.Empty;
        start += key.Length;
        int end = json.IndexOf('"', start);
        return end > start ? json.Substring(start, end - start) : string.Empty;
    }

    // ------------------------------------------------------------------
    // Handlers
    // ------------------------------------------------------------------

    private void HandleWelcome(WelcomeMessage welcome)
    {
        if (welcome == null) return;
        if (welcome.speed_ms > 0) speedMs = welcome.speed_ms;
        if (uiManager != null)
        {
            uiManager.SetSessionInfo(
                welcome.algorithm_label != null ? welcome.algorithm_label : welcome.algorithm,
                welcome.scenario, welcome.num_agents, welcome.trained, welcome.episode_limit);
            uiManager.SetSpeedMs(speedMs);
        }
        if (eventLog != null)
        {
            eventLog.LogInfo("connected: " + welcome.algorithm_label + " | " + welcome.scenario
                + " | agents=" + welcome.num_agents
                + (welcome.trained ? "" : " | (untrained)"));
        }
        if (networkManager != null) networkManager.SendSpeed(speedMs);
    }

    private void HandleScenario(ScenarioMessage scenario)
    {
        if (scenario == null || environmentManager == null) return;
        CurrentScenario = scenario;

        if (eventLog != null)
        {
            eventLog.Clear();
            eventLog.LogInfo("scenario " + scenario.name + "  " + scenario.width + "x" + scenario.height
                + "  agents=" + (scenario.agents != null ? scenario.agents.Length : 0)
                + "  obstacles=" + (scenario.obstacles != null ? scenario.obstacles.Length : 0)
                + "  max_steps=" + scenario.max_steps);
        }

        // Python is the source of truth for the world: rebuild it from the
        // scenario message so the 3D layout can never drift from the logical one.
        environmentManager.ClearWorld();
        List<Vector2Int> obstacles = new List<Vector2Int>();
        if (scenario.obstacles != null)
        {
            foreach (ObstacleEntry entry in scenario.obstacles)
            {
                obstacles.Add(new Vector2Int(entry.row, entry.col));
            }
        }
        environmentManager.InitializeGrid(scenario.height, scenario.width, obstacles);

        if (scenario.agents != null)
        {
            foreach (ScenarioAgent agent in scenario.agents)
            {
                environmentManager.SpawnAgent(
                    agent.id, agent.start[0], agent.start[1], agent.goal[0], agent.goal[1]);
            }
        }

        CameraController camera = FindObjectOfType<CameraController>();
        if (camera != null)
        {
            camera.FrameGrid(scenario.height, scenario.width, scenario.agents != null ? scenario.agents.Length : 0);
        }
        if (uiManager != null) uiManager.SetScenarioInfo(scenario.name, scenario.height, scenario.width, scenario.max_steps);
    }

    private void HandleState(StateMessage state)
    {
        if (state == null) return;
        CurrentState = state;

        if (environmentManager != null && state.agents != null)
        {
            foreach (AgentState agent in state.agents)
            {
                if (agent.position == null || agent.position.Length < 2) continue;
                environmentManager.UpdateAgentPosition(agent.id, agent.position[0], agent.position[1]);
            }
        }
        if (uiManager != null) uiManager.UpdateState(state);
    }

    private void HandleStepResult(StepResultMessage result)
    {
        if (result == null) return;
        if (environmentManager != null && result.collisions != null)
        {
            foreach (CollisionEntryData collision in result.collisions)
            {
                environmentManager.NotifyCollision(collision.agent_id);
            }
        }
        if (uiManager != null) uiManager.ShowStepResult(result);
        if (eventLog != null) eventLog.LogStep(result);
    }

    private void HandleEpisodeEnd(EpisodeEndMessage end)
    {
        if (end == null) return;
        if (uiManager != null) uiManager.ShowEpisodeEnd(end);
        if (eventLog != null) eventLog.LogEpisodeEnd(end);
    }

    private void HandleError(ErrorMessage error)
    {
        if (error == null) return;
        LastError = error.code + ": " + error.detail;
        Debug.LogError("[GridMind] server error " + LastError);
        if (uiManager != null) uiManager.ShowError(LastError);
        if (eventLog != null) eventLog.LogInfo("SERVER ERROR: " + LastError);
    }

    // ------------------------------------------------------------------
    // Controls (wired to UI buttons)
    // ------------------------------------------------------------------

    /// <summary>Sends a control command to the Python server.</summary>
    public void SendControl(string command)
    {
        if (networkManager == null) return;
        networkManager.SendControl(command);
    }

    /// <summary>Changes the streaming pace.</summary>
    public void SetSpeed(int millisecondsPerStep)
    {
        speedMs = Mathf.Max(0, millisecondsPerStep);
        if (networkManager != null) networkManager.SendSpeed(speedMs);
        if (uiManager != null) uiManager.SetSpeedMs(speedMs);
    }
}
