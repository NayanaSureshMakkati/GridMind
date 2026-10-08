using UnityEngine;

/// <summary>
/// Isometric / elevated third-person camera for the GridMind demo
/// (Developer 4, Phase 6).
///
/// * frames the whole grid automatically from the scenario size,
/// * keeps the entire environment visible from an elevated angle,
/// * optional follow mode for one agent (keys 1-9 select, F toggles),
/// * right-drag = orbit, mouse wheel = zoom, R = reframe.
///
/// The camera only observes: agent positions come from Python through
/// <see cref="CommunicationManager"/> and are never influenced here.
/// </summary>
public class CameraController : MonoBehaviour
{
    [Header("View")]
    [SerializeField] private float elevationDegrees = 55f;
    [SerializeField] private float azimuthDegrees = 35f;
    [SerializeField] private float distanceFactor = 1.65f;
    [SerializeField] private float minDistance = 6f;
    [SerializeField] private float maxDistance = 60f;
    [SerializeField] private float orbitSpeed = 120f;
    [SerializeField] private float zoomSpeed = 6f;
    [SerializeField] private float followLerp = 6f;

    [Header("Selection")]
    [SerializeField] private bool enableFollow = true;
    [SerializeField] private int maxSelectableAgents = 5;

    private Vector3 pivot = new Vector3(4.5f, 0f, -4.5f);
    private float distance = 12f;
    private int selectedAgent = -1;
    private bool following;
    private Transform followTarget;

    private int gridHeight = 10;
    private int gridWidth = 10;
    private int agentCount = 3;

    private void Start()
    {
        FrameGrid(gridHeight, gridWidth, agentCount);
    }

    /// <summary>
    /// Centers and zooms the camera so a height x width grid stays fully visible.
    /// Called by <see cref="CommunicationManager"/> whenever a scenario arrives.
    /// </summary>
    public void FrameGrid(int height, int width, int agents)
    {
        gridHeight = Mathf.Max(1, height);
        gridWidth = Mathf.Max(1, width);
        agentCount = Mathf.Max(0, agents);

        float centreX = (gridWidth - 1) * 0.5f * CoordinateConverter.CellSize;
        float centreZ = -(gridHeight - 1) * 0.5f * CoordinateConverter.CellSize;
        pivot = new Vector3(centreX, 0f, centreZ);

        float span = Mathf.Max(gridHeight, gridWidth) * CoordinateConverter.CellSize;
        distance = Mathf.Clamp(span * distanceFactor, minDistance, maxDistance);

        selectedAgent = agentCount > 0 ? 0 : -1;
        following = false;
        followTarget = null;
        ApplyTransform();
    }

    private void Update()
    {
        HandleInput();
        if (following && followTarget != null)
        {
            pivot = Vector3.Lerp(pivot, followTarget.position, followLerp * Time.deltaTime);
        }
        ApplyTransform();
    }

    private void HandleInput()
    {
        // right-drag to orbit (left button stays free for the UI panel)
        if (Input.GetMouseButton(1))
        {
            azimuthDegrees += Input.GetAxis("Mouse X") * orbitSpeed * Time.deltaTime;
            elevationDegrees -= Input.GetAxis("Mouse Y") * orbitSpeed * Time.deltaTime;
            elevationDegrees = Mathf.Clamp(elevationDegrees, 15f, 85f);
        }

        // wheel to zoom
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.0001f)
        {
            distance = Mathf.Clamp(distance - scroll * zoomSpeed, minDistance, maxDistance);
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            FrameGrid(gridHeight, gridWidth, agentCount);
        }
        if (enableFollow && Input.GetKeyDown(KeyCode.F))
        {
            ToggleFollow(selectedAgent);
        }
        for (int index = 0; index < maxSelectableAgents; index++)
        {
            if (Input.GetKeyDown((KeyCode)((int)KeyCode.Alpha1 + index)))
            {
                ToggleFollow(index);
            }
        }
    }

    /// <summary>Follows the given agent id (id &lt; 0 or no follow support stops it).</summary>
    public void ToggleFollow(int agentId)
    {
        if (!enableFollow || agentId < 0 || agentId >= agentCount)
        {
            following = false;
            followTarget = null;
            return;
        }
        // EnvironmentManager names its visual agents "Agent_<id>".
        GameObject agentObject = GameObject.Find("Agent_" + agentId);
        if (agentObject == null)
        {
            following = false;
            followTarget = null;
            return;
        }
        followTarget = agentObject.transform;
        selectedAgent = agentId;
        following = true;
    }

    /// <summary>Current follow target id, or -1 when the camera is free.</summary>
    public int FollowedAgentId
    {
        get { return following ? selectedAgent : -1; }
    }

    private void ApplyTransform()
    {
        Quaternion rotation = Quaternion.Euler(elevationDegrees, azimuthDegrees, 0f);
        Vector3 offset = rotation * Vector3.back;
        transform.position = pivot + offset * distance;
        transform.rotation = Quaternion.LookRotation(pivot - transform.position, Vector3.up);
    }
}
