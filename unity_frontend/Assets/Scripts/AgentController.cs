using UnityEngine;

/// <summary>
/// Visual behaviour of one agent (Developer 1 Phase 15 + 3D polish).
///
/// Interpolates towards the authoritative logical position streamed by Python,
/// turns a heading "nose" towards the direction of the last movement, and
/// flashes red for a moment when Python reports a collision for this agent.
/// It never decides anything: positions, actions and collisions all come from
/// the backend (Master Context 47).
/// </summary>
public class AgentController : MonoBehaviour
{
    public int agentId;

    [SerializeField] private float moveSpeed = 6.0f;
    [SerializeField] private float turnSpeed = 720.0f;
    [SerializeField] private float flashDuration = 0.35f;
    [SerializeField] private Color flashColor = new Color(1f, 0.15f, 0.1f);

    private Vector3 targetWorldPosition;
    private Quaternion targetRotation;
    private Renderer agentRenderer;
    private Color baseColor = Color.white;
    private float flashTimer;

    private void Awake()
    {
        targetWorldPosition = transform.position;
        targetRotation = transform.rotation;
        agentRenderer = GetComponentInChildren<Renderer>();
        if (agentRenderer != null) baseColor = agentRenderer.material.color;
    }

    /// <summary>
    /// Receives authoritative logical updates from Python, sets the movement
    /// target and turns to face the direction of travel.
    /// </summary>
    public void SetLogicalPosition(int row, int col)
    {
        Vector3 next = CoordinateConverter.GridToWorld(row, col);
        Vector3 delta = next - targetWorldPosition;
        targetWorldPosition = next;

        if (delta.sqrMagnitude > 0.0001f)
        {
            // Yaw only: keep the capsule upright, aim the nose along the move.
            float yaw = Mathf.Atan2(delta.x, delta.z) * Mathf.Rad2Deg;
            targetRotation = Quaternion.Euler(0f, yaw, 0f);
        }
    }

    /// <summary>
    /// Colour the agent returns to after a collision flash (its palette colour).
    /// </summary>
    public void SetBaseColor(Color color)
    {
        baseColor = color;
        if (agentRenderer != null && flashTimer <= 0f)
        {
            agentRenderer.material.color = color;
        }
    }

    /// <summary>Flashes the collision colour for a short moment.</summary>
    public void Flash()
    {
        flashTimer = flashDuration;
        if (agentRenderer != null) agentRenderer.material.color = flashColor;
    }

    private void Update()
    {
        if (Vector3.Distance(transform.position, targetWorldPosition) > 0.001f)
        {
            transform.position = Vector3.MoveTowards(
                transform.position,
                targetWorldPosition,
                moveSpeed * Time.deltaTime
            );
        }

        if (transform.rotation != targetRotation)
        {
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation,
                targetRotation,
                turnSpeed * Time.deltaTime
            );
        }

        if (flashTimer > 0f)
        {
            flashTimer -= Time.deltaTime;
            if (flashTimer <= 0f && agentRenderer != null)
            {
                agentRenderer.material.color = baseColor;
            }
        }
    }
}