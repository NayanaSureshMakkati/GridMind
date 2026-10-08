using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Transport layer for the Python &lt;-&gt; Unity link (Developer 4, Phase 3).
///
/// Responsibilities (and ONLY these):
///   * connect to the Python server (localhost, newline-delimited JSON),
///   * receive messages on a background thread and queue them,
///   * expose the queue to the main thread through <see cref="Dequeue"/>,
///   * send control commands and (optionally) client-supplied actions.
///
/// It contains NO reinforcement-learning logic and never decides actions:
/// Python owns the logical simulation and the policy
/// (see docs/communication_protocol.md). Interpretation of messages belongs to
/// <see cref="CommunicationManager"/>; scene changes belong to
/// <see cref="EnvironmentManager"/>.
/// </summary>
public class NetworkManager : MonoBehaviour
{
    [Header("Server (Python backend)")]
    [SerializeField] private string host = "127.0.0.1";
    [SerializeField] private int port = 8765;
    [SerializeField] private bool connectOnStart = true;
    [SerializeField] private bool autoReconnect = true;
    [SerializeField] private float reconnectDelaySeconds = 2.0f;
    [SerializeField] private bool verboseLogging = true;

    /// <summary>Protocol version this client speaks (must match the server).</summary>
    public const int ProtocolVersion = 1;

    /// <summary>Raised on the main thread when the connection is established.</summary>
    public event Action OnConnected;

    /// <summary>Raised on the main thread when the connection is lost.</summary>
    public event Action OnDisconnected;

    /// <summary>Raised on the main thread for every received message line.</summary>
    public event Action<string> OnRawMessage;

    /// <summary>True while the socket is connected.</summary>
    public bool IsConnected { get { return client != null && client.Connected; } }

    public string Host { get { return host; } }
    public int Port { get { return port; } }

    private readonly Queue<string> incoming = new Queue<string>();
    private readonly object incomingLock = new object();
    private readonly Queue<string> outgoing = new Queue<string>();
    private readonly object outgoingLock = new object();

    private TcpClient client;
    private NetworkStream stream;
    private Thread receiveThread;
    private volatile bool running;
    private float reconnectTimer;
    private bool connectionEventPending;

    private void Start()
    {
        Application.runInBackground = true; // keep receiving while Unity is unfocused
        if (connectOnStart)
        {
            Connect();
        }
    }

    private void Update()
    {
        if (!IsConnected)
        {
            if (autoReconnect && !running)
            {
                reconnectTimer -= Time.deltaTime;
                if (reconnectTimer <= 0f)
                {
                    reconnectTimer = reconnectDelaySeconds;
                    Connect();
                }
            }
            return;
        }

        // main-thread event notification for the UI
        if (connectionEventPending)
        {
            connectionEventPending = false;
            if (OnConnected != null) OnConnected();
        }

        DrainIncoming();
        FlushOutgoing();
    }

    private void OnDestroy()
    {
        Disconnect();
    }

    private void OnApplicationQuit()
    {
        Disconnect();
    }

    // ------------------------------------------------------------------
    // Connection
    // ------------------------------------------------------------------

    /// <summary>Connects to the Python server and sends the hello handshake.</summary>
    public void Connect()
    {
        if (IsConnected) return;
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            client.NoDelay = true;
            stream = client.GetStream();
            running = true;
            connectionEventPending = true;
            reconnectTimer = reconnectDelaySeconds;

            receiveThread = new Thread(ReceiveLoop);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            Send("{\"type\":\"hello\",\"client\":\"unity\",\"protocol\":" + ProtocolVersion + "}");
            Log("connected to " + host + ":" + port);
        }
        catch (Exception exception)
        {
            Log("connect failed: " + exception.Message + " (is the Python server running?)");
            Cleanup();
        }
    }

    /// <summary>Closes the connection (safe to call repeatedly).</summary>
    public void Disconnect()
    {
        running = false;
        Cleanup();
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(200);
        }
        receiveThread = null;
        if (OnDisconnected != null) OnDisconnected();
    }

    private void Cleanup()
    {
        try { if (stream != null) stream.Close(); } catch (Exception) { }
        try { if (client != null) client.Close(); } catch (Exception) { }
        stream = null;
        client = null;
    }

    private void ReceiveLoop()
    {
        byte[] buffer = new byte[1 << 16];
        StringBuilder pending = new StringBuilder();

        while (running)
        {
            int read;
            try
            {
                read = stream.Read(buffer, 0, buffer.Length);
            }
            catch (Exception)
            {
                break; // socket closed / interrupted
            }
            if (read <= 0)
            {
                break; // server closed the connection
            }

            pending.Append(Encoding.UTF8.GetString(buffer, 0, read));
            string text = pending.ToString();
            int newlineIndex;
            while ((newlineIndex = text.IndexOf('\n')) >= 0)
            {
                string line = text.Substring(0, newlineIndex).Trim();
                text = text.Substring(newlineIndex + 1);
                if (line.Length > 0)
                {
                    lock (incomingLock) { incoming.Enqueue(line); }
                }
            }
            pending.Length = 0;
            pending.Append(text);
        }

        running = false;
        if (autoReconnect) reconnectTimer = reconnectDelaySeconds;
        Log("receive loop ended (disconnected)");
    }

    // ------------------------------------------------------------------
    // Receiving (main thread)
    // ------------------------------------------------------------------

    /// <summary>Returns the next received message line, or null when idle.</summary>
    public string Dequeue()
    {
        lock (incomingLock)
        {
            return incoming.Count > 0 ? incoming.Dequeue() : null;
        }
    }

    private void DrainIncoming()
    {
        string line = Dequeue();
        while (line != null)
        {
            if (OnRawMessage != null) OnRawMessage(line);
            line = Dequeue();
        }
    }

    // ------------------------------------------------------------------
    // Sending (main thread)
    // ------------------------------------------------------------------

    /// <summary>Queues a raw JSON message for delivery to the server.</summary>
    public void Send(string json)
    {
        lock (outgoingLock) { outgoing.Enqueue(json); }
        if (IsConnected) FlushOutgoing();
    }

    private void FlushOutgoing()
    {
        while (true)
        {
            string json;
            lock (outgoingLock)
            {
                if (outgoing.Count == 0) return;
                json = outgoing.Dequeue();
            }
            try
            {
                byte[] payload = Encoding.UTF8.GetBytes(json + "\n");
                stream.Write(payload, 0, payload.Length);
                stream.Flush();
            }
            catch (Exception exception)
            {
                Log("send failed: " + exception.Message);
                running = false;
                Cleanup();
                return;
            }
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// <summary>Sends a control command (start/pause/resume/reset/step/set_speed/quit).</summary>
    public void SendControl(string command)
    {
        Send("{\"type\":\"control\",\"command\":\"" + command + "\"}");
    }

    /// <summary>Changes the streaming pace (milliseconds per logical step).</summary>
    public void SendSpeed(int speedMs)
    {
        Send("{\"type\":\"control\",\"command\":\"set_speed\",\"speed_ms\":" + speedMs + "}");
    }

    /// <summary>
    /// Sends client-supplied actions for one step. Only valid when the server
    /// runs with "--action-source external" (manual debugging): the actions are
    /// then executed by the PYTHON environment, not chosen here.
    /// </summary>
    public void SendActions(Dictionary<int, int> actions)
    {
        if (actions == null || actions.Count == 0) return;
        StringBuilder builder = new StringBuilder();
        builder.Append("{\"type\":\"actions\",\"actions\":{");
        bool first = true;
        foreach (KeyValuePair<int, int> pair in actions)
        {
            if (!first) builder.Append(',');
            builder.Append('"').Append(pair.Key).Append("\":").Append(pair.Value);
            first = false;
        }
        builder.Append("}}");
        Send(builder.ToString());
    }

    private void Log(string message)
    {
        if (verboseLogging) Debug.Log("[GridMind Network] " + message);
    }
}
