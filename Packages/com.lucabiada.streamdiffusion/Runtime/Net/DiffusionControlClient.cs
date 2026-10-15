using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace StreamDiffusion
{
    public enum ConnectionState { Disconnected, Connecting, Connected, Error }

    /// <summary>
    /// WebSocket JSON client for the StreamDiffusion control server (protocol v1).
    /// Requests carry an increasing "id"; the server's ack / error / video_ready / state replies
    /// echo it, so <see cref="RequestAsync"/> can be awaited. Server pushes (progress, stats,
    /// phase, warning) are raised through <see cref="MessageReceived"/>.
    ///
    /// Await from the main thread: Unity's synchronization context resumes continuations there.
    /// <see cref="MessageReceived"/> is raised on a background thread; use <see cref="Drain"/>
    /// from Update to receive events on the main thread instead.
    /// </summary>
    public sealed class DiffusionControlClient : IDisposable
    {
        public const int ProtocolVersion = 1;

        ClientWebSocket _ws;
        CancellationTokenSource _cts;
        readonly ConcurrentQueue<JObject> _events = new ConcurrentQueue<JObject>();
        readonly ConcurrentDictionary<int, TaskCompletionSource<JObject>> _pending = new ConcurrentDictionary<int, TaskCompletionSource<JObject>>();
        readonly SemaphoreSlim _sendLock = new SemaphoreSlim(1, 1);
        int _nextId;

        public ConnectionState State { get; private set; } = ConnectionState.Disconnected;
        public string LastError { get; private set; }
        public JObject HelloAck { get; private set; }

        /// <summary>Raised on a background thread for every server message.</summary>
        public event Action<JObject> MessageReceived;

        public async Task ConnectAsync(string host, int port, string token = null, string clientName = "unity", int timeoutMs = 5000)
        {
            await DisconnectAsync();
            State = ConnectionState.Connecting;
            _cts = new CancellationTokenSource();
            _ws = new ClientWebSocket();
            try
            {
                using (var cts = CancellationTokenSource.CreateLinkedTokenSource(_cts.Token))
                {
                    cts.CancelAfter(timeoutMs);
                    await _ws.ConnectAsync(new Uri($"ws://{host}:{port}"), cts.Token);
                }
                _ = Task.Run(() => ReceiveLoop(_ws, _cts.Token));
                var ack = await RequestAsync("hello", new JObject
                {
                    ["protocol"] = ProtocolVersion,
                    ["client_name"] = clientName,
                    ["token"] = token,
                }, timeoutMs);
                if ((string)ack["type"] != "hello_ack")
                    throw new InvalidOperationException($"server refused hello: {ack["code"]} {ack["message"]}");
                HelloAck = ack;
                State = ConnectionState.Connected;
            }
            catch (Exception ex)
            {
                LastError = ex.Message;
                State = ConnectionState.Error;
                await DisconnectAsync(keepError: true);
                throw;
            }
        }

        async Task ReceiveLoop(ClientWebSocket ws, CancellationToken ct)
        {
            var buf = new byte[64 * 1024];
            var sb = new StringBuilder();
            try
            {
                while (!ct.IsCancellationRequested && ws.State == WebSocketState.Open)
                {
                    sb.Clear();
                    WebSocketReceiveResult r;
                    do
                    {
                        r = await ws.ReceiveAsync(new ArraySegment<byte>(buf), ct);
                        if (r.MessageType == WebSocketMessageType.Close) return;
                        sb.Append(Encoding.UTF8.GetString(buf, 0, r.Count));
                    } while (!r.EndOfMessage);

                    JObject m;
                    try { m = JObject.Parse(sb.ToString()); }
                    catch { continue; }

                    var idTok = m["id"];
                    string type = (string)m["type"];
                    bool isReply = type == "ack" || type == "hello_ack" || type == "video_ready" || type == "error"
                                   || type == "pong" || type == "state" || type == "ndi_sources" || type == "engine_status" || type == "loras" || type == "models";
                    if (isReply && idTok != null && idTok.Type == JTokenType.Integer
                        && _pending.TryRemove((int)idTok, out var tcs))
                    {
                        tcs.TrySetResult(m);
                    }
                    _events.Enqueue(m);
                    MessageReceived?.Invoke(m);
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                LastError = ex.Message;
                State = ConnectionState.Error;
            }
            finally
            {
                if (State == ConnectionState.Connected) State = ConnectionState.Disconnected;
                foreach (var kv in _pending)
                    if (_pending.TryRemove(kv.Key, out var t)) t.TrySetCanceled();
            }
        }

        /// <summary>Send a request and await the matching reply (ack / error / video_ready / state ...).</summary>
        public async Task<JObject> RequestAsync(string type, JObject payload = null, int timeoutMs = 30000)
        {
            if (_ws == null || _ws.State != WebSocketState.Open) throw new InvalidOperationException("not connected");
            int id = Interlocked.Increment(ref _nextId);
            var msg = payload != null ? (JObject)payload.DeepClone() : new JObject();
            msg["type"] = type;
            msg["id"] = id;
            var tcs = new TaskCompletionSource<JObject>(TaskCreationOptions.RunContinuationsAsynchronously);
            _pending[id] = tcs;
            var ct = _cts != null ? _cts.Token : CancellationToken.None;
            var bytes = Encoding.UTF8.GetBytes(msg.ToString(Newtonsoft.Json.Formatting.None));
            await _sendLock.WaitAsync();
            try { await _ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, ct); }
            finally { _sendLock.Release(); }

            var delay = Task.Delay(timeoutMs, ct);
            var done = await Task.WhenAny(tcs.Task, delay);
            if (done != tcs.Task)
            {
                _pending.TryRemove(id, out _);
                // The delay also completes (cancelled) when the connection is closed on purpose;
                // that is not a timeout and must not be reported as one.
                if (delay.IsCanceled || ct.IsCancellationRequested)
                    throw new OperationCanceledException($"connection closed while waiting for '{type}'");
                throw new TimeoutException($"'{type}' timed out after {timeoutMs} ms");
            }
            return await tcs.Task;
        }

        /// <summary>Fire-and-forget (no reply awaited). Safe for high-rate parameter updates.</summary>
        public void Send(string type, JObject payload = null)
        {
            _ = RequestFireAndForget(type, payload);
        }

        async Task RequestFireAndForget(string type, JObject payload)
        {
            if (_ws == null || _ws.State != WebSocketState.Open) return;
            var msg = payload != null ? (JObject)payload.DeepClone() : new JObject();
            msg["type"] = type;
            var bytes = Encoding.UTF8.GetBytes(msg.ToString(Newtonsoft.Json.Formatting.None));
            await _sendLock.WaitAsync();
            try { await _ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, _cts.Token); }
            catch { /* connection lost; state handled by the receive loop */ }
            finally { _sendLock.Release(); }
        }

        /// <summary>Drain queued server messages on the calling (main) thread.</summary>
        public void Drain(List<JObject> into)
        {
            while (_events.TryDequeue(out var m)) into.Add(m);
        }

        public async Task DisconnectAsync(bool keepError = false)
        {
            var ws = _ws;
            _ws = null;
            if (ws != null)
            {
                try
                {
                    _cts?.Cancel();
                    if (ws.State == WebSocketState.Open)
                        await ws.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "bye", CancellationToken.None);
                }
                catch { }
                ws.Dispose();
            }
            _cts?.Dispose();
            _cts = null;
            if (!keepError) State = ConnectionState.Disconnected;
        }

        public void Dispose()
        {
            _ = DisconnectAsync();
        }
    }
}
