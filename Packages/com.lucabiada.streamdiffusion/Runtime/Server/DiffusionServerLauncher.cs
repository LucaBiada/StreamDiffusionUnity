using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace StreamDiffusion
{
    /// <summary>
    /// Starts and stops the local Python server (run_server.py) from the venv inside the base
    /// folder. If something already listens on the port, it connects to that instead and never
    /// kills it. The server also watches Unity's pid (--parent-pid), so a crashed player cannot
    /// leave an orphan holding the GPU.
    /// </summary>
    public sealed class DiffusionServerLauncher : IDisposable
    {
        public event Action<string> LogLine;

        Process _proc;
        readonly Queue<string> _tail = new Queue<string>();
        const int TailLines = 400;

        public bool OwnsProcess => _proc != null && !_proc.HasExited;

        /// <summary>The persistent server log (flushed per line, survives a crash of Unity or the GPU).</summary>
        public string LogFile { get; private set; }
        public string[] RecentLines { get { lock (_tail) return _tail.ToArray(); } }

        public static bool PortOpen(string host, int port, int timeoutMs = 300)
        {
            try
            {
                using (var c = new TcpClient())
                {
                    var t = c.ConnectAsync(host, port);
                    return t.Wait(timeoutMs) && c.Connected;
                }
            }
            catch { return false; }
        }

        /// <returns>true when a server is reachable (started by us, or already running).</returns>
        public async Task<bool> StartAsync(DiffusionPaths paths, string host, int port, string token,
            CancellationToken ct, int timeoutMs = 60000)
        {
            if (PortOpen(host, port))
            {
                Emit($"server already running on {host}:{port}, using it");
                return true;
            }
            if (OwnsProcess) Stop();

            if (!File.Exists(paths.VenvPython))
                throw new InvalidOperationException("Python environment is not installed (venv not found)");
            string script = DiffusionPaths.ServerScriptDir();
            if (script == null)
                throw new InvalidOperationException("diffusion-server/run_server.py not found next to the application");
            string full = Path.GetFullPath(paths.VenvPython);
            if (!full.StartsWith(paths.Base, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("refusing to launch an interpreter outside the base folder");

            var args = new StringBuilder();
            args.Append("-u \"").Append(Path.Combine(script, "run_server.py")).Append('"');
            args.Append(" --basefolder \"").Append(paths.RepoDir).Append('"');
            args.Append(" --host ").Append(host == "localhost" ? "127.0.0.1" : host);
            args.Append(" --port ").Append(port);
            args.Append(" --allow-shutdown --parent-pid ").Append(Process.GetCurrentProcess().Id);
            Directory.CreateDirectory(Path.Combine(paths.Base, "logs"));
            LogFile = Path.Combine(paths.Base, "logs", "server-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + ".log");
            args.Append(" --log-file \"").Append(LogFile).Append('"');
            if (!string.IsNullOrEmpty(token)) args.Append(" --token ").Append(token);

            var psi = new ProcessStartInfo(paths.VenvPython, args.ToString())
            {
                WorkingDirectory = script,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";

            _proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            _proc.OutputDataReceived += (_, e) => { if (e.Data != null) Emit(e.Data); };
            _proc.ErrorDataReceived += (_, e) => { if (e.Data != null) Emit(e.Data); };
            if (!_proc.Start()) throw new InvalidOperationException("could not start the server process");
            _proc.BeginOutputReadLine();
            _proc.BeginErrorReadLine();
            Emit($"server started (pid {_proc.Id}); waiting for {host}:{port}");

            var sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                ct.ThrowIfCancellationRequested();
                if (_proc.HasExited)
                    throw new InvalidOperationException($"server exited during startup (code {_proc.ExitCode}); last log: " + LastLines(6));
                if (PortOpen(host, port)) { Emit("server is listening"); return true; }
                await Task.Delay(250, ct);
            }
            Stop();
            throw new TimeoutException("server did not start listening within " + timeoutMs / 1000 + " s");
        }

        /// <summary>Kills the process tree if we started it. A server we merely connected to is left alone.</summary>
        public void Stop()
        {
            var p = _proc;
            _proc = null;
            if (p == null) return;
            try
            {
                if (!p.HasExited)
                {
                    Emit("stopping server (pid " + p.Id + ")");
                    ProcessRunner.KillTree(p);
                }
                p.Dispose();
            }
            catch { }
        }

        string LastLines(int n)
        {
            var all = RecentLines;
            int from = Math.Max(0, all.Length - n);
            return string.Join(" | ", all, from, all.Length - from);
        }

        void Emit(string line)
        {
            lock (_tail)
            {
                _tail.Enqueue(line);
                while (_tail.Count > TailLines) _tail.Dequeue();
            }
            LogLine?.Invoke(line);
        }

        public void Dispose() => Stop();
    }
}
