using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace StreamDiffusion
{
    /// <summary>Runs an external process, streams its output lines, supports cancellation.</summary>
    public static class ProcessRunner
    {
        public static async Task<int> RunAsync(
            string exe, string arguments, string workingDirectory,
            IDictionary<string, string> env, Action<string> onLine, CancellationToken ct)
        {
            var psi = new ProcessStartInfo(exe, arguments)
            {
                WorkingDirectory = workingDirectory ?? "",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";
            psi.EnvironmentVariables["PIP_DISABLE_PIP_VERSION_CHECK"] = "1";
            if (env != null)
                foreach (var kv in env) psi.EnvironmentVariables[kv.Key] = kv.Value;

            var tcs = new TaskCompletionSource<int>(TaskCreationOptions.RunContinuationsAsynchronously);
            var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (_, e) => { if (e.Data != null) onLine?.Invoke(e.Data); };
            proc.ErrorDataReceived += (_, e) => { if (e.Data != null) onLine?.Invoke(e.Data); };
            proc.Exited += (_, __) => tcs.TrySetResult(proc.ExitCode);

            if (!proc.Start()) throw new InvalidOperationException("could not start " + exe);
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();

            using (ct.Register(() => { KillTree(proc); tcs.TrySetCanceled(); }))
            {
                try { return await tcs.Task; }
                finally { proc.Dispose(); }
            }
        }

        /// <summary>Run and return the first non-empty stdout line (for version probes).</summary>
        public static async Task<(int code, string output)> ProbeAsync(string exe, string arguments, int timeoutMs = 15000)
        {
            var sb = new StringBuilder();
            using (var cts = new CancellationTokenSource(timeoutMs))
            {
                try
                {
                    int code = await RunAsync(exe, arguments, null, null, l => { if (sb.Length == 0 && l.Trim().Length > 0) sb.Append(l.Trim()); }, cts.Token);
                    return (code, sb.ToString());
                }
                catch { return (-1, ""); }
            }
        }

        public static void KillTree(Process p)
        {
            try
            {
                if (p == null || p.HasExited) return;
                // taskkill /T kills the whole tree (pip -> build subprocesses).
                using (var k = Process.Start(new ProcessStartInfo("taskkill", $"/T /F /PID {p.Id}")
                { CreateNoWindow = true, UseShellExecute = false }))
                    k?.WaitForExit(5000);
            }
            catch { }
        }
    }
}
