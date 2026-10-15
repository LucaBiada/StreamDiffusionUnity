using System;
using System.Diagnostics;
using System.IO;
using Newtonsoft.Json.Linq;

namespace StreamDiffusion
{
    /// <summary>
    /// Flight recorder for engine builds that run next to Unity. While engines build it keeps a
    /// marker file (build-session.json) and an independent process (tools/gpu_recorder.py) that
    /// logs GPU state once per second to disk, so a GPU reset or crash of Unity still leaves data.
    /// Afterwards: python tools/collect_diagnostics.py --base &lt;base&gt;.
    /// </summary>
    public static class EngineBuildDiagnostics
    {
        public static string DiagDir(DiffusionPaths paths) => Path.Combine(paths.Base, "logs", "diagnostics");
        static string MarkerPath(DiffusionPaths paths) => Path.Combine(DiagDir(paths), "build-session.json");

        static JObject _marker;
        static string _markerPath;
        static DateTime _started;

        public static bool Active => _marker != null;

        /// <returns>the recorder CSV path, or null when it could not be started.</returns>
        public static string Begin(DiffusionPaths paths, string modelId, int width, int height, int steps,
            bool tinyVae, string policy, string serverLog, string unityVersion, string projectPath, string editorLog)
        {
            try
            {
                Directory.CreateDirectory(DiagDir(paths));
                _started = DateTime.UtcNow;
                string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
                string csv = Path.Combine(DiagDir(paths), $"gpu-{stamp}-{width}x{height}.csv");
                _markerPath = MarkerPath(paths);
                _marker = new JObject
                {
                    ["status"] = "in_progress",
                    ["startedUtc"] = _started.ToString("o"),
                    ["modelId"] = modelId, ["width"] = width, ["height"] = height, ["steps"] = steps,
                    ["tinyVae"] = tinyVae, ["policy"] = policy,
                    ["unityVersion"] = unityVersion, ["unityPid"] = Process.GetCurrentProcess().Id,
                    ["projectPath"] = projectPath, ["editorLogPath"] = editorLog,
                    ["serverLogPath"] = serverLog, ["recorderCsv"] = csv,
                };
                Write();

                string server = DiffusionPaths.ServerScriptDir();
                string script = server == null ? null : Path.Combine(server, "tools", "gpu_recorder.py");
                if (script != null && File.Exists(script) && File.Exists(paths.VenvPython))
                {
                    var psi = new ProcessStartInfo(paths.VenvPython,
                        $"-u \"{script}\" --out \"{csv}\" --marker \"{_markerPath}\" --max-minutes 60")
                    {
                        UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = server,
                    };
                    var p = Process.Start(psi);
                    _marker["recorderPid"] = p != null ? p.Id : 0;
                    Write();
                }
                return csv;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning("[StreamDiffusion] diagnostics could not start: " + e.Message);
                _marker = null;
                return null;
            }
        }

        public static void End(bool success, string message)
        {
            if (_marker == null) return;
            try
            {
                _marker["status"] = success ? "completed" : "failed";
                _marker["result"] = success ? "ok" : "error";
                _marker["message"] = message ?? "";
                _marker["endedUtc"] = DateTime.UtcNow.ToString("o");
                _marker["seconds"] = (DateTime.UtcNow - _started).TotalSeconds;
                Write();
            }
            catch { }
            _marker = null;
        }

        static void Write()
        {
            string tmp = _markerPath + ".tmp";
            File.WriteAllText(tmp, _marker.ToString());
            if (File.Exists(_markerPath)) File.Delete(_markerPath);
            File.Move(tmp, _markerPath);
        }

        /// <summary>
        /// Call at startup: a marker still "in_progress" whose Unity process is gone means the last
        /// build ended abnormally (Unity or the GPU went down).
        /// </summary>
        public static string CheckInterrupted(DiffusionPaths paths)
        {
            try
            {
                string path = MarkerPath(paths);
                if (!File.Exists(path)) return null;
                var m = JObject.Parse(File.ReadAllText(path));
                if ((string)m["status"] != "in_progress") return null;
                int pid = (int?)m["unityPid"] ?? 0;
                if (pid == Process.GetCurrentProcess().Id) return null;
                try { if (pid > 0 && !Process.GetProcessById(pid).HasExited) return null; } catch { }
                m["status"] = "interrupted";
                File.WriteAllText(path, m.ToString());
                string server = DiffusionPaths.ServerScriptDir() ?? "<diffusion-server>";
                return "the previous engine build (" + m["width"] + "x" + m["height"] + ") ended abnormally. Data was recorded; run: \""
                       + paths.VenvPython + "\" \"" + Path.Combine(server, "tools", "collect_diagnostics.py") + "\" --base \"" + paths.Base + "\"";
            }
            catch { return null; }
        }
    }
}
