using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace StreamDiffusion.Editor
{
    /// <summary>
    /// Copies the Python server (the package's Server~ folder) next to the built exe so the player can
    /// launch it. The Python environment itself is NOT shipped; it is installed on first run.
    /// </summary>
    public sealed class DiffusionBuildPostprocess : IPostprocessBuildWithReport
    {
        public int callbackOrder => 100;

        static readonly string[] SkipDirs = { "__pycache__", "tests", "models", "logs", ".pytest_cache" };

        public void OnPostprocessBuild(BuildReport report)
        {
            if (report.summary.platform != BuildTarget.StandaloneWindows64) return;
            string src = DiffusionPaths.ServerScriptDir();
            if (string.IsNullOrEmpty(src) || !File.Exists(Path.Combine(src, "run_server.py")))
            {
                Debug.LogWarning("[SD] the Python server (Server~ in the package, or diffusion-server/ next to the project) was not found; not copied into the build.");
                return;
            }
            string exeDir = Path.GetDirectoryName(report.summary.outputPath);
            string dst = Path.Combine(exeDir, "diffusion-server");
            if (Directory.Exists(dst)) Directory.Delete(dst, true);
            Copy(src, dst);
            Debug.Log($"[SD] copied diffusion-server to {dst}");
        }

        static void Copy(string src, string dst)
        {
            Directory.CreateDirectory(dst);
            foreach (var f in Directory.GetFiles(src))
                if (!f.EndsWith(".pyc")) File.Copy(f, Path.Combine(dst, Path.GetFileName(f)), true);
            foreach (var d in Directory.GetDirectories(src))
            {
                string name = Path.GetFileName(d);
                if (System.Array.IndexOf(SkipDirs, name) >= 0) continue;
                Copy(d, Path.Combine(dst, name));
            }
        }
    }
}
