using System;
using System.IO;
using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>Resolves every on-disk location the installer and launcher use.</summary>
    public sealed class DiffusionPaths
    {
        public readonly string Base;           // user-chosen folder, no spaces
        public string RepoDir => Path.Combine(Base, "StreamDiffusion");
        public string InstallerDir => Path.Combine(RepoDir, "StreamDiffusion-installer");
        public string VenvDir => Path.Combine(RepoDir, "venv");
        public string VenvPython => Path.Combine(VenvDir, "Scripts", "python.exe");
        public string EngineDir => Path.Combine(RepoDir, "engines", "td");
        public string ModelDir => Path.Combine(RepoDir, "models", "Model");
        public string LoraDir => Path.Combine(RepoDir, "models", "LoRA");
        public string PrivatePythonDir => Path.Combine(Base, "python");
        public string PrivatePython => Path.Combine(PrivatePythonDir, "python.exe");
        public string MinGitDir => Path.Combine(Base, "git");
        public string MinGitExe => Path.Combine(MinGitDir, "cmd", "git.exe");
        public string DownloadsDir => Path.Combine(Base, "downloads");
        public string MarkerFile => Path.Combine(Base, "unity-install.json");

        public DiffusionPaths(string basefolder)
        {
            Base = Path.GetFullPath(string.IsNullOrWhiteSpace(basefolder) ? DefaultBase() : basefolder);
            if (Base.Contains(" "))
                throw new ArgumentException($"base folder must not contain spaces: '{Base}'");
        }

        public static string DefaultBase()
        {
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(local, "StreamDiffusionUnity", "diffusion");
        }

        /// <summary>
        /// Folder holding run_server.py. Player: next to the exe (post-build copy). Editor: the
        /// package's Server~ folder, or a diffusion-server/ next to the Unity project.
        /// </summary>
        public static string ServerScriptDir()
        {
            string data = Application.dataPath; // <exe>_Data or <project>/Assets
            var candidates = new System.Collections.Generic.List<string>
            {
                Path.Combine(data, "..", "diffusion-server"),               // player: beside the exe
            };
#if UNITY_EDITOR
            // editor: the server shipped inside the package (Server~), wherever the package lives
            var info = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(DiffusionPaths).Assembly);
            if (info != null) candidates.Add(Path.Combine(info.resolvedPath, "Server~"));
#endif
            candidates.Add(Path.Combine(data, "..", "..", "diffusion-server")); // editor: beside the project (development layout)
            candidates.Add(Path.Combine(Application.streamingAssetsPath, "diffusion-server"));
            foreach (var c in candidates)
            {
                var full = Path.GetFullPath(c);
                if (File.Exists(Path.Combine(full, "run_server.py"))) return full;
            }
            return null;
        }
    }
}
