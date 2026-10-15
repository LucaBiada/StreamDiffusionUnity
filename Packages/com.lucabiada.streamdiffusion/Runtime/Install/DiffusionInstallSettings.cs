using System;
using System.IO;
using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>
    /// Where the Python environment comes from. Defaults point at the project's own fork so the
    /// install keeps working even if upstream moves. Override per build with
    /// StreamingAssets/streamdiffusion-install.json (same field names).
    /// </summary>
    [Serializable]
    public sealed class DiffusionInstallSettings
    {
        // StreamDiffusion fork that holds the pinned SDTD_040_stable branch.
        public string repoUrl = "https://github.com/LucaBiada/StreamDiffusion.git";
        public string repoBranch = "SDTD_040_stable";
        public string repoCommit = "25054bf21fad2ac13a65258651dee13d8d8e627b";

        // sd_installer CLI (installs venv, PyTorch, pins, TensorRT).
        // The project's own fork (a fork of dotsimulate/StreamDiffusion-installer with all branches). If it is
        // not reachable or lacks the branch, the installer falls back to installerFallbackUrl and says so.
        public string installerUrl = "https://github.com/LucaBiada/StreamDiffusion-installer.git";
        public string installerFallbackUrl = "https://github.com/dotsimulate/StreamDiffusion-installer.git";
        public string installerBranch = "SDTD_041_installer";
        public string installerCommit = "fa8179ee9363c2208239bae21cfcd201b6dcd3a8";

        public string cuda = "cu128";
        public string pythonVersionPrefix = "3.11.";
        public string pythonInstallerUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe";
        public string minGitUrl = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip";

        /// <summary>Empty = default (%LOCALAPPDATA%\StreamDiffusionUnity\diffusion). Must not contain spaces.</summary>
        public string basefolder = "";

        /// <summary>Allow downloading Python / MinGit when they are missing. The UI asks the user first.</summary>
        public bool allowBootstrapDownloads = false;

        public static DiffusionInstallSettings Load()
        {
            var s = new DiffusionInstallSettings();
            try
            {
                var path = Path.Combine(Application.streamingAssetsPath, "streamdiffusion-install.json");
                if (File.Exists(path)) JsonUtility.FromJsonOverwrite(File.ReadAllText(path), s);
                var envBase = Environment.GetEnvironmentVariable("SDU_BASEFOLDER");
                if (!string.IsNullOrEmpty(envBase)) s.basefolder = envBase;
            }
            catch (Exception ex) { Debug.LogWarning("[SD] could not read install settings: " + ex.Message); }
            return s;
        }
    }
}
