using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace StreamDiffusion
{
    public enum GpuTimeoutState
    {
        /// <summary>Not Windows (or the value could not be read): nothing to do.</summary>
        Unsupported,
        /// <summary>Windows is using its short default (2 s). Long TensorRT builds can reset the GPU.</summary>
        Short,
        /// <summary>A long timeout is configured and active.</summary>
        Fixed,
        /// <summary>The value was changed but Windows has not been restarted since, so it is not active yet.</summary>
        RebootPending,
    }

    /// <summary>
    /// The Windows GPU watchdog (TDR) resets the GPU when a job runs longer than TdrDelay (default 2 s).
    /// TensorRT engine builds saturate the GPU for minutes and, next to a running 3D app, have caused such
    /// resets (black screen, Unity closes). This raises the timeout to 60 s. It is a system setting under
    /// HKLM, so it always needs administrator permission: it is only ever done after the user confirms, and
    /// Windows then shows its own UAC prompt. Nothing is changed silently.
    /// </summary>
    public static class GpuTimeoutFix
    {
        public const int TargetSeconds = 60;
        public const int GoodEnoughSeconds = 10;

        const string Key = @"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers";

        /// <summary>The two commands, for people who prefer to run them themselves (administrator prompt).</summary>
        public static readonly string ManualCommands =
            "reg add " + Key + " /v TdrDelay /t REG_DWORD /d " + TargetSeconds + " /f" + Environment.NewLine +
            "reg add " + Key + " /v TdrDdiDelay /t REG_DWORD /d " + TargetSeconds + " /f";

        static GpuTimeoutState _cached;
        static int? _cachedValue;
        static DateTime _cachedAt = DateTime.MinValue;

        public static bool IsWindows =>
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            true;
#else
            false;
#endif

        static string MarkerPath(string baseFolder) => Path.Combine(baseFolder ?? "", "unity-gpu-timeout.json");

        /// <summary>The configured TdrDelay in seconds, or null when it is not set (Windows default) or unreadable.</summary>
        public static int? ReadTdrDelaySeconds()
        {
            if (!IsWindows) return null;
            try
            {
                var psi = new ProcessStartInfo("reg", "query " + Key + " /v TdrDelay")
                {
                    UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true,
                };
                using (var p = Process.Start(psi))
                {
                    string output = p.StandardOutput.ReadToEnd();
                    if (!p.WaitForExit(3000)) { try { p.Kill(); } catch { } return null; }
                    int i = output.IndexOf("0x", StringComparison.OrdinalIgnoreCase);
                    if (i < 0) return null;
                    int j = i + 2;
                    while (j < output.Length && Uri.IsHexDigit(output[j])) j++;
                    return Convert.ToInt32(output.Substring(i + 2, j - i - 2), 16);
                }
            }
            catch { return null; }
        }

        /// <summary>Current state (cached for a few seconds: reading it starts a small process).</summary>
        public static GpuTimeoutState GetState(string baseFolder, out int? seconds)
        {
            if ((DateTime.UtcNow - _cachedAt).TotalSeconds < 5)
            {
                seconds = _cachedValue;
                return _cached;
            }
            var state = Evaluate(baseFolder, out var value);
            _cached = state; _cachedValue = value; _cachedAt = DateTime.UtcNow;
            seconds = value;
            return state;
        }

        public static GpuTimeoutState GetState(string baseFolder) => GetState(baseFolder, out _);

        public static void Invalidate() => _cachedAt = DateTime.MinValue;

        static GpuTimeoutState Evaluate(string baseFolder, out int? seconds)
        {
            seconds = null;
            if (!IsWindows) return GpuTimeoutState.Unsupported;
            seconds = ReadTdrDelaySeconds();
            if (seconds == null || seconds < GoodEnoughSeconds) return GpuTimeoutState.Short;

            // configured: is it active? only if Windows has been (re)started since we changed it
            try
            {
                string marker = MarkerPath(baseFolder);
                if (File.Exists(marker))
                {
                    var changed = File.GetLastWriteTimeUtc(marker);
                    var bootUtc = DateTime.UtcNow - TimeSpan.FromMilliseconds((uint)Environment.TickCount);
                    if (changed > bootUtc) return GpuTimeoutState.RebootPending;
                }
            }
            catch { }
            return GpuTimeoutState.Fixed;
        }

        public static string Describe(GpuTimeoutState state, int? seconds)
        {
            switch (state)
            {
                case GpuTimeoutState.Short:
                    return "Windows is using its default 2 second GPU timeout. Long TensorRT builds can reset the GPU " +
                           "(black screen, the app closes). Raise it to " + TargetSeconds + " s once.";
                case GpuTimeoutState.RebootPending:
                    return "GPU timeout set to " + seconds + " s. Restart Windows (Restart, not Shut down) to activate it.";
                case GpuTimeoutState.Fixed:
                    return "GPU timeout is " + seconds + " s. Good for engine builds.";
                default:
                    return "Not applicable on this system.";
            }
        }

        /// <summary>
        /// Raises TdrDelay / TdrDdiDelay to <see cref="TargetSeconds"/>. Windows asks the user for
        /// administrator permission (UAC); declining leaves everything unchanged.
        /// </summary>
        public static async Task<(bool ok, string message)> ApplyAsync(string baseFolder)
        {
            if (!IsWindows) return (false, "Not applicable on this system.");
            try
            {
                string args = "/c reg add " + Key + " /v TdrDelay /t REG_DWORD /d " + TargetSeconds + " /f && " +
                              "reg add " + Key + " /v TdrDdiDelay /t REG_DWORD /d " + TargetSeconds + " /f";
                var psi = new ProcessStartInfo("cmd.exe", args)
                {
                    UseShellExecute = true,           // required for the elevation prompt
                    Verb = "runas",
                    WindowStyle = ProcessWindowStyle.Hidden,
                };
                int exit = await Task.Run(() =>
                {
                    using (var p = Process.Start(psi))
                    {
                        if (p == null) return -1;
                        if (!p.WaitForExit(60000)) return -2;
                        return p.ExitCode;
                    }
                });
                if (exit == -2) return (false, "Timed out waiting for the administrator command.");
                Invalidate();
                int? now = ReadTdrDelaySeconds();
                if (now != null && now >= GoodEnoughSeconds)
                {
                    try
                    {
                        Directory.CreateDirectory(baseFolder ?? ".");
                        File.WriteAllText(MarkerPath(baseFolder), "{\"changedUtc\":\"" + DateTime.UtcNow.ToString("o") +
                                                                   "\",\"tdrDelay\":" + now + "}");
                    }
                    catch { }
                    Invalidate();
                    return (true, "GPU timeout set to " + now + " s. Restart Windows (Restart, not Shut down) to activate it.");
                }
                return (false, "The setting was not changed (exit code " + exit + ").");
            }
            catch (Win32Exception ex) when (ex.NativeErrorCode == 1223)
            {
                return (false, "Cancelled: administrator permission was not granted. Nothing was changed.");
            }
            catch (Exception ex)
            {
                return (false, "Could not change the setting: " + ex.Message);
            }
        }
    }
}
