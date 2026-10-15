using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace StreamDiffusion
{
    public enum InstallState { NotInstalled, Partial, Installed }

    /// <summary>
    /// Installs the Python side into a base folder, mirroring StreamDiffusionTD's three steps:
    /// (1) source: git clone pinned repos, (2) venv + PyTorch + pins via sd_installer, (3) TensorRT.
    /// Works identically in the Editor and in a built player (System.Diagnostics.Process only).
    ///
    /// Safety rules (learned the hard way): every pip-based step after the venv exists runs with
    /// the venv's own python.exe, never PATH python, and the step is refused if the interpreter
    /// is not inside the base folder.
    /// </summary>
    public sealed class DiffusionInstaller
    {
        public event Action<string> LogLine;
        public event Action<float, string> Progress;

        readonly DiffusionInstallSettings _s;
        readonly DiffusionPaths _p;
        string _python;   // 3.11 interpreter used to create the venv
        string _git;
        string _logFile;

        public bool IsRunning { get; private set; }
        public string LastError { get; private set; }
        public DiffusionPaths Paths => _p;
        public DiffusionInstallSettings Settings => _s;

        public DiffusionInstaller(DiffusionInstallSettings settings)
        {
            _s = settings;
            _p = new DiffusionPaths(settings.basefolder);
        }

        // ---------------------------------------------------------------- state
        public static InstallState Inspect(DiffusionPaths p)
        {
            bool venv = File.Exists(p.VenvPython);
            bool marker = File.Exists(p.MarkerFile);
            if (venv && marker) return InstallState.Installed;
            if (Directory.Exists(p.RepoDir) || venv) return InstallState.Partial;
            return InstallState.NotInstalled;
        }

        public InstallState State => Inspect(_p);

        // ---------------------------------------------------------------- public steps
        public async Task<bool> InstallAsync(CancellationToken ct)
        {
            if (IsRunning) return false;
            IsRunning = true;
            LastError = null;
            Directory.CreateDirectory(Path.Combine(_p.Base, "logs"));
            _logFile = Path.Combine(_p.Base, "logs", $"install-{DateTime.Now:yyyyMMdd-HHmmss}.log");
            try
            {
                Report(0.00f, "Preparing");
                Directory.CreateDirectory(_p.Base);
                Directory.CreateDirectory(_p.DownloadsDir);

                Report(0.02f, "Looking for Python " + _s.pythonVersionPrefix + "x");
                _python = await FindOrInstallPythonAsync(ct);

                Report(0.08f, "Looking for git");
                _git = await FindOrInstallGitAsync(ct);

                Report(0.12f, "Step 1/3: downloading StreamDiffusion source");
                await EnsureRepoAsync(_s.repoUrl, _s.repoBranch, _s.repoCommit, _p.RepoDir, ct);
                string installerUrl = await PickReachableUrlAsync(_s.installerUrl, _s.installerFallbackUrl, _s.installerBranch, ct);
                await EnsureRepoAsync(installerUrl, _s.installerBranch, _s.installerCommit, _p.InstallerDir, ct);

                // Idempotent: if an environment already verifies, do not re-run the heavy install
                // (re-running sd_installer on a live venv can break shared opencv files).
                bool envOk = false;
                if (File.Exists(_p.VenvPython))
                {
                    Report(0.20f, "Checking the existing environment");
                    envOk = (await VerifyAsync(ct)).ok;
                    Log(envOk ? "existing environment is healthy; skipping install" : "existing environment needs work");
                }

                if (!envOk)
                {
                    Report(0.25f, "Step 2/3: creating environment (PyTorch, pins). This downloads several GB");
                    try
                    {
                        await RunStepAsync("sd_installer install", _python,
                            $"-u -m sd_installer --base-folder \"{_p.RepoDir}\" install --cuda {_s.cuda}", _p.InstallerDir, ct);
                    }
                    catch (InvalidOperationException ex)
                    {
                        // sd_installer exits 1 when its own verify step fails; try the known fixes below.
                        Log("install step reported a problem (" + ex.Message + "); attempting repair");
                    }
                    Report(0.60f, "Repairing known issues");
                    await RepairAsync(ct);
                }

                RequireInsideBase(_p.VenvPython);
                bool trtOk = await TensorRtOkAsync(ct);
                if (!trtOk)
                {
                    Report(0.70f, "Step 3/3: installing TensorRT (about 3 GB)");
                    await RunStepAsync("sd_installer install-tensorrt", _p.VenvPython,
                        $"-u -m sd_installer --base-folder \"{_p.RepoDir}\" install-tensorrt", _p.InstallerDir, ct);
                }
                else Log("TensorRT already installed; skipping");

                Report(0.92f, "Installing server dependencies");
                await InstallServerDepsAsync(ct);

                Report(0.96f, "Verifying");
                var v = await VerifyAsync(ct);
                if (!v.ok) throw new InvalidOperationException($"verify failed: {v.failed} check(s) failed, see the log");

                WriteMarker(v.passed);
                Report(1.00f, "Installed");
                return true;
            }
            catch (OperationCanceledException)
            {
                LastError = "cancelled";
                Report(-1f, "Cancelled");
                return false;
            }
            catch (Exception ex)
            {
                LastError = ex.Message;
                Log("ERROR: " + ex.Message);
                Report(-1f, "Failed: " + ex.Message);
                return false;
            }
            finally { IsRunning = false; }
        }

        public async Task<(bool ok, int passed, int failed)> VerifyAsync(CancellationToken ct)
        {
            if (!File.Exists(_p.VenvPython)) return (false, 0, 1);
            RequireInsideBase(_p.VenvPython);
            int passed = 0, failed = 0;
            var rx = new Regex(@"Results:\s*(\d+)\s*passed,\s*(\d+)\s*failed");
            string work = Directory.Exists(_p.InstallerDir) ? _p.InstallerDir : _p.RepoDir;
            int code = await ProcessRunner.RunAsync(_p.VenvPython,
                $"-u -m sd_installer --base-folder \"{_p.RepoDir}\" verify", work, BuildEnv(),
                line =>
                {
                    Log(line);
                    var m = rx.Match(line);
                    if (m.Success) { passed = int.Parse(m.Groups[1].Value); failed = int.Parse(m.Groups[2].Value); }
                }, ct);
            return (code == 0 && failed == 0 && passed > 0, passed, failed);
        }

        /// <summary>Known fixes from StreamDiffusionTD's error table, applied only when verify is red.</summary>
        public async Task<bool> RepairAsync(CancellationToken ct)
        {
            if (!File.Exists(_p.VenvPython)) return false;
            RequireInsideBase(_p.VenvPython);
            var v = await VerifyAsync(ct);
            if (v.ok) return true;

            Log("repair: running sd_installer repair");
            await ProcessRunner.RunAsync(_p.VenvPython,
                $"-u -m sd_installer --base-folder \"{_p.RepoDir}\" repair -y", _p.InstallerDir, BuildEnv(), Log, ct);
            v = await VerifyAsync(ct);
            if (v.ok) return true;

            // opencv variants share the cv2/ folder: uninstalling one removes the others' files
            // while pip still lists the package. Remove all and reinstall the pinned one.
            Log("repair: restoring opencv-python (shared cv2 folder)");
            await ProcessRunner.RunAsync(_p.VenvPython,
                "-m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python", _p.RepoDir, BuildEnv(), Log, ct);
            string cv2Dir = Path.Combine(_p.VenvDir, "Lib", "site-packages", "cv2");
            try { if (Directory.Exists(cv2Dir)) Directory.Delete(cv2Dir, true); } catch (Exception e) { Log("could not remove cv2 dir: " + e.Message); }
            await ProcessRunner.RunAsync(_p.VenvPython,
                "-m pip install --no-deps --no-cache-dir opencv-python==4.8.1.78", _p.RepoDir, BuildEnv(), Log, ct);
            v = await VerifyAsync(ct);
            return v.ok;
        }

        async Task<bool> TensorRtOkAsync(CancellationToken ct)
        {
            var sb = new StringBuilder();
            int code = await ProcessRunner.RunAsync(_p.VenvPython,
                "-c \"import tensorrt;print(tensorrt.__version__)\"", _p.RepoDir, BuildEnv(), l => sb.Append(l), ct);
            return code == 0 && sb.ToString().StartsWith("10.");
        }

        // ---------------------------------------------------------------- bootstrap
        async Task<string> FindOrInstallPythonAsync(CancellationToken ct)
        {
            var found = await FindPythonAsync();
            if (found != null) { Log("Python: " + found); return found; }
            if (!_s.allowBootstrapDownloads)
                throw new InvalidOperationException(
                    $"Python {_s.pythonVersionPrefix}x was not found. Install Python 3.11.9 or allow the bootstrap download.");

            Log("Python not found; downloading the official installer (private per-user install, no PATH changes)");
            string exe = Path.Combine(_p.DownloadsDir, "python-installer.exe");
            await DownloadAsync(_s.pythonInstallerUrl, exe, "Python", ct);
            int code = await ProcessRunner.RunAsync(exe,
                $"/quiet InstallAllUsers=0 TargetDir=\"{_p.PrivatePythonDir}\" Include_launcher=0 Include_test=0 " +
                "Include_doc=0 Include_pip=1 PrependPath=0 AssociateFiles=0 Shortcuts=0",
                _p.DownloadsDir, null, Log, ct);
            if (code != 0 || !File.Exists(_p.PrivatePython))
                throw new InvalidOperationException("Python installer failed (exit " + code + ")");
            return _p.PrivatePython;
        }

        async Task<string> FindPythonAsync()
        {
            var candidates = new List<(string exe, string args)>
            {
                (_p.PrivatePython, ""),
                ("py", "-3.11"),
                ("python", ""),
            };
            foreach (var (exe, pre) in candidates)
            {
                if (exe.Contains(Path.DirectorySeparatorChar) && !File.Exists(exe)) continue;
                var (code, outp) = await ProcessRunner.ProbeAsync(exe,
                    (pre + " -c \"import sys;print(sys.executable+'|'+'%d.%d.%d'%sys.version_info[:3])\"").Trim());
                if (code != 0 || !outp.Contains("|")) continue;
                var parts = outp.Split('|');
                if (parts[1].StartsWith(_s.pythonVersionPrefix) && File.Exists(parts[0])) return parts[0];
            }
            return null;
        }

        async Task<string> FindOrInstallGitAsync(CancellationToken ct)
        {
            if (File.Exists(_p.MinGitExe)) { Log("git: " + _p.MinGitExe); return _p.MinGitExe; }
            var (code, outp) = await ProcessRunner.ProbeAsync("git", "--version");
            if (code == 0 && outp.StartsWith("git version")) { Log("git: PATH (" + outp + ")"); return "git"; }
            if (!_s.allowBootstrapDownloads)
                throw new InvalidOperationException("git was not found. Install git or allow the bootstrap download.");

            Log("git not found; downloading MinGit (portable, no installation)");
            string zip = Path.Combine(_p.DownloadsDir, "mingit.zip");
            await DownloadAsync(_s.minGitUrl, zip, "MinGit", ct);
            if (Directory.Exists(_p.MinGitDir)) Directory.Delete(_p.MinGitDir, true);
            ZipFile.ExtractToDirectory(zip, _p.MinGitDir);
            if (!File.Exists(_p.MinGitExe)) throw new InvalidOperationException("MinGit extraction failed");
            return _p.MinGitExe;
        }

        async Task DownloadAsync(string url, string dest, string label, CancellationToken ct)
        {
            using (var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) })
            using (var resp = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct))
            {
                resp.EnsureSuccessStatusCode();
                long total = resp.Content.Headers.ContentLength ?? -1;
                using (var src = await resp.Content.ReadAsStreamAsync())
                using (var dst = File.Create(dest))
                {
                    var buf = new byte[256 * 1024];
                    long done = 0; int n; long lastLog = 0;
                    while ((n = await src.ReadAsync(buf, 0, buf.Length, ct)) > 0)
                    {
                        await dst.WriteAsync(buf, 0, n, ct);
                        done += n;
                        if (done - lastLog > 4 * 1024 * 1024)
                        {
                            lastLog = done;
                            Log($"{label}: {done / 1048576} / {(total > 0 ? (total / 1048576).ToString() : "?")} MB");
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------- repos
        /// <summary>
        /// The primary URL when it is reachable and has the branch; otherwise the fallback (with a clear log line).
        /// Keeps installs working before the project's fork exists and if it is ever removed.
        /// </summary>
        async Task<string> PickReachableUrlAsync(string primary, string fallback, string branch, CancellationToken ct)
        {
            var seen = new StringBuilder();
            int code = await ProcessRunner.RunAsync(_git, $"-c credential.helper= ls-remote --heads \"{primary}\" {branch}", null, BuildEnv(),
                l => seen.AppendLine(l), ct);
            if (code == 0 && seen.ToString().Contains("refs/heads/" + branch)) return primary;
            if (string.IsNullOrWhiteSpace(fallback) || fallback == primary) return primary;   // EnsureRepoAsync explains
            Log($"WARNING: {primary} is not reachable or has no branch '{branch}' (fork not created yet, or branches not copied). " +
                $"Using {fallback} instead.");
            return fallback;
        }

        async Task EnsureRepoAsync(string url, string branch, string commit, string dir, CancellationToken ct)
        {
            Log($"source: {url} @ {branch} {commit.Substring(0, 7)}");
            // Fail early and clearly if the branch is not in this remote (e.g. a fork that only has main).
            var seen = new StringBuilder();
            int code = await ProcessRunner.RunAsync(_git, $"-c credential.helper= ls-remote --heads \"{url}\" {branch}", null, BuildEnv(),
                l => seen.AppendLine(l), ct);
            if (code != 0) throw new InvalidOperationException($"cannot reach {url}");
            if (!seen.ToString().Contains("refs/heads/" + branch))
                throw new InvalidOperationException(
                    $"{url} has no branch '{branch}'. Push it to the fork, or change repoUrl/repoBranch in streamdiffusion-install.json.");

            if (!Directory.Exists(Path.Combine(dir, ".git")))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(dir));
                await Git($"clone --branch {branch} \"{url}\" \"{dir}\"", null, ct);
            }
            else
            {
                // An existing clone may point at a different remote (e.g. the former upstream):
                // always follow the configured URL so the fork is the single source of truth.
                await Git($"remote set-url origin \"{url}\"", dir, ct);
                await Git($"fetch origin {branch}", dir, ct);
            }
            await Git($"checkout --quiet {commit}", dir, ct);
        }

        async Task Git(string args, string cwd, CancellationToken ct)
        {
            int code = await ProcessRunner.RunAsync(_git, args, cwd, BuildEnv(), Log, ct);
            if (code != 0) throw new InvalidOperationException($"git {args.Split(' ')[0]} failed (exit {code})");
        }

        // ---------------------------------------------------------------- python steps
        async Task InstallServerDepsAsync(CancellationToken ct)
        {
            RequireInsideBase(_p.VenvPython);
            string server = DiffusionPaths.ServerScriptDir();
            string req = server != null ? Path.Combine(server, "requirements.txt") : null;
            string args = req != null && File.Exists(req) ? $"-m pip install -r \"{req}\"" : "-m pip install \"websockets>=14\"";
            await RunStepAsync("server dependencies", _p.VenvPython, args, _p.RepoDir, ct);
        }

        async Task RunStepAsync(string label, string exe, string args, string cwd, CancellationToken ct)
        {
            Log($"== {label}");
            int code = await ProcessRunner.RunAsync(exe, args, cwd, BuildEnv(), Log, ct);
            if (code != 0) throw new InvalidOperationException($"{label} failed (exit {code})");
        }

        void RequireInsideBase(string exe)
        {
            string full = Path.GetFullPath(exe);
            if (!full.StartsWith(_p.Base, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"refusing to run '{exe}': interpreter is outside the base folder");
        }

        Dictionary<string, string> BuildEnv()
        {
            var env = new Dictionary<string, string>();
            if (_git != null && _git != "git")
            {
                string gitDir = Path.GetDirectoryName(_git);              // ...\cmd
                string path = Environment.GetEnvironmentVariable("PATH") ?? "";
                env["PATH"] = gitDir + Path.PathSeparator + path;
            }
            env["GIT_TERMINAL_PROMPT"] = "0";
            env["GCM_INTERACTIVE"] = "never";      // never pop a sign-in window for a repository that is not there
            return env;
        }

        void WriteMarker(int passed)
        {
            var m = new Marker
            {
                installedAtUtc = DateTime.UtcNow.ToString("o"),
                repoUrl = _s.repoUrl, repoBranch = _s.repoBranch, repoCommit = _s.repoCommit,
                installerCommit = _s.installerCommit, cuda = _s.cuda, verifyPassed = passed,
            };
            File.WriteAllText(_p.MarkerFile, JsonUtility.ToJson(m, true));
        }

        [Serializable]
        class Marker
        {
            public string installedAtUtc, repoUrl, repoBranch, repoCommit, installerCommit, cuda;
            public int verifyPassed;
        }

        // ---------------------------------------------------------------- logging
        void Log(string line)
        {
            try { File.AppendAllText(_logFile ?? Path.Combine(_p.Base, "install.log"), line + "\n"); } catch { }
            ParseProgress(line);
            LogLine?.Invoke(line);
        }

        void Report(float fraction, string message)
        {
            Log("[" + (fraction < 0 ? "x" : (int)(fraction * 100) + "%") + "] " + message);
            StartStepWindow(fraction);
            _lastFraction = fraction;
            _lastMessage = message;
            Progress?.Invoke(fraction, message);
        }

        // ---------------------------------------------------------------- live progress
        // The two long steps (PyTorch+packages, TensorRT) are single pip runs that print nothing while
        // a wheel downloads, so the bar is advanced from what the log does tell us: the sd_installer
        // sub-steps "[n/8]", the "Downloading X (NNN MB)" line that starts each wheel, and the typical
        // download speed measured on earlier files. It never goes backwards and stops short of the next step.
        float _stepFrom = -1f, _stepTo = -1f, _stepExpectedMb;
        float _stepFloor;                       // from the [n/N] sub-step lines
        double _stepDoneMb;                     // wheels fully downloaded in this step
        string _curFile = "";
        double _curSizeMb;
        DateTime _curStarted;
        double _speedMbPerS = 6.0;              // refined from finished downloads
        float _lastFraction;
        string _lastMessage = "";
        string _subStep = "";
        DateTime _runStarted;

        void StartStepWindow(float at)
        {
            if (Math.Abs(at - 0.25f) < 0.006f) { _stepFrom = 0.25f; _stepTo = 0.58f; _stepExpectedMb = 5200f; }
            else if (Math.Abs(at - 0.70f) < 0.011f) { _stepFrom = 0.70f; _stepTo = 0.91f; _stepExpectedMb = 3400f; }
            else { _stepFrom = _stepTo = -1f; _stepExpectedMb = 0f; }
            _stepFloor = 0f;
            _stepDoneMb = 0;
            _curFile = ""; _curSizeMb = 0;
            _subStep = "";
        }

        static readonly System.Text.RegularExpressions.Regex RxSubStep =
            new System.Text.RegularExpressions.Regex(@"^\[(\d+)/(\d+)\]\s*(.*)$", System.Text.RegularExpressions.RegexOptions.Compiled);
        static readonly System.Text.RegularExpressions.Regex RxDownload =
            new System.Text.RegularExpressions.Regex(@"^\s*Downloading\s+(\S+)\s+\(([\d.]+)\s*(kB|MB|GB)\)\s*$",
                System.Text.RegularExpressions.RegexOptions.Compiled);
        static readonly System.Text.RegularExpressions.Regex RxDone =
            new System.Text.RegularExpressions.Regex(@"([\d.]+)/([\d.]+)\s*(kB|MB|GB)\s+([\d.]+)\s*(kB|MB|GB)/s",
                System.Text.RegularExpressions.RegexOptions.Compiled);

        static double ToMb(string value, string unit)
        {
            double v = double.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
            return unit == "GB" ? v * 1024.0 : unit == "kB" ? v / 1024.0 : v;
        }

        void ParseProgress(string line)
        {
            if (_stepFrom < 0f || string.IsNullOrEmpty(line)) return;
            try
            {
                var m = RxSubStep.Match(line);
                if (m.Success)
                {
                    int n = int.Parse(m.Groups[1].Value), total = Math.Max(1, int.Parse(m.Groups[2].Value));
                    _stepFloor = Math.Max(_stepFloor, (_stepTo - _stepFrom) * Math.Max(0, n - 1) / total);
                    _subStep = "(" + n + "/" + total + ") " + m.Groups[3].Value.Trim();
                    return;
                }
                m = RxDownload.Match(line);
                if (m.Success)
                {
                    // a new wheel starts: the previous one is complete
                    _stepDoneMb += _curSizeMb;
                    _curFile = m.Groups[1].Value;
                    _curSizeMb = ToMb(m.Groups[2].Value, m.Groups[3].Value);
                    _curStarted = DateTime.UtcNow;
                    return;
                }
                m = RxDone.Match(line);
                if (m.Success && _curSizeMb > 1.0)
                {
                    double seconds = Math.Max(0.5, (DateTime.UtcNow - _curStarted).TotalSeconds);
                    if (_curSizeMb >= 20.0)                        // small files say nothing about the line speed
                        _speedMbPerS = 0.5 * _speedMbPerS + 0.5 * Math.Max(0.5, _curSizeMb / seconds);
                    _stepDoneMb += _curSizeMb;
                    _curSizeMb = 0;
                    _curFile = "";
                }
            }
            catch { }
        }

        /// <summary>
        /// The bar value (0..1, or -1 when unknown) and a one-line message, interpolated from the log and the
        /// clock. Poll it while <see cref="IsRunning"/>; it is cheap.
        /// </summary>
        public float LiveProgress(out string message)
        {
            message = _lastMessage;
            float f = _lastFraction;
            if (_stepFrom >= 0f && IsRunning)
            {
                double mb = _stepDoneMb;
                if (_curSizeMb > 0)
                {
                    double elapsed = (DateTime.UtcNow - _curStarted).TotalSeconds;
                    double est = Math.Min(_curSizeMb * 0.97, _speedMbPerS * elapsed);
                    mb += est;
                    if (_curSizeMb >= 5.0)      // metadata and tiny wheels flash by; only show real downloads
                        message = (string.IsNullOrEmpty(_subStep) ? "" : _subStep + " - ") + "downloading " + ShortName(_curFile) +
                                  " (" + (est < _curSizeMb ? est.ToString("0") + "/" : "") + _curSizeMb.ToString("0") + " MB)";
                    else if (!string.IsNullOrEmpty(_subStep)) message = _subStep;
                }
                else if (!string.IsNullOrEmpty(_subStep)) message = _subStep;

                float byDownload = (float)Math.Min(0.97, mb / Math.Max(1.0, _stepExpectedMb));
                float within = Math.Max(_stepFloor / Math.Max(0.0001f, _stepTo - _stepFrom), byDownload);
                f = Math.Max(f, _stepFrom + (_stepTo - _stepFrom) * Math.Min(0.99f, within));
            }
            return f;
        }

        static string ShortName(string wheel)
        {
            if (string.IsNullOrEmpty(wheel)) return "";
            int i = wheel.IndexOf('-');
            return i > 0 ? wheel.Substring(0, i) : wheel;
        }
    }
}
