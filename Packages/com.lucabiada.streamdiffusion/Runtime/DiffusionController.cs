using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace StreamDiffusion
{
    public enum DiffusionPhase { Disconnected, Installing, StartingServer, Connecting, Connected, Loading, Ready, Running, Error, AwaitingBuildConfirmation }

    public enum DiffusionSourceMode { Texture, Camera }

    /// <summary>
    /// Runtime component: owns the control client and the video transport, sends a source
    /// texture to the diffusion server and exposes the generated frames as ResultTexture.
    /// Unity is the source of truth: after (re)connecting it replays model, prompt and params.
    /// </summary>
    [AddComponentMenu("StreamDiffusion/Diffusion Controller")]
    public sealed class DiffusionController : MonoBehaviour
    {
        [Header("Startup")]
        [Tooltip("Connect, load the model and start streaming as soon as the scene starts. Needed for builds with no UI.")]
        public bool autoStart = false;
        [Tooltip("If the Python environment is not installed, install it first (long: several GB).")]
        public bool autoInstallIfMissing = false;
        [Tooltip("Read -sd-host -sd-port -sd-token -sd-basefolder -sd-autostart -sd-install -sd-allow-downloads from the command line.")]
        public bool applyCommandLine = true;

        [Header("Engine builds")]
        [Tooltip("A model / resolution / step count that has no cached TensorRT engines must BUILD them: minutes of " +
                 "sustained GPU load (it once reset the GPU driver while the Editor was open). Ask = pause and wait for " +
                 "a confirmation; Allow = build without asking; Block = refuse (use tools/build_engines.py instead).")]
        public EngineBuildPolicy engineBuildPolicy = EngineBuildPolicy.Ask;

        [Header("Logging")]
        [Tooltip("Print status messages (phases, errors, warnings) to the Unity console with an [SD] prefix.")]
        public bool logToConsole = true;
        [Tooltip("Also print the Python server's own output (verbose).")]
        public bool logServerOutput = false;

        [Header("Server")]
        public string host = "127.0.0.1";
        public int port = 8765;
        public string token = "";

        [Header("Local server")]
        [Tooltip("Start the Python server ourselves when host is local and nothing is listening.")]
        public bool launchLocalServer = true;
        [Tooltip("Empty = LocalAppData/StreamDiffusionUnity/diffusion. Must not contain spaces.")]
        public string basefolder = "";
        [Tooltip("Let the installer download Python 3.11.9 / MinGit when missing (asks nothing: set only after the user agreed).")]
        public bool allowBootstrapDownloads = false;

        [Header("Video")]
        public int width = 512;
        public int height = 512;
        [Tooltip("Read the picture back from the GPU synchronously instead of asynchronously. Lower latency (the async " +
                 "readback delivers one to two frames later) at the cost of a short stall of the main thread each frame. " +
                 "Measured: about 12 ms less round-trip latency with SDXL-Turbo, same frame rate. On by default.")]
        public bool syncReadback = true;
        [Tooltip("Cap on frames sent per second (0 = every frame).")]
        public int maxSendFps = 60;
        [Tooltip("Unity GPU readback is bottom-up in memory. With this on, the server flips rows before the model " +
                 "and again on the result so the model sees an upright image. Leave on.")]
        public bool flipY = true;
        [Tooltip("Where the image sent to the model comes from.")]
        public DiffusionSourceMode sourceMode = DiffusionSourceMode.Texture;
        [Tooltip("Texture mode: any Texture (RenderTexture, video, webcam...), blitted into the send target each frame.")]
        public Texture sourceTexture;
        [Tooltip("Camera mode: this camera is rendered into the send target (it can keep rendering to the screen too).")]
        public Camera sourceCamera;

        [Header("Model (applied on load / reload)")]
        public string modelId = "stabilityai/sdxl-turbo";
        public string acceleration = "tensorrt";
        [Tooltip("Tiny VAE decoder: much faster, slightly softer. Changing it rebuilds an engine (minutes, cached).")]
        public bool useTinyVae = true;

        [Header("Advanced engine options (reload)")]
        [Tooltip("Img2Img follows the input picture. Txt2Img ignores the input and generates from the prompt alone " +
                 "(CFG type is then forced to None). Changing it builds a separate engine set.")]
        public DiffusionMode mode = DiffusionMode.Img2Img;
        [Tooltip("How the guidance scale is applied. Self (default) is the fastest stream-batch method; None disables " +
                 "guidance (needed for Txt2Img); Initialize and Full cost more per frame. Changes the engine batch size.")]
        public DiffusionCfgType cfgType = DiffusionCfgType.Self;
        [Tooltip("Lcm (default) or Tcd. Does not need a rebuild but is applied on reload.")]
        public DiffusionScheduler scheduler = DiffusionScheduler.Lcm;
        [Tooltip("How the noise schedule is spaced. Normal is the default; applied on reload, no rebuild.")]
        public DiffusionSampler sampler = DiffusionSampler.Normal;

        [Header("FX processors (live): image effects before / after the model")]
        [Tooltip("Applied to the input picture before the model. Order matters. Live: add, reorder and tune while streaming.")]
        public List<FxEntry> fxPre = new List<FxEntry>();
        [Tooltip("Applied to the result after the model. Order matters. Live.")]
        public List<FxEntry> fxPost = new List<FxEntry>();

        JObject FxPayload()
        {
            JArray Chain(List<FxEntry> list)
            {
                var arr = new JArray();
                if (list == null) return arr;
                foreach (var e in list)
                {
                    if (e == null || string.IsNullOrEmpty(e.type)) continue;
                    var p = new JObject();
                    foreach (var q in e.parameters)
                    {
                        if (q == null || string.IsNullOrEmpty(q.name)) continue;
                        var def = FxCatalog.Find(e.type);
                        float v = q.value;
                        if (def != null)
                            foreach (var d in def.parameters)
                                if (d.name == q.name)
                                {
                                    if (d.integer) v = Mathf.Round(v);
                                    if (d.odd) { int iv = Mathf.RoundToInt(v); if (iv % 2 == 0) iv++; v = iv; }
                                }
                        p[q.name] = v;
                    }
                    arr.Add(new JObject { ["type"] = e.type, ["enabled"] = e.enabled, ["params"] = p });
                }
                return arr;
            }
            return new JObject { ["pre"] = Chain(fxPre), ["post"] = Chain(fxPost) };
        }

        [Header("Windows GPU timeout")]
        [Tooltip("At the end of an install, ask Windows (UAC administrator prompt) to raise the GPU timeout (TdrDelay) " +
                 "from the 2 s default to 60 s so long TensorRT builds do not reset the GPU. Declining changes nothing. " +
                 "Command-line switch to turn it off: -sd-no-gpu-fix")]
        public bool fixGpuTimeoutDuringInstall = true;

        [Header("IP-Adapter (image prompt). The adapter is an engine option (reload); scale and the reference are live")]
        [Tooltip("Steers the result with the look of a reference image (style, subject, colours) in addition to the " +
                 "text prompt. Needs its own engine (cached) and the model family must match the adapter.")]
        public bool useIpAdapter = false;
        [Tooltip("IP-Adapter weights as 'repo/folder/file.bin' on Hugging Face, or a local path.")]
        public string ipAdapterModel = "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin";
        [Tooltip("The matching image encoder folder.")]
        public string ipAdapterEncoder = "h94/IP-Adapter/sdxl_models/image_encoder";
        [Range(0f, 2f)] public float ipAdapterScale = 0.7f;
        [Tooltip("Off keeps the adapter loaded but ignores the reference (live).")]
        public bool ipAdapterEnabled = true;
        [Tooltip("The reference image (any Texture, e.g. a RenderTexture or a picture). Encoded when it is sent.")]
        public Texture referenceTexture;
        [Tooltip("Resend the reference every N seconds (use it for a live reference). 0 = only at start and when " +
                 "'Update reference now' is pressed (UpdateReferenceNow() from a script).")]
        [Min(0f)] public float referenceRefreshSeconds = 0f;

        /// <summary>The reference image as sent to the server (for previews).</summary>
        public RenderTexture ReferenceSendTexture { get; private set; }
        bool _referenceRequested, _referenceSentOnce;
        float _nextReference;

        /// <summary>Send the reference image again (it is re-encoded once on the server).</summary>
        public void UpdateReferenceNow() => _referenceRequested = true;

        /// <summary>The IP-Adapter the loaded model was built with (null = nothing loaded yet).</summary>
        public string LoadedIpAdapterKey { get; private set; }
        /// <summary>"regular", "plus" or "faceid", read from the adapter file name (the server reads the real token count from the file).</summary>
        public string IpAdapterTypeName()
        {
            string m = (ipAdapterModel ?? "").ToLowerInvariant();
            return m.Contains("faceid") ? "faceid" : m.Contains("plus") ? "plus" : "regular";
        }

        public string IpAdapterKey() => useIpAdapter && !string.IsNullOrWhiteSpace(ipAdapterModel) ? ipAdapterModel.Trim() : "";

        [Header("ControlNet (structure guidance from the input picture). The SET is an engine option (reload); scale is live")]
        [Tooltip("Each entry steers the result with an edge / depth / pose map made from the input picture. Adding or " +
                 "removing one (or changing its preprocessor) needs 'Apply and reload' (engine build, cached); " +
                 "the scale and the on/off switch are live. The ControlNet must match the model family.")]
        public List<ControlNetEntry> controlNets = new List<ControlNetEntry>();
        [Tooltip("SameAsInput: ControlNets read the picture sent to the model. Texture: they read 'controlTexture' " +
                 "instead (e.g. a depth map from another camera); the model input stays the source above.")]
        public ControlSourceMode controlSource = ControlSourceMode.SameAsInput;
        [Tooltip("The separate control image (any Texture, e.g. a RenderTexture). Used when controlSource is Texture. " +
                 "Use the 'passthrough' preprocessor when it already is a depth / edge / pose map.")]
        public Texture controlTexture;

        [Tooltip("Show what the first ControlNet actually sees (the edge / depth / pose map made from the control " +
                 "image) in the output card. Costs a little GPU readback, so switch it off when not tuning.")]
        public bool showControlMap = false;

        /// <summary>The processed control image coming back from the server (for the preview).</summary>
        public RenderTexture ControlMapTexture { get; private set; }
        Texture2D _cpvTex;
        public bool ControlMapActive => showControlMap && controlNets != null && controlNets.Count > 0;

        /// <summary>The control image as sent to the server (for previews).</summary>
        public RenderTexture ControlSendTexture { get; private set; }
        public bool ControlSeparateActive => controlSource == ControlSourceMode.Texture && controlTexture != null;

        [Header("LoRA (style adapters). The SET is an engine option (reload); each weight is live")]
        [Tooltip("Files from <base>\\StreamDiffusion\\models\\LoRA. They must match the model family (SDXL LoRAs for SDXL). " +
                 "Adding or removing one needs 'Apply and reload' (engine rebuild, cached); weights change live.")]
        public List<LoraEntry> loras = new List<LoraEntry>();

        [Header("StreamV2V: temporal consistency (engine option: reload, then cached)")]
        [Tooltip("Cached attention across frames (StreamV2V): the picture stays coherent from frame to frame instead of " +
                 "flickering. Needs its own UNet engine (minutes, cached) and more VRAM; the resolution is fixed to the build.")]
        public bool useStreamV2V = false;
        [Tooltip("Feature injection: blends features of earlier frames into the current one (needs StreamV2V).")]
        public bool featureInjection = true;
        [Tooltip("How many frames of attention the engine can remember at most. Baked into the engine.")]
        [Range(1, 8)] public int v2vMaxCacheFrames = 4;

        [Header("StreamV2V live tuning (no rebuild)")]
        [Tooltip("How strongly previous frames are blended in (0 = not at all, 1 = fully).")]
        [Range(0f, 1f)] public float fiStrength = 0.75f;
        [Tooltip("Features are only injected when a frame is at least this similar to the cached one.")]
        [Range(0f, 1f)] public float fiThreshold = 0.98f;
        [Tooltip("Store a frame in the memory every N frames (higher = longer memory, less detail per frame).")]
        [Range(1, 10)] public int v2vCacheInterval = 1;
        [Tooltip("How many remembered frames the attention looks at (up to the engine maximum). 1 degrades the picture " +
                 "(measured): use 2 or more, 4 is a good start.")]
        [Range(1, 8)] public int v2vCacheFrames = 4;

        [Header("Prompt / parameters (live: edits apply immediately while running)")]
        public string negativePrompt = "";
        public float guidanceScale = 1.0f;
        public float delta = 1.0f;
        [Tooltip("Total size of the noise schedule the step positions index into. Advanced; leave at 50.")]
        public int numInferenceSteps = 50;

        [Header("Step schedule (the UI's STEP SCHEDULE card)")]
        [Tooltip("One entry per denoising step; each value (1-49) is that step's position in the noise-reduction " +
                 "sequence. LOWER = denoising starts sooner, more randomness (more 'AI'). HIGHER = the result stays " +
                 "closer to the input image. Positions change live. The NUMBER of entries changes the model's batch " +
                 "size and takes effect on 'Apply step count (reload model)' (button in this inspector or the UI).")]
        [UnityEngine.Serialization.FormerlySerializedAs("tIndexList")]
        [Range(1, 49)]
        public int[] stepSchedule = { 1 };

        /// <summary>Same array as <see cref="stepSchedule"/> under its StreamDiffusion name (t_index_list).</summary>
        public int[] tIndexList
        {
            get => stepSchedule;
            set => stepSchedule = value;
        }

        [Header("Prompts (live): first row = the main prompt; two or more rows are blended by weight")]
        public List<PromptBlendEntry> promptBlend = new List<PromptBlendEntry>
        {
            new PromptBlendEntry("a fancy banana, studio photo", 1f),
        };
        [Tooltip("How the prompts are mixed when there are two or more.")]
        public BlendInterpolation promptInterpolation = BlendInterpolation.Slerp;
        public bool normalizePromptWeights = true;

        [Header("Seeds (live): first row = the main seed; two or more rows are blended by weight")]
        public List<SeedBlendEntry> seedBlend = new List<SeedBlendEntry>
        {
            new SeedBlendEntry(2416333, 1f),
        };
        public BlendInterpolation seedInterpolation = BlendInterpolation.Average;
        public bool normalizeSeedWeights = true;

        [Header("NDI output (live; off by default, costs nothing while off)")]
        [Tooltip("Publish the generated stream as an NDI source so other software (Resolume, OBS, vMix...) can use it. " +
                 "Sent from the Python server at the model's resolution. Needs the free NDI Runtime installed.")]
        public bool ndiOutputEnabled = false;
        [Tooltip("The name other software shows for this source.")]
        public string ndiOutputName = "StreamDiffusion";

        [Header("Performance (live)")]
        [Tooltip("Skip inference when the new frame is nearly identical to the last one (saves GPU on static input).")]
        public bool similarFilterEnabled = false;
        [Range(0.5f, 1f)] public float similarFilterThreshold = 0.98f;
        [Range(1, 60)] public int similarFilterMaxSkip = 10;
        [Tooltip("Add noise to the source before denoising. Leave on unless you know you need it off.")]
        public bool addNoise = true;

        // ---- main prompt / seed = first list entry (single value in, blend when there are more) ----
        void EnsureLists()
        {
            if (promptBlend == null) promptBlend = new List<PromptBlendEntry>();
            if (promptBlend.Count == 0) promptBlend.Add(new PromptBlendEntry("", 1f));
            if (seedBlend == null) seedBlend = new List<SeedBlendEntry>();
            if (seedBlend.Count == 0) seedBlend.Add(new SeedBlendEntry(0, 1f));
        }

        /// <summary>The main prompt: first entry of <see cref="promptBlend"/>.</summary>
        public string prompt
        {
            get { EnsureLists(); return promptBlend[0].text ?? ""; }
            set { EnsureLists(); promptBlend[0].text = value ?? ""; }
        }

        /// <summary>The main seed: first entry of <see cref="seedBlend"/>.</summary>
        public int seed
        {
            get { EnsureLists(); return seedBlend[0].seed; }
            set { EnsureLists(); seedBlend[0].seed = value; }
        }

        /// <summary>True when two or more non-empty prompts are being blended.</summary>
        public bool PromptBlendActive
        {
            get
            {
                if (promptBlend == null) return false;
                int n = 0;
                foreach (var e in promptBlend) if (!string.IsNullOrWhiteSpace(e.text)) n++;
                return n >= 2;
            }
        }

        /// <summary>True when two or more seeds are being blended.</summary>
        public bool SeedBlendActive => seedBlend != null && seedBlend.Count >= 2;

        // ---- step schedule ----
        /// <summary>Number of denoising steps the loaded engine was built for (0 = nothing loaded).</summary>
        public int LoadedStepCount { get; private set; }

        /// <summary>The user changed the number of steps; a model reload is needed for it to take effect.</summary>
        public bool StepCountPending => LoadedStepCount > 0 && (tIndexList?.Length ?? 0) != LoadedStepCount;

        /// <summary>
        /// Positions to send live: while the step count differs from the loaded engine, only the
        /// first LoadedStepCount entries (the server ignores a count change until restart anyway).
        /// </summary>
        int[] EffectiveTIndexList()
        {
            var list = (tIndexList != null && tIndexList.Length > 0) ? tIndexList : new[] { 1 };
            if (LoadedStepCount > 0 && list.Length != LoadedStepCount)
            {
                var cut = new int[LoadedStepCount];
                for (int i = 0; i < cut.Length; i++) cut[i] = list[Mathf.Min(i, list.Length - 1)];
                return cut;
            }
            return list;
        }

        // ---- observable state ----
        public DiffusionPhase Phase { get; private set; } = DiffusionPhase.Disconnected;
        public string StatusMessage { get; private set; } = "";
        public string LastError { get; private set; }
        public RenderTexture ResultTexture { get; private set; }
        public RenderTexture SendTexture { get; private set; }
        public JObject Stats { get; private set; }
        public JObject ServerInfo { get; private set; }
        public float LastLatencyMs { get; private set; }
        public float ReceiveFps { get; private set; }
        public ulong FramesSent { get; private set; }
        public ulong FramesReceived { get; private set; }

        // ---- NDI output status (from the server) ----
        /// <summary>Server reports it can create NDI senders (runtime + library present).</summary>
        public bool NdiAvailable { get; private set; } = true;
        public string NdiUnavailableReason { get; private set; }
        public string NdiError { get; private set; }
        public int NdiConnections { get; private set; }
        public long NdiFrames { get; private set; }
        /// <summary>True when the server confirms the NDI source is live.</summary>
        public bool NdiLive { get; private set; }

        public event Action<string> Log;
        public event Action<DiffusionPhase> PhaseChanged;

        DiffusionControlClient _client;
        DiffusionInstaller _installer;
        DiffusionServerLauncher _launcher;
        CancellationTokenSource _installCts;
        IFrameTransport _transport;
        Texture2D _recvTex;
        readonly List<JObject> _inbox = new List<JObject>();
        float _nextSend;
        float _fpsWindowStart;
        int _fpsWindowCount;
        ulong _frameId;
        bool _applyingState;
        // last values pushed to the server, to detect inspector / script / UI edits
        string _sentPrompt, _sentNegative;
        string _sentParams;
        string _sentPromptKey;
        string _sentNdiKey;
        float _nextLiveCheck;

        // ======================================================================== environment + server

        public DiffusionInstallSettings InstallSettings { get; private set; }

        /// <summary>Installer for the Python environment (created on first use).</summary>
        public DiffusionInstaller GetInstaller()
        {
            // Follow the inspector: a changed base folder / download flag rebuilds the installer
            // (never while an install is running).
            bool stale = _installer != null && !_installer.IsRunning &&
                         (_installerBase != (basefolder ?? "") || _installerAllowDl != allowBootstrapDownloads);
            if (_installer == null || stale)
            {
                InstallSettings = DiffusionInstallSettings.Load();
                if (!string.IsNullOrWhiteSpace(basefolder)) InstallSettings.basefolder = basefolder;
                InstallSettings.allowBootstrapDownloads |= allowBootstrapDownloads;
                _installerBase = basefolder ?? "";
                _installerAllowDl = allowBootstrapDownloads;
                _installer = new DiffusionInstaller(InstallSettings);
                _installer.LogLine += l => Emit(l);
            }
            return _installer;
        }

        string _installerBase;
        bool _installerAllowDl;

        public InstallState EnvironmentState => GetInstaller().State;

        public async Task<bool> InstallEnvironmentAsync()
        {
            var inst = GetInstaller();
            _installCts?.Cancel();
            _installCts = new CancellationTokenSource();
            SetPhase(DiffusionPhase.Installing, "installing environment");
            bool ok = await inst.InstallAsync(_installCts.Token);
            SetPhase(ok ? DiffusionPhase.Disconnected : DiffusionPhase.Error,
                ok ? "environment installed" : "install failed: " + inst.LastError);
            if (ok) await GpuTimeoutInstallStepAsync(inst.Paths.Base);
            return ok;
        }

        public void CancelInstall() => _installCts?.Cancel();

        /// <summary>
        /// Last install step: the Windows GPU timeout. If it is still the short default, ask Windows for
        /// administrator permission to raise it (the UAC prompt is the confirmation; declining is fine and
        /// the Environment page and engine-build card keep offering it).
        /// </summary>
        async Task GpuTimeoutInstallStepAsync(string baseFolder)
        {
            if (!GpuTimeoutFix.IsWindows) return;
            var state = GpuTimeoutFix.GetState(baseFolder, out var seconds);
            Emit("GPU timeout: " + GpuTimeoutFix.Describe(state, seconds));
            if (state != GpuTimeoutState.Short) return;
            if (!fixGpuTimeoutDuringInstall)
            {
                Emit("GPU timeout: not changed automatically. Use 'Fix GPU timeout' on the Environment page, or run:" +
                     Environment.NewLine + GpuTimeoutFix.ManualCommands);
                return;
            }
            Emit("GPU timeout: asking Windows for administrator permission to raise it to " + GpuTimeoutFix.TargetSeconds + " s...");
            var (ok, message) = await GpuTimeoutFix.ApplyAsync(baseFolder);
            Emit("GPU timeout: " + message);
            if (!ok)
                Emit("GPU timeout: you can do it later from the Environment page, or run (administrator prompt):" +
                     Environment.NewLine + GpuTimeoutFix.ManualCommands);
        }

        bool IsLocalHost => host == "127.0.0.1" || host == "localhost";

        /// <summary>Make sure a server is reachable: start the local one if needed.</summary>
        public async Task EnsureLocalServerAsync()
        {
            if (!launchLocalServer || !IsLocalHost) return;
            var inst = GetInstaller();
            if (inst.State != InstallState.Installed)
                throw new InvalidOperationException("Python environment is not installed yet (use Install first).");
            string interrupted = EngineBuildDiagnostics.CheckInterrupted(inst.Paths);
            if (interrupted != null) Debug.LogWarning("[StreamDiffusion] " + interrupted);
            _launcher ??= new DiffusionServerLauncher();
            _launcher.LogLine -= OnServerLine;
            _launcher.LogLine += OnServerLine;
            SetPhase(DiffusionPhase.StartingServer, "starting local server");
            await _launcher.StartAsync(inst.Paths, host, port, string.IsNullOrEmpty(token) ? null : token, CancellationToken.None);
        }

        /// <summary>What a long operation (server start, model load, engine build) is doing, and how far along.</summary>
        public ActivityTracker Activity { get; } = new ActivityTracker();

        void OnServerLine(string line)
        {
            Activity.Feed(line);
            Emit("[server] " + line);
        }

        /// <summary>Console (if enabled) plus the <see cref="Log"/> event the UI subscribes to.</summary>
        void Emit(string message)
        {
            bool isServer = message.StartsWith("[server]");
            if (isServer ? logServerOutput : logToConsole) Debug.Log("[SD] " + message);
            Log?.Invoke(message);
        }

        /// <summary>Ask a server we started to exit cleanly, then make sure the process is gone.</summary>
        public async Task StopServerAsync()
        {
            try
            {
                if (_client != null && _client.State == ConnectionState.Connected && _launcher != null && _launcher.OwnsProcess)
                    await _client.RequestAsync("shutdown", null, 3000);
            }
            catch { }
            _launcher?.Stop();
        }

        // ======================================================================== public API

        /// <summary>Connect, configure video, load the model, set prompt/params and start streaming.</summary>
        public async Task ConnectAndStartAsync()
        {
            // Idempotent: a second click / call while a session is running or starting is ignored.
            switch (Phase)
            {
                case DiffusionPhase.Running:
                    return;
                case DiffusionPhase.Ready:
                    await StartAsync();
                    return;
                case DiffusionPhase.Installing:
                case DiffusionPhase.StartingServer:
                case DiffusionPhase.Connecting:
                case DiffusionPhase.Connected:
                case DiffusionPhase.Loading:
                case DiffusionPhase.AwaitingBuildConfirmation:
                    Emit("already starting; ignoring a second start request");
                    return;
            }
            try
            {
                await EnsureLocalServerAsync();
                await ConnectAsync();
                await ConfigureVideoAsync();
                await LoadModelAsync();
                await StartAsync();
            }
            catch (OperationCanceledException)
            {
                // closed on purpose (stop / disconnect / quit): not an error
                if (Phase != DiffusionPhase.Disconnected) SetPhase(DiffusionPhase.Disconnected, "cancelled");
            }
            catch (Exception ex)
            {
                Fail(ex);
                throw;
            }
        }

        public async Task ConnectAsync()
        {
            SetPhase(DiffusionPhase.Connecting, $"connecting to ws://{host}:{port}");
            _client ??= new DiffusionControlClient();
            await _client.ConnectAsync(host, port, string.IsNullOrEmpty(token) ? null : token, "unity");
            ServerInfo = _client.HelloAck;
            var cap = ServerInfo["capabilities"]?["ndi_output"];
            NdiAvailable = cap == null || (bool?)cap["available"] != false;
            NdiUnavailableReason = (string)cap?["reason"];
            SetPhase(DiffusionPhase.Connected, $"connected: {(string)ServerInfo["gpu"]}");
        }

        public async Task ConfigureVideoAsync()
        {
            var r = await _client.RequestAsync("configure_video", new JObject
            {
                ["transport"] = "shmem", ["width"] = width, ["height"] = height, ["channels"] = 4,
            });
            if ((string)r["type"] != "video_ready") throw new InvalidOperationException($"configure_video: {r}");

            _transport?.Dispose();
            _transport = new SharedMemTransport((string)r["in_name"], (string)r["out_name"], (string)r["ctl_name"], (string)r["ref_name"],
                (string)r["cpv_name"]);
            if (_transport is SharedMemTransport smt) smt.SyncReadback = syncReadback;
            _referenceSentOnce = false;
            AllocateTextures();
            Emit($"video: {_transport.Description}");
        }

        // ---- engine cache check ----
        /// <summary>Set while a build is waiting for confirmation (UI / inspector / script answers it).</summary>
        // not serialized: Unity would otherwise hand back a non-null empty object after a domain reload
        [field: NonSerialized] public EngineStatus PendingBuild { get; private set; }
        [field: NonSerialized] public EngineStatus LastEngineStatus { get; private set; }
        TaskCompletionSource<bool> _buildDecision;
        string _confirmedEngineKey;
        EngineStatus _buildToRecord;   // set when this load will build engines; consumed by LoadModelAsync

        string EngineKey()
        {
            int steps = stepSchedule != null && stepSchedule.Length > 0 ? stepSchedule.Length : 1;
            return $"{modelId}|{width}x{height}|{steps}|{useTinyVae}|{acceleration}|{V2VKey()}|{mode}|{cfgType}";
        }

        /// <summary>Everything that changes the UNet engine besides the size: StreamV2V and its options.</summary>
        public string V2VKey() => (useStreamV2V ? $"v2v-{featureInjection}-{v2vMaxCacheFrames}" : "plain") + "|" + LoraSetKey()
                                  + "|" + ControlNetSetKey() + "|" + IpAdapterKey();

        /// <summary>Which ControlNets and preprocessors are active (scales excluded: they are live).</summary>
        public string ControlNetSetKey()
        {
            if (controlNets == null || controlNets.Count == 0) return "";
            var parts = new List<string>();
            foreach (var c in controlNets)
                if (c != null && !string.IsNullOrWhiteSpace(c.modelId)) parts.Add(c.modelId.Trim() + ":" + c.preprocessor);
            parts.Sort(StringComparer.Ordinal);
            return string.Join(";", parts);
        }

        /// <summary>The ControlNet set the loaded model was built with (null = nothing loaded yet).</summary>
        public string LoadedControlNetSetKey { get; private set; }

        JArray ControlNetArray()
        {
            var arr = new JArray();
            if (controlNets == null) return arr;
            foreach (var c in controlNets)
            {
                if (c == null || string.IsNullOrWhiteSpace(c.modelId)) continue;
                var o = new JObject
                {
                    ["model_id"] = c.modelId.Trim(), ["preprocessor"] = c.preprocessor,
                    ["conditioning_scale"] = c.scale, ["enabled"] = c.enabled,
                };
                if (c.preprocessor == "canny")
                    o["preprocessor_params"] = new JObject { ["low_threshold"] = c.cannyLow, ["high_threshold"] = c.cannyHigh };
                arr.Add(o);
            }
            return arr;
        }

        /// <summary>Which LoRA files are active (weights excluded: they are live).</summary>
        public string LoraSetKey()
        {
            if (loras == null || loras.Count == 0) return "";
            var names = new List<string>();
            foreach (var l in loras) if (l != null && !string.IsNullOrWhiteSpace(l.path)) names.Add(l.path.Replace('\\', '/'));
            names.Sort(StringComparer.Ordinal);
            return string.Join(";", names);
        }

        /// <summary>The LoRA set the loaded model was built with (null = nothing loaded yet).</summary>
        public string LoadedLoraSetKey { get; private set; }

        public LoraFile[] AvailableLoras { get; private set; } = new LoraFile[0];
        public string LoraFolder { get; private set; }

        /// <summary>Ask the server which LoRA files are in its models/LoRA folder.</summary>
        public async Task<LoraFile[]> RefreshLorasAsync()
        {
            if (_client == null || _client.State != ConnectionState.Connected)
            {
                // not connected: a local install can be listed straight from disk
                if (IsLocalHost) ScanLorasLocal();
                return AvailableLoras;
            }
            var r = await _client.RequestAsync("list_loras", null, 10000);
            if ((string)r["type"] != "loras") return AvailableLoras;
            LoraFolder = (string)r["folder"];
            var list = new List<LoraFile>();
            if (r["loras"] is JArray a)
                foreach (var t in a)
                    list.Add(new LoraFile { name = (string)t["name"], path = (string)t["path"], sizeMb = (float?)t["size_mb"] ?? 0f });
            AvailableLoras = list.ToArray();
            return AvailableLoras;
        }

        string IpAdapterBuildArgs() =>
            IpAdapterKey().Length == 0 ? "" :
            " --ipadapter " + Quote(ipAdapterModel.Trim()) + " --ipadapter-encoder " + Quote((ipAdapterEncoder ?? "").Trim());

        string ControlNetBuildArgs()
        {
            var sb = new StringBuilder();
            if (controlNets != null)
                foreach (var c in controlNets)
                    if (c != null && !string.IsNullOrWhiteSpace(c.modelId))
                        sb.Append(" --controlnet ").Append(Quote(c.modelId.Trim() + "=" + c.preprocessor));
            return sb.ToString();
        }

        static DiffusionCfgType ParseCfg(string s)
        {
            foreach (DiffusionCfgType v in Enum.GetValues(typeof(DiffusionCfgType)))
                if (string.Equals(DiffusionOptionNames.Of(v), s, StringComparison.OrdinalIgnoreCase)) return v;
            return DiffusionCfgType.Self;
        }

        static DiffusionSampler ParseSampler(string s)
        {
            foreach (DiffusionSampler v in Enum.GetValues(typeof(DiffusionSampler)))
                if (string.Equals(DiffusionOptionNames.Of(v), s, StringComparison.OrdinalIgnoreCase)) return v;
            return DiffusionSampler.Normal;
        }

        public ModelFile[] AvailableModels { get; private set; } = new ModelFile[0];
        public string ModelFolder { get; private set; }

        /// <summary>List base models: from the server when connected, from disk for a local install otherwise.</summary>
        public async Task<ModelFile[]> RefreshModelsAsync()
        {
            if (_client != null && _client.State == ConnectionState.Connected)
            {
                var r = await _client.RequestAsync("list_models", null, 10000);
                if ((string)r["type"] == "models")
                {
                    ModelFolder = (string)r["folder"];
                    var list = new List<ModelFile>();
                    if (r["models"] is JArray a)
                        foreach (var t in a)
                            list.Add(new ModelFile
                            {
                                name = (string)t["name"], path = (string)t["path"],
                                kind = (string)t["kind"], sizeMb = (float?)t["size_mb"] ?? 0f,
                            });
                    AvailableModels = list.ToArray();
                }
                return AvailableModels;
            }
            if (IsLocalHost) ScanModelsLocal();
            return AvailableModels;
        }

        void ScanModelsLocal()
        {
            var list = new List<ModelFile>();
            try
            {
                string dir = GetInstaller().Paths.ModelDir;
                ModelFolder = dir;
                if (Directory.Exists(dir))
                {
                    foreach (var d in Directory.GetDirectories(dir, "*", SearchOption.AllDirectories))
                    {
                        if (!File.Exists(Path.Combine(d, "model_index.json"))) continue;
                        long size = 0;
                        foreach (var f in Directory.GetFiles(d, "*", SearchOption.AllDirectories)) size += new FileInfo(f).Length;
                        list.Add(new ModelFile
                        {
                            name = d.Substring(dir.Length).TrimStart(Path.DirectorySeparatorChar, '/').Replace(Path.DirectorySeparatorChar, '/'),
                            path = d.Replace(Path.DirectorySeparatorChar, '/'), kind = "folder", sizeMb = size / (1024f * 1024f),
                        });
                    }
                    foreach (var f in Directory.GetFiles(dir, "*", SearchOption.AllDirectories))
                    {
                        string ext = Path.GetExtension(f).ToLowerInvariant();
                        if (ext != ".safetensors" && ext != ".ckpt") continue;
                        bool insideFolderModel = false;
                        foreach (var m in list) if (m.kind == "folder" && f.Replace(Path.DirectorySeparatorChar, '/').StartsWith(m.path + "/")) insideFolderModel = true;
                        if (insideFolderModel) continue;
                        list.Add(new ModelFile
                        {
                            name = f.Substring(dir.Length).TrimStart(Path.DirectorySeparatorChar, '/').Replace(Path.DirectorySeparatorChar, '/'),
                            path = f.Replace(Path.DirectorySeparatorChar, '/'), kind = "file",
                            sizeMb = new FileInfo(f).Length / (1024f * 1024f),
                        });
                    }
                }
            }
            catch { }
            AvailableModels = list.ToArray();
        }

        /// <summary>Opens the base-model folder in Explorer (created if missing). Local installs only.</summary>
        public string OpenModelFolder()
        {
            if (!IsLocalHost) return "The server is on another machine: put the files in its models/Model folder.";
            try
            {
                string dir = GetInstaller().Paths.ModelDir;
                Directory.CreateDirectory(dir);
                System.Diagnostics.Process.Start("explorer.exe", "\"" + Path.GetFullPath(dir) + "\"");
                return null;
            }
            catch (Exception ex) { return "Could not open the folder: " + ex.Message; }
        }

        // ---- Windows GPU timeout (TdrDelay): see GpuTimeoutFix ----
        public static int? ReadTdrDelaySeconds() => GpuTimeoutFix.ReadTdrDelaySeconds();

        /// <summary>True when the GPU timeout is still the short Windows default.</summary>
        public static bool GpuTimeoutIsShort() => GpuTimeoutFix.GetState(DiffusionPaths.DefaultBase()) == GpuTimeoutState.Short;

        /// <summary>State of the Windows GPU timeout for this install (cached for a few seconds).</summary>
        public GpuTimeoutState GpuTimeout => GpuTimeoutFix.GetState(GetInstaller().Paths.Base);

        /// <summary>
        /// Raise the Windows GPU timeout to 60 s. Windows shows its administrator prompt; nothing happens without it.
        /// </summary>
        public async Task<string> FixGpuTimeoutAsync()
        {
            var (ok, message) = await GpuTimeoutFix.ApplyAsync(GetInstaller().Paths.Base);
            Emit("GPU timeout: " + message);
            return message;
        }

        /// <summary>Opens the LoRA folder in Explorer (created if missing). Local installs only.</summary>
        public string OpenLoraFolder()
        {
            if (!IsLocalHost) return "The server is on another machine: put the files in its models/LoRA folder.";
            try
            {
                string dir = GetInstaller().Paths.LoraDir;
                Directory.CreateDirectory(dir);
                System.Diagnostics.Process.Start("explorer.exe", "\"" + Path.GetFullPath(dir) + "\"");
                return null;
            }
            catch (Exception ex) { return "Could not open the folder: " + ex.Message; }
        }

        void ScanLorasLocal()
        {
            var list = new List<LoraFile>();
            try
            {
                string dir = GetInstaller().Paths.LoraDir;
                LoraFolder = dir;
                if (Directory.Exists(dir))
                    foreach (var f in Directory.GetFiles(dir, "*", SearchOption.AllDirectories))
                    {
                        string ext = Path.GetExtension(f).ToLowerInvariant();
                        if (ext != ".safetensors" && ext != ".pt" && ext != ".ckpt" && ext != ".bin") continue;
                        list.Add(new LoraFile
                        {
                            name = f.Substring(dir.Length).TrimStart('\\', '/').Replace('\\', '/'),
                            path = f.Replace('\\', '/'),
                            sizeMb = new FileInfo(f).Length / (1024f * 1024f),
                        });
                    }
            }
            catch { }
            AvailableLoras = list.ToArray();
        }

        JObject V2VLoadFields(JObject o)
        {
            o["mode"] = DiffusionOptionNames.Of(mode);
            o["cfg_type"] = DiffusionOptionNames.Of(mode == DiffusionMode.Txt2Img ? DiffusionCfgType.None : cfgType);
            o["scheduler"] = DiffusionOptionNames.Of(scheduler);
            o["sampler"] = DiffusionOptionNames.Of(sampler);
            var ld = new JObject();
            if (loras != null)
                foreach (var l in loras)
                    if (l != null && !string.IsNullOrWhiteSpace(l.path)) ld[l.path.Replace('\\', '/')] = l.weight;
            if (ld.Count > 0) o["lora_dict"] = ld;
            var cns = ControlNetArray();
            if (cns.Count > 0) { o["use_controlnet"] = true; o["controlnets"] = cns; }
            if (IpAdapterKey().Length > 0)
            {
                o["use_ipadapter"] = true;
                o["ipadapters"] = new JArray(new JObject
                {
                    ["ipadapter_model_path"] = ipAdapterModel.Trim().Replace('\\', '/'),
                    ["image_encoder_path"] = (ipAdapterEncoder ?? "").Trim().Replace('\\', '/'),
                    ["scale"] = ipAdapterScale, ["enabled"] = ipAdapterEnabled, ["type"] = IpAdapterTypeName(),
                });
            }

            o["use_cached_attn"] = useStreamV2V;
            if (useStreamV2V)
            {
                o["use_feature_injection"] = featureInjection;
                o["max_cache_maxframes"] = v2vMaxCacheFrames;
                o["cache_maxframes"] = Mathf.Min(v2vCacheFrames, v2vMaxCacheFrames);
                o["cache_interval"] = v2vCacheInterval;
                o["fi_strength"] = fiStrength;
                o["fi_threshold"] = fiThreshold;
            }
            return o;
        }

        /// <summary>Ask the server whether the current setup would need to build TensorRT engines.</summary>
        public async Task<EngineStatus> QueryEngineStatusAsync()
        {
            if (_client == null || _client.State != ConnectionState.Connected) return null;
            int steps = stepSchedule != null && stepSchedule.Length > 0 ? stepSchedule.Length : 1;
            var r = await _client.RequestAsync("engine_status", V2VLoadFields(new JObject
            {
                ["model_id"] = modelId, ["width"] = width, ["height"] = height, ["steps"] = steps,
                ["use_tiny_vae"] = useTinyVae, ["acceleration"] = acceleration,
            }), 10000);
            if ((string)r["type"] != "engine_status") return null;
            var st = new EngineStatus
            {
                cached = (bool?)r["cached"] ?? true,
                missing = r["missing"] != null ? r["missing"].ToObject<string[]>() : new string[0],
                estimateMinutes = (float?)r["estimate_minutes"] ?? 0f,
                detail = (string)r["detail"] ?? "",
                width = (int?)r["width"] ?? width, height = (int?)r["height"] ?? height, steps = (int?)r["steps"] ?? steps,
            };
            LastEngineStatus = st;
            return st;
        }

        async Task EnsureEnginesAsync()
        {
            string key = EngineKey();
            if (key == _confirmedEngineKey) return;
            var st = await QueryEngineStatusAsync();
            if (st == null || st.cached) { _confirmedEngineKey = key; return; }

            string what = $"{st.width}x{st.height}, {st.steps} step(s): {st.MissingText} (about {st.estimateMinutes:0} min)";
            if (engineBuildPolicy == EngineBuildPolicy.Allow)
            {
                Emit("building TensorRT engines without asking (policy Allow): " + what);
                _buildToRecord = st;
                _confirmedEngineKey = key;
                return;
            }
            if (engineBuildPolicy == EngineBuildPolicy.Block)
                throw new EngineBuildBlockedException(
                    "TensorRT engines for this setup are not built yet (" + what + "). Engine Build Policy is Block: " +
                    "build them with tools/build_engines.py (Unity closed), or choose a cached setup.");

            PendingBuild = st;
            _buildDecision = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            SetPhase(DiffusionPhase.AwaitingBuildConfirmation, "engines must be built first: " + what);
            bool ok;
            try { ok = await _buildDecision.Task; }
            finally { PendingBuild = null; _buildDecision = null; }
            if (!ok) throw new OperationCanceledException("engine build declined");
            _buildToRecord = st;
            _confirmedEngineKey = key;
        }

        /// <summary>Go ahead with the pending engine build.</summary>
        public void ConfirmEngineBuild() => _buildDecision?.TrySetResult(true);

        /// <summary>Do not build; the load is abandoned and nothing changes on the server.</summary>
        public void DeclineEngineBuild() => _buildDecision?.TrySetResult(false);

        /// <summary>
        /// Writes a .bat that builds the engines for the current setup with this app closed (safer
        /// than building next to the Editor). Returns its path.
        /// </summary>
        public string CreateEngineBuildScript()
        {
            var paths = GetInstaller().Paths;
            string server = DiffusionPaths.ServerScriptDir();
            if (server == null) throw new InvalidOperationException("diffusion-server was not found next to the application");
            int steps = stepSchedule != null && stepSchedule.Length > 0 ? stepSchedule.Length : 1;
            string logs = Path.Combine(paths.Base, "logs");
            Directory.CreateDirectory(logs);
            string bat = Path.Combine(paths.Base, $"build-engines-{width}x{height}-{steps}step.bat");

            var sb = new StringBuilder();
            sb.AppendLine("@echo off");
            sb.AppendLine("echo Builds the TensorRT engines this StreamDiffusionUnity setup needs.");
            sb.AppendLine("echo Close Unity and other GPU-heavy programs first: the GPU stays busy for several minutes.");
            sb.AppendLine("pause");
            string tool = Path.Combine(server, "tools", "build_engines.py");
            string monitor = Path.Combine(logs, "engine-build-gpu.csv");
            string noTiny = (useTinyVae ? "" : " --no-tiny-vae") +
                            " --mode " + DiffusionOptionNames.Of(mode) + ControlNetBuildArgs() + IpAdapterBuildArgs() +
                            " --cfg-type " + DiffusionOptionNames.Of(mode == DiffusionMode.Txt2Img ? DiffusionCfgType.None : cfgType) +
                            " --scheduler " + DiffusionOptionNames.Of(scheduler) +
                            " --sampler " + DiffusionOptionNames.Of(sampler) +
                            (useStreamV2V ? " --v2v" + (featureInjection ? "" : " --no-fi") + " --max-cache-frames " + v2vMaxCacheFrames : "");
            sb.AppendLine(Quote(paths.VenvPython) + " " + Quote(tool) +
                          " --basefolder " + Quote(paths.RepoDir) +
                          " --model " + Quote(modelId) +
                          " --width " + width + " --height " + height + " --steps " + steps + noTiny +
                          " --monitor-file " + Quote(monitor));
            sb.AppendLine("echo.");
            sb.AppendLine("echo Done. You can start Unity again.");
            sb.AppendLine("pause");
            File.WriteAllText(bat, sb.ToString());
            return bat;
        }

        static string Quote(string s) => "\"" + s + "\"";

        public async Task LoadModelAsync()
        {
            await EnsureEnginesAsync();
            StartBuildRecording();
            try { await LoadModelCoreAsync(); }
            catch (Exception e) { EndBuildRecording(false, e.GetType().Name + ": " + e.Message); throw; }
            EndBuildRecording(true, "engines built and model loaded");
        }

        bool _recording;

        /// <summary>Starts the GPU flight recorder when this load will build engines (no-op otherwise).</summary>
        void StartBuildRecording()
        {
            if (_buildToRecord == null || _recording) return;
            int bs = _buildToRecord.steps;
            _buildToRecord = null;
            string csv = EngineBuildDiagnostics.Begin(GetInstaller().Paths, modelId, width, height, bs, useTinyVae,
                engineBuildPolicy.ToString(), _launcher?.LogFile, Application.unityVersion,
                Directory.GetParent(Application.dataPath)?.FullName,
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Unity", "Editor", "Editor.log"));
            _recording = csv != null;
            if (_recording) Emit("engine build: GPU recorder running -> " + csv);
        }

        void EndBuildRecording(bool ok, string message)
        {
            if (!_recording) return;
            _recording = false;
            EngineBuildDiagnostics.End(ok, message);
        }

        async Task LoadModelCoreAsync()
        {
            SetPhase(DiffusionPhase.Loading, "loading model (first time builds TensorRT engines, can take 20+ minutes)");
            var r = await _client.RequestAsync("load_model", V2VLoadFields(new JObject
            {
                ["model_id"] = modelId, ["width"] = width, ["height"] = height, ["acceleration"] = acceleration,
                ["use_tiny_vae"] = useTinyVae,
                ["t_index_list"] = new JArray(tIndexList),
                ["guidance_scale"] = guidanceScale, ["delta"] = delta,
                ["num_inference_steps"] = numInferenceSteps, ["seed"] = seed,
                ["prompt"] = prompt, ["negative_prompt"] = negativePrompt,
            }), timeoutMs: 3 * 60 * 60 * 1000);
            if ((string)r["type"] == "error") throw new InvalidOperationException($"load_model: {r["message"]}");
            LoadedStepCount = tIndexList != null && tIndexList.Length > 0 ? tIndexList.Length : 1;
            LoadedLoraSetKey = LoraSetKey();
            LoadedControlNetSetKey = ControlNetSetKey();
            LoadedIpAdapterKey = IpAdapterKey();
            _referenceSentOnce = false;               // the new model has not seen the reference yet
            await ApplyStateAsync();
            SetPhase(DiffusionPhase.Ready, $"model ready ({r["seconds"]} s)");
        }

        public async Task StartAsync()
        {
            var r = await _client.RequestAsync("start");
            if ((string)r["type"] == "error") throw new InvalidOperationException($"start: {r["message"]}");
            SetPhase(DiffusionPhase.Running, "running");
        }

        public async Task StopAsync()
        {
            if (_client == null || _client.State != ConnectionState.Connected) return;
            await _client.RequestAsync("stop");
            SetPhase(DiffusionPhase.Ready, "stopped");
        }

        public async Task DisconnectAsync()
        {
            _transport?.Dispose();
            _transport = null;
            if (_client != null) await _client.DisconnectAsync();
            SetPhase(DiffusionPhase.Disconnected, "disconnected");
        }

        /// <summary>Push the current prompt and live parameters to the server.</summary>
        public async Task ApplyStateAsync()
        {
            if (_client == null || _client.State != ConnectionState.Connected || _applyingState) return;
            _applyingState = true;
            _sentPromptKey = PromptKey(); _sentParams = ParamsKey();
            try
            {
                await _client.RequestAsync("set_prompt", PromptPayload());
                var pl = ParamsPayload();
                pl["max_fps"] = 0;
                await _client.RequestAsync("set_params", new JObject { ["params"] = pl });
                SendNdi();
            }
            finally { _applyingState = false; }
        }

        /// <summary>
        /// Re-apply the model / resolution settings: stops, reconfigures video if the size
        /// changed, reloads the model (first time per setting builds engines) and restarts.
        /// </summary>
        public async Task ReloadModelAsync()
        {
            bool wasRunning = Phase == DiffusionPhase.Running;
            if (_client == null || _client.State != ConnectionState.Connected)
            {
                await ConnectAndStartAsync();
                return;
            }
            var before = Phase;
            try
            {
                await EnsureEnginesAsync();   // may pause for confirmation; declining leaves everything as it was
                StartBuildRecording();         // record from before the old model is released
                // Quiesce: no frames are sent (Update only sends while Running), the stream is stopped and
                // the old model is fully released, so a build never runs next to a resident pipeline.
                SetPhase(DiffusionPhase.Loading, "releasing the running model");
                await _client.RequestAsync("stop");
                var un = await _client.RequestAsync("unload", null, 120000);
                if ((string)un["type"] == "ack" && un["torch_before_gb"] != null)
                    Emit($"model released: this process {(string)un["torch_before_gb"]} -> {(string)un["torch_after_gb"]} GB, " +
                         $"whole GPU {(string)un["gpu_before_gb"]} -> {(string)un["gpu_after_gb"]} GB");
                await ConfigureVideoAsync();
                await LoadModelAsync();
                await StartAsync();
            }
            catch (OperationCanceledException)
            {
                EndBuildRecording(false, "cancelled");
                SetPhase(before == DiffusionPhase.AwaitingBuildConfirmation ? DiffusionPhase.Ready : before,
                         "engine build cancelled; nothing was changed");
            }
            catch (EngineBuildBlockedException ex)
            {
                EndBuildRecording(false, ex.Message);
                // refused before anything was touched: report it, keep streaming with the current model
                LastError = ex.Message;
                Emit(ex.Message);
                SetPhase(before, "engine build blocked by policy; nothing was changed");
            }
            catch (Exception ex) { EndBuildRecording(false, ex.Message); Fail(ex); throw; }
        }

        // ---- presets ----
        public DiffusionPreset CapturePreset()
        {
            return new DiffusionPreset
            {
                modelId = modelId, acceleration = acceleration, useTinyVae = useTinyVae, width = width, height = height,
                prompt = prompt, negativePrompt = negativePrompt,
                promptBlendEnabled = PromptBlendActive,
                promptBlend = promptBlend.ConvertAll(e => new PromptBlendEntry(e.text, e.weight)),
                promptInterpolation = promptInterpolation, normalizePromptWeights = normalizePromptWeights,
                guidanceScale = guidanceScale, delta = delta, numInferenceSteps = numInferenceSteps,
                tIndexList = (int[])tIndexList.Clone(), seed = seed,
                seedBlendEnabled = SeedBlendActive,
                seedBlend = seedBlend.ConvertAll(e => new SeedBlendEntry(e.seed, e.weight)),
                seedInterpolation = seedInterpolation, normalizeSeedWeights = normalizeSeedWeights,
                similarFilterEnabled = similarFilterEnabled, similarFilterThreshold = similarFilterThreshold,
                similarFilterMaxSkip = similarFilterMaxSkip, addNoise = addNoise, maxSendFps = maxSendFps,
                loras = loras.ConvertAll(l => new LoraEntry(l.path, l.weight)),
                fxPre = fxPre.ConvertAll(e => e.Clone()), fxPost = fxPost.ConvertAll(e => e.Clone()),
                useIpAdapter = useIpAdapter, ipAdapterModel = ipAdapterModel, ipAdapterEncoder = ipAdapterEncoder,
                ipAdapterScale = ipAdapterScale, ipAdapterEnabled = ipAdapterEnabled,
                referenceRefreshSeconds = referenceRefreshSeconds,
                controlNets = controlNets.ConvertAll(c => new ControlNetEntry(c.modelId, c.preprocessor, c.scale)
                    { enabled = c.enabled, cannyLow = c.cannyLow, cannyHigh = c.cannyHigh }),
                mode = DiffusionOptionNames.Of(mode), cfgType = DiffusionOptionNames.Of(cfgType),
                scheduler = DiffusionOptionNames.Of(scheduler), sampler = DiffusionOptionNames.Of(sampler),
                useStreamV2V = useStreamV2V, featureInjection = featureInjection, v2vMaxCacheFrames = v2vMaxCacheFrames,
                fiStrength = fiStrength, fiThreshold = fiThreshold, v2vCacheInterval = v2vCacheInterval, v2vCacheFrames = v2vCacheFrames,
            };
        }

        /// <summary>Apply a preset. Live values take effect immediately; model/size need ReloadModelAsync.</summary>
        public void ApplyPreset(DiffusionPreset p)
        {
            if (p == null) return;
            modelId = p.modelId; acceleration = p.acceleration; useTinyVae = p.useTinyVae; width = p.width; height = p.height;
            negativePrompt = p.negativePrompt;
            // Older presets stored a single prompt + an on/off flag; the list is the source of truth now.
            var pl = (p.promptBlendEnabled && p.promptBlend != null && p.promptBlend.Count > 0)
                ? p.promptBlend.ConvertAll(e => new PromptBlendEntry(e.text, e.weight))
                : new List<PromptBlendEntry> { new PromptBlendEntry(p.prompt, 1f) };
            promptBlend = pl;
            promptInterpolation = p.promptInterpolation; normalizePromptWeights = p.normalizePromptWeights;
            guidanceScale = p.guidanceScale; delta = p.delta; numInferenceSteps = p.numInferenceSteps;
            tIndexList = p.tIndexList != null && p.tIndexList.Length > 0 ? (int[])p.tIndexList.Clone() : new[] { 1 };
            seedBlend = (p.seedBlendEnabled && p.seedBlend != null && p.seedBlend.Count > 0)
                ? p.seedBlend.ConvertAll(e => new SeedBlendEntry(e.seed, e.weight))
                : new List<SeedBlendEntry> { new SeedBlendEntry(p.seed, 1f) };
            seedInterpolation = p.seedInterpolation; normalizeSeedWeights = p.normalizeSeedWeights;
            similarFilterEnabled = p.similarFilterEnabled; similarFilterThreshold = p.similarFilterThreshold;
            similarFilterMaxSkip = p.similarFilterMaxSkip; addNoise = p.addNoise; maxSendFps = p.maxSendFps;
            loras = p.loras != null ? p.loras.ConvertAll(l => new LoraEntry(l.path, l.weight)) : new List<LoraEntry>();
            fxPre = p.fxPre != null ? p.fxPre.ConvertAll(e => e.Clone()) : new List<FxEntry>();
            fxPost = p.fxPost != null ? p.fxPost.ConvertAll(e => e.Clone()) : new List<FxEntry>();
            useIpAdapter = p.useIpAdapter;
            if (!string.IsNullOrWhiteSpace(p.ipAdapterModel)) ipAdapterModel = p.ipAdapterModel;
            if (!string.IsNullOrWhiteSpace(p.ipAdapterEncoder)) ipAdapterEncoder = p.ipAdapterEncoder;
            ipAdapterScale = p.ipAdapterScale; ipAdapterEnabled = p.ipAdapterEnabled;
            referenceRefreshSeconds = p.referenceRefreshSeconds;
            controlNets = p.controlNets != null
                ? p.controlNets.ConvertAll(c => new ControlNetEntry(c.modelId, c.preprocessor, c.scale)
                    { enabled = c.enabled, cannyLow = c.cannyLow, cannyHigh = c.cannyHigh })
                : new List<ControlNetEntry>();
            mode = string.Equals(p.mode, "txt2img", StringComparison.OrdinalIgnoreCase) ? DiffusionMode.Txt2Img : DiffusionMode.Img2Img;
            cfgType = ParseCfg(p.cfgType);
            scheduler = string.Equals(p.scheduler, "tcd", StringComparison.OrdinalIgnoreCase) ? DiffusionScheduler.Tcd : DiffusionScheduler.Lcm;
            sampler = ParseSampler(p.sampler);
            useStreamV2V = p.useStreamV2V; featureInjection = p.featureInjection; v2vMaxCacheFrames = p.v2vMaxCacheFrames;
            fiStrength = p.fiStrength; fiThreshold = p.fiThreshold; v2vCacheInterval = p.v2vCacheInterval; v2vCacheFrames = p.v2vCacheFrames;
        }

        // ---- scripting API: set a value, the live-sync loop sends it ----
        public void SetPrompt(string value) { prompt = value; }
        public void SetNegativePrompt(string value) { negativePrompt = value; }
        public void SetGuidance(float value) { guidanceScale = value; }
        public void SetDelta(float value) { delta = value; }
        public void SetSteps(int value) { numInferenceSteps = value; }
        public void SetSeed(int value) { seed = value; }
        public void SetTIndexList(params int[] value) { tIndexList = value; }

        static string Interp(BlendInterpolation i) =>
            i == BlendInterpolation.Slerp ? "slerp" : i == BlendInterpolation.Average ? "average" : "cosine_weighted";

        string PromptKey()
        {
            var sb = new System.Text.StringBuilder();
            EnsureLists();
            sb.Append(negativePrompt).Append('\u001f').Append(promptInterpolation).Append(normalizePromptWeights);
            foreach (var e in promptBlend) sb.Append('\u001e').Append(e.text).Append('\u001f').Append(e.weight.ToString("R"));
            return sb.ToString();
        }

        string ParamsKey()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append(guidanceScale.ToString("R")).Append('|').Append(delta.ToString("R")).Append('|').Append(numInferenceSteps)
              .Append('|').Append(string.Join(",", EffectiveTIndexList())).Append('|').Append(flipY)
              .Append('|').Append(seedInterpolation).Append(normalizeSeedWeights)
              .Append('|').Append(similarFilterEnabled).Append(similarFilterThreshold.ToString("R")).Append(similarFilterMaxSkip)
              .Append('|').Append(addNoise)
              .Append('|').Append(fiStrength.ToString("R")).Append(fiThreshold.ToString("R"))
              .Append(v2vCacheInterval).Append('/').Append(v2vCacheFrames)
              .Append('|').Append(ControlSeparateActive).Append(ControlMapActive)
              .Append('|').Append(FxPayload().ToString(Newtonsoft.Json.Formatting.None))
              .Append('|').Append(ipAdapterScale.ToString("R")).Append(ipAdapterEnabled);
            if (loras != null)
                foreach (var l in loras) if (l != null) sb.Append('|').Append(l.path).Append('=').Append(l.weight.ToString("R"));
            if (controlNets != null)
                foreach (var c in controlNets)
                    if (c != null)
                        sb.Append('|').Append(c.modelId).Append('=').Append(c.scale.ToString("R")).Append(c.enabled)
                          .Append(c.cannyLow).Append('/').Append(c.cannyHigh);
            EnsureLists();
            foreach (var e in seedBlend) sb.Append(';').Append(e.seed).Append(':').Append(e.weight.ToString("R"));
            return sb.ToString();
        }

        JObject PromptPayload()
        {
            var o = new JObject { ["prompt"] = prompt, ["negative_prompt"] = negativePrompt };
            if (PromptBlendActive)
            {
                var arr = new JArray();
                foreach (var e in promptBlend)
                    if (!string.IsNullOrWhiteSpace(e.text))
                        arr.Add(new JObject { ["text"] = e.text, ["weight"] = e.weight });
                o["prompts"] = arr;
                o["interpolation"] = Interp(promptInterpolation);
                o["normalize"] = normalizePromptWeights;
            }
            else
            {
                o["prompts"] = null; // back to the single prompt
            }
            return o;
        }

        JObject ParamsPayload()
        {
            var seeds = new JArray();
            if (SeedBlendActive)
                foreach (var e in seedBlend) seeds.Add(new JObject { ["seed"] = e.seed, ["weight"] = e.weight });
            return new JObject
            {
                ["guidance_scale"] = guidanceScale, ["delta"] = delta, ["num_inference_steps"] = numInferenceSteps,
                ["t_index_list"] = new JArray(EffectiveTIndexList()), ["seed"] = seed, ["flip_y"] = flipY,
                ["seed_list"] = seeds,
                ["seed_interpolation"] = Interp(seedInterpolation),
                ["normalize_seed_weights"] = normalizeSeedWeights,
                ["similar_filter"] = new JObject
                {
                    ["enabled"] = similarFilterEnabled, ["threshold"] = similarFilterThreshold, ["max_skip"] = similarFilterMaxSkip,
                },
                ["add_noise"] = addNoise,
                ["fi_strength"] = fiStrength, ["fi_threshold"] = fiThreshold,
                ["cache_interval"] = v2vCacheInterval,
                ["cache_maxframes"] = Mathf.Min(v2vCacheFrames, v2vMaxCacheFrames),
                ["lora_weights"] = LoraWeights(),
                ["controlnets"] = ControlNetArray(),
                ["control_source"] = ControlSeparateActive ? "separate" : "input",
                ["fx"] = FxPayload(),
                ["control_preview"] = ControlMapActive,
                ["ipadapter"] = new JObject { ["scale"] = ipAdapterScale, ["enabled"] = ipAdapterEnabled },
            };
        }

        JObject LoraWeights()
        {
            var o = new JObject();
            if (loras != null)
                foreach (var l in loras)
                    if (l != null && !string.IsNullOrWhiteSpace(l.path)) o[l.path.Replace('\\', '/')] = l.weight;
            return o;
        }

        /// <summary>Send the prompt now (also called by the live-sync loop when it changed).</summary>
        public void SendPrompt()
        {
            _sentPromptKey = PromptKey();
            _client?.Send("set_prompt", PromptPayload());
        }

        /// <summary>Send the live parameters now (also called by the live-sync loop when they changed).</summary>
        public void SendParams()
        {
            _sentParams = ParamsKey();
            _client?.Send("set_params", new JObject { ["params"] = ParamsPayload() });
        }

        /// <summary>
        /// Inspector edits, scripts and UI all just change the public fields; this notices the
        /// difference (at most every 50 ms) and sends only what changed.
        /// </summary>
        string NdiKey() => ndiOutputEnabled + "|" + ndiOutputName;

        /// <summary>Send the NDI on/off + name now (also called by the live-sync loop).</summary>
        public void SendNdi()
        {
            _sentNdiKey = NdiKey();
            _client?.Send("set_ndi_output", new JObject
            {
                ["enabled"] = ndiOutputEnabled,
                ["name"] = string.IsNullOrWhiteSpace(ndiOutputName) ? "StreamDiffusion" : ndiOutputName.Trim(),
            });
        }

        void LiveSync()
        {
            if (_client == null || _client.State != ConnectionState.Connected) return;
            if (Time.unscaledTime < _nextLiveCheck) return;
            _nextLiveCheck = Time.unscaledTime + 0.05f;

            // NDI does not depend on the model: switch it as soon as we are connected
            if (NdiKey() != _sentNdiKey) SendNdi();

            if (Phase != DiffusionPhase.Ready && Phase != DiffusionPhase.Running) return;
            if (PromptKey() != _sentPromptKey) SendPrompt();
            if (ParamsKey() != _sentParams) SendParams();
        }

        // ======================================================================== startup / command line

        static string Arg(string name)
        {
            var a = Environment.GetCommandLineArgs();
            for (int i = 0; i < a.Length; i++)
                if (a[i] == name) return i + 1 < a.Length && !a[i + 1].StartsWith("-") ? a[i + 1] : "";
            return null;
        }

        async void Start()
        {
            bool install = false;
            if (applyCommandLine)
            {
                var v = Arg("-sd-host"); if (!string.IsNullOrEmpty(v)) host = v;
                v = Arg("-sd-port"); if (int.TryParse(v, out var p)) port = p;
                v = Arg("-sd-token"); if (!string.IsNullOrEmpty(v)) token = v;
                v = Arg("-sd-basefolder"); if (!string.IsNullOrEmpty(v)) basefolder = v;
                if (Arg("-sd-allow-downloads") != null) allowBootstrapDownloads = true;
                if (Arg("-sd-autostart") != null) autoStart = true;
                if (Arg("-sd-no-gpu-fix") != null) fixGpuTimeoutDuringInstall = false;
                install = Arg("-sd-install") != null;
            }

            if (!autoStart && !install) return;
            try
            {
                if (install || (autoInstallIfMissing && EnvironmentState != InstallState.Installed))
                {
                    if (!await InstallEnvironmentAsync()) return;
                }
                if (autoStart) await ConnectAndStartAsync();
            }
            catch (Exception ex) { Debug.LogError("[SD] start failed: " + ex.Message); }
        }

        // ======================================================================== frame loop

        void Update()
        {
            DrainServerMessages();
            LiveSync();
            if (_transport is SharedMemTransport live && live.SyncReadback != syncReadback) live.SyncReadback = syncReadback;
            if (Phase != DiffusionPhase.Running || _transport == null) return;

            // send
            if (Time.unscaledTime >= _nextSend && CaptureSource())
            {
                _nextSend = maxSendFps > 0 ? Time.unscaledTime + 1f / maxSendFps : 0f;
                ulong ts = UnixNowNs();
                if (_transport.Send(SendTexture, _frameId + 1, ts)) { _frameId++; FramesSent++; }
                if (ControlSeparateActive && ControlSendTexture != null)
                {
                    Graphics.Blit(controlTexture, ControlSendTexture);
                    _transport.SendControl(ControlSendTexture, _frameId, ts);
                }
            }

            // IP-Adapter reference: sent at start, on request, and every N seconds when set
            if (useIpAdapter && LoadedIpAdapterKey != null && LoadedIpAdapterKey.Length > 0 &&
                referenceTexture != null && ReferenceSendTexture != null)
            {
                bool periodic = referenceRefreshSeconds > 0f && Time.unscaledTime >= _nextReference;
                if (!_referenceSentOnce || _referenceRequested || periodic)
                {
                    Graphics.Blit(referenceTexture, ReferenceSendTexture);
                    if (_transport.SendReference(ReferenceSendTexture, _frameId, UnixNowNs()))
                    {
                        _referenceSentOnce = true;
                        _referenceRequested = false;
                        _nextReference = Time.unscaledTime + Mathf.Max(0.05f, referenceRefreshSeconds);
                    }
                }
            }

            // processed control map (preview)
            if (ControlMapActive && _cpvTex != null && _transport.TryReceiveControlMap(_cpvTex))
            {
                _cpvTex.Apply(false);
                Graphics.Blit(_cpvTex, ControlMapTexture);
            }

            // receive
            if (_transport.TryReceive(_recvTex, out ulong fid, out ulong tsNs))
            {
                _recvTex.Apply(false);
                Graphics.Blit(_recvTex, ResultTexture);
                FramesReceived++;
                LastLatencyMs = (UnixNowNs() - tsNs) / 1e6f;
                _fpsWindowCount++;
                float now = Time.unscaledTime;
                if (now - _fpsWindowStart >= 0.5f)
                {
                    ReceiveFps = _fpsWindowCount / (now - _fpsWindowStart);
                    _fpsWindowCount = 0;
                    _fpsWindowStart = now;
                }
            }
        }

        /// <summary>Fill SendTexture from the configured source. False if there is nothing to send.</summary>
        bool CaptureSource()
        {
            if (SendTexture == null) return false;
            if (sourceMode == DiffusionSourceMode.Camera)
            {
                if (sourceCamera == null) return false;
                var request = new RenderPipeline.StandardRequest { destination = SendTexture };
                if (RenderPipeline.SupportsRenderRequest(sourceCamera, request))
                {
                    RenderPipeline.SubmitRenderRequest(sourceCamera, request);
                }
                else
                {
                    var prev = sourceCamera.targetTexture;
                    sourceCamera.targetTexture = SendTexture;
                    sourceCamera.Render();
                    sourceCamera.targetTexture = prev;
                }
                return true;
            }
            if (sourceTexture == null) return false;
            Graphics.Blit(sourceTexture, SendTexture);
            return true;
        }

        void DrainServerMessages()
        {
            if (_client == null) return;
            _inbox.Clear();
            _client.Drain(_inbox);
            foreach (var m in _inbox)
            {
                switch ((string)m["type"])
                {
                    case "stats":
                        Stats = m;
                        NdiFrames = (long?)m["ndi_frames"] ?? 0;
                        NdiConnections = (int?)m["ndi_connections"] ?? 0;
                        break;
                    case "ndi_status":
                        NdiLive = (bool?)m["enabled"] == true;
                        NdiError = (string)m["error"];
                        if (!string.IsNullOrEmpty(NdiError))
                        {
                            Emit("NDI output: " + NdiError);
                            // the server could not start it: reflect that so we do not keep asking
                            if (!NdiLive) { ndiOutputEnabled = false; _sentNdiKey = NdiKey(); }
                        }
                        break;
                    case "progress": StatusMessage = (string)m["message"]; break;
                    case "activity":
                        Activity.SetGpu((float?)m["gpu_util"] ?? -1f, (float?)m["gpu_mem_gb"] ?? 0f, (float?)m["gpu_temp"] ?? 0f);
                        break;
                    case "warning": Emit("warning: " + (string)m["message"]); break;
                    case "error":
                        LastError = (string)m["message"];
                        Emit($"server error [{m["code"]}]: {LastError}");
                        break;
                    case "server_closing":
                        SetPhase(DiffusionPhase.Disconnected, "server closed");
                        break;
                }
            }
            if (_client.State == ConnectionState.Error && Phase != DiffusionPhase.Error && Phase != DiffusionPhase.Disconnected)
            {
                LastError = _client.LastError;
                SetPhase(DiffusionPhase.Error, "connection lost: " + _client.LastError);
            }
        }

        // ======================================================================== helpers

        void AllocateTextures()
        {
            ReleaseTextures();
            SendTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB)
            { name = "SD_Send", useMipMap = false, autoGenerateMips = false };
            SendTexture.Create();
            ResultTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB)
            { name = "SD_Result", useMipMap = false, autoGenerateMips = false };
            ResultTexture.Create();
            ControlSendTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB)
            { name = "SD_Control", useMipMap = false, autoGenerateMips = false };
            ControlSendTexture.Create();
            ReferenceSendTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB)
            { name = "SD_Reference", useMipMap = false, autoGenerateMips = false };
            ReferenceSendTexture.Create();
            ControlMapTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB)
            { name = "SD_ControlMap", useMipMap = false, autoGenerateMips = false };
            ControlMapTexture.Create();
            _cpvTex = new Texture2D(width, height, TextureFormat.RGBA32, false, linear: false) { name = "SD_ControlMapRecv" };
            _recvTex = new Texture2D(width, height, TextureFormat.RGBA32, false, linear: false) { name = "SD_Recv" };
        }

        void ReleaseTextures()
        {
            if (SendTexture != null) { SendTexture.Release(); Destroy(SendTexture); SendTexture = null; }
            if (ResultTexture != null) { ResultTexture.Release(); Destroy(ResultTexture); ResultTexture = null; }
            if (ControlSendTexture != null) { ControlSendTexture.Release(); Destroy(ControlSendTexture); ControlSendTexture = null; }
            if (ReferenceSendTexture != null) { ReferenceSendTexture.Release(); Destroy(ReferenceSendTexture); ReferenceSendTexture = null; }
            if (ControlMapTexture != null) { ControlMapTexture.Release(); Destroy(ControlMapTexture); ControlMapTexture = null; }
            if (_cpvTex != null) { Destroy(_cpvTex); _cpvTex = null; }
            if (_recvTex != null) { Destroy(_recvTex); _recvTex = null; }
        }

        void SetPhase(DiffusionPhase phase, string message)
        {
            Phase = phase;
            StatusMessage = message;
            if (phase == DiffusionPhase.StartingServer || phase == DiffusionPhase.Loading)
            {
                if (!Activity.Get().active) Activity.Begin(message);
                else Activity.SetTitle(message);
            }
            else Activity.End();
            Emit($"[{phase}] {message}");
            PhaseChanged?.Invoke(phase);
        }

        void Fail(Exception ex)
        {
            LastError = ex.Message;
            SetPhase(DiffusionPhase.Error, ex.Message);
        }

        static readonly long UnixEpochTicks = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).Ticks;
        static ulong UnixNowNs() => (ulong)(DateTime.UtcNow.Ticks - UnixEpochTicks) * 100UL;

        void Awake()
        {
            Application.quitting += OnQuitting;
        }

        void OnQuitting()
        {
            _launcher?.Stop();
        }

        void OnDestroy()
        {
            Application.quitting -= OnQuitting;
            _installCts?.Cancel();
            _launcher?.Dispose();
            _transport?.Dispose();
            _client?.Dispose();
            ReleaseTextures();
        }
    }
}
