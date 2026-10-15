using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace StreamDiffusion
{
    /// <summary>
    /// Turns the server's output into "what is it doing and how far along is it" for long operations
    /// (model download, TensorRT engine build, engine loading). Progress is derived from real
    /// milestones in the log; inside a long silent step (the TensorRT kernel search prints nothing
    /// for minutes) it is interpolated from typical durations and clearly marked as an estimate.
    /// Thread safe: server lines arrive on thread-pool threads, the UI reads on the main thread.
    /// </summary>
    public sealed class ActivityTracker
    {
        public struct Snapshot
        {
            public bool active;
            public string title;        // e.g. "Building TensorRT engine: unet"
            public string detail;       // e.g. "step 3 of 3 - optimising kernels (silent for a few minutes)"
            public float fraction;      // 0..1, or -1 when not known
            public bool estimated;      // true when the fraction is interpolated inside a silent step
            public float elapsed;       // seconds since the operation started
            public float sinceOutput;   // seconds since the server last printed anything
            public float gpuUtil;       // -1 unknown
            public float gpuMemGb;
            public float gpuTemp;
        }

        // typical seconds per stage for each engine (measured: SDXL-turbo 1024x512 on an RTX 5090)
        static readonly Dictionary<string, float[]> Typical = new Dictionary<string, float[]>
        {
            { "vae_decoder.engine", new[] { 6f, 1f, 84f } },
            { "vae_encoder.engine", new[] { 2f, 1f, 126f } },
            { "unet.engine",        new[] { 55f, 93f, 181f } },
        };
        const float LoadWeight = 12f;     // loading weights before the build
        const float FinalWeight = 8f;     // loading the finished engines

        static readonly Regex RxMissing = new Regex(@"-\s+(UNet|VAE decoder|VAE encoder) engine:", RegexOptions.Compiled);
        static readonly Regex RxExporting = new Regex(@"Exporting model:.*[\\/]([A-Za-z0-9_]+\.engine)\.onnx", RegexOptions.Compiled);
        static readonly Regex RxOnnxExport = new Regex(@"\[BUILD\] ONNX export \(([A-Za-z0-9_.]+)\)", RegexOptions.Compiled);
        static readonly Regex RxOnnxOpt = new Regex(@"\[BUILD\] ONNX optimize \(([A-Za-z0-9_.]+)\)", RegexOptions.Compiled);
        static readonly Regex RxComplete = new Regex(@"\[BUILD\] ([A-Za-z0-9_]+\.engine) complete", RegexOptions.Compiled);
        static readonly Regex RxTrtDone = new Regex(@"\[BUILD\] TRT engine build \(([A-Za-z0-9_.]+)\)", RegexOptions.Compiled);
        static readonly Regex RxPct = new Regex(@"(\d{1,3})%\|", RegexOptions.Compiled);
        static readonly Regex RxComponents = new Regex(@"Loading pipeline components.*?(\d+)/(\d+)", RegexOptions.Compiled);

        readonly object _lock = new object();
        bool _active;
        DateTime _started, _lastOutput;
        string _title = "", _detail = "";
        float _fraction = -1f;
        bool _estimated;
        readonly HashSet<string> _missing = new HashSet<string>();
        readonly HashSet<string> _done = new HashSet<string>();
        string _engine;           // currently building
        int _stage;               // 0 export, 1 optimise, 2 TRT build
        DateTime _stageStart;
        bool _downloading;
        float _downloadFraction = -1f;
        float _gpuUtil = -1f, _gpuMem, _gpuTemp;

        public void Begin(string title)
        {
            lock (_lock)
            {
                _active = true;
                _started = _lastOutput = DateTime.UtcNow;
                _title = title; _detail = ""; _fraction = -1f; _estimated = false;
                _missing.Clear(); _done.Clear();
                _engine = null; _downloading = false; _downloadFraction = -1f;
                _gpuUtil = -1f;
            }
        }

        public void End()
        {
            lock (_lock) { _active = false; }
        }

        public void SetTitle(string title, string detail = "")
        {
            lock (_lock) { _title = title; _detail = detail; }
        }

        public void SetGpu(float util, float memGb, float temp)
        {
            lock (_lock) { _gpuUtil = util; _gpuMem = memGb; _gpuTemp = temp; }
        }

        /// <summary>Feed one line of server output.</summary>
        public void Feed(string line)
        {
            if (string.IsNullOrEmpty(line)) return;
            lock (_lock)
            {
                _lastOutput = DateTime.UtcNow;
                if (!_active) return;

                Match m;
                if ((m = RxMissing.Match(line)).Success)
                {
                    _missing.Add(m.Groups[1].Value == "UNet" ? "unet.engine"
                        : m.Groups[1].Value == "VAE decoder" ? "vae_decoder.engine" : "vae_encoder.engine");
                    return;
                }
                if ((m = RxComplete.Match(line)).Success)
                {
                    _done.Add(m.Groups[1].Value);
                    if (_engine == m.Groups[1].Value) _engine = null;
                    _downloading = false;
                    return;
                }
                if ((m = RxExporting.Match(line)).Success)
                {
                    if (_engine != m.Groups[1].Value || _stage != 0) { _engine = m.Groups[1].Value; _stage = 0; _stageStart = DateTime.UtcNow; }
                    _downloading = false;
                    _missing.Add(_engine);
                    return;
                }
                if ((m = RxOnnxExport.Match(line)).Success) { _engine = m.Groups[1].Value; _stage = 1; _stageStart = DateTime.UtcNow; return; }
                if ((m = RxOnnxOpt.Match(line)).Success) { _engine = m.Groups[1].Value; _stage = 2; _stageStart = DateTime.UtcNow; return; }
                if (line.Contains("Loading TensorRT engine")) { _engine = null; _stage = 3; _detail = "loading engines into the GPU"; return; }

                // downloads: tqdm "Fetching 12 files:  45%|" / "Downloading ...  45%|"
                if ((line.Contains("Download") || line.Contains("Fetching")) && (m = RxPct.Match(line)).Success)
                {
                    _downloading = true;
                    _downloadFraction = Math.Min(100, int.Parse(m.Groups[1].Value)) / 100f;
                    return;
                }
                if ((m = RxComponents.Match(line)).Success)
                {
                    _detail = "loading model components " + m.Groups[1].Value + "/" + m.Groups[2].Value;
                }
            }
        }

        public Snapshot Get()
        {
            lock (_lock)
            {
                var s = new Snapshot { active = _active, gpuUtil = _gpuUtil, gpuMemGb = _gpuMem, gpuTemp = _gpuTemp };
                if (!_active) return s;
                var now = DateTime.UtcNow;
                s.elapsed = (float)(now - _started).TotalSeconds;
                s.sinceOutput = (float)(now - _lastOutput).TotalSeconds;
                s.title = _title;
                s.detail = _detail;
                s.fraction = -1f;

                if (_downloading)
                {
                    s.title = "Downloading model files";
                    s.detail = (_downloadFraction * 100f).ToString("0") + " %";
                    s.fraction = _downloadFraction;
                    return s;
                }

                if (_missing.Count > 0)
                {
                    float total = LoadWeight + FinalWeight, done = LoadWeight;
                    foreach (var e in _missing)
                    {
                        float w = EngineWeight(e);
                        total += w;
                        if (_done.Contains(e)) done += w;
                    }
                    if (_engine != null && _missing.Contains(_engine) && !_done.Contains(_engine) && Typical.TryGetValue(_engine, out var t))
                    {
                        for (int i = 0; i < _stage && i < 3; i++) done += t[i];
                        if (_stage < 3)
                        {
                            float into = (float)(now - _stageStart).TotalSeconds;
                            done += Math.Min(t[_stage] * 0.95f, into);
                            s.estimated = true;
                        }
                        int doneCount = _done.Count + 1;
                        string eng = _engine.Replace(".engine", "");
                        s.title = $"Building TensorRT engine {doneCount} of {_missing.Count}: {eng}";
                        s.detail = _stage == 0 ? "exporting the network (ONNX)"
                                 : _stage == 1 ? "optimising the graph"
                                 : "searching GPU kernels (prints nothing for minutes; this is normal)";
                    }
                    else if (_stage == 3)
                    {
                        done = total - FinalWeight / 2f;
                        s.title = "Loading engines";
                        s.detail = "loading the finished engines into the GPU";
                    }
                    else if (_done.Count == _missing.Count)
                    {
                        s.title = "Loading engines";
                        s.detail = "loading the finished engines into the GPU";
                        done = total - FinalWeight;
                    }
                    s.fraction = Math.Max(0f, Math.Min(0.99f, done / total));
                }
                return s;
            }
        }

        static float EngineWeight(string engine)
        {
            return Typical.TryGetValue(engine, out var t) ? t[0] + t[1] + t[2] : 60f;
        }
    }
}
