using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>
    /// Sample helper: feeds the controller from an optional TestPatternSource, can draw a tiny
    /// IMGUI preview, and adds two test-harness command-line switches (the controller itself
    /// already handles -sd-host/-sd-port/-sd-autostart/-sd-install...):
    ///   -sd-report              log a status line every 3 s
    ///   -sd-screenshot &lt;path&gt;   save a screenshot once streaming has run for a while
    /// Runs before the controller so the source is set before an autoStart begins.
    /// </summary>
    [DefaultExecutionOrder(-50)]
    [RequireComponent(typeof(DiffusionController))]
    [AddComponentMenu("StreamDiffusion/Quick Start (sample)")]
    public sealed class DiffusionQuickStart : MonoBehaviour
    {
        public TestPatternSource pattern;
        public bool showPreview = false;

        DiffusionController _c;
        float _nextReport;
        bool _report, _shotDone;
        string _shotPath;

        static string Arg(string name)
        {
            var a = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < a.Length; i++)
                if (a[i] == name) return i + 1 < a.Length && !a[i + 1].StartsWith("-") ? a[i + 1] : "";
            return null;
        }

        void Start()
        {
            _c = GetComponent<DiffusionController>();
            if (pattern != null)
            {
                _c.sourceMode = DiffusionSourceMode.Texture;
                _c.sourceTexture = pattern.Output;
            }
            _report = Arg("-sd-report") != null || Arg("-sd-autostart") != null || Arg("-sd-install") != null;
            var shot = Arg("-sd-screenshot");
            if (!string.IsNullOrEmpty(shot)) _shotPath = shot;
        }

        void Update()
        {
            if (_c == null) return;
            if (_shotPath != null && !_shotDone && _c.Phase == DiffusionPhase.Running && Time.unscaledTime > 25f)
            {
                _shotDone = true;
                ScreenCapture.CaptureScreenshot(_shotPath);
                Debug.Log("[SD] screenshot -> " + _shotPath);
            }
            if (!_report || Time.unscaledTime < _nextReport) return;
            _nextReport = Time.unscaledTime + 3f;
            Debug.Log($"[SD] status phase={_c.Phase} sent={_c.FramesSent} recv={_c.FramesReceived} fps={_c.ReceiveFps:0.0} latency={_c.LastLatencyMs:0.0}ms" +
                      (_c.Stats != null ? $" server_infer={_c.Stats["infer_ms"]}ms drops={_c.Stats["queue_drops"]}" : ""));
        }

        void OnGUI()
        {
            if (!showPreview || _c == null || _c.ResultTexture == null) return;
            GUI.Label(new Rect(10, 6, 700, 20),
                $"{_c.Phase}  {_c.StatusMessage}   recv={_c.FramesReceived}  fps={_c.ReceiveFps:0.0}  latency={_c.LastLatencyMs:0.0} ms");
            GUI.DrawTexture(new Rect(10, 30, 512, 512), _c.ResultTexture, ScaleMode.ScaleToFit, false);
        }
    }
}
