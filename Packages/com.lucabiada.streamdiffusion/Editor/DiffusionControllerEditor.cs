using UnityEditor;
using UnityEngine;

namespace StreamDiffusion.Editor
{
    /// <summary>
    /// Inspector for DiffusionController: the normal fields (every control the runtime panel has,
    /// plus a few it does not) with live status and buttons underneath, so a project can be
    /// operated and shipped with no UI at all.
    /// </summary>
    [CustomEditor(typeof(DiffusionController))]
    public sealed class DiffusionControllerEditor : UnityEditor.Editor
    {
        public override bool RequiresConstantRepaint() => Application.isPlaying;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var c = (DiffusionController)target;

            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Environment", EditorStyles.boldLabel);
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                var inst = c.GetInstaller();
                EditorGUILayout.LabelField("State", c.EnvironmentState.ToString());
                EditorGUILayout.LabelField("Base folder", inst.Paths.Base);
                using (new EditorGUI.DisabledScope(!Application.isPlaying && false))
                {
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        if (inst.IsRunning)
                        {
                            if (GUILayout.Button("Cancel install")) c.CancelInstall();
                        }
                        else if (GUILayout.Button(c.EnvironmentState == InstallState.Installed ? "Re-verify / repair" : "Install environment"))
                        {
                            _ = c.InstallEnvironmentAsync();
                        }
                    }
                }
                if (!string.IsNullOrEmpty(inst.LastError))
                    EditorGUILayout.HelpBox(inst.LastError, MessageType.Warning);
            }

            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Stream", EditorStyles.boldLabel);
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                if (!Application.isPlaying)
                {
                    EditorGUILayout.HelpBox("Enter Play mode to connect. Enable 'Auto Start' to stream as soon as the scene starts " +
                                            "(this is what a build with no UI uses).", MessageType.Info);
                    return;
                }

                EditorGUILayout.LabelField("Phase", c.Phase.ToString());
                if (!string.IsNullOrEmpty(c.StatusMessage)) EditorGUILayout.LabelField("Status", c.StatusMessage, EditorStyles.wordWrappedLabel);
                if (c.ServerInfo != null) EditorGUILayout.LabelField("GPU", (string)c.ServerInfo["gpu"]);
                if (c.Phase == DiffusionPhase.Running)
                {
                    EditorGUILayout.LabelField("Receive fps", c.ReceiveFps.ToString("0.0"));
                    EditorGUILayout.LabelField("Latency", c.LastLatencyMs.ToString("0") + " ms");
                    EditorGUILayout.LabelField("Frames sent / received", $"{c.FramesSent} / {c.FramesReceived}");
                }
                if (c.SendTexture != null && c.ResultTexture != null)
                {
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        GUILayout.FlexibleSpace();
                        using (new EditorGUILayout.VerticalScope(GUILayout.Width(160)))
                        {
                            GUILayout.Label("Input", EditorStyles.miniLabel);
                            GUILayout.Label(c.SendTexture, GUILayout.Width(160), GUILayout.Height(160));
                        }
                        using (new EditorGUILayout.VerticalScope(GUILayout.Width(160)))
                        {
                            GUILayout.Label("Result", EditorStyles.miniLabel);
                            GUILayout.Label(c.ResultTexture, GUILayout.Width(160), GUILayout.Height(160));
                        }
                        GUILayout.FlexibleSpace();
                    }
                }
                if (c.PendingBuild != null)
                {
                    var pb = c.PendingBuild;
                    EditorGUILayout.HelpBox(
                        $"Engine build needed for {pb.width}x{pb.height}, {pb.steps} step(s): {pb.MissingText}. " +
                        $"Roughly {pb.estimateMinutes:0} minutes of heavy GPU use. Building while StreamDiffusionUnity is open could reset the GPU " +
                        "driver; the safer way is the build script (run with StreamDiffusionUnity closed).",
                        MessageType.Warning);
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        if (GUILayout.Button("Build now")) c.ConfirmEngineBuild();
                        if (GUILayout.Button("Cancel")) c.DeclineEngineBuild();
                        if (GUILayout.Button("Create build script"))
                        {
                            string bat = c.CreateEngineBuildScript();
                            EditorUtility.RevealInFinder(bat);
                        }
                    }
                }
                if (c.StepCountPending)
                {
                    EditorGUILayout.HelpBox(
                        $"Step schedule has {c.stepSchedule.Length} step(s) but the loaded model was built for {c.LoadedStepCount}. " +
                        "The new step count takes effect after a model reload (the engine for a new count is built once, then cached).",
                        MessageType.Info);
                    if (GUILayout.Button("Apply step count (reload model)")) _ = c.ReloadModelAsync();
                }
                if (c.ndiOutputEnabled || c.NdiLive)
                    EditorGUILayout.LabelField("NDI output", c.NdiLive
                        ? $"live '{c.ndiOutputName}', {c.NdiConnections} receiver(s), {c.NdiFrames} frames"
                        : (c.NdiAvailable ? "starting..." : "runtime not available"));
                if (!string.IsNullOrEmpty(c.NdiError))
                    EditorGUILayout.HelpBox("NDI: " + c.NdiError, MessageType.Warning);
                if (!string.IsNullOrEmpty(c.LastError))
                    EditorGUILayout.HelpBox(c.LastError, MessageType.Error);

                bool loaded = c.Phase == DiffusionPhase.Running || c.Phase == DiffusionPhase.Ready;
                if (loaded && GUILayout.Button("Apply and reload model (model, acceleration, tiny VAE, resolution, step count)"))
                    _ = c.ReloadModelAsync();

                using (new EditorGUILayout.HorizontalScope())
                {
                    bool running = c.Phase == DiffusionPhase.Running;
                    bool ready = c.Phase == DiffusionPhase.Ready;
                    if (running && GUILayout.Button("Stop")) _ = c.StopAsync();
                    else if (ready && GUILayout.Button("Start")) _ = c.StartAsync();
                    else if (!running && !ready && GUILayout.Button("Connect & Start")) _ = c.ConnectAndStartAsync();
                    if (GUILayout.Button("Disconnect")) _ = c.DisconnectAsync();
                    if (GUILayout.Button("Stop server")) _ = c.StopServerAsync();
                }
            }
        }
    }
}
