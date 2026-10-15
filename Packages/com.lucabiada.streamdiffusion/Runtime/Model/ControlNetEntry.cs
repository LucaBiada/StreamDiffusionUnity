using System;
using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>
    /// One ControlNet: which model, which preprocessor turns the input picture into the control image
    /// (the model and preprocessor are load-time), and how strongly it steers (live).
    /// </summary>
    [Serializable]
    public class ControlNetEntry
    {
        [Tooltip("Hugging Face repo id of the ControlNet (must match the model family: SDXL for SDXL-Turbo).")]
        public string modelId = "diffusers/controlnet-canny-sdxl-1.0";
        [Tooltip("canny, depth, soft_edge, lineart, standard_lineart, openpose, mediapipe_pose or passthrough (tile).")]
        public string preprocessor = "canny";
        [Range(0f, 2f)] public float scale = 0.5f;
        public bool enabled = true;
        [Header("canny")]
        [Range(0, 255)] public int cannyLow = 100;
        [Range(0, 255)] public int cannyHigh = 200;

        public ControlNetEntry() { }
        public ControlNetEntry(string modelId, string preprocessor, float scale)
        {
            this.modelId = modelId; this.preprocessor = preprocessor; this.scale = scale;
        }

        public string DisplayName
        {
            get
            {
                if (string.IsNullOrEmpty(modelId)) return "(empty)";
                int i = modelId.Replace('\\', '/').LastIndexOf('/');
                return i >= 0 ? modelId.Substring(i + 1) : modelId;
            }
        }
    }
}
