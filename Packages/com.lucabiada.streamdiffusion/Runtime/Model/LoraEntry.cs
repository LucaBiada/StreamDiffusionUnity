using System;
using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>One LoRA: its file and how strongly it is applied. The weight is live, the SET is an engine option.</summary>
    [Serializable]
    public class LoraEntry
    {
        [Tooltip("Path to the .safetensors file (or a Hugging Face repo id).")]
        public string path = "";
        [Range(-1f, 2f)] public float weight = 1f;

        public LoraEntry() { }
        public LoraEntry(string path, float weight) { this.path = path; this.weight = weight; }

        public string DisplayName
        {
            get
            {
                if (string.IsNullOrEmpty(path)) return "(empty)";
                string p = path.Replace('\\', '/');
                int i = p.LastIndexOf('/');
                return i >= 0 ? p.Substring(i + 1) : p;
            }
        }
    }

    /// <summary>A LoRA file the server found in its models/LoRA folder.</summary>
    public class LoraFile
    {
        public string name, path;
        public float sizeMb;
    }
}
