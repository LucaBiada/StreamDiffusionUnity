using System;
using System.Collections.Generic;
using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>One named number of an FX processor (Unity cannot serialise a dictionary).</summary>
    [Serializable]
    public class FxParam
    {
        public string name;
        public float value;
        public FxParam() { }
        public FxParam(string name, float value) { this.name = name; this.value = value; }
    }

    /// <summary>
    /// One FX processor in a chain: an image effect applied to the picture before the model ("pre") or to
    /// the result after it ("post"). Effects are live: they can be added, reordered and tuned while streaming.
    /// </summary>
    [Serializable]
    public class FxEntry
    {
        [Tooltip("blur, sharpen or feedback (see FxCatalog).")]
        public string type = "blur";
        public bool enabled = true;
        public List<FxParam> parameters = new List<FxParam>();

        public FxEntry() { }
        public FxEntry(string type) { this.type = type; EnsureDefaults(); }

        public float Get(string name, float fallback)
        {
            foreach (var p in parameters) if (p != null && p.name == name) return p.value;
            return fallback;
        }

        public void Set(string name, float value)
        {
            foreach (var p in parameters) if (p != null && p.name == name) { p.value = value; return; }
            parameters.Add(new FxParam(name, value));
        }

        /// <summary>Adds any parameter of the type that is missing, with its default.</summary>
        public void EnsureDefaults()
        {
            var def = FxCatalog.Find(type);
            if (def == null) return;
            foreach (var p in def.parameters)
            {
                bool have = false;
                foreach (var q in parameters) if (q != null && q.name == p.name) { have = true; break; }
                if (!have) parameters.Add(new FxParam(p.name, p.defaultValue));
            }
        }

        public FxEntry Clone()
        {
            var c = new FxEntry { type = type, enabled = enabled };
            foreach (var p in parameters) if (p != null) c.parameters.Add(new FxParam(p.name, p.value));
            return c;
        }
    }

    /// <summary>The effects the server's FX chain understands, with the ranges the UI sliders use.</summary>
    public static class FxCatalog
    {
        public sealed class Param
        {
            public string name, label;
            public float min, max, defaultValue;
            public bool integer, odd;
        }

        public sealed class Def
        {
            public string type, label, description;
            /// <summary>True when the effect only makes sense on the input picture (before the model).</summary>
            public bool preOnly;
            public Param[] parameters;
        }

        public static readonly Def[] All =
        {
            new Def
            {
                type = "blur", label = "Blur", description = "Gaussian blur. Soften the input before the model, or the result after it.",
                parameters = new[]
                {
                    new Param { name = "blur_intensity", label = "Intensity", min = 0.1f, max = 10f, defaultValue = 2f },
                    new Param { name = "kernel_size", label = "Kernel size", min = 3f, max = 61f, defaultValue = 15f, integer = true, odd = true },
                },
            },
            new Def
            {
                type = "sharpen", label = "Sharpen", description = "Unsharp mask with edge and detail boost. Mostly useful after the model.",
                parameters = new[]
                {
                    new Param { name = "sharpen_intensity", label = "Intensity", min = 0f, max = 5f, defaultValue = 1.5f },
                    new Param { name = "unsharp_radius", label = "Radius", min = 0.1f, max = 5f, defaultValue = 1f },
                    new Param { name = "edge_enhancement", label = "Edge enhancement", min = 0f, max = 2f, defaultValue = 0.5f },
                    new Param { name = "detail_boost", label = "Detail boost", min = 0f, max = 2f, defaultValue = 0.3f },
                    new Param { name = "noise_reduction", label = "Noise reduction", min = 0f, max = 1f, defaultValue = 0.1f },
                },
            },
            new Def
            {
                type = "feedback", label = "Feedback", preOnly = true,
                description = "Blends the previous result into the input picture: trails, evolving images, video-like drift.",
                parameters = new[]
                {
                    new Param { name = "feedback_strength", label = "Strength", min = 0f, max = 1f, defaultValue = 0.5f },
                },
            },
        };

        public static Def Find(string type)
        {
            foreach (var d in All) if (d.type == type) return d;
            return null;
        }
    }
}
