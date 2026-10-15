using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace StreamDiffusion
{
    public enum BlendInterpolation { Slerp, Average, CosineWeighted }

    [Serializable]
    public sealed class PromptBlendEntry
    {
        public string text = "";
        public float weight = 1f;
        public PromptBlendEntry() { }
        public PromptBlendEntry(string text, float weight) { this.text = text; this.weight = weight; }
    }

    [Serializable]
    public sealed class SeedBlendEntry
    {
        public int seed;
        public float weight = 1f;
        public SeedBlendEntry() { }
        public SeedBlendEntry(int seed, float weight) { this.seed = seed; this.weight = weight; }
    }

    /// <summary>Everything an artist tunes, as one JSON-serialisable snapshot.</summary>
    [Serializable]
    public sealed class DiffusionPreset
    {
        public int version = 1;

        // model / video (reload-class: applied when the model is (re)loaded)
        public string modelId = "stabilityai/sdxl-turbo";
        public string acceleration = "tensorrt";
        public bool useTinyVae = true;
        public int width = 512, height = 512;

        // prompt
        public string prompt = "";
        public string negativePrompt = "";
        public bool promptBlendEnabled;
        public List<PromptBlendEntry> promptBlend = new List<PromptBlendEntry>();
        public BlendInterpolation promptInterpolation = BlendInterpolation.Slerp;
        public bool normalizePromptWeights = true;

        // generation
        public float guidanceScale = 1f, delta = 1f;
        public int numInferenceSteps = 50;
        public int[] tIndexList = { 1 };
        public int seed;
        public bool seedBlendEnabled;
        public List<SeedBlendEntry> seedBlend = new List<SeedBlendEntry>();
        public BlendInterpolation seedInterpolation = BlendInterpolation.Average;
        public bool normalizeSeedWeights = true;

        // performance
        public bool similarFilterEnabled;
        public float similarFilterThreshold = 0.98f;
        public int similarFilterMaxSkip = 10;
        public bool addNoise = true;
        public int maxSendFps = 60;

        // advanced engine options (strings = the library's names; older presets load the defaults)
        public string mode = "img2img";
        public string cfgType = "self";
        public string scheduler = "lcm";
        public string sampler = "normal";

        // FX chains (live; older presets have none)
        public System.Collections.Generic.List<FxEntry> fxPre = new System.Collections.Generic.List<FxEntry>();
        public System.Collections.Generic.List<FxEntry> fxPost = new System.Collections.Generic.List<FxEntry>();

        // IP-Adapter (older presets load the defaults)
        public bool useIpAdapter = false;
        public string ipAdapterModel = "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin";
        public string ipAdapterEncoder = "h94/IP-Adapter/sdxl_models/image_encoder";
        public float ipAdapterScale = 0.7f;
        public bool ipAdapterEnabled = true;
        public float referenceRefreshSeconds = 0f;

        public System.Collections.Generic.List<ControlNetEntry> controlNets = new System.Collections.Generic.List<ControlNetEntry>();

        public System.Collections.Generic.List<LoraEntry> loras = new System.Collections.Generic.List<LoraEntry>();

        // StreamV2V (older presets load with these defaults)
        public bool useStreamV2V = false;
        public bool featureInjection = true;
        public int v2vMaxCacheFrames = 4;
        public float fiStrength = 0.75f;
        public float fiThreshold = 0.98f;
        public int v2vCacheInterval = 1;
        public int v2vCacheFrames = 4;
    }

    /// <summary>JSON presets under persistentDataPath/StreamDiffusion/Presets (works in builds).</summary>
    public static class DiffusionPresetStore
    {
        public static string Directory
        {
            get
            {
                var d = Path.Combine(Application.persistentDataPath, "StreamDiffusion", "Presets");
                System.IO.Directory.CreateDirectory(d);
                return d;
            }
        }

        static string Sanitize(string name)
        {
            foreach (var c in Path.GetInvalidFileNameChars()) name = name.Replace(c, '_');
            return name.Trim();
        }

        public static string PathFor(string name) => Path.Combine(Directory, Sanitize(name) + ".json");

        public static string[] List()
        {
            var files = System.IO.Directory.GetFiles(Directory, "*.json");
            var names = new List<string>();
            foreach (var f in files) names.Add(Path.GetFileNameWithoutExtension(f));
            names.Sort(StringComparer.OrdinalIgnoreCase);
            return names.ToArray();
        }

        public static void Save(string name, DiffusionPreset preset)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("preset name is empty");
            File.WriteAllText(PathFor(name), JsonUtility.ToJson(preset, true));
        }

        public static DiffusionPreset Load(string name)
        {
            var path = PathFor(name);
            if (!File.Exists(path)) return null;
            return JsonUtility.FromJson<DiffusionPreset>(File.ReadAllText(path));
        }

        public static bool Delete(string name)
        {
            var path = PathFor(name);
            if (!File.Exists(path)) return false;
            File.Delete(path);
            return true;
        }
    }
}
