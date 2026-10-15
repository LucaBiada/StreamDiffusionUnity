namespace StreamDiffusion
{
    public enum DiffusionMode { Img2Img, Txt2Img }

    /// <summary>Where the ControlNet control image comes from.</summary>
    public enum ControlSourceMode
    {
        /// <summary>The picture sent to the model is also the control image (edges/depth are taken from it).</summary>
        SameAsInput,
        /// <summary>A separate texture (e.g. a depth map rendered by another camera) drives the ControlNets.</summary>
        Texture,
    }
    public enum DiffusionCfgType { Self, Initialize, Full, None }
    public enum DiffusionScheduler { Lcm, Tcd }
    public enum DiffusionSampler { Normal, Simple, SgmUniform, Ddim, Beta, Karras }

    /// <summary>Wire names (the StreamDiffusion library's strings) for the engine option enums.</summary>
    public static class DiffusionOptionNames
    {
        public static string Of(DiffusionMode v) => v == DiffusionMode.Txt2Img ? "txt2img" : "img2img";

        public static string Of(DiffusionCfgType v) =>
            v == DiffusionCfgType.Initialize ? "initialize" : v == DiffusionCfgType.Full ? "full"
            : v == DiffusionCfgType.None ? "none" : "self";

        public static string Of(DiffusionScheduler v) => v == DiffusionScheduler.Tcd ? "tcd" : "lcm";

        public static string Of(DiffusionSampler v)
        {
            switch (v)
            {
                case DiffusionSampler.Simple: return "simple";
                case DiffusionSampler.SgmUniform: return "sgm_uniform";
                case DiffusionSampler.Ddim: return "ddim";
                case DiffusionSampler.Beta: return "beta";
                case DiffusionSampler.Karras: return "karras";
                default: return "normal";
            }
        }
    }
}
