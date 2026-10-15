using System;

namespace StreamDiffusion
{
    /// <summary>What to do when a model reload needs TensorRT engines that are not cached yet.</summary>
    public enum EngineBuildPolicy
    {
        /// <summary>Pause and ask (UI card, inspector buttons or ConfirmEngineBuild / DeclineEngineBuild).</summary>
        Ask,
        /// <summary>Build without asking. For unattended setups where the engines are known to be missing.</summary>
        Allow,
        /// <summary>Never build from the running app: fail with a message (use the engine build tool instead).</summary>
        Block,
    }

    /// <summary>Thrown when a load needs engine builds and the policy is Block.</summary>
    public sealed class EngineBuildBlockedException : InvalidOperationException
    {
        public EngineBuildBlockedException(string message) : base(message) { }
    }

    /// <summary>The server's answer to "would this setup have to build engines?".</summary>
    [Serializable]
    public sealed class EngineStatus
    {
        public bool cached = true;
        public string[] missing = new string[0];
        public float estimateMinutes;
        public string detail = "";
        public int width, height, steps;

        public string MissingText => missing == null || missing.Length == 0 ? "-" : string.Join(", ", missing);
    }
}
