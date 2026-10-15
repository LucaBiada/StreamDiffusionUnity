namespace StreamDiffusion
{
    /// <summary>A base model found in the server's models/Model folder (checkpoint file or diffusers folder).</summary>
    public class ModelFile
    {
        public string name, path, kind;
        public float sizeMb;
    }
}
