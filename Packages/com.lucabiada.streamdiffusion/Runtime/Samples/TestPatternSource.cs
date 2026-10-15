using Unity.Collections;
using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>
    /// Animated CPU test pattern with orientation markers: RED square top-left, BLUE square
    /// top-right (as displayed), GREEN square bottom-left, plus a moving white disc. Use it as the
    /// DiffusionController source to verify orientation and colour without a camera.
    /// </summary>
    [AddComponentMenu("StreamDiffusion/Test Pattern Source")]
    public sealed class TestPatternSource : MonoBehaviour
    {
        public int size = 512;
        public Texture2D Output { get; private set; }

        void Awake()
        {
            Output = new Texture2D(size, size, TextureFormat.RGBA32, false, linear: false) { name = "SD_TestPattern" };
        }

        void Update()
        {
            var raw = Output.GetRawTextureData<byte>();
            int w = size, h = size;
            float t = Time.unscaledTime;
            int cx = (int)((0.5f + 0.35f * Mathf.Sin(t * 1.3f)) * w);
            int cy = (int)((0.5f + 0.35f * Mathf.Cos(t * 0.9f)) * h);
            int r2 = (w / 10) * (w / 10);
            int m = w / 8;
            for (int y = 0; y < h; y++)
            {
                // Texture row 0 is the BOTTOM of the displayed texture in Unity.
                bool top = y >= h - m, bottom = y < m;
                for (int x = 0; x < w; x++)
                {
                    int i = (y * w + x) * 4;
                    byte r = (byte)(40 + 120 * x / w), g = (byte)(40 + 120 * y / h), b = (byte)(80 + 60 * Mathf.Sin(t + x * 0.02f) * 0.5f + 30);
                    if (top && x < m) { r = 255; g = 0; b = 0; }                    // top-left: red
                    else if (top && x >= w - m) { r = 0; g = 0; b = 255; }          // top-right: blue
                    else if (bottom && x < m) { r = 0; g = 255; b = 0; }            // bottom-left: green
                    int dx = x - cx, dy = y - cy;
                    if (dx * dx + dy * dy < r2) { r = g = b = 255; }
                    raw[i] = r; raw[i + 1] = g; raw[i + 2] = b; raw[i + 3] = 255;
                }
            }
            Output.Apply(false);
        }
    }
}
