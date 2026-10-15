using UnityEngine;
using UnityEngine.UIElements;

namespace StreamDiffusion.UI
{
    /// <summary>
    /// The StreamDiffusionUnity title lockup (wordmark + subtitle) as a transparent, white image (Resources/UI/StreamDiffusionUnityWordmark.png,
    /// rendered by branding/make_png_wordmark.py at the exact display size). Falls back to plain text when the image is
    /// missing.
    /// </summary>
    public sealed class DiffusionWordmark : VisualElement
    {
        public const string Resource = "UI/StreamDiffusionUnityWordmark";

        /// <summary>The image is rendered at exactly the size it is shown (1 texel = 1 UI pixel), so nothing is resampled.</summary>
        public DiffusionWordmark(float fallbackHeight, Color tint)
        {
            style.flexShrink = 0;
            pickingMode = PickingMode.Ignore;

            var tex = Resources.Load<Texture2D>(Resource);
            if (tex != null)
            {
                style.width = tex.width;
                style.height = tex.height;
                style.backgroundImage = new StyleBackground(tex);
                style.unityBackgroundImageTintColor = tint;
                style.backgroundSize = new BackgroundSize(BackgroundSizeType.Contain);
                style.backgroundRepeat = new BackgroundRepeat(Repeat.NoRepeat, Repeat.NoRepeat);
                style.backgroundPositionX = new BackgroundPosition(BackgroundPositionKeyword.Left);
                style.backgroundPositionY = new BackgroundPosition(BackgroundPositionKeyword.Center);
            }
            else
            {
                style.height = fallbackHeight;
                var l = new Label("StreamDiffusionUnity");
                l.style.fontSize = fallbackHeight * 0.8f;
                l.style.color = tint;
                Add(l);
            }
        }
    }
}
