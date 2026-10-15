using System;
using UnityEngine;

namespace StreamDiffusion
{
    /// <summary>
    /// Moves pixels between Unity and the diffusion server. Implementations never touch control
    /// logic; the controller picks one from the server's configure_video reply.
    /// </summary>
    public interface IFrameTransport : IDisposable
    {
        string Description { get; }

        /// <summary>Queue the render texture for sending. Returns false if skipped (backpressure).</summary>
        bool Send(RenderTexture source, ulong frameId, ulong timestampNs);

        /// <summary>Queue a separate control image (ControlNet). False if the transport has no control channel or skipped.</summary>
        bool SendControl(RenderTexture source, ulong frameId, ulong timestampNs);

        /// <summary>Queue an IP-Adapter reference image. False if the transport has no reference channel or skipped.</summary>
        bool SendReference(RenderTexture source, ulong frameId, ulong timestampNs);

        /// <summary>Copy the newest processed control image (what the ControlNet sees) into <paramref name="target"/>.</summary>
        bool TryReceiveControlMap(Texture2D target);

        /// <summary>
        /// Copy the newest unseen result into <paramref name="target"/> (RGBA32, Apply is the caller's job).
        /// </summary>
        bool TryReceive(Texture2D target, out ulong frameId, out ulong timestampNs);
    }
}
