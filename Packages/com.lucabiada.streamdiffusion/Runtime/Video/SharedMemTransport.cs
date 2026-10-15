using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace StreamDiffusion
{
    /// <summary>
    /// Same-machine transport over named shared memory (see ShmChannel). Sending uses
    /// AsyncGPUReadback so the main thread never stalls on the GPU.
    /// </summary>
    public sealed unsafe class SharedMemTransport : IFrameTransport
    {
        readonly ShmChannel _in;   // we publish source frames here
        readonly ShmChannel _out;  // server publishes results here
        readonly ShmChannel _ctl;  // optional: separate control image for ControlNet
        readonly ShmChannel _ref;  // optional: IP-Adapter reference image
        readonly ShmChannel _cpv;  // optional: processed control image coming back (preview)
        ulong _lastCpv;
        int _inFlight, _ctlInFlight, _refInFlight;
        ulong _lastReceived;
        bool _disposed;

        public int Width => _in.Width;
        public int Height => _in.Height;
        public string Description => $"shmem {_in.Name} / {_out.Name} ({_in.Width}x{_in.Height}x{_in.Channels})";
        public int Dropped { get; private set; }

        /// <summary>
        /// Read the picture back synchronously (the CPU waits for the GPU) instead of AsyncGPUReadback, which
        /// delivers one to two frames later. Lower latency, costs a small stall on the main thread.
        /// </summary>
        public bool SyncReadback { get; set; }
        Texture2D _stage;

        public SharedMemTransport(string inName, string outName, string ctlName = null, string refName = null,
            string cpvName = null)
        {
            _in = ShmChannel.Open(inName);
            _out = ShmChannel.Open(outName);
            if (!string.IsNullOrEmpty(ctlName))
            {
                try { _ctl = ShmChannel.Open(ctlName); } catch { _ctl = null; }
            }
            if (!string.IsNullOrEmpty(refName))
            {
                try { _ref = ShmChannel.Open(refName); } catch { _ref = null; }
            }
            if (!string.IsNullOrEmpty(cpvName))
            {
                try { _cpv = ShmChannel.Open(cpvName); } catch { _cpv = null; }
            }
            if (_in.Channels != 4 || _out.Channels != 4)
                throw new NotSupportedException("only RGBA (4 channel) rings are supported");
        }

        public bool Send(RenderTexture source, ulong frameId, ulong timestampNs)
        {
            if (_disposed) return false;
            if (_inFlight >= 2) { Dropped++; return false; }
            if (source.width != _in.Width || source.height != _in.Height)
                throw new InvalidOperationException($"send texture is {source.width}x{source.height}, ring is {_in.Width}x{_in.Height}");
            if (SyncReadback)
            {
                if (_stage == null || _stage.width != source.width || _stage.height != source.height)
                {
                    if (_stage != null) UnityEngine.Object.Destroy(_stage);
                    _stage = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, false) { name = "SD_SyncStage" };
                }
                var prev = RenderTexture.active;
                RenderTexture.active = source;
                _stage.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0, false);
                RenderTexture.active = prev;
                var raw = _stage.GetRawTextureData<byte>();
                _in.Publish((byte*)raw.GetUnsafeReadOnlyPtr(), raw.Length, frameId, timestampNs);
                return true;
            }
            _inFlight++;
            AsyncGPUReadback.Request(source, 0, TextureFormat.RGBA32, req =>
            {
                _inFlight--;
                if (_disposed || req.hasError) return;
                var data = req.GetData<byte>();
                _in.Publish((byte*)data.GetUnsafeReadOnlyPtr(), data.Length, frameId, timestampNs);
            });
            return true;
        }

        public bool SendControl(RenderTexture source, ulong frameId, ulong timestampNs)
        {
            if (_disposed || _ctl == null) return false;
            if (_ctlInFlight >= 2) return false;
            if (source.width != _ctl.Width || source.height != _ctl.Height) return false;
            _ctlInFlight++;
            AsyncGPUReadback.Request(source, 0, TextureFormat.RGBA32, req =>
            {
                _ctlInFlight--;
                if (_disposed || req.hasError) return;
                var data = req.GetData<byte>();
                _ctl.Publish((byte*)data.GetUnsafeReadOnlyPtr(), data.Length, frameId, timestampNs);
            });
            return true;
        }

        public bool SendReference(RenderTexture source, ulong frameId, ulong timestampNs)
        {
            if (_disposed || _ref == null) return false;
            if (_refInFlight >= 1) return false;
            if (source.width != _ref.Width || source.height != _ref.Height) return false;
            _refInFlight++;
            AsyncGPUReadback.Request(source, 0, TextureFormat.RGBA32, req =>
            {
                _refInFlight--;
                if (_disposed || req.hasError) return;
                var data = req.GetData<byte>();
                _ref.Publish((byte*)data.GetUnsafeReadOnlyPtr(), data.Length, frameId, timestampNs);
            });
            return true;
        }

        public bool TryReceiveControlMap(Texture2D target)
        {
            if (_disposed || _cpv == null || target.width != _cpv.Width || target.height != _cpv.Height) return false;
            NativeArray<byte> raw = target.GetRawTextureData<byte>();
            if (!_cpv.TryReadLatest(_lastCpv, (byte*)raw.GetUnsafePtr(), out ulong idx, out _, out _)) return false;
            _lastCpv = idx;
            return true;
        }

        public bool TryReceive(Texture2D target, out ulong frameId, out ulong timestampNs)
        {
            frameId = timestampNs = 0;
            if (_disposed || target.width != _out.Width || target.height != _out.Height) return false;
            NativeArray<byte> raw = target.GetRawTextureData<byte>();
            if (!_out.TryReadLatest(_lastReceived, (byte*)raw.GetUnsafePtr(), out ulong idx, out frameId, out timestampNs))
                return false;
            _lastReceived = idx;
            return true;
        }

        public void Dispose()
        {
            _disposed = true;
            _in.Dispose();
            _out.Dispose();
            if (_stage != null) UnityEngine.Object.Destroy(_stage);
            _ctl?.Dispose();
            _ref?.Dispose();
            _cpv?.Dispose();
        }
    }
}
