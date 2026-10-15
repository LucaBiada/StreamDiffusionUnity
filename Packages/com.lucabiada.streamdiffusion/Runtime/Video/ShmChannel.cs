using System;
using System.IO.MemoryMappedFiles;
using System.Threading;

namespace StreamDiffusion
{
    /// <summary>
    /// One shared-memory frame ring (Windows named mapping). Byte-compatible with
    /// diffusion-server/server/transports/shmem.py. The server creates the regions; the client
    /// opens them: it publishes into the "in" ring and reads from the "out" ring.
    /// Header (64 bytes, little endian): magic u32 "SDSH", version u32, width, height, channels,
    /// num_slots, flags, reserved (u32 each), write_idx u64 @32, frame_id u64 @40,
    /// timestamp_ns u64 @48, shutdown u32 @56. Per-slot meta (u64 frame_id, u64 ts) at 64+slot*16;
    /// pixel data starts at (64 + 16*num_slots) rounded up to 64.
    /// </summary>
    public sealed unsafe class ShmChannel : IDisposable
    {
        const uint Magic = 0x48534453;
        const uint FormatVersion = 1;
        const int HeaderSize = 64;
        const int OffWriteIdx = 32;
        const int OffFrameId = 40;
        const int OffTimestamp = 48;
        const int OffShutdown = 56;
        const int SlotMetaSize = 16;

        MemoryMappedFile _mmf;
        MemoryMappedViewAccessor _view;
        byte* _base;
        readonly object _lock = new object();

        public string Name { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int Channels { get; private set; }
        public int NumSlots { get; private set; }
        public int SlotBytes { get; private set; }
        int _dataOffset;
        public bool IsOpen => _base != null;

        public static ShmChannel Open(string name)
        {
            var ch = new ShmChannel { Name = name };
            ch._mmf = MemoryMappedFile.OpenExisting(name, MemoryMappedFileRights.ReadWrite);
            ch._view = ch._mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.ReadWrite);
            byte* p = null;
            ch._view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            ch._base = p + ch._view.PointerOffset;

            uint magic = *(uint*)ch._base;
            uint version = *(uint*)(ch._base + 4);
            if (magic != Magic || version != FormatVersion)
            {
                ch.Dispose();
                throw new InvalidOperationException($"'{name}' is not a StreamDiffusion frame ring (magic {magic:X}, version {version})");
            }
            ch.Width = (int)*(uint*)(ch._base + 8);
            ch.Height = (int)*(uint*)(ch._base + 12);
            ch.Channels = (int)*(uint*)(ch._base + 16);
            ch.NumSlots = (int)*(uint*)(ch._base + 20);
            ch.SlotBytes = ch.Width * ch.Height * ch.Channels;
            ch._dataOffset = (HeaderSize + SlotMetaSize * ch.NumSlots + 63) / 64 * 64;
            return ch;
        }

        public ulong WriteIndex => (ulong)Volatile.Read(ref *(long*)(_base + OffWriteIdx));

        public bool ServerShutdown => *(uint*)(_base + OffShutdown) != 0;

        /// <summary>Copy one frame (Width*Height*Channels bytes) into the ring and publish it.</summary>
        public ulong Publish(byte* src, int length, ulong frameId, ulong timestampNs)
        {
            if (length != SlotBytes) throw new ArgumentException($"frame is {length} bytes, ring expects {SlotBytes}");
            lock (_lock)
            {
                ulong idx = *(ulong*)(_base + OffWriteIdx) + 1;
                int slot = (int)(idx % (ulong)NumSlots);
                Buffer.MemoryCopy(src, _base + _dataOffset + (long)slot * SlotBytes, SlotBytes, SlotBytes);
                byte* meta = _base + HeaderSize + slot * SlotMetaSize;
                *(ulong*)meta = frameId;
                *(ulong*)(meta + 8) = timestampNs;
                *(ulong*)(_base + OffFrameId) = frameId;
                *(ulong*)(_base + OffTimestamp) = timestampNs;
                Thread.MemoryBarrier();
                *(ulong*)(_base + OffWriteIdx) = idx; // publish last
                return idx;
            }
        }

        /// <summary>
        /// Copy the newest frame with index &gt; lastIndex into dst (must hold SlotBytes). Never blocks.
        /// Returns false when there is no newer frame or the copy was torn (caller just retries next tick).
        /// </summary>
        public bool TryReadLatest(ulong lastIndex, byte* dst, out ulong index, out ulong frameId, out ulong timestampNs)
        {
            index = frameId = timestampNs = 0;
            for (int attempt = 0; attempt < 4; attempt++)
            {
                ulong idx1 = *(ulong*)(_base + OffWriteIdx);
                if (idx1 == 0 || idx1 <= lastIndex) return false;
                int slot = (int)(idx1 % (ulong)NumSlots);
                byte* meta = _base + HeaderSize + slot * SlotMetaSize;
                ulong fid = *(ulong*)meta;
                ulong ts = *(ulong*)(meta + 8);
                Buffer.MemoryCopy(_base + _dataOffset + (long)slot * SlotBytes, dst, SlotBytes, SlotBytes);
                Thread.MemoryBarrier();
                ulong idx2 = *(ulong*)(_base + OffWriteIdx);
                if (idx2 - idx1 < (ulong)(NumSlots - 1))
                {
                    index = idx1; frameId = fid; timestampNs = ts;
                    return true;
                }
            }
            return false;
        }

        public void Dispose()
        {
            if (_base != null && _view != null)
            {
                _view.SafeMemoryMappedViewHandle.ReleasePointer();
                _base = null;
            }
            _view?.Dispose();
            _mmf?.Dispose();
            _view = null;
            _mmf = null;
        }
    }
}
