"""Shared-memory video transport (same machine, CPU memory).

One named region carries frames in ONE direction (producer -> consumer). The server creates
both regions (input and output); Unity opens them with System.IO.MemoryMappedFile using the
same names. Names embed the resolution (``..._512x512``) so a resize creates a *new* region
instead of resizing a mapping another process still holds open.

Layout (little-endian). Header = 64 bytes:

    off  size  field
      0   u32   magic        0x48534453 ("SDSH")
      4   u32   version      1
      8   u32   width
     12   u32   height
     16   u32   channels     3 or 4 (RGB / RGBA, uint8)
     20   u32   num_slots    ring size (3)
     24   u32   flags        bit0: producer data is bottom-up (informational)
     28   u32   reserved
     32   u64   write_idx    number of frames published (0 = none); written LAST
     40   u64   frame_id     id of the newest frame (producer assigned)
     48   u64   timestamp_ns producer time.time_ns() of the newest frame
     56   u32   shutdown     producer sets 1 when closing
     60   u32   reserved

    64 + slot*16 : per-slot meta: u64 frame_id, u64 timestamp_ns
    data_offset  : (64 + 16*num_slots) rounded up to 64, then num_slots * (W*H*C) bytes

Publishing frame N writes slot ``N % num_slots`` then bumps ``write_idx``. A reader copies the
newest slot and re-reads ``write_idx``; if the producer lapped it (>= num_slots-1 frames in
between) the copy is discarded and retried, so a returned frame is never torn.
"""

import struct
import time
from multiprocessing import shared_memory
from typing import Optional

import numpy as np

from .base import Frame

MAGIC = 0x48534453
VERSION = 1
HEADER_SIZE = 64
SLOT_META = struct.Struct("<QQ")
_HDR_STATIC = struct.Struct("<8I")  # magic, version, w, h, ch, slots, flags, reserved
_OFF_WRITE_IDX = 32
_OFF_FRAME_ID = 40
_OFF_TS = 48
_OFF_SHUTDOWN = 56
_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")


def region_name(prefix: str, direction: str, width: int, height: int) -> str:
    return f"{prefix}_{direction}_{width}x{height}"


def data_offset(num_slots: int) -> int:
    raw = HEADER_SIZE + SLOT_META.size * num_slots
    return (raw + 63) // 64 * 64


def region_size(width: int, height: int, channels: int, num_slots: int) -> int:
    return data_offset(num_slots) + num_slots * width * height * channels


class ShmChannel:
    """One shared-memory frame ring. Either side may call ``publish`` or ``read_latest``;
    by convention exactly one process publishes to a given channel."""

    def __init__(self, shm: shared_memory.SharedMemory, owner: bool):
        self._shm = shm
        self._owner = owner
        buf = shm.buf
        magic, version, w, h, ch, slots, flags, _ = _HDR_STATIC.unpack_from(buf, 0)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"not a shmem frame ring (magic={magic:#x} version={version})")
        self.width, self.height, self.channels, self.num_slots, self.flags = w, h, ch, slots, flags
        self.slot_bytes = w * h * ch
        self._data_off = data_offset(slots)
        self._views = [
            np.ndarray(
                (h, w, ch), dtype=np.uint8, buffer=buf, offset=self._data_off + i * self.slot_bytes
            )
            for i in range(slots)
        ]
        self.name = shm.name

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def create(
        cls, name: str, width: int, height: int, channels: int = 4, num_slots: int = 3, flags: int = 0
    ) -> "ShmChannel":
        size = region_size(width, height, channels, num_slots)
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            # Stale region from a crashed run: attach and re-initialise it.
            shm = shared_memory.SharedMemory(name=name, create=False)
            if shm.size < size:
                shm.close()
                raise
        shm.buf[:HEADER_SIZE] = bytes(HEADER_SIZE)
        _HDR_STATIC.pack_into(shm.buf, 0, MAGIC, VERSION, width, height, channels, num_slots, flags, 0)
        return cls(shm, owner=True)

    @classmethod
    def open(cls, name: str) -> "ShmChannel":
        return cls(shared_memory.SharedMemory(name=name, create=False), owner=False)

    # ---- producer -----------------------------------------------------------------------
    def publish(self, pixels: np.ndarray, frame_id: int, timestamp_ns: Optional[int] = None) -> int:
        if pixels.shape != (self.height, self.width, self.channels) or pixels.dtype != np.uint8:
            raise ValueError(
                f"frame shape/dtype {pixels.shape}/{pixels.dtype} != "
                f"{(self.height, self.width, self.channels)}/uint8"
            )
        ts = time.time_ns() if timestamp_ns is None else timestamp_ns
        buf = self._shm.buf
        idx = _U64.unpack_from(buf, _OFF_WRITE_IDX)[0] + 1
        slot = idx % self.num_slots
        self._views[slot][...] = pixels
        SLOT_META.pack_into(buf, HEADER_SIZE + slot * SLOT_META.size, frame_id, ts)
        _U64.pack_into(buf, _OFF_FRAME_ID, frame_id)
        _U64.pack_into(buf, _OFF_TS, ts)
        _U64.pack_into(buf, _OFF_WRITE_IDX, idx)  # publish last
        return idx

    # ---- consumer -----------------------------------------------------------------------
    def write_index(self) -> int:
        return _U64.unpack_from(self._shm.buf, _OFF_WRITE_IDX)[0]

    def is_shutdown(self) -> bool:
        return _U32.unpack_from(self._shm.buf, _OFF_SHUTDOWN)[0] != 0

    def read_latest(self, last_index: int = 0) -> Optional[Frame]:
        buf = self._shm.buf
        for _ in range(4):
            idx1 = _U64.unpack_from(buf, _OFF_WRITE_IDX)[0]
            if idx1 == 0 or idx1 <= last_index:
                return None
            slot = idx1 % self.num_slots
            frame_id, ts = SLOT_META.unpack_from(buf, HEADER_SIZE + slot * SLOT_META.size)
            pixels = self._views[slot].copy()
            idx2 = _U64.unpack_from(buf, _OFF_WRITE_IDX)[0]
            if idx2 - idx1 < self.num_slots - 1:  # slot was not recycled while we copied
                return Frame(pixels=pixels, frame_id=frame_id, timestamp_ns=ts, index=idx1)
        return None

    # ---- lifetime -----------------------------------------------------------------------
    def close(self) -> None:
        if self._shm is None:
            return
        try:
            if self._owner:
                _U32.pack_into(self._shm.buf, _OFF_SHUTDOWN, 1)
        except Exception:
            pass
        self._views = []  # release exported buffer pointers before close()
        try:
            self._shm.close()
        except BufferError:
            pass
        if self._owner:
            try:
                self._shm.unlink()
            except Exception:
                pass
        self._shm = None


class ShmSource:
    """VideoSource over a ShmChannel (reads the newest frame)."""

    def __init__(self, channel: ShmChannel):
        self.channel = channel

    def read_latest(self, last_index: int = 0) -> Optional[Frame]:
        return self.channel.read_latest(last_index)

    def close(self) -> None:
        self.channel.close()


class ShmSink:
    """VideoSink over a ShmChannel (publishes frames)."""

    def __init__(self, channel: ShmChannel):
        self.channel = channel

    def write(self, pixels: np.ndarray, frame_id: int, timestamp_ns: int) -> None:
        self.channel.publish(pixels, frame_id, timestamp_ns)

    def close(self) -> None:
        self.channel.close()
