"""NDI output (server -> any NDI receiver: Resolume, OBS, vMix, TouchDesigner, ...).

The inference thread only hands over the newest frame; a dedicated thread converts it to BGRA
and sends it, so a slow receiver or the NDI runtime can never stall diffusion. Sending is not
clocked (``clock_video=False``): the NDI stream runs at the rate frames are produced.

Requires the free NDI Runtime from ndi.video (installed with NDI Tools) and `cyndilib`.
Every failure is reported, never raised into the inference loop.
"""

import logging
import threading
import time
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("ndi")


def ndi_capability() -> Dict[str, Any]:
    """Cheap check used in hello_ack: can this process create NDI senders at all?"""
    try:
        from cyndilib.sender import Sender  # noqa: F401
        from cyndilib.video_frame import VideoSendFrame  # noqa: F401

        return {"available": True}
    except Exception as exc:  # missing package or missing NDI runtime DLL
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


class NdiOutput:
    def __init__(self, name: str, width: int = 512, height: int = 512, fps: int = 30):
        self.name = name
        self._w, self._h = int(width), int(height)
        self._fps = fps
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._pending: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self.frames_sent = 0
        self.connections = 0
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ public
    def start(self, timeout: float = 5.0) -> Tuple[bool, Optional[str]]:
        self._thread = threading.Thread(target=self._run, name="ndi-out", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        if self.error:
            return False, self.error
        return True, None

    def submit(self, rgb: np.ndarray) -> None:
        """Newest frame wins. `rgb` is uint8 (H, W, >=3), top row first. Copied here because
        the caller's buffer (a pinned host buffer) is reused for the next frame."""
        if self._stop.is_set() or self.error:
            return
        frame = np.array(rgb[..., :3], dtype=np.uint8, copy=True, order="C")
        with self._lock:
            self._pending = frame
        self._wake.set()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frames": self.frames_sent,
            "connections": self.connections,
            "width": self._w,
            "height": self._h,
            "error": self.error,
        }

    # ------------------------------------------------------------------ sender thread
    def _make_sender(self, w: int, h: int):
        from cyndilib.sender import Sender
        from cyndilib.video_frame import VideoSendFrame
        from cyndilib.wrapper.ndi_structs import FourCC

        sender = Sender(self.name, clock_video=False)
        vf = VideoSendFrame()
        vf.set_resolution(w, h)
        vf.set_frame_rate(Fraction(self._fps, 1))
        vf.set_fourcc(FourCC.BGRA)
        sender.set_video_frame(vf)
        sender.open()
        return sender

    def _run(self) -> None:
        sender = None
        try:
            w, h = self._w, self._h
            sender = self._make_sender(w, h)
            bgra = np.zeros((h, w, 4), dtype=np.uint8)
            bgra[..., 3] = 255
            sender.write_video(bgra.reshape(-1))  # visible immediately, black until the first frame
            self._ready.set()
            last_conn = 0.0
            log.info("NDI output '%s' open (%dx%d)", self.name, w, h)

            while not self._stop.is_set():
                self._wake.wait(1.0)
                self._wake.clear()
                with self._lock:
                    frame, self._pending = self._pending, None

                now = time.perf_counter()
                if now - last_conn > 1.0:
                    last_conn = now
                    try:
                        self.connections = int(sender.get_num_connections(0.0))
                    except Exception:
                        pass

                if frame is None:
                    continue
                fh, fw = frame.shape[0], frame.shape[1]
                if (fw, fh) != (w, h):
                    # resolution changed: a sender's frame size is fixed, so rebuild it
                    sender.close()
                    w, h = fw, fh
                    sender = self._make_sender(w, h)
                    bgra = np.zeros((h, w, 4), dtype=np.uint8)
                    bgra[..., 3] = 255
                    self._w, self._h = w, h
                    log.info("NDI output '%s' resized to %dx%d", self.name, w, h)
                bgra[..., 0] = frame[..., 2]
                bgra[..., 1] = frame[..., 1]
                bgra[..., 2] = frame[..., 0]
                sender.write_video(bgra.reshape(-1))
                self.frames_sent += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("NDI output failed: %s", self.error)
            self._ready.set()
        finally:
            try:
                if sender is not None:
                    sender.close()
            except Exception:
                pass
