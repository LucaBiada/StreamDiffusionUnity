"""Video transport interfaces (plan section 4.5).

A transport only moves pixels. Both ends use the same small contract so that switching
transports never touches inference or control logic.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Frame:
    pixels: np.ndarray  # uint8, shape (H, W, C), C = 3 or 4 (RGB / RGBA)
    frame_id: int  # producer-assigned id, echoed on the result
    timestamp_ns: int  # producer wall-clock time (time.time_ns), used for latency
    index: int = 0  # transport sequence number (monotonic; gaps = dropped frames)


class VideoSource(Protocol):
    def read_latest(self, last_index: int = 0) -> Optional[Frame]:
        """Return the newest frame with index > last_index, or None. Never blocks."""

    def close(self) -> None: ...


class VideoSink(Protocol):
    def write(self, pixels: np.ndarray, frame_id: int, timestamp_ns: int) -> None: ...

    def close(self) -> None: ...
