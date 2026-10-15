"""Crash-survivable diagnostics for the server process.

- tee_output(path): everything printed to stdout/stderr (our logging, the library's progress,
  polygraphy / TensorRT [I] lines) also goes to a file, flushed on every write, so the log
  outlives Unity (which normally owns the server's pipes) and survives a GPU reset.
- enable_faulthandler(path): Python-level dump of all thread stacks on a fatal native error.
- watch_stacks(label): while a long operation (model load / engine build) runs, dump all
  thread stacks to the same file every N seconds. After a crash the last dump shows which
  call the process was in.
"""

import datetime
import faulthandler
import sys
import threading
from typing import Optional

_fault_fh = None
_log_fh = None
_lock = threading.Lock()


class _Tee:
    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        try:
            self._stream.write(s)
        except Exception:
            pass
        fh = _log_fh
        if fh is not None:
            try:
                with _lock:
                    fh.write(s)
                    fh.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):          # fileno, encoding, ...
        return getattr(self._stream, name)


def tee_output(path: str) -> None:
    global _log_fh
    _log_fh = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    _log_fh.write(f"# server log started {datetime.datetime.now().isoformat(timespec='seconds')} pid-log\n")
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)


def enable_faulthandler(path: str) -> None:
    global _fault_fh
    _fault_fh = open(path, "a", buffering=1)
    faulthandler.enable(file=_fault_fh, all_threads=True)


def watch_stacks(label: str, every: float = 60.0) -> None:
    """Start periodic all-thread stack dumps (no-op if faulthandler output is not configured)."""
    if _fault_fh is None:
        return
    try:
        _fault_fh.write(f"\n--- {datetime.datetime.now():%H:%M:%S} {label}: stack dump every {every:.0f} s until it ends ---\n")
        faulthandler.dump_traceback_later(every, repeat=True, file=_fault_fh)
    except Exception:
        pass


def stop_watch(label: str = "") -> None:
    if _fault_fh is None:
        return
    try:
        faulthandler.cancel_dump_traceback_later()
        _fault_fh.write(f"--- {datetime.datetime.now():%H:%M:%S} {label}: ended normally ---\n")
    except Exception:
        pass
