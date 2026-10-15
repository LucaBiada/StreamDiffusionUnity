"""Wires worker + control server and owns process lifecycle."""

import asyncio
import ctypes
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

from .control import ControlServer
from .pipeline import FakePipeline, PipelineBase, StreamDiffusionPipeline
from .worker import InferenceWorker

log = logging.getLogger("app")


def query_gpu() -> Dict[str, Any]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().splitlines()[0]
        name, mem = [p.strip() for p in out.split(",")]
        return {"name": name, "vram_total_gb": round(float(mem) / 1024.0, 1)}
    except Exception:
        return {"name": "unknown", "vram_total_gb": 0.0}


def _watch_parent(pid: int, on_gone) -> None:
    """Windows: exit when the launching process (Unity) is gone, so no orphan keeps the VRAM."""
    if os.name != "nt":
        return
    SYNCHRONIZE = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        log.warning("parent pid %s not found; not watching", pid)
        return

    def run() -> None:
        ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        log.warning("parent process %s exited; shutting down", pid)
        on_gone()

    threading.Thread(target=run, name="parent-watch", daemon=True).start()


async def serve_forever(args) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    if args.fake_pipeline:
        factory = lambda: FakePipeline()  # noqa: E731
    else:
        factory = lambda: StreamDiffusionPipeline(args.basefolder)  # noqa: E731

    server_ref: Dict[str, ControlServer] = {}
    worker = InferenceWorker(factory, emit=lambda ev: server_ref["s"].post_event(ev))
    server = ControlServer(
        worker,
        host=args.host,
        port=args.port,
        token=args.token,
        allow_shutdown=args.allow_shutdown,
        gpu_info=query_gpu(),
        on_shutdown=lambda: loop.call_soon_threadsafe(stop.set),
        engine_dir=(os.path.join(args.basefolder, "engines", "td") if args.basefolder else None),
        models_dir=(os.path.join(args.basefolder, "models") if args.basefolder else None),
    )
    server_ref["s"] = server
    worker.start()
    if args.parent_pid:
        _watch_parent(args.parent_pid, lambda: loop.call_soon_threadsafe(stop.set))
    try:
        await server.run(stop)
    finally:
        worker.stop()
        worker.join(timeout=10)
