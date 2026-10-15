"""Reference client for the control + shared-memory protocol (no Unity needed).

Doubles as the executable spec for the Unity client: same message order, same shared-memory
rules. Usage against a running server:

    <venv>\\python.exe tools\\sd_client.py --port 8765 --frames 100 --width 512 --height 512 --save out.png
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from server.transports.shmem import ShmChannel  # noqa: E402


class SdClient:
    def __init__(self, url: str, token: Optional[str] = None):
        self.url, self.token = url, token
        self.ws = None
        self._id = 0
        self.events: List[Dict[str, Any]] = []
        self.latest_stats: Optional[Dict[str, Any]] = None
        self._waiters: Dict[int, asyncio.Future] = {}
        self._reader: Optional[asyncio.Task] = None
        self.chan_in: Optional[ShmChannel] = None
        self.chan_out: Optional[ShmChannel] = None

    async def connect(self) -> Dict[str, Any]:
        self.ws = await websockets.connect(self.url, max_size=2**20)
        self._reader = asyncio.create_task(self._read_loop())
        ack = await self.request("hello", protocol=1, client_name="sd_client", token=self.token)
        return ack

    async def _read_loop(self) -> None:
        try:
            async for raw in self.ws:
                m = json.loads(raw)
                if m.get("type") == "stats":
                    self.latest_stats = m
                self.events.append(m)
                rid = m.get("id")
                if rid in self._waiters and m.get("type") in ("ack", "hello_ack", "video_ready", "error", "pong", "state", "ndi_sources", "engine_status", "loras", "models"):
                    fut = self._waiters.pop(rid)
                    if not fut.done():
                        fut.set_result(m)
        except websockets.ConnectionClosed:
            pass

    async def request(self, type_: str, timeout: float = 60.0, **payload: Any) -> Dict[str, Any]:
        self._id += 1
        rid = self._id
        fut = asyncio.get_running_loop().create_future()
        self._waiters[rid] = fut
        await self.ws.send(json.dumps({"type": type_, "id": rid, **payload}))
        return await asyncio.wait_for(fut, timeout)

    async def configure_video(self, width: int, height: int, channels: int = 4) -> Dict[str, Any]:
        r = await self.request("configure_video", transport="shmem", width=width, height=height, channels=channels)
        if r["type"] != "video_ready":
            raise RuntimeError(r)
        # Server creates both regions; we write `in` and read `out`.
        self.chan_in = ShmChannel.open(r["in_name"])
        self.chan_out = ShmChannel.open(r["out_name"])
        self.chan_ctl = ShmChannel.open(r["ctl_name"]) if r.get("ctl_name") else None
        self.chan_ref = ShmChannel.open(r["ref_name"]) if r.get("ref_name") else None
        self.chan_cpv = ShmChannel.open(r["cpv_name"]) if r.get("cpv_name") else None
        return r

    def send_frame(self, pixels: np.ndarray, frame_id: int) -> None:
        self.chan_in.publish(pixels, frame_id, time.time_ns())

    def send_reference(self, pixels: np.ndarray, frame_id: int) -> None:
        """Publish an IP-Adapter reference image."""
        self.chan_ref.publish(pixels, frame_id, time.time_ns())

    def send_control(self, pixels: np.ndarray, frame_id: int) -> None:
        """Publish a separate control image (ControlNet)."""
        self.chan_ctl.publish(pixels, frame_id, time.time_ns())

    async def wait_result(self, frame_id: int, timeout: float = 5.0):
        """Wait until the output ring shows `frame_id` (or a later frame)."""
        t0 = time.perf_counter()
        last = 0
        while time.perf_counter() - t0 < timeout:
            fr = self.chan_out.read_latest(last)
            if fr is not None:
                last = fr.index
                if fr.frame_id >= frame_id:
                    return fr
            await asyncio.sleep(0.001)
        return None

    async def close(self) -> None:
        for c in (self.chan_in, self.chan_out, getattr(self, "chan_ctl", None), getattr(self, "chan_ref", None), getattr(self, "chan_cpv", None)):
            if c is not None:
                c.close()
        if self.ws is not None:
            await self.ws.close()
        if self._reader:
            self._reader.cancel()


async def run_demo(args) -> int:
    c = SdClient(f"ws://{args.host}:{args.port}", args.token)
    ack = await c.connect()
    print("hello_ack: gpu=%s vram=%sGB caps=%s" % (ack["gpu"], ack["vram_total_gb"], ack["capabilities"]["transports"]))
    v = await c.configure_video(args.width, args.height)
    print("video_ready:", v["in_name"], v["out_name"])
    t0 = time.perf_counter()
    r = await c.request(
        "load_model", timeout=3600, model_id=args.model, width=args.width, height=args.height, acceleration=args.acceleration
    )
    print("load_model:", r["type"], r.get("seconds"), "s", r.get("message", ""))
    if r["type"] == "error":
        return 1
    await c.request("set_prompt", prompt=args.prompt, negative_prompt="")
    await c.request("start")
    # Synthetic moving gradient
    h, w = args.height, args.width
    ys, xs = np.mgrid[0:h, 0:w]
    if args.saturate:
        return await _saturate(c, args)
    sent = got = 0
    lat = []
    t_start = time.perf_counter()
    last_png = None
    for i in range(1, args.frames + 1):
        phase = (i % 120) / 120.0
        img = np.empty((h, w, 4), np.uint8)
        img[..., 0] = ((xs / w + phase) % 1.0 * 255).astype(np.uint8)
        img[..., 1] = ((ys / h + phase * 0.5) % 1.0 * 255).astype(np.uint8)
        img[..., 2] = ((xs * ys / (w * h) + phase) % 1.0 * 255).astype(np.uint8)
        img[..., 3] = 255
        c.send_frame(img, i)
        sent += 1
        fr = await c.wait_result(i, timeout=5.0)
        if fr is not None:
            got += 1
            lat.append((time.time_ns() - fr.timestamp_ns) / 1e6)
            last_png = fr.pixels
    dt = time.perf_counter() - t_start
    print(f"sent={sent} received={got} in {dt:.2f}s  e2e_fps={got / dt:.1f}  latency_ms mean={np.mean(lat):.1f} p95={np.percentile(lat, 95):.1f}")
    print("last stats:", c.latest_stats)
    if args.save and last_png is not None:
        from PIL import Image

        Image.fromarray(last_png[..., :3]).save(args.save)
        print("saved", args.save)
    await c.request("stop")
    await c.close()
    return 0


async def _saturate(c: "SdClient", args) -> int:
    """Publish as fast as the server consumes and count results for N seconds (no per-frame wait)."""
    h, w = args.height, args.width
    rng = np.random.default_rng(1)
    base = (rng.random((h // 16 + 1, w // 16 + 1, 3)) * 255).astype(np.uint8)
    base = np.kron(base, np.ones((16, 16, 1), np.uint8))[:h, :w]
    frames = []
    for k in range(8):
        f = np.empty((h, w, 4), np.uint8)
        f[..., :3] = np.roll(base, k * 7, axis=1)
        f[..., 3] = 255
        frames.append(f)
    got, last, lat, fid = 0, 0, [], 0
    t_end = time.perf_counter() + args.saturate
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        fid += 1
        c.send_frame(frames[fid % 8], fid)
        fr = c.chan_out.read_latest(last)
        while fr is not None:
            got += 1
            last = fr.index
            lat.append((time.time_ns() - fr.timestamp_ns) / 1e6)
            fr = c.chan_out.read_latest(last)
        await asyncio.sleep(0)
        time.sleep(0.002)  # ~500 fps offered load, far above the model's rate
    dt = time.perf_counter() - t_start
    print(f"offered={fid} processed={got} in {dt:.1f}s  server_fps={got / dt:.1f}  latency_ms mean={np.mean(lat):.1f} p95={np.percentile(lat, 95):.1f}")
    print("last stats:", c.latest_stats)
    await c.request("stop")
    await c.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=None)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--model", default="stabilityai/sdxl-turbo")
    ap.add_argument("--acceleration", default="tensorrt")
    ap.add_argument("--prompt", default="a fancy banana, studio photo")
    ap.add_argument("--save", default=None)
    ap.add_argument("--saturate", type=float, default=0, help="seconds of max-rate load instead of lockstep")
    return asyncio.run(run_demo(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
