"""WebSocket control server and message router (plan section 3)."""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Set

import websockets
from websockets.asyncio.server import ServerConnection, serve

from . import __version__
from .protocol import (
    ALL_RUNTIME_PARAMS,
    DEFAULT_PARAMS,
    LOAD_MODEL_KEYS,
    PROTOCOL_VERSION,
    err,
    msg,
)
from .engines import engine_status
from .transports.ndi import ndi_capability
from .worker import InferenceWorker

log = logging.getLogger("control")


class ControlServer:
    def __init__(
        self,
        worker: InferenceWorker,
        host: str,
        port: int,
        token: Optional[str] = None,
        allow_shutdown: bool = False,
        gpu_info: Optional[Dict[str, Any]] = None,
        on_shutdown: Optional[Callable[[], None]] = None,
        engine_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
    ):
        self.worker = worker
        self.host, self.port = host, port
        self.token = token
        self.allow_shutdown = allow_shutdown
        self.gpu_info = gpu_info or {}
        self.on_shutdown = on_shutdown
        self.engine_dir = engine_dir
        self.models_dir = models_dir
        self.clients: Set[ServerConnection] = set()
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Mirror of the desired/current state, owned by the asyncio thread.
        self.state: Dict[str, Any] = {
            "phase": "idle",
            "model": None,
            "params": dict(DEFAULT_PARAMS),
            "prompt": "",
            "negative_prompt": "",
            "prompts": None,
            "interpolation": "slerp",
            "video": None,
            "ndi": {"enabled": False, "name": "StreamDiffusion", "frames": 0, "connections": 0, "error": None},
            "last_error": None,
            "stats": None,
        }

    # ------------------------------------------------------------------ worker -> clients
    def post_event(self, ev: Dict[str, Any]) -> None:
        """Thread-safe; called from the worker thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._on_event, ev)

    def _on_event(self, ev: Dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "phase":
            self.state["phase"] = ev["phase"]
        elif t == "video_ready":
            self.state["video"] = {k: ev[k] for k in ("transport", "in_name", "out_name", "width", "height", "channels")}
        elif t == "ndi_status":
            self.state["ndi"] = {k: ev[k] for k in ("enabled", "name", "frames", "connections", "error") if k in ev}
        elif t == "error":
            self.state["last_error"] = ev.get("message")
        elif t == "stats":
            self.state["stats"] = {k: v for k, v in ev.items() if k != "type"}
        self._broadcast(ev)

    def _broadcast(self, ev: Dict[str, Any]) -> None:
        if self.clients:
            websockets.broadcast(self.clients, json.dumps(ev))

    # ------------------------------------------------------------------ connection handling
    async def _handler(self, ws: ServerConnection) -> None:
        authed = False
        peer = ws.remote_address
        log.info("client connected %s", peer)
        try:
            async for raw in ws:
                try:
                    m = json.loads(raw)
                    if not isinstance(m, dict) or "type" not in m:
                        raise ValueError("message must be a JSON object with a 'type'")
                except (ValueError, TypeError) as exc:
                    await ws.send(json.dumps(err("bad_message", str(exc))))
                    continue
                req_id = m.get("id")
                if m["type"] == "hello":
                    reply = self._hello(m)
                    if reply["type"] == "hello_ack":
                        authed = True
                        self.clients.add(ws)
                    await ws.send(json.dumps(reply))
                    if reply["type"] == "error":
                        await ws.close(code=4001, reason=reply["code"])
                        return
                    continue
                if not authed:
                    await ws.send(json.dumps(err("not_authenticated", "send hello first", req_id)))
                    continue
                try:
                    reply = self._route(m)
                except Exception as exc:  # never let a bad message kill the loop
                    log.exception("route error")
                    reply = err("internal_error", f"{type(exc).__name__}: {exc}", req_id)
                if reply is not None:
                    await ws.send(json.dumps(reply))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            log.info("client disconnected %s", peer)

    def _hello(self, m: Dict[str, Any]) -> Dict[str, Any]:
        if int(m.get("protocol", 0)) != PROTOCOL_VERSION:
            return err("protocol_mismatch", f"server speaks protocol {PROTOCOL_VERSION}", m.get("id"), fatal=True)
        if self.token and m.get("token") != self.token:
            return err("bad_token", "invalid token", m.get("id"), fatal=True)
        return msg(
            "hello_ack",
            id=m.get("id"),
            protocol=PROTOCOL_VERSION,
            server_version=__version__,
            gpu=self.gpu_info.get("name", "unknown"),
            vram_total_gb=self.gpu_info.get("vram_total_gb", 0.0),
            capabilities={
                "transports": ["shmem"],
                "ndi_output": ndi_capability(),
                "accelerations": ["tensorrt", "none"],
                "runtime_params": sorted(ALL_RUNTIME_PARAMS),
                "load_model_keys": sorted(LOAD_MODEL_KEYS),
            },
            state=self._state_snapshot(),
        )

    def _state_snapshot(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.state))

    # ------------------------------------------------------------------ routing
    def _route(self, m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        t, req_id = m["type"], m.get("id")
        w = self.worker

        if t == "ping":
            return msg("pong", t=m.get("t"), id=req_id)
        if t == "get_state":
            return msg("state", id=req_id, **self._state_snapshot())
        if t == "list_models":
            from .engines import scan_models

            return msg("models", id=req_id, **scan_models(self.models_dir))
        if t == "list_loras":
            from .engines import scan_loras

            return msg("loras", id=req_id, **scan_loras(self.models_dir))
        if t == "list_ndi_sources":
            return msg("ndi_sources", id=req_id, available=False, sources=[])
        if t == "configure_video":
            transport = m.get("transport", "shmem")
            if transport != "shmem":
                return err("unsupported_transport", f"transport '{transport}' is not available", req_id)
            if "width" not in m or "height" not in m:
                return err("bad_message", "configure_video needs width and height", req_id)
            w.submit("configure_video", id=req_id, width=m["width"], height=m["height"], channels=m.get("channels", 4))
            return None  # video_ready / error follows
        if t == "load_model":
            cfg = {k: v for k, v in m.items() if k not in ("type", "id")}
            self.state["model"] = {k: v for k, v in cfg.items() if k in LOAD_MODEL_KEYS}
            w.submit("load_model", id=req_id, cfg=cfg)
            return None  # progress... then ack / error
        if t == "cancel_load":
            return err("not_supported", "an in-progress engine build cannot be cancelled in this version", req_id)
        if t == "set_params":
            params = m.get("params")
            if params is None:
                params = {k: v for k, v in m.items() if k not in ("type", "id")}
            unknown = sorted(k for k in params if k not in ALL_RUNTIME_PARAMS)
            known = {k: v for k, v in params.items() if k in ALL_RUNTIME_PARAMS}
            self.state["params"].update(known)
            w.submit("set_params", id=req_id, params=known)
            if unknown:
                self._broadcast(msg("warning", id=req_id, message=f"ignored unknown params: {unknown}"))
            return None
        if t == "set_prompt":
            self.state["prompt"] = m.get("prompt", self.state["prompt"])
            self.state["negative_prompt"] = m.get("negative_prompt", self.state["negative_prompt"])
            prompts = m.get("prompts", self.state["prompts"])
            if prompts is not None and not (
                isinstance(prompts, list) and all(isinstance(p, dict) and "text" in p for p in prompts)
            ):
                return err("bad_message", "prompts must be a list of {text, weight}", req_id)
            self.state["prompts"] = prompts or None
            self.state["interpolation"] = m.get("interpolation", self.state["interpolation"])
            w.submit(
                "set_prompt",
                id=req_id,
                prompt=self.state["prompt"],
                negative_prompt=self.state["negative_prompt"],
                prompts=self.state["prompts"],
                interpolation=self.state["interpolation"],
                normalize=bool(m.get("normalize", True)),
            )
            return None
        if t == "engine_status":
            # Would loading this setup have to BUILD TensorRT engines? (a heavy, minutes-long GPU job)
            try:
                width, height = int(m["width"]), int(m["height"])
            except (KeyError, TypeError, ValueError):
                return err("bad_message", "engine_status needs integer width and height", req_id)
            accel = str(m.get("acceleration", "tensorrt"))
            if accel != "tensorrt":
                res = {"cached": True, "missing": [], "estimate_minutes": 0.0, "width": width, "height": height,
                       "steps": int(m.get("steps", 1)), "detail": "no TensorRT engines needed"}
            else:
                res = engine_status(
                    self.engine_dir,
                    str(m.get("model_id", "stabilityai/sdxl-turbo")),
                    width, height, int(m.get("steps", 1)), bool(m.get("use_tiny_vae", True)),
                    features=dict(
                        {k: m[k] for k in ("mode", "cfg_type", "use_cached_attn", "use_feature_injection",
                                           "max_cache_maxframes", "lora_dict", "controlnets") if k in m},
                        ipadapter=(m["ipadapters"][0].get("ipadapter_model_path")
                                   if isinstance(m.get("ipadapters"), list) and m["ipadapters"]
                                   and isinstance(m["ipadapters"][0], dict) else None),
                    ),
                )
            return msg("engine_status", id=req_id, **res)
        if t == "set_ndi_output":
            enabled = m.get("enabled", False)
            name = m.get("name", self.state["ndi"].get("name", "StreamDiffusion"))
            if not isinstance(enabled, bool) or not isinstance(name, str) or len(name) > 64:
                return err("bad_message", "set_ndi_output needs enabled (bool) and name (string, max 64)", req_id)
            w.submit("set_ndi", id=req_id, enabled=enabled, name=name)
            return None  # ndi_status + ack / error follow
        if t in ("start", "stop", "unload"):
            w.submit(t, id=req_id)
            return None
        if t == "shutdown":
            if not self.allow_shutdown:
                return err("not_allowed", "server was not started with --allow-shutdown", req_id)
            if self.on_shutdown:
                self._loop.call_later(0.1, self.on_shutdown)  # type: ignore[union-attr]
            return msg("ack", id=req_id)
        return err("unknown_type", f"unknown message type '{t}'", req_id)

    # ------------------------------------------------------------------ lifecycle
    async def run(self, stop: asyncio.Event) -> None:
        self._loop = asyncio.get_running_loop()
        async with serve(self._handler, self.host, self.port, max_size=2**20) as server:
            self._server = server
            log.info("control server listening on ws://%s:%d", self.host, self.port)
            await stop.wait()
            self._broadcast(msg("server_closing"))
