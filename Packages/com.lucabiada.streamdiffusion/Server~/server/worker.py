"""Inference worker thread (plan section 4.3).

Owns the CUDA context and the pipeline. All model-affecting commands arrive through a queue
and are applied *between* frames; there are no locks around the model. Video is pulled from
the shared-memory ring, which already has "latest frame wins" semantics, so the mailbox is the
ring itself (a future NDI transport adds a receiver thread feeding the same read_latest()).
"""

import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import diag
from .pipeline import PipelineBase
from .protocol import DEFAULT_PARAMS, MODEL_RUNTIME_PARAMS
from .transports.ndi import NdiOutput
from .transports.shmem import ShmChannel, region_name

log = logging.getLogger("worker")

Emit = Callable[[Dict[str, Any]], None]


def _resize_rgb(img: np.ndarray, width: int, height: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        ys = (np.arange(height) * img.shape[0] // height).astype(np.intp)
        xs = (np.arange(width) * img.shape[1] // width).astype(np.intp)
        return img[ys][:, xs]


class InferenceWorker(threading.Thread):
    def __init__(self, pipeline_factory: Callable[[], PipelineBase], emit: Emit, name_prefix: Optional[str] = None):
        super().__init__(name="inference", daemon=True)
        self._factory = pipeline_factory
        self._emit = emit
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop_flag = threading.Event()
        self._prefix = name_prefix or f"sdu{uuid.uuid4().hex[:6]}"

        self.pipeline: Optional[PipelineBase] = None
        self.phase = "idle"  # idle | loading | ready | running
        self.params: Dict[str, Any] = dict(DEFAULT_PARAMS)
        self.prompt = ""
        self.negative = ""
        self.prompts = None          # weighted blend: [{"text", "weight"}], or None for a single prompt
        self.interp = "slerp"
        self.norm_prompt = True

        # NDI output (optional; independent of the Unity video link)
        self.ndi: Optional[NdiOutput] = None
        self.ndi_enabled = False
        self.ndi_name = "StreamDiffusion"

        self.chan_in: Optional[ShmChannel] = None
        self.chan_out: Optional[ShmChannel] = None
        self.chan_ctl: Optional[ShmChannel] = None     # optional separate control image (ControlNet)
        self.chan_ref: Optional[ShmChannel] = None     # optional IP-Adapter reference image
        self.chan_cpv: Optional[ShmChannel] = None     # processed control image (preview), server -> client
        self._cpv_count = 0
        self._ref_idx = 0
        self._ctl_idx = 0
        self._ctl_pixels: Optional[np.ndarray] = None
        self._out_rgba: Optional[np.ndarray] = None
        self._last_idx = 0

        # stats window
        self._frames = 0
        self._infer_ms_sum = 0.0
        self._latency_ms_sum = 0.0
        self._drops = 0
        self._win_start = time.perf_counter()
        self._win_in_idx = 0
        self._last_frame_t = 0.0
        self._consecutive_errors = 0

    # ------------------------------------------------------------------ public (any thread)
    def submit(self, cmd: str, **kw: Any) -> None:
        self._q.put({"cmd": cmd, **kw})

    def stop(self) -> None:
        self._stop_flag.set()
        self._q.put({"cmd": "_wake"})

    # ------------------------------------------------------------------ helpers
    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self._emit({"type": "phase", "phase": phase})

    def _progress(self, req_id: Any):
        def cb(stage: str, fraction: Optional[float], message: str) -> None:
            self._emit({"type": "progress", "id": req_id, "stage": stage, "fraction": fraction, "message": message})

        return cb

    def _heartbeat(self, stop: threading.Event, t0: float) -> None:
        """While a model loads / engines build, tell the client every 2 s that we are alive and what the GPU is doing."""
        import subprocess

        q = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"]
        while not stop.wait(2.0):
            msg = {"type": "activity", "elapsed_s": round(time.perf_counter() - t0, 1)}
            try:
                out = subprocess.run(q, capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                util, mem, temp = [float(x) for x in out.split(",")]
                msg.update({"gpu_util": util, "gpu_mem_gb": round(mem / 1024.0, 2), "gpu_temp": temp})
            except Exception:
                pass
            try:
                self._emit(msg)
            except Exception:
                return

    def _error(self, code: str, message: str, req_id: Any = None, fatal: bool = False) -> None:
        out = {"type": "error", "code": code, "message": message, "fatal": fatal}
        if req_id is not None:
            out["id"] = req_id
        self._emit(out)

    # ------------------------------------------------------------------ commands
    def _cmd_configure_video(self, c: Dict[str, Any]) -> None:
        w, h, ch = int(c["width"]), int(c["height"]), int(c.get("channels", 4))
        if not (16 <= w <= 4096 and 16 <= h <= 4096 and ch in (3, 4)):
            self._error("bad_video_config", f"unsupported video config {w}x{h}x{ch}", c.get("id"))
            return
        was_running = self.phase == "running"
        if was_running:
            self._set_phase("ready")
        self._close_video()
        in_name = region_name(self._prefix, "in", w, h)
        out_name = region_name(self._prefix, "out", w, h)
        ctl_name = region_name(self._prefix, "ctl", w, h)
        ref_name = region_name(self._prefix, "ref", w, h)
        cpv_name = region_name(self._prefix, "cpv", w, h)
        try:
            self.chan_in = ShmChannel.create(in_name, w, h, ch)
            self.chan_out = ShmChannel.create(out_name, w, h, ch)
            self.chan_ctl = ShmChannel.create(ctl_name, w, h, ch)
            self.chan_ref = ShmChannel.create(ref_name, w, h, ch)
            self.chan_cpv = ShmChannel.create(cpv_name, w, h, ch)
        except Exception as exc:
            self._close_video()
            self._error("video_failed", f"could not create shared memory: {exc}", c.get("id"))
            return
        self._out_rgba = np.zeros((h, w, ch), dtype=np.uint8)
        if ch == 4:
            self._out_rgba[..., 3] = 255
        self._last_idx = 0
        self._ctl_idx = 0
        self._ctl_pixels = None
        self._ref_idx = 0
        self._emit(
            {
                "type": "video_ready",
                "id": c.get("id"),
                "transport": "shmem",
                "in_name": in_name,
                "out_name": out_name,
                "ctl_name": ctl_name,
                "ref_name": ref_name,
                "cpv_name": cpv_name,
                "width": w,
                "height": h,
                "channels": ch,
                "num_slots": self.chan_in.num_slots,
                "format": "uint8 HWC, header 64 bytes, see server/transports/shmem.py",
            }
        )
        if was_running:
            self._set_phase("running")

    def _cmd_load_model(self, c: Dict[str, Any]) -> None:
        req = c.get("id")
        self._set_phase("loading")
        t0 = time.perf_counter()
        cfg = dict(c.get("cfg", {}))
        label = f"load_model {cfg.get('model_id')} {cfg.get('width')}x{cfg.get('height')}"
        log.info("%s starting (engines are built here when missing)", label)
        diag.watch_stacks(label, every=60.0)
        hb_stop = threading.Event()
        threading.Thread(target=self._heartbeat, args=(hb_stop, t0), daemon=True, name="load-heartbeat").start()
        try:
            if self.pipeline is None:
                self.pipeline = self._factory()
            unknown = self.pipeline.load(cfg, self._progress(req))
        except Exception as exc:  # keep the server alive so the client can retry
            log.exception("load failed")
            hb_stop.set()
            diag.stop_watch(label + " FAILED")
            self._set_phase("idle")
            self._error("load_failed", f"{type(exc).__name__}: {exc}", req)
            return
        hb_stop.set()
        diag.stop_watch(label)
        log.info("%s finished in %.1f s", label, time.perf_counter() - t0)
        if unknown:
            self._emit({"type": "warning", "id": req, "message": f"ignored unknown load_model keys: {unknown}"})
        for w in getattr(self.pipeline, "load_warnings", []) or []:
            self._emit({"type": "warning", "id": req, "message": w})
        self._apply_pending()
        self._set_phase("ready")
        self._emit({"type": "ack", "id": req, "seconds": round(time.perf_counter() - t0, 1)})

    def _apply_pending(self) -> None:
        if not self.pipeline or not self.pipeline.loaded:
            return
        try:
            if self.prompt or self.prompts:
                self.pipeline.set_prompt(self.prompt, self.negative, self.prompts, self.interp, self.norm_prompt)
            self.pipeline.set_params({k: v for k, v in self.params.items() if k in MODEL_RUNTIME_PARAMS})
        except Exception as exc:
            self._error("param_failed", f"{type(exc).__name__}: {exc}")

    def _cmd_set_params(self, c: Dict[str, Any]) -> None:
        p = c.get("params", {})
        self.params.update(p)
        if self.pipeline and self.pipeline.loaded:
            try:
                self.pipeline.set_params(p)
            except Exception as exc:
                self._error("param_failed", f"{type(exc).__name__}: {exc}", c.get("id"))
                return
        if c.get("id") is not None:
            self._emit({"type": "ack", "id": c["id"]})

    def _cmd_set_prompt(self, c: Dict[str, Any]) -> None:
        self.prompt = c.get("prompt", self.prompt)
        self.negative = c.get("negative_prompt", self.negative)
        if "prompts" in c:
            self.prompts = c["prompts"] or None
        self.interp = c.get("interpolation", self.interp)
        self.norm_prompt = c.get("normalize", self.norm_prompt)
        if self.pipeline and self.pipeline.loaded:
            try:
                self.pipeline.set_prompt(self.prompt, self.negative, self.prompts, self.interp, self.norm_prompt)
            except Exception as exc:
                self._error("prompt_failed", f"{type(exc).__name__}: {exc}", c.get("id"))
                return
        if c.get("id") is not None:
            self._emit({"type": "ack", "id": c["id"]})

    def _cmd_set_ndi(self, c: Dict[str, Any]) -> None:
        enabled = bool(c.get("enabled", False))
        name = str(c.get("name") or self.ndi_name).strip() or "StreamDiffusion"
        req = c.get("id")

        if self.ndi is not None and (not enabled or name != self.ndi_name):
            self.ndi.close()
            self.ndi = None
        self.ndi_name = name
        self.ndi_enabled = enabled

        error = None
        if enabled and self.ndi is None:
            w = self.chan_out.width if self.chan_out else 512
            h = self.chan_out.height if self.chan_out else 512
            out = NdiOutput(name, w, h)
            ok, err = out.start()
            if ok:
                self.ndi = out
            else:
                out.close()
                self.ndi_enabled = False
                error = err
                self._error("ndi_failed", f"NDI output could not start: {err}", req)
        self._emit_ndi_status(error)
        if req is not None and error is None:
            self._emit({"type": "ack", "id": req})

    def _emit_ndi_status(self, error: Optional[str] = None) -> None:
        st = self.ndi.status() if self.ndi is not None else {}
        self._emit({
            "type": "ndi_status",
            "enabled": self.ndi_enabled,
            "name": self.ndi_name,
            "frames": st.get("frames", 0),
            "connections": st.get("connections", 0),
            "error": error or st.get("error"),
        })

    def _cmd_start(self, c: Dict[str, Any]) -> None:
        if not (self.pipeline and self.pipeline.loaded):
            self._error("not_ready", "load_model must complete before start", c.get("id"))
            return
        if self.chan_in is None:
            self._error("not_ready", "configure_video must be called before start", c.get("id"))
            return
        self._last_idx = self.chan_in.write_index()  # ignore stale frames
        self._reset_stats()
        self._set_phase("running")
        if c.get("id") is not None:
            self._emit({"type": "ack", "id": c["id"]})

    def _cmd_stop(self, c: Dict[str, Any]) -> None:
        if self.phase == "running":
            self._set_phase("ready")
        if c.get("id") is not None:
            self._emit({"type": "ack", "id": c["id"]})

    def _cmd_unload(self, c: Dict[str, Any]) -> None:
        stats = {}
        if self.pipeline:
            stats = self.pipeline.unload() or {}
        self._set_phase("idle")
        if c.get("id") is not None:
            self._emit({"type": "ack", "id": c["id"], **stats})

    def _close_video(self) -> None:
        for ch in (self.chan_in, self.chan_out, self.chan_ctl, self.chan_ref, self.chan_cpv):
            if ch is not None:
                ch.close()
        self.chan_in = self.chan_out = self.chan_ctl = self.chan_ref = self.chan_cpv = None
        self._ctl_pixels = None
        self._out_rgba = None

    # ------------------------------------------------------------------ stats
    def _reset_stats(self) -> None:
        self._frames = 0
        self._infer_ms_sum = 0.0
        self._latency_ms_sum = 0.0
        self._drops = 0
        self._win_start = time.perf_counter()
        self._win_in_idx = self.chan_in.write_index() if self.chan_in else 0

    def _maybe_emit_stats(self, force: bool = False) -> None:
        now = time.perf_counter()
        dt = now - self._win_start
        if dt < 0.5 and not force:
            return
        in_idx = self.chan_in.write_index() if self.chan_in else 0
        used, total = (0.0, 0.0)
        if self.pipeline:
            try:
                used, total = self.pipeline.vram()
            except Exception:
                pass
        n = max(self._frames, 1)
        self._emit(
            {
                "type": "stats",
                "phase": self.phase,
                "fps_in": round((in_idx - self._win_in_idx) / dt, 1) if dt > 0 else 0.0,
                "fps_out": round(self._frames / dt, 1) if dt > 0 else 0.0,
                "infer_ms": round(self._infer_ms_sum / n, 2) if self._frames else 0.0,
                "latency_ms": round(self._latency_ms_sum / n, 2) if self._frames else 0.0,
                "queue_drops": self._drops,
                "vram_used_gb": round(used, 2),
                "vram_total_gb": round(total, 2),
                "ndi_enabled": self.ndi_enabled,
                "ndi_frames": self.ndi.frames_sent if self.ndi else 0,
                "ndi_connections": self.ndi.connections if self.ndi else 0,
            }
        )
        self._frames = 0
        self._infer_ms_sum = 0.0
        self._latency_ms_sum = 0.0
        self._win_start = now
        self._win_in_idx = in_idx

    # ------------------------------------------------------------------ frame loop
    def _process_one(self) -> bool:
        assert self.chan_in and self.chan_out and self.pipeline and self._out_rgba is not None
        max_fps = float(self.params.get("max_fps") or 0)
        if max_fps > 0 and (time.perf_counter() - self._last_frame_t) < 1.0 / max_fps:
            return False
        fr = self.chan_in.read_latest(self._last_idx)
        if fr is None:
            return False
        if fr.index - self._last_idx > 1 and self._last_idx != 0:
            self._drops += fr.index - self._last_idx - 1
        self._last_idx = fr.index
        t0 = time.perf_counter()
        flip = bool(self.params.get("flip_y", False))
        if self.chan_ref is not None:
            rf = self.chan_ref.read_latest(self._ref_idx)
            if rf is not None:                 # a new IP-Adapter reference image: encode it once
                self._ref_idx = rf.index
                try:
                    self.pipeline.set_style_image(np.array(rf.pixels, copy=True), flip)
                except Exception as exc:
                    log.warning("IP-Adapter reference update failed: %s", exc)
        control = None
        if self.params.get("control_source") == "separate" and self.chan_ctl is not None:
            cf = self.chan_ctl.read_latest(self._ctl_idx)
            if cf is not None:
                self._ctl_idx = cf.index
                self._ctl_pixels = np.array(cf.pixels, copy=True)
            control = self._ctl_pixels            # newest control frame seen so far (None until one arrives)
        out = self.pipeline.process(fr.pixels, flip, control)
        if self.ndi is not None:
            # `out` is bottom-up when flip_y is on (Unity's memory order); NDI wants top-down
            self.ndi.submit(out[::-1] if flip else out)
        out = _resize_rgb(out, self.chan_out.width, self.chan_out.height)
        self._out_rgba[..., :3] = out
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self.chan_out.publish(self._out_rgba, fr.frame_id, fr.timestamp_ns)
        if self.chan_cpv is not None and self.params.get("control_preview"):
            self._cpv_count += 1
            if self._cpv_count % 3 == 0:                 # ~1/3 of the frame rate is plenty for a preview
                try:
                    cp = self.pipeline.get_control_preview()
                    if cp is not None:
                        cp = _resize_rgb(cp, self.chan_cpv.width, self.chan_cpv.height)
                        cp = cp[::-1] if flip else cp        # same row order convention as the result
                        buf = np.empty((self.chan_cpv.height, self.chan_cpv.width, self.chan_cpv.channels), np.uint8)
                        buf[..., :3] = cp
                        if buf.shape[2] == 4:
                            buf[..., 3] = 255
                        self.chan_cpv.publish(buf, fr.frame_id, fr.timestamp_ns)
                except Exception as exc:
                    log.debug("control preview failed: %s", exc)
        self._last_frame_t = time.perf_counter()
        self._frames += 1
        self._infer_ms_sum += infer_ms
        self._latency_ms_sum += (time.time_ns() - fr.timestamp_ns) / 1e6
        self._consecutive_errors = 0
        return True

    # ------------------------------------------------------------------ thread main
    _HANDLERS = {
        "configure_video": "_cmd_configure_video",
        "load_model": "_cmd_load_model",
        "set_params": "_cmd_set_params",
        "set_prompt": "_cmd_set_prompt",
        "set_ndi": "_cmd_set_ndi",
        "start": "_cmd_start",
        "stop": "_cmd_stop",
        "unload": "_cmd_unload",
    }

    def run(self) -> None:
        try:
            while not self._stop_flag.is_set():
                busy = self.phase == "running"
                try:
                    c = self._q.get(timeout=0.0 if busy else 0.05)
                except queue.Empty:
                    c = None
                if c is not None:
                    name = self._HANDLERS.get(c["cmd"])
                    if name:
                        try:
                            getattr(self, name)(c)
                        except Exception as exc:
                            log.exception("command %s failed", c["cmd"])
                            self._error("command_failed", f"{c['cmd']}: {type(exc).__name__}: {exc}", c.get("id"))
                    continue
                if busy:
                    try:
                        if not self._process_one():
                            time.sleep(0.0005)
                    except Exception as exc:
                        log.exception("inference error")
                        self._consecutive_errors += 1
                        self._error("inference_error", f"{type(exc).__name__}: {exc}")
                        if self._consecutive_errors >= 5:
                            self._set_phase("ready")
                            self._error("inference_stopped", "stopped after 5 consecutive errors")
                            self._consecutive_errors = 0
                self._maybe_emit_stats()
        finally:
            if self.ndi is not None:
                self.ndi.close()
                self.ndi = None
            self._close_video()
            if self.pipeline:
                try:
                    self.pipeline.unload()
                except Exception:
                    pass
