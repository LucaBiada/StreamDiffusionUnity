"""Pipeline adapter over the dotsimulate StreamDiffusion fork (plan section 4.4).

``StreamDiffusionPipeline`` is the real thing (torch / TensorRT, imported lazily).
``FakePipeline`` has the same interface without a GPU, for protocol and transport tests.

Contract of ``process``: input uint8 RGBA/RGB HWC, output uint8 HWC RGB of the *model*
resolution. The returned array may alias an internal buffer; copy before the next call.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .protocol import LOAD_MODEL_KEYS, MODEL_RUNTIME_PARAMS

log = logging.getLogger("pipeline")

ProgressFn = Callable[[str, Optional[float], str], None]


class PipelineBase:
    width: int = 512
    height: int = 512
    loaded: bool = False

    def load(self, cfg: Dict[str, Any], progress: ProgressFn) -> List[str]:
        raise NotImplementedError

    def set_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        prompts: Optional[List[Dict[str, Any]]] = None,
        interpolation: str = "slerp",
        normalize: bool = True,
    ) -> None:
        """Single prompt, or a weighted blend when `prompts` = [{"text", "weight"}, ...]."""
        raise NotImplementedError

    def set_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def process(self, frame: np.ndarray, flip_y: bool = False, control: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def set_style_image(self, frame: np.ndarray, flip_y: bool = False) -> None:
        """New IP-Adapter reference image. Default: ignored (models without an image adapter)."""
        return None

    def get_control_preview(self) -> Optional[np.ndarray]:
        """The processed control image (what the first ControlNet sees) as upright RGB uint8, or None."""
        return None

    def unload(self) -> None:
        raise NotImplementedError

    def vram(self) -> Tuple[float, float]:
        return (0.0, 0.0)  # used, total in GB


def split_load_keys(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    known = {k: v for k, v in cfg.items() if k in LOAD_MODEL_KEYS}
    unknown = sorted(k for k in cfg if k not in LOAD_MODEL_KEYS)
    return known, unknown


# FX processor types accepted from clients (image domain, run on the GPU / CPU without an engine).
FX_TYPES = {"blur", "sharpen", "feedback", "passthrough"}


class FakePipeline(PipelineBase):
    """Inverts colours. Emulates a slow load and a small per-frame cost."""

    def __init__(self, load_seconds: float = 0.3, frame_ms: float = 2.0):
        self.load_seconds = load_seconds
        self.frame_ms = frame_ms
        self.params: Dict[str, Any] = {}
        self.prompt = ""
        self.negative = ""
        self._style = None
        self._last_control = None

    def get_control_preview(self) -> Optional[np.ndarray]:
        return self._last_control

    def load(self, cfg: Dict[str, Any], progress: ProgressFn) -> List[str]:
        self._style = None
        self._last_control = None
        known, unknown = split_load_keys(cfg)
        self.width = int(known.get("width", 512))
        self.height = int(known.get("height", 512))
        steps = 4
        for i in range(steps):
            progress("build", (i + 1) / steps, f"fake load step {i + 1}/{steps}")
            time.sleep(self.load_seconds / steps)
        self.loaded = True
        return unknown

    def set_prompt(self, prompt, negative_prompt="", prompts=None, interpolation="slerp", normalize=True) -> None:
        self.prompt, self.negative = prompt, negative_prompt
        self.prompts, self.interpolation, self.normalize = prompts, interpolation, normalize

    def set_params(self, params: Dict[str, Any]) -> None:
        self.params.update({k: v for k, v in params.items() if k in MODEL_RUNTIME_PARAMS})

    def process(self, frame: np.ndarray, flip_y: bool = False, control: Optional[np.ndarray] = None) -> np.ndarray:
        rgb = frame[..., :3]
        if flip_y:
            rgb = rgb[::-1]  # the model sees an upright image (Unity readback is bottom-up)
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            ys = (np.arange(self.height) * rgb.shape[0] // self.height).astype(np.intp)
            xs = (np.arange(self.width) * rgb.shape[1] // self.width).astype(np.intp)
            rgb = rgb[ys][:, xs]
        out = 255 - rgb
        if self._style is not None and control is None and self._style.shape[:2] == rgb.shape[:2]:
            out = 255 - (self._style[::-1] if flip_y else self._style)   # test model: the reference image wins
        src_c = control if control is not None else frame
        up_c = src_c[..., :3][::-1] if flip_y else src_c[..., :3]
        self._last_control = np.ascontiguousarray(255 - up_c)     # test model: the "processed" control = inverted
        if control is not None:      # test model: a separate control image replaces the input
            crgb = control[..., :3]
            if flip_y:
                crgb = crgb[::-1]
            out = 255 - crgb
        if flip_y:
            out = out[::-1]
        if self.frame_ms:
            time.sleep(self.frame_ms / 1000.0)
        return np.ascontiguousarray(out)

    def set_style_image(self, frame: np.ndarray, flip_y: bool = False) -> None:
        self._style = np.ascontiguousarray(frame[..., :3])      # kept as sent; process() applies the flips

    def unload(self) -> None:
        self.loaded = False


class StreamDiffusionPipeline(PipelineBase):
    """Real pipeline. Must be created and used on the inference thread (owns the CUDA context)."""

    def __init__(self, basefolder: str):
        self.basefolder = Path(basefolder)
        self.wrapper = None
        self._torch = None
        self.v2v = False
        self._fx_sent = {"pre": "[]", "post": "[]"}   # last FX chains forwarded to the library (as JSON)
        self.ip_loaded = False      # an IP-Adapter is part of the loaded model
        self.cn_models = []         # ControlNet model ids that actually loaded (order = engine order)
        self.lora_paths = []        # LoRAs that actually loaded (normalised paths)
        self.load_warnings = []     # human-readable problems of the last load (shown by the client)

    # defaults mirror StreamDiffusionTD 0.4 td_config.yaml
    def _build_config(self, known: Dict[str, Any]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "model_id": "stabilityai/sdxl-turbo",
            "t_index_list": [1],
            "width": 512,
            "height": 512,
            "device": "cuda",
            "dtype": "float16",
            "guidance_scale": 1.0,
            "num_inference_steps": 50,
            "seed": 2416333,
            "delta": 1.0,
            "prompt": "a fancy banana",
            "negative_prompt": "",
            "mode": "img2img",
            "frame_buffer_size": 1,
            "use_denoising_batch": True,
            "use_tiny_vae": True,
            "acceleration": "tensorrt",
            "cfg_type": "self",
            "do_add_noise": True,
            "warmup": 10,
            "use_safety_checker": False,
            "build_engines_if_missing": True,
            "static_shapes": True,
            "scheduler": "lcm",
            "sampler": "normal",
            "engine_dir": str(self.basefolder / "engines" / "td"),
            "output_type": "np",  # uint8 HWC RGB through a pinned buffer
            "use_controlnet": False,
            "use_ipadapter": False,
        }
        cfg.update(known)
        return cfg

    def load(self, cfg: Dict[str, Any], progress: ProgressFn) -> List[str]:
        import torch

        self._torch = torch
        known, unknown = split_load_keys(cfg)
        full = self._build_config(known)
        progress(
            "build",
            None,
            f"Loading {full['model_id']} {full['width']}x{full['height']} ({full['acceleration']}). "
            "First time builds TensorRT engines and can take 20+ minutes.",
        )
        from streamdiffusion.config import create_wrapper_from_config

        self.unload()
        self.v2v = bool(full.get("use_cached_attn", False))
        self.load_warnings = []
        for key, allowed, default in (
            ("mode", ("img2img", "txt2img"), "img2img"),
            ("cfg_type", ("self", "initialize", "full", "none"), "self"),
            ("scheduler", ("lcm", "tcd"), "lcm"),
            ("sampler", ("simple", "sgm_uniform", "normal", "ddim", "beta", "karras"), "normal"),
        ):
            if full.get(key) not in allowed:
                self.load_warnings.append(f"{key} '{full.get(key)}' is not supported; using '{default}'")
                full[key] = default
        cns = []
        for c in (full.get("controlnets") or []):
            if isinstance(c, dict) and str(c.get("model_id", "")).strip():
                cns.append({
                    "model_id": str(c["model_id"]).strip(),
                    "preprocessor": str(c.get("preprocessor") or "passthrough"),
                    "conditioning_scale": float(c.get("conditioning_scale", 1.0)),
                    "enabled": bool(c.get("enabled", True)),
                    "preprocessor_params": dict(c.get("preprocessor_params") or {}) or None,
                })
        if cns:
            full["controlnets"] = cns
            full["use_controlnet"] = True
            mid = str(full.get("model_id", "")).lower()
            full["pipeline_type"] = "sdxl" if ("xl" in mid or "sdxl" in mid) else "sd1.5"
        else:
            full.pop("controlnets", None)
            full["use_controlnet"] = False
        self.cn_models = []
        ips = []
        from . import ipadapter as ipa

        for c in (full.get("ipadapters") or []):
            n = ipa.normalize(c) if isinstance(c, dict) else None
            if n:
                ips.append(n)
        if ips:
            ipa.patch_library()
            for c in ips:
                if c.get("type") == "faceid":
                    name = c.get("insightface_model_name", ipa.DEFAULT_FACE_MODEL)
                    progress("build", 0.05, f"face model {name} (one time, ~280 MB)")
                    if not ipa.ensure_face_model(name):
                        self.load_warnings.append(
                            f"face model '{name}' could not be downloaded; FaceID will try its own downloader")
        ips = ips[:1]                      # one image adapter at a time
        if ips:
            full["ipadapters"] = ips
            full["use_ipadapter"] = True
        else:
            full.pop("ipadapters", None)
            full["use_ipadapter"] = False
        self.ip_loaded = False
        self._fx_sent = {"pre": "[]", "post": "[]"}
        if full["mode"] == "txt2img" and full["cfg_type"] != "none":
            self.load_warnings.append("txt2img needs cfg_type 'none'; using it")
            full["cfg_type"] = "none"
        want = [str(p).replace("\\", "/") for p in (full.get("lora_dict") or {})]
        self.wrapper = create_wrapper_from_config(full)
        torch.cuda.synchronize()
        self.width, self.height = int(full["width"]), int(full["height"])
        self.loaded = True
        self.ip_loaded = bool(ips) and bool(getattr(self.wrapper, "use_ipadapter", True))
        if cns:
            try:
                mod = getattr(self.wrapper.stream, "_controlnet_module", None)
                present = [getattr(c, "model_id", None) for c in (getattr(mod, "controlnets", None) or [])]
                self.cn_models = [m for m in present if m] or [c["model_id"] for c in cns]
            except Exception:
                self.cn_models = [c["model_id"] for c in cns]
            if len(self.cn_models) != len(cns):
                self.load_warnings.append(
                    f"{len(cns) - len(self.cn_models)} ControlNet(s) were dropped (duplicate or failed to load)")
        try:
            order = getattr(self.wrapper.stream, "_lora_order", None) or []
            self.lora_paths = [str(name).replace("\\", "/") for name, _adapter in order]
        except Exception:
            self.lora_paths = []
        for p in want:
            if p not in self.lora_paths:
                self.load_warnings.append(
                    f"LoRA not loaded: {p} (wrong model family, unreadable file or unsupported format)")
        try:
            from . import engines

            engines.record_build(
                full["engine_dir"], str(full["model_id"]), self.width, self.height,
                len(full.get("t_index_list") or [1]),
                dict({k: full[k] for k in ("mode", "cfg_type", "use_cached_attn", "use_feature_injection",
                                           "max_cache_maxframes", "lora_dict") if k in full},
                     controlnets=[c["model_id"] for c in cns],
                     ipadapter=ips[0].get("source_path", ips[0]["ipadapter_model_path"]) if ips else None),
            )
        except Exception:
            pass
        progress("ready", 1.0, "model loaded")
        return unknown

    def set_prompt(self, prompt, negative_prompt="", prompts=None, interpolation="slerp", normalize=True) -> None:
        if prompts:
            plist = [(str(p["text"]), float(p.get("weight", 1.0))) for p in prompts if str(p.get("text", "")).strip()]
        else:
            plist = []
        if not plist:
            plist = [(prompt, 1.0)]
        kw = dict(prompt_list=plist, negative_prompt=negative_prompt)
        if len(plist) > 1:
            kw["prompt_interpolation_method"] = interpolation
            kw["normalize_prompt_weights"] = bool(normalize)
        self.wrapper.update_stream_params(**kw)

    def set_params(self, params: Dict[str, Any]) -> None:
        kw: Dict[str, Any] = {}
        for k in ("guidance_scale", "delta", "num_inference_steps", "t_index_list"):
            if k in params:
                kw[k] = params[k]
        fx = params.get("fx")
        if isinstance(fx, dict):
            import json

            for stage, kwname in (("pre", "image_preprocessing_config"), ("post", "image_postprocessing_config")):
                chain = fx.get(stage)
                if not isinstance(chain, list):
                    continue
                clean = []
                for d in chain:
                    if isinstance(d, dict) and str(d.get("type")) in FX_TYPES:
                        clean.append({"type": str(d["type"]), "enabled": bool(d.get("enabled", True)),
                                      "params": {str(k): v for k, v in (d.get("params") or {}).items()}})
                key = json.dumps(clean, sort_keys=True)
                if key != self._fx_sent.get(stage):          # only talk to the library when the chain changed
                    old_types = [d["type"] for d in json.loads(self._fx_sent.get(stage) or "[]")]
                    new_types = [d["type"] for d in clean]
                    if old_types and old_types != new_types[:len(old_types)]:
                        # the library cannot swap a processor's TYPE in place (it keeps the old one): clear first
                        try:
                            self.wrapper.update_stream_params(**{kwname: []})
                        except Exception:
                            pass
                    self._fx_sent[stage] = key
                    kw[kwname] = clean
        ipp = params.get("ipadapter")
        if self.ip_loaded and isinstance(ipp, dict) and ipp:
            d = {}
            if "scale" in ipp:
                d["scale"] = float(ipp["scale"])
            if "enabled" in ipp:
                d["enabled"] = bool(ipp["enabled"])
            if d:
                kw["ipadapter_config"] = d
        cnp = params.get("controlnets")
        if self.cn_models and isinstance(cnp, list):
            have = {}
            for c in cnp:
                if isinstance(c, dict) and c.get("model_id") in self.cn_models:
                    have[c["model_id"]] = c
            cfgs = []
            for mid in self.cn_models:          # always the full loaded set, in order: the library removes others
                c = have.get(mid, {})
                d = {"model_id": mid,
                     "conditioning_scale": float(c.get("conditioning_scale", 1.0)),
                     "enabled": bool(c.get("enabled", True))}
                if c.get("preprocessor_params"):
                    d["preprocessor_params"] = dict(c["preprocessor_params"])
                cfgs.append(d)
            kw["controlnet_config"] = cfgs
        lw = params.get("lora_weights")
        if isinstance(lw, dict) and lw and getattr(self, "lora_paths", None):
            kw["lora_weights"] = {str(k).replace("\\", "/"): float(v) for k, v in lw.items()}
        # StreamV2V live knobs; the library ignores them when the model has no cached attention
        if getattr(self, "v2v", False):
            for k, cast in (("fi_strength", float), ("fi_threshold", float),
                            ("cache_interval", int), ("cache_maxframes", int)):
                if k in params:
                    kw[k] = cast(params[k])
        seeds = params.get("seed_list") or []
        if seeds:
            kw["seed_list"] = [(int(s["seed"]), float(s.get("weight", 1.0))) for s in seeds]
            kw["seed_interpolation_method"] = params.get("seed_interpolation", "average")
            kw["normalize_seed_weights"] = bool(params.get("normalize_seed_weights", True))
        elif "seed" in params:
            kw["seed"] = int(params["seed"])
        if kw:
            self.wrapper.update_stream_params(**kw)

        sf = params.get("similar_filter")
        if isinstance(sf, dict):
            stream = self.wrapper.stream
            if sf.get("enabled"):
                stream.enable_similar_image_filter(float(sf.get("threshold", 0.98)), int(sf.get("max_skip", 10)))
            else:
                stream.disable_similar_image_filter()
        if "add_noise" in params:
            self.wrapper.stream.do_add_noise = bool(params["add_noise"])

    def get_control_preview(self) -> Optional[np.ndarray]:
        """Processed control image of the first ControlNet (upright RGB uint8 HWC) or None."""
        if not self.cn_models or self.wrapper is None:
            return None
        try:
            mod = getattr(self.wrapper.stream, "_controlnet_module", None)
            imgs = getattr(mod, "controlnet_images", None)
            if not imgs or imgs[0] is None:
                return None
            t = imgs[0]
            if t.dim() == 4:
                t = t[0]
            t = t.detach().float()
            if float(t.min()) < -0.01:                 # [-1, 1] -> [0, 1]
                t = (t + 1.0) * 0.5
            t = t.clamp(0.0, 1.0)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            return np.ascontiguousarray((t[:3].permute(1, 2, 0) * 255.0).round().byte().cpu().numpy())
        except Exception:
            return None

    def set_style_image(self, frame: np.ndarray, flip_y: bool = False) -> None:
        """New IP-Adapter reference image (RGB uint8 HWC, bottom-up when flip_y)."""
        if not self.ip_loaded or self.wrapper is None:
            return
        from PIL import Image

        up = frame[..., :3][::-1] if flip_y else frame[..., :3]
        upd = self.wrapper.stream._param_updater
        # The embedding preprocessor skips the CLIP encode when the input tensor lands on the same memory
        # address as the previous one (a streaming shortcut). A new image is usually allocated at the same
        # address, so without this the FIRST reference would stick forever. Forget the shortcut first.
        for pre, _key in getattr(upd, "_embedding_preprocessors", []) or []:
            if hasattr(pre, "_last_input_ptr"):
                pre._last_input_ptr = -1
                pre._cached_embeds = None
        self.wrapper.update_style_image(Image.fromarray(np.ascontiguousarray(up)))
        # The image tokens are concatenated into the prompt embeddings when those are built, so a new
        # style image only takes effect after the prompt embeddings are re-blended (TouchDesigner does the
        # same after every style update). Without this the reference is encoded but never used.
        upd._apply_prompt_blending(upd._last_prompt_interpolation_method)

    def process(self, frame: np.ndarray, flip_y: bool = False, control: Optional[np.ndarray] = None) -> np.ndarray:
        torch = self._torch
        F = torch.nn.functional
        if self._fx_sent.get("post", "[]") != "[]":
            # The library's post-FX orchestrator returns its previous result whenever the decoded image sits at
            # the same GPU address as last frame (it always does), which freezes the output on the first frame
            # and ignores later parameter changes. Forget that cache before every frame.
            orch = getattr(self.wrapper.stream, "postprocessing_orchestrator", None)
            if orch is not None:
                orch.clear_cache()
        t = torch.from_numpy(frame[..., :3]).to("cuda", non_blocking=True)  # HWC uint8
        x = t.permute(2, 0, 1).unsqueeze(0).to(torch.float16).div_(127.5).sub_(1.0)  # [-1, 1]
        if flip_y:
            x = x.flip(2)
        if x.shape[2] != self.height or x.shape[3] != self.width:
            x = F.interpolate(x, size=(self.height, self.width), mode="bilinear", antialias=True)
        if self.cn_models:
            # the control image is the (upright) input picture; each ControlNet runs its own preprocessor on it
            from PIL import Image

            src = control if control is not None else frame      # separate control image, else the input itself
            up = src[..., :3][::-1] if flip_y else src[..., :3]
            pil = Image.fromarray(np.ascontiguousarray(up))
            for i in range(len(self.cn_models)):
                self.wrapper.update_control_image(i, pil)
        out = self.wrapper(image=x)  # numpy uint8 HWC RGB (pinned buffer view)
        if flip_y:
            out = out[::-1]
        return out

    @staticmethod
    def gpu_used_gb() -> float:
        """Whole-GPU memory in use (all processes, incl. the Editor) via nvidia-smi; -1 if unavailable."""
        import subprocess

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()[0]
            return float(out) / 1024.0
        except Exception:
            return -1.0

    def _torch_used_gb(self) -> float:
        try:
            return self._torch.cuda.memory_reserved() / 1024**3 if self._torch is not None else 0.0
        except Exception:
            return 0.0

    def unload(self) -> Dict[str, float]:
        """Release the model and give the memory back; returns before/after figures (GB)."""
        import gc
        import time

        self.loaded = False
        stats = {
            "torch_before_gb": round(self._torch_used_gb(), 2),
            "gpu_before_gb": round(self.gpu_used_gb(), 2),
        }
        if self.wrapper is not None:
            self.wrapper = None
            for _ in range(2):
                gc.collect()
            if self._torch is not None:
                try:
                    self._torch.cuda.synchronize()
                    self._torch.cuda.empty_cache()
                    self._torch.cuda.ipc_collect()
                except Exception:
                    pass
            # Wait for the driver to report the memory as released (bounded), so a following
            # engine build never starts while the old pipeline is still being torn down.
            last = self.gpu_used_gb()
            deadline = time.time() + 15
            while time.time() < deadline:
                time.sleep(1.0)
                now = self.gpu_used_gb()
                if now >= 0 and abs(now - last) < 0.05:
                    break
                last = now
        stats["torch_after_gb"] = round(self._torch_used_gb(), 2)
        stats["gpu_after_gb"] = round(self.gpu_used_gb(), 2)
        log.info(
            "unload: torch reserved %.2f -> %.2f GB, whole GPU %.2f -> %.2f GB",
            stats["torch_before_gb"], stats["torch_after_gb"], stats["gpu_before_gb"], stats["gpu_after_gb"],
        )
        return stats

    def vram(self) -> Tuple[float, float]:
        if self._torch is None:
            return (0.0, 0.0)
        free, total = self._torch.cuda.mem_get_info()
        return ((total - free) / 1024**3, total / 1024**3)
