"""Control protocol constants and small helpers (plan section 3)."""

from typing import Any, Dict

PROTOCOL_VERSION = 1

# Parameters that can change at runtime without rebuilding engines.
# Values are forwarded to wrapper.update_stream_params(...) unless in SERVER_SIDE_PARAMS.
MODEL_RUNTIME_PARAMS = {
    "guidance_scale",
    "delta",
    "num_inference_steps",
    "t_index_list",
    "seed",
    # seed blending: [{"seed": int, "weight": float}, ...] (empty list = single seed)
    "seed_list",
    "seed_interpolation",        # average | slerp | cosine_weighted
    "normalize_seed_weights",
    # similar-frame filter: {"enabled": bool, "threshold": float, "max_skip": int}
    "similar_filter",
    "add_noise",
    # StreamV2V temporal consistency (only effective when the model was loaded with use_cached_attn)
    "fi_strength",       # feature injection blend, 0..1
    "fi_threshold",      # cosine-similarity gate, 0..1
    "cache_interval",    # store a frame in the attention cache every N frames
    "cache_maxframes",   # how many past frames the attention looks at (<= max_cache_maxframes)
    # ControlNet live values: [{model_id, conditioning_scale, enabled, preprocessor_params}] for the ControlNets
    # that were loaded (the SET and the preprocessors are load-time; unknown entries are ignored, never added live).
    "controlnets",
    # IP-Adapter live values: {"scale": float, "enabled": bool}. The adapter and its encoder are load-time.
    "ipadapter",
    # FX processors, applied live (no engine rebuild):
    #   {"pre": [{"type", "enabled", "params": {...}}, ...],   # on the input picture, before the model
    #    "post": [...]}                                          # on the result, after the model
    # Each list is the complete chain, in order; a shorter list removes the extra processors.
    "fx",
    # LoRA live weights: {lora_path: weight}. The set of LoRAs is load-time (lora_dict); weights are live.
    "lora_weights",
}
# Parameters handled by the server itself.
SERVER_SIDE_PARAMS = {
    "flip_y",  # flip input before the model and the result after it
    "max_fps",  # 0 = unlimited
    "control_source",  # "input" (the model input is also the control image) | "separate" (the control ring)
    "control_preview",  # publish the processed control image (edge/depth/pose map) of the first ControlNet
}
ALL_RUNTIME_PARAMS = MODEL_RUNTIME_PARAMS | SERVER_SIDE_PARAMS

DEFAULT_PARAMS: Dict[str, Any] = {
    "guidance_scale": 1.0,
    "delta": 1.0,
    "num_inference_steps": 50,
    "t_index_list": [1],
    "seed": 2416333,
    "seed_list": [],
    "seed_interpolation": "average",
    "normalize_seed_weights": True,
    "similar_filter": {"enabled": False, "threshold": 0.98, "max_skip": 10},
    "add_noise": True,
    "lora_weights": {},
    "ipadapter": {},
    "fx": {"pre": [], "post": []},
    "controlnets": [],
    "fi_strength": 0.75,
    "fi_threshold": 0.98,
    "cache_interval": 1,
    "cache_maxframes": 1,
    "flip_y": False,
    "control_source": "input",
    "control_preview": False,
    "max_fps": 0,
}

# load_model keys that are accepted (everything else is reported as a warning).
LOAD_MODEL_KEYS = {
    "model_id",
    "width",
    "height",
    "mode",
    "acceleration",
    "use_tiny_vae",
    "cfg_type",
    "scheduler",
    "sampler",
    "static_shapes",
    "warmup",
    "do_add_noise",
    "use_lcm_lora",
    "t_index_list",
    "guidance_scale",
    "delta",
    "num_inference_steps",
    "seed",
    "prompt",
    "negative_prompt",
    # StreamV2V (cached attention). Changes the UNet engine: reload class, cached by feature key.
    "lora_dict",             # {path or HF id: weight}; the SET of LoRAs is baked into the engine
    "use_controlnet",
    "use_ipadapter",
    "ipadapters",            # [{ipadapter_model_path, image_encoder_path, scale, enabled, type}] (one is used)
    "controlnets",           # [{model_id, preprocessor, conditioning_scale, enabled, preprocessor_params}]
    "pipeline_type",
    "use_cached_attn",
    "use_feature_injection",
    "max_cache_maxframes",
    "cache_maxframes",
    "cache_interval",
    "fi_strength",
    "fi_threshold",
}


def msg(type_: str, **payload: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": type_}
    out.update(payload)
    return out


def err(code: str, message: str, req_id: Any = None, fatal: bool = False) -> Dict[str, Any]:
    out = msg("error", code=code, message=message, fatal=fatal)
    if req_id is not None:
        out["id"] = req_id
    return out
