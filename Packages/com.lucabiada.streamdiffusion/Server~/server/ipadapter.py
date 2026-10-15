"""IP-Adapter helpers: adapter kind (regular / plus / faceid), image-token count, library patch.

The library derives the adapter flavour from the checkpoint itself (``"latents" in image_proj`` -> plus,
16 image tokens; FaceID checkpoints carry InsightFace embeddings), but the TensorRT UNet engine needs the
token count *before* the adapter is built, from ``num_image_tokens`` in the config. ``config.py`` drops that
key when it prepares the adapter config, so a Plus adapter would be compiled with 4 tokens and fail at
runtime. This module reads the token count from the checkpoint and puts it back.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger("sd.ipadapter")

KINDS = ("regular", "plus", "faceid")
DEFAULT_FACE_MODEL = "buffalo_l"


def kind_from_path(path: str) -> str:
    low = path.lower().replace("\\", "/")
    if "faceid" in low:
        return "faceid"
    if "plus" in low:
        return "plus"
    return "regular"


def resolve_file(path: str) -> Optional[str]:
    """Local file for a path that is either a file on disk or ``repo/folder/file`` on Hugging Face."""
    p = path.replace("\\", "/")
    if os.path.isfile(p):
        return p
    parts = p.split("/")
    if len(parts) < 3:
        return None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id="/".join(parts[:2]), filename="/".join(parts[2:]))
    except Exception as e:  # noqa: BLE001
        log.warning("cannot resolve %s: %s", path, e)
        return None


def as_torch_checkpoint(path: str) -> str:
    """The library only reads torch ``.bin`` adapters (``torch.load``); the SDXL Plus adapters are published
    as ``.safetensors``. Convert once into a cache folder next to the Hugging Face cache and use that."""
    local = resolve_file(path)
    if local is None or not local.endswith(".safetensors"):
        return path
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        out_dir = os.path.join(os.path.dirname(HF_HUB_CACHE), "sdunity_ipadapter_bin")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.basename(local)[: -len(".safetensors")] + ".bin")
        if not os.path.isfile(out) or os.path.getmtime(out) < os.path.getmtime(local):
            import torch
            from safetensors.torch import load_file

            tensors = load_file(local)
            sd: Dict[str, Dict[str, Any]] = {}
            for k, v in tensors.items():
                section, _, name = k.partition(".")
                sd.setdefault(section, {})[name] = v
            tmp = out + ".tmp"
            torch.save(sd, tmp)
            os.replace(tmp, out)
            log.info("converted %s -> %s", local, out)
        return out.replace("\\", "/")
    except Exception as e:  # noqa: BLE001
        log.warning("cannot convert %s: %s", path, e)
        return path


def _checkpoint_keys(path: str) -> Optional[Dict[str, Any]]:
    """The ``image_proj`` section key names of an adapter checkpoint (resolved through the HF cache)."""
    local = resolve_file(path)
    if local is None:
        return None
    try:
        if local.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(local, framework="pt", device="cpu") as f:
                return {k.split(".", 1)[1]: None for k in f.keys() if k.startswith("image_proj.")}
        import torch

        sd = torch.load(local, map_location="cpu", weights_only=True)
        ip = sd.get("image_proj")
        return {k: None for k in ip} if isinstance(ip, dict) else None
    except Exception as e:  # noqa: BLE001
        log.warning("cannot read %s for token detection: %s", path, e)
        return None


def image_tokens(model_path: str, kind: str) -> int:
    """Image tokens the adapter appends to the prompt (4 regular, 16 plus), as the library will see it."""
    keys = _checkpoint_keys(model_path)
    if keys is not None:
        return 16 if "latents" in keys else 4
    return 16 if kind == "plus" else 4


def normalize(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One client adapter entry -> the dict the library config expects (None when unusable)."""
    model = cfg.get("ipadapter_model_path")
    enc = cfg.get("image_encoder_path")
    if not model or not enc:
        return None
    model = str(model).replace("\\", "/")
    kind = str(cfg.get("type") or "").lower()
    if kind not in KINDS:
        kind = kind_from_path(model)
    out = {
        "source_path": model,                       # what the client asked for (engine bookkeeping)
        "ipadapter_model_path": as_torch_checkpoint(model),
        "image_encoder_path": str(enc).replace("\\", "/"),
        "scale": float(cfg.get("scale", 1.0)),
        "enabled": bool(cfg.get("enabled", True)),
        "type": kind,
        "num_image_tokens": image_tokens(model, kind),
    }
    if kind == "faceid":
        out["insightface_model_name"] = str(cfg.get("insightface_model_name") or DEFAULT_FACE_MODEL)
    return out


_patched = False


def patch_library() -> None:
    """Keep ``num_image_tokens`` when ``streamdiffusion.config`` prepares the adapter configs."""
    global _patched
    if _patched:
        return
    import streamdiffusion.config as C

    original = C._prepare_ipadapter_configs

    def prepare(config):
        out = original(config)
        for src, dst in zip(config.get("ipadapters") or [], out):
            for k in ("num_image_tokens", "insightface_model_name"):
                if src.get(k) is not None:
                    dst[k] = src[k]
        return out

    C._prepare_ipadapter_configs = prepare
    _patched = True


# ---------------------------------------------------------------------------------------------- face model
FACE_MODEL_URLS = {
    "buffalo_l": [
        # the user's own mirror first (a GitHub release asset of the fork), then the original
        "https://github.com/LucaBiada/StreamDiffusion/releases/download/models-v1/buffalo_l.zip",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    ],
}
_FACE_FILES = ("det_10g.onnx", "w600k_r50.onnx")


def face_model_dir(name: str = DEFAULT_FACE_MODEL) -> str:
    return os.path.join(os.path.expanduser("~"), ".insightface", "models", "models", name)


def ensure_face_model(name: str = DEFAULT_FACE_MODEL, progress=None) -> bool:
    """Make sure the InsightFace pack that FaceID needs is on disk, downloading it robustly.

    insightface's own downloader writes straight into the final zip: one dropped connection leaves a
    broken file and a confusing error. This one resumes, retries, checks the zip and extracts atomically.
    Returns True when the pack is present.
    """
    target = face_model_dir(name)
    if all(os.path.isfile(os.path.join(target, f)) for f in _FACE_FILES):
        return True
    import shutil
    import time
    import zipfile

    import requests

    tmp_dir = os.path.join(os.path.dirname(os.path.dirname(target)), "_download")
    os.makedirs(tmp_dir, exist_ok=True)
    zpath = os.path.join(tmp_dir, name + ".zip.part")
    for url in FACE_MODEL_URLS.get(name, []):
        for attempt in range(6):
            try:
                have = os.path.getsize(zpath) if os.path.isfile(zpath) else 0
                headers = {"Range": f"bytes={have}-"} if have else {}
                with requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
                    if r.status_code == 404:
                        break                          # not published at this mirror: next URL
                    if r.status_code == 416:           # already complete
                        r.close()
                    elif r.status_code in (200, 206):
                        if r.status_code == 200:
                            have = 0
                        total = have + int(r.headers.get("Content-Length", 0))
                        mode = "ab" if (have and r.status_code == 206) else "wb"
                        with open(zpath, mode) as f:
                            got = have
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                f.write(chunk)
                                got += len(chunk)
                                if progress and total:
                                    progress(got / total)
                    else:
                        raise RuntimeError(f"HTTP {r.status_code}")
                done = False
                if zipfile.is_zipfile(zpath):
                    with zipfile.ZipFile(zpath) as z:
                        if z.testzip() is None:
                            stage = target + ".extract"
                            shutil.rmtree(stage, ignore_errors=True)
                            z.extractall(stage)
                            shutil.rmtree(target, ignore_errors=True)
                            os.replace(stage, target)
                            done = True
                os.remove(zpath)                      # installed, or corrupt: start over
                if done:
                    log.info("face model %s installed at %s (from %s)", name, target, url)
                    return True
            except Exception as e:  # noqa: BLE001
                log.warning("face model download (%s) attempt %d failed: %s", url, attempt + 1, e)
                time.sleep(2 + attempt * 2)
    return False
