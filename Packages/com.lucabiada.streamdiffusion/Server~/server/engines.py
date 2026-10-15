"""TensorRT engine cache inspection.

Answers "would loading this model at this size / step count have to BUILD engines?" without
loading anything. A build is the heavy operation (minutes of sustained GPU load); knowing in
advance lets the UI ask first, and lets users build engines outside the app.

Layout written by the StreamDiffusion fork under <engine_dir>/<org>/ :
    <model>--tiny_vae-<True|False>--...--trt<ver>--cc<arch>--res-<H>x<W>/vae_decoder.engine, vae_encoder.engine
    <model>--h<hash>--res-<H>x<W>/unet.engine
and <engine_dir>/build_log.jsonl with one record per built engine (engine_dir, engine_file,
batch_size, opt_resolution, ...). The UNet directory hash covers batch size and features, so the
batch size is read from build_log.jsonl.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _split_model(model_id: str):
    p = model_id.replace("\\", "/").strip("/")
    if "/" in p:
        org, name = p.split("/", 1)
        return org, name.replace("/", "--")
    return ".", p


def _read_log(engine_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    log = engine_dir / "build_log.jsonl"
    if not log.exists():
        return out
    try:
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


MANIFEST = "unity_builds.json"


def feature_key(features: Optional[Dict[str, Any]]) -> str:
    """Compact key of the options that change the UNet engine but are not visible in the folder
    name. 'default' = plain img2img / cfg self / no cached attention (what existed before)."""
    f = features or {}
    mode = str(f.get("mode", "img2img"))
    cfg = str(f.get("cfg_type", "self"))
    v2v = bool(f.get("use_cached_attn", False))
    loras = sorted(str(p).replace("\\", "/") for p in (f.get("lora_dict") or {}))
    cns = sorted(str(c.get("model_id") if isinstance(c, dict) else c) for c in (f.get("controlnets") or []))
    ip = f.get("ipadapter")
    if isinstance(ip, dict):
        ip = ip.get("ipadapter_model_path") or ip.get("model_path")
    ip = str(ip).replace("\\", "/") if ip else ""
    if mode == "img2img" and cfg == "self" and not v2v and not loras and not cns and not ip:
        return "default"
    parts = [mode, cfg]
    if ip:
        import hashlib

        parts.append("ip" + hashlib.sha1(ip.encode("utf-8")).hexdigest()[:8])
    if cns:
        import hashlib

        parts.append("cn" + hashlib.sha1("|".join(cns).encode("utf-8")).hexdigest()[:8])
    if loras:
        import hashlib

        parts.append("lora" + hashlib.sha1("|".join(loras).encode("utf-8")).hexdigest()[:8])
    if v2v:
        parts.append("v2v")
        parts.append("fi" if f.get("use_feature_injection", True) else "nofi")
        parts.append("c" + str(int(f.get("max_cache_maxframes", 4))))
    return "-".join(parts)


def _manifest_path(engine_dir: str) -> Path:
    return Path(engine_dir) / MANIFEST


def _read_manifest(engine_dir: str) -> List[str]:
    try:
        return list(json.loads(_manifest_path(engine_dir).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []


def _entry(model_id: str, width: int, height: int, steps: int, key: str) -> str:
    return f"{model_id}|{int(width)}x{int(height)}|{int(steps)}|{key}"


def record_build(engine_dir: str, model_id: str, width: int, height: int, steps: int,
                 features: Optional[Dict[str, Any]]) -> None:
    """Remember that this exact setup loaded successfully (so its engines exist)."""
    try:
        entries = _read_manifest(engine_dir)
        e = _entry(model_id, width, height, steps, feature_key(features))
        if e not in entries:
            entries.append(e)
            _manifest_path(engine_dir).write_text(json.dumps(entries, indent=1), encoding="utf-8")
    except OSError:
        pass


def scan_loras(models_dir: Optional[str]) -> Dict[str, Any]:
    """List LoRA weight files under <models>/LoRA (and LCM_LoRA). Never raises."""
    out: Dict[str, Any] = {"folder": None, "loras": []}
    if not models_dir:
        return out
    try:
        root = Path(models_dir) / "LoRA"
        out["folder"] = str(root)
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix.lower() in (".safetensors", ".pt", ".ckpt", ".bin"):
                    out["loras"].append({
                        "name": p.relative_to(root).as_posix(),
                        "path": str(p).replace("\\", "/"),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    })
    except OSError:
        pass
    return out


def scan_models(models_dir: Optional[str]) -> Dict[str, Any]:
    """List base models under <models>/Model: single-file checkpoints and diffusers folders. Never raises."""
    out: Dict[str, Any] = {"folder": None, "models": []}
    if not models_dir:
        return out
    try:
        root = Path(models_dir) / "Model"
        out["folder"] = str(root)
        if not root.is_dir():
            return out
        seen_dirs: List[Path] = []
        for p in sorted(root.rglob("*")):
            if any(d in p.parents or d == p for d in seen_dirs):
                continue
            if p.is_dir() and (p / "model_index.json").exists():
                seen_dirs.append(p)
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                out["models"].append({"name": p.relative_to(root).as_posix(), "path": str(p).replace("\\", "/"),
                                      "size_mb": round(size / (1024 * 1024), 1), "kind": "folder"})
            elif p.is_file() and p.suffix.lower() in (".safetensors", ".ckpt"):
                out["models"].append({"name": p.relative_to(root).as_posix(), "path": str(p).replace("\\", "/"),
                                      "size_mb": round(p.stat().st_size / (1024 * 1024), 1), "kind": "file"})
    except OSError:
        pass
    return out


def is_local_model(model_id: str) -> bool:
    """A path (file or folder) rather than a Hugging Face repo id."""
    m = str(model_id)
    return (len(m) > 1 and m[1] == ":") or m.startswith(("/", "\\")) or m.lower().endswith((".safetensors", ".ckpt"))


def estimate_minutes(width: int, height: int, steps: int, missing: List[str]) -> float:
    """Very rough wall-clock estimate from this machine's measured builds (RTX 5090:
    UNet 512x512 1 step ~3-6 min, VAE ~1 min). Real time grows faster than pixel count."""
    ratio = max(0.25, (width * height) / (512 * 512))
    total = 0.0
    if "unet" in missing:
        total += 5.0 * (ratio ** 1.4) * (0.7 + 0.3 * max(1, steps))
    if "vae_decoder" in missing or "vae_encoder" in missing:
        total += 1.5 * (ratio ** 1.2)
    return round(total, 1)


def engine_status(
    engine_dir: Optional[str],
    model_id: str,
    width: int,
    height: int,
    steps: int = 1,
    tiny_vae: bool = True,
    features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return {cached, missing[], estimate_minutes, detail}. Never raises."""
    steps = max(1, int(steps))
    result: Dict[str, Any] = {
        "cached": True,
        "missing": [],
        "estimate_minutes": 0.0,
        "width": width,
        "height": height,
        "steps": steps,
        "detail": "",
    }
    if not engine_dir:
        result["detail"] = "no engine directory configured (fake pipeline or acceleration off)"
        return result
    try:
        root = Path(engine_dir)
        if is_local_model(model_id):
            # the library names local-model engine folders itself; trust only setups loaded before
            ok = _entry(model_id, width, height, steps, feature_key(features)) in _read_manifest(engine_dir)
            result["feature_key"] = feature_key(features)
            result["missing"] = [] if ok else ["unet", "vae_decoder", "vae_encoder"]
            result["cached"] = ok
            result["estimate_minutes"] = 0.0 if ok else estimate_minutes(width, height, steps, ["unet", "vae_decoder"])
            result["detail"] = ("all engines are cached" if ok else
                                "local model not loaded at this setup before: engines will be built")
            return result
        org, name = _split_model(model_id)
        base = root / org
        res = f"res-{int(height)}x{int(width)}"   # the library names it H x W
        missing: List[str] = []

        # VAE (decoder + encoder) for this size
        vae_ok = False
        for d in base.glob(f"{name}--tiny_vae-{bool(tiny_vae)}--*--{res}"):
            if (d / "vae_decoder.engine").exists() and (d / "vae_encoder.engine").exists():
                vae_ok = True
                break
        if not vae_ok:
            missing += ["vae_decoder", "vae_encoder"]

        # UNet for this size and step count (batch)
        log = _read_log(root)
        unet_ok = False
        key = feature_key(features)
        result["feature_key"] = key
        if key != "default":
            # the folder hash hides the feature set; trust only setups this app has built before
            unet_ok = _entry(model_id, width, height, steps, key) in _read_manifest(engine_dir)
        for d in ([] if key != "default" else base.glob(f"{name}--h*--{res}")):
            if not (d / "unet.engine").exists():
                continue
            batches = [
                int(r.get("batch_size", 0))
                for r in log
                if r.get("engine_dir") == d.name and r.get("engine_file") == "unet.engine"
            ]
            if not batches or steps in batches:   # no log entry = cannot tell, accept
                unet_ok = True
                break
        if not unet_ok:
            missing.append("unet")

        if missing and features and features.get("controlnets"):
            result["estimate_minutes"] = 0.0     # recomputed below with the ControlNet engines added
        result["missing"] = missing
        result["cached"] = not missing
        result["estimate_minutes"] = estimate_minutes(width, height, steps, missing) if missing else 0.0
        if missing and features and features.get("controlnets"):
            result["estimate_minutes"] = round(result["estimate_minutes"] + 5.0 * len(features["controlnets"]), 1)
        result["detail"] = (
            "all engines are cached" if not missing else "engines to build: " + ", ".join(missing)
        )
    except Exception as exc:  # inspection must never break the server
        result["cached"] = True
        result["detail"] = f"could not inspect the engine cache: {type(exc).__name__}: {exc}"
    return result
