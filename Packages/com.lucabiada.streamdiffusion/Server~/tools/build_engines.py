"""Build TensorRT engines for a model / resolution / step count, outside the Unity app.

Why: an engine build keeps the GPU busy for minutes. Doing it with Unity closed (and other
GPU-heavy programs closed) means a driver reset cannot take the Editor down with it, and the
result is cached for later runs.

Run with the venv interpreter of the StreamDiffusion base folder:

    <base>\\venv\\Scripts\\python.exe tools\\build_engines.py --basefolder <base>\\StreamDiffusion ^
        --model stabilityai/sdxl-turbo --width 1024 --height 576 --steps 1

--dry-run only reports whether the engines are already cached (no GPU work).
--monitor-file F records GPU utilisation / memory / temperature / clocks once a second, so a
failure can be matched against what the GPU was doing.
"""

import argparse
import datetime
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from server.engines import engine_status  # noqa: E402


def stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{stamp()}] {msg}", flush=True)


def start_monitor(path: str):
    """Sample nvidia-smi once a second into a CSV until stopped."""
    stop = threading.Event()

    def run() -> None:
        query = "timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm,power.draw"
        with open(path, "a", encoding="utf-8") as f:
            f.write("# " + query + "\n")
            while not stop.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    ).stdout.strip()
                    f.write(out + "\n")
                    f.flush()
                    os.fsync(f.fileno())  # survive a hard crash
                except Exception as exc:
                    f.write(f"# monitor error: {exc}\n")
                    f.flush()
                stop.wait(1.0)

    t = threading.Thread(target=run, name="gpu-monitor", daemon=True)
    t.start()
    return stop


def main() -> int:
    ap = argparse.ArgumentParser(description="Build TensorRT engines outside Unity")
    ap.add_argument("--basefolder", required=True, help="StreamDiffusion checkout (contains src/, engines/, venv/)")
    ap.add_argument("--model", default="stabilityai/sdxl-turbo")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1, help="number of denoising steps (batch size of the UNet engine)")
    ap.add_argument("--no-tiny-vae", action="store_true")
    ap.add_argument("--mode", default="img2img", choices=["img2img", "txt2img"])
    ap.add_argument("--cfg-type", default="self", choices=["self", "initialize", "full", "none"])
    ap.add_argument("--scheduler", default="lcm", choices=["lcm", "tcd"])
    ap.add_argument("--sampler", default="normal",
                    choices=["simple", "sgm_uniform", "normal", "ddim", "beta", "karras"])
    ap.add_argument("--controlnet", action="append", default=[],
                    help="ControlNet as MODEL_ID=PREPROCESSOR (repeatable), e.g. diffusers/controlnet-canny-sdxl-1.0=canny")
    ap.add_argument("--ipadapter", default=None, help="IP-Adapter weights, e.g. h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin")
    ap.add_argument("--ipadapter-encoder", default=None, help="its image encoder, e.g. h94/IP-Adapter/sdxl_models/image_encoder")
    ap.add_argument("--v2v", action="store_true", help="build the StreamV2V (cached attention) UNet")
    ap.add_argument("--no-fi", action="store_true", help="with --v2v: without feature injection")
    ap.add_argument("--max-cache-frames", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="only report the cache state")
    ap.add_argument("--monitor-file", default=None, help="CSV file for 1 Hz GPU samples")
    args = ap.parse_args()

    base = Path(args.basefolder).resolve()
    if " " in str(base):
        print("ERROR: basefolder must not contain spaces", file=sys.stderr)
        return 2
    engine_dir = str(base / "engines" / "td")
    steps = max(1, args.steps)
    tiny = not args.no_tiny_vae

    cn_list = []
    for item in args.controlnet:
        mid, _, pre = item.partition("=")
        cn_list.append({"model_id": mid.strip(), "preprocessor": (pre or "passthrough").strip(),
                        "conditioning_scale": 0.5, "enabled": True})
    feats = {"controlnets": [c["model_id"] for c in cn_list], "ipadapter": args.ipadapter, "mode": args.mode, "cfg_type": args.cfg_type, "use_cached_attn": args.v2v, "use_feature_injection": not args.no_fi,
             "max_cache_maxframes": args.max_cache_frames}
    st = engine_status(engine_dir, args.model, args.width, args.height, steps, tiny, features=feats)
    log(f"engine cache for {args.model} {args.width}x{args.height}, {steps} step(s), tiny_vae={tiny}: {st['detail']}"
        + (f" (rough estimate {st['estimate_minutes']} min)" if st["missing"] else ""))
    if args.dry_run:
        return 0 if st["cached"] else 3
    if st["cached"]:
        log("nothing to build")
        return 0

    log("Close Unity and other GPU-heavy programs. The GPU will be under sustained load for several minutes.")
    mon = start_monitor(args.monitor_file) if args.monitor_file else None
    t0 = time.time()
    try:
        import torch  # noqa: F401  (import late: --dry-run must stay GPU-free)

        from server.pipeline import StreamDiffusionPipeline
        from streamdiffusion.config import create_wrapper_from_config

        pipe = StreamDiffusionPipeline(str(base))
        cfg = pipe._build_config({
            "model_id": args.model,
            "width": args.width,
            "height": args.height,
            "use_tiny_vae": tiny,
            "t_index_list": [1 + i * (48 // steps) for i in range(steps)],
            "mode": args.mode, "cfg_type": args.cfg_type, "scheduler": args.scheduler, "sampler": args.sampler,
            **({"use_cached_attn": True, "use_feature_injection": not args.no_fi,
                "max_cache_maxframes": args.max_cache_frames} if args.v2v else {}),
        })
        if args.ipadapter and args.ipadapter_encoder:
            cfg["use_ipadapter"] = True
            from server import ipadapter as ipa

            n = ipa.normalize({"ipadapter_model_path": args.ipadapter, "image_encoder_path": args.ipadapter_encoder})
            cfg["ipadapters"] = [n]
            ipa.patch_library()
        if cn_list:
            cfg["use_controlnet"] = True
            cfg["controlnets"] = cn_list
            cfg["pipeline_type"] = "sdxl" if "xl" in args.model.lower() else "sd1.5"
        cfg["compile_engines_only"] = True   # build and stop; do not load the model
        log("starting engine build (the library logs its own progress below)")
        create_wrapper_from_config(cfg)
        log(f"build finished in {time.time() - t0:.0f} s")
    except Exception as exc:
        log(f"BUILD FAILED after {time.time() - t0:.0f} s: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if mon is not None:
            mon.set()

    from server.engines import record_build

    record_build(engine_dir, args.model, args.width, args.height, steps, feats)   # the build succeeded
    st = engine_status(engine_dir, args.model, args.width, args.height, steps, tiny, features=feats)
    log("result: " + st["detail"])
    return 0 if st["cached"] else 4


if __name__ == "__main__":
    sys.exit(main())
