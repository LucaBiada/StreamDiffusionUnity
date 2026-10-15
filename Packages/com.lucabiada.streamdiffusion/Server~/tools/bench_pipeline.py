"""Headless benchmark for the StreamDiffusion fork (Phase 1).

Builds the same default configuration StreamDiffusionTD 0.4 uses (SDXL-turbo, 512x512,
TensorRT, tiny VAE, img2img) through the library API, feeds synthetic frames and reports
ms/frame and fps. Writes results to JSON and saves the last output as PNG.

Run with the venv interpreter, never the system one:
    <basefolder>\\venv\\Scripts\\python.exe bench_pipeline.py --basefolder <basefolder>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basefolder", required=True, help="StreamDiffusion checkout (contains src/, venv/)")
    ap.add_argument("--model", default="stabilityai/sdxl-turbo")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--acceleration", default="tensorrt")
    ap.add_argument("--prompt", default="a fancy banana, studio photo")
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    base = Path(args.basefolder).resolve()
    if " " in str(base):
        print("ERROR: basefolder must not contain spaces", file=sys.stderr)
        return 2

    import torch  # noqa: E402  (imported late so --help works anywhere)

    from streamdiffusion.config import create_wrapper_from_config  # noqa: E402

    config = {
        "model_id": args.model,
        "t_index_list": [1],
        "width": args.width,
        "height": args.height,
        "device": "cuda",
        "dtype": "float16",
        "guidance_scale": 1.0,
        "num_inference_steps": 50,
        "seed": 2416333,
        "delta": 1.0,
        "prompt": args.prompt,
        "negative_prompt": "",
        "mode": "img2img",
        "frame_buffer_size": 1,
        "use_denoising_batch": True,
        "use_tiny_vae": True,
        "acceleration": args.acceleration,
        "cfg_type": "self",
        "do_add_noise": True,
        "warmup": args.warmup,
        "use_safety_checker": False,
        "build_engines_if_missing": True,
        "static_shapes": True,
        "scheduler": "lcm",
        "sampler": "normal",
        "engine_dir": str(base / "engines" / "td"),
        "output_type": "pt",
        "use_controlnet": False,
        "use_ipadapter": False,
    }

    t0 = time.perf_counter()
    print(f"[bench] building wrapper: {args.model} {args.width}x{args.height} {args.acceleration}", flush=True)
    wrapper = create_wrapper_from_config(config)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    print(f"[bench] wrapper ready in {load_s:.1f}s", flush=True)

    # Synthetic moving gradient input as a CUDA tensor [1,3,H,W] in [0,1].
    h, w = args.height, args.width
    ys = torch.linspace(0, 1, h, device="cuda").view(1, 1, h, 1).expand(1, 1, h, w)
    xs = torch.linspace(0, 1, w, device="cuda").view(1, 1, 1, w).expand(1, 1, h, w)

    def make_frame(i: int) -> "torch.Tensor":
        phase = (i % 120) / 120.0
        r = (xs + phase) % 1.0
        g = (ys + phase * 0.5) % 1.0
        b = (xs * ys + phase) % 1.0
        return torch.cat([r, g, b], dim=1).contiguous()

    # Extra warm frames so the stream batch is full and the first outputs are meaningful.
    for i in range(args.warmup):
        wrapper(image=make_frame(i))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    last = None
    for i in range(args.frames):
        frame = make_frame(i)
        torch.cuda.synchronize()
        t = time.perf_counter()
        last = wrapper(image=frame)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000.0)

    times_sorted = sorted(times)
    mean = sum(times) / len(times)
    result = {
        "model": args.model,
        "resolution": [args.width, args.height],
        "acceleration": args.acceleration,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": round(load_s, 1),
        "frames": len(times),
        "ms_mean": round(mean, 2),
        "ms_p50": round(times_sorted[len(times) // 2], 2),
        "ms_p95": round(times_sorted[int(len(times) * 0.95)], 2),
        "ms_min": round(times_sorted[0], 2),
        "fps_mean": round(1000.0 / mean, 1),
        "vram_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
    }
    print("[bench] RESULT " + json.dumps(result), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    try:  # save last frame for a visual sanity check
        import numpy as np
        from PIL import Image

        t = last
        if hasattr(t, "detach"):
            t = t.detach().float().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0)
            arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
        else:
            arr = np.array(t)
        Image.fromarray(arr).save(str(Path(args.out).with_suffix(".png")))
    except Exception as exc:  # non-fatal
        print(f"[bench] could not save image: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
