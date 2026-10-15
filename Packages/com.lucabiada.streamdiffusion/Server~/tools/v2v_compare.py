"""Measure what StreamV2V (cached attention + feature injection) actually changes.

Loads the real pipeline twice (plain, then StreamV2V; both engines must be cached), feeds the
same moving synthetic scene to each setting and prints the frame-to-frame change of the OUTPUT
(flicker) next to the frame-to-frame change of the INPUT (motion), plus how far each setting's
output is from the plain model's output.

    <venv>\\python.exe tools\\v2v_compare.py --basefolder E:\\AI\\SDUnity\\StreamDiffusion
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


STATIC = False


def scene(t: int, size=(512, 512)):
    import numpy as np

    if STATIC:
        t_img = 0           # same picture every frame; only the sensor noise changes
    else:
        t_img = t
    t_noise = t
    t = t_img

    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 90 + 60 * np.sin(xx / 90.0 + t * 0.05)
    img[..., 1] = 100 + 60 * np.sin(yy / 70.0 - t * 0.04)
    img[..., 2] = 120 + 40 * np.sin((xx + yy) / 110.0 + t * 0.03)
    for i in range(5):
        cx = w * (0.5 + 0.35 * np.sin(t * 0.045 + i * 1.3))
        cy = h * (0.5 + 0.35 * np.cos(t * 0.037 + i * 0.9))
        r = 38 + 8 * i
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        col = np.array([220 - 30 * i, 80 + 35 * i, 60 + 30 * i], np.float32)
        img[m] = col
    rng = np.random.default_rng(t_noise)    # sensor-like noise, different every frame
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run(pipe, frames, label, settings=None):
    import numpy as np

    if settings:
        pipe.set_params(settings)
    outs = []
    for t in range(frames):
        outs.append(np.array(pipe.process(scene(t)), copy=True))
    return outs


def flicker(seq):
    import numpy as np

    d = [np.abs(seq[i].astype(np.int16) - seq[i - 1].astype(np.int16)).mean() for i in range(1, len(seq))]
    return float(np.mean(d))


def dist(a, b):
    import numpy as np

    return float(np.mean([np.abs(x.astype(np.int16) - y.astype(np.int16)).mean() for x, y in zip(a, b)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basefolder", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--prompt", default="a colorful abstract painting")
    ap.add_argument("--steps", default="1", help="comma separated t_index_list, e.g. 1 or 12,30")
    ap.add_argument("--static", action="store_true", help="static picture, only noise changes between frames")
    args = ap.parse_args()
    global STATIC
    STATIC = args.static

    base = str(Path(args.basefolder).resolve())
    from server.pipeline import StreamDiffusionPipeline

    tl = [int(x) for x in args.steps.split(",")]
    common = {"width": 512, "height": 512, "t_index_list": tl, "prompt": args.prompt, "seed": 2416333}
    noop = lambda *a, **k: None  # noqa: E731

    inputs = [scene(t) for t in range(args.frames)]
    import numpy as np

    motion = float(np.mean([np.abs(inputs[i].astype(np.int16) - inputs[i - 1].astype(np.int16)).mean()
                            for i in range(1, len(inputs))]))
    print(f"input motion (mean |frame diff|): {motion:.2f}")

    pipe = StreamDiffusionPipeline(base)
    pipe.load(dict(common), noop)
    for t in range(10):                                   # settle
        pipe.process(scene(t))
    plain = run(pipe, args.frames, "plain")
    print(f"plain          : flicker {flicker(plain):.2f}")
    pipe.unload()

    v2v_cfg = dict(common, use_cached_attn=True, use_feature_injection=True, max_cache_maxframes=4,
                   cache_maxframes=1, cache_interval=1, fi_strength=0.75, fi_threshold=0.98)
    pipe = StreamDiffusionPipeline(base)
    pipe.load(v2v_cfg, noop)
    for t in range(10):
        pipe.process(scene(t))

    cases = [
        ("v2v frames=1 fi=0.75 thr=0.98", {"fi_strength": 0.75, "fi_threshold": 0.98, "cache_maxframes": 1, "cache_interval": 1}),
        ("v2v frames=4 fi=0.00", {"cache_maxframes": 4, "fi_strength": 0.0}),
        ("v2v frames=4 fi=0.75 thr=0.98", {"fi_strength": 0.75, "fi_threshold": 0.98}),
        ("v2v frames=4 fi=1.00 thr=0.50", {"fi_strength": 1.0, "fi_threshold": 0.5}),
        ("v2v frames=4 fi=1.00 thr=0.50 int=3", {"cache_interval": 3}),
        ("v2v frames=4 fi=0.00 int=3", {"fi_strength": 0.0}),
    ]
    res = {}
    for label, st in cases:
        pipe.set_params(st)
        for t in range(6):                                # let the cache refill under the new setting
            pipe.process(scene(t))
        o = run(pipe, args.frames, label)
        res[label] = o
        print(f"{label:38s}: flicker {flicker(o):.2f}   mean |out - plain| {dist(o, plain):.2f}")
    a, b = res["v2v frames=4 fi=0.00"], res["v2v frames=4 fi=1.00 thr=0.50"]
    print(f"fi 0 vs fi 1 (thr 0.5) output distance: {dist(a, b):.2f}")
    pipe.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
