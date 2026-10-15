"""Receive from an NDI source the way Resolume/OBS would, and save one frame.

    <venv>\\python.exe tools\\ndi_probe.py --name StreamDiffusion --seconds 4 --save frame.png
"""

import argparse
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="StreamDiffusion", help="substring of the NDI source name")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    from cyndilib.finder import Finder
    from cyndilib.receiver import Receiver, ReceiveFrameType
    from cyndilib.video_frame import VideoRecvFrame
    from cyndilib.wrapper.ndi_recv import RecvBandwidth, RecvColorFormat

    finder = Finder()
    finder.open()
    src = None
    t0 = time.time()
    while time.time() - t0 < 8 and src is None:
        for n in finder.get_source_names():
            if args.name in n:
                src = finder.get_source(n)
                print("found source:", n, flush=True)
                break
        else:
            time.sleep(0.2)
    if src is None:
        print("NOT FOUND. visible sources:", finder.get_source_names())
        return 1

    receiver = Receiver(color_format=RecvColorFormat.RGBX_RGBA, bandwidth=RecvBandwidth.highest)
    vf = VideoRecvFrame()
    receiver.set_video_frame(vf)
    receiver.set_source(src)
    t0 = time.time()
    while not receiver.is_connected() and time.time() - t0 < 5:
        time.sleep(0.05)
    print("connected:", receiver.is_connected(), flush=True)

    frames, last = 0, None
    t_end = time.time() + args.seconds
    t_first = None
    while time.time() < t_end:
        r = receiver.receive(ReceiveFrameType.recv_video, 200)
        if r & ReceiveFrameType.recv_video:
            frames += 1
            if t_first is None:
                t_first = time.time()
            w, h = vf.get_resolution()
            dest = np.empty(vf.get_data_size(), dtype=np.uint8)
            if vf.fill_p_data(dest):
                stride = vf.get_line_stride()
                if stride >= w * 4 and dest.size >= stride * h:
                    last = dest[: stride * h].reshape(h, stride)[:, : w * 4].reshape(h, w, 4).copy()
    dt = (time.time() - t_first) if t_first else 0
    print(f"received {frames} frames in {args.seconds:.0f}s"
          + (f" (~{frames / dt:.1f} fps)" if dt > 0 else "")
          + (f", resolution {last.shape[1]}x{last.shape[0]}" if last is not None else ""))
    if last is not None and args.save:
        from PIL import Image

        Image.fromarray(last[..., :3]).save(args.save)
        print("saved", args.save)
    receiver.disconnect() if hasattr(receiver, "disconnect") else None
    finder.close()
    return 0 if frames > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
