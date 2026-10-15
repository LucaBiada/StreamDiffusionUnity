"""Entry point: headless StreamDiffusion server.

Run with the venv interpreter of the StreamDiffusion base folder, never the system Python:

    <basefolder>\\venv\\Scripts\\python.exe run_server.py --basefolder <basefolder>

Use --fake-pipeline to run the protocol/transport without a GPU (tests).
"""

import argparse
import asyncio
import logging
import os
import sys

# Make `server` importable when launched from another working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description="StreamDiffusion control server")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default localhost only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=None, help="shared secret required in hello")
    ap.add_argument("--basefolder", default=None, help="StreamDiffusion checkout containing src/ and venv/")
    ap.add_argument("--allow-shutdown", action="store_true", help="accept the shutdown message (local launch)")
    ap.add_argument("--parent-pid", type=int, default=0, help="exit when this process exits (Windows)")
    ap.add_argument("--fake-pipeline", action="store_true", help="no GPU: invert colours (tests)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-file", default=None,
                    help="also write all output here (flushed per line; survives a crash of the launcher)")
    args = ap.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        from server import diag

        diag.tee_output(args.log_file)
        diag.enable_faulthandler(args.log_file + ".fault")

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    if not args.fake_pipeline and not args.basefolder:
        ap.error("--basefolder is required unless --fake-pipeline is given")
    if args.basefolder and " " in args.basefolder:
        ap.error("--basefolder must not contain spaces")

    from server.app import serve_forever

    try:
        asyncio.run(serve_forever(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
