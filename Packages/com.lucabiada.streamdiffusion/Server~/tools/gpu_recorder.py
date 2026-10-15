"""Independent GPU flight recorder for engine builds.

Runs as its own process (NOT a child that dies with Unity or the server) so that a GPU
driver reset, a crash of the Editor or of the server leaves a complete record. It writes:

  <out>                 1 Hz CSV: utilisation, memory, temperatures, clocks, power, P-state,
                        PCIe link, throttle reasons (every line is flushed and fsynced)
  <out>.apps.txt        every 5 s: which processes hold GPU memory
  <out>.snapshot.txt    once at start: driver / GPU / display mode, full `nvidia-smi -q`,
                        Windows TDR + hardware scheduling registry values

It stops when the marker file's status is no longer "in_progress" (the build ended), or after
--max-minutes. If Unity crashes the marker stays "in_progress", so recording continues
through the reset and the recovery, which is exactly the part worth seeing.

    <venv>\\python.exe gpu_recorder.py --out gpu.csv --marker build-session.json --max-minutes 45
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

CANDIDATE_FIELDS = [
    "timestamp", "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
    "temperature.gpu", "temperature.memory", "clocks.sm", "clocks.mem", "power.draw", "power.limit",
    "pstate", "fan.speed", "pcie.link.gen.current", "pcie.link.width.current",
    "clocks_throttle_reasons.active", "clocks_event_reasons.active",
]


def smi(*args, timeout=8):
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def valid_fields():
    ok = []
    for f in CANDIDATE_FIELDS:
        code, out = smi(f"--query-gpu={f}", "--format=csv,noheader,nounits")
        if code == 0 and "not a valid field" not in out.lower() and "error" not in out.lower():
            ok.append(f)
    return ok


def write_snapshot(path):
    lines = [f"recorder snapshot {datetime.datetime.now().isoformat()}", ""]
    code, out = smi("--query-gpu=name,driver_version,vbios_version,pci.bus_id,display_active,display_mode,"
                    "driver_model.current,driver_model.pending,compute_mode",
                    "--format=csv")
    lines += ["== GPU / driver ==", out, ""]
    for key in ("TdrDelay", "TdrDdiDelay", "TdrLevel", "TdrLimitCount", "TdrLimitTime", "HwSchMode"):
        try:
            r = subprocess.run(
                ["reg", "query", r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers", "/v", key],
                capture_output=True, text=True, timeout=5)
            val = [l.strip() for l in r.stdout.splitlines() if key in l]
            lines.append(f"{key}: " + (val[0] if val else "(not set, Windows default)"))
        except Exception as exc:
            lines.append(f"{key}: error {exc}")
    lines += ["", "== nvidia-smi -q ==", smi("-q", timeout=20)[1]]
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n".join(lines))


def marker_status(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("status")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker", default=None)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_snapshot(args.out + ".snapshot.txt")
    fields = valid_fields()
    apps_path = args.out + ".apps.txt"

    t_start = time.time()
    t_end = t_start + args.max_minutes * 60.0
    ended_for = 0
    n = 0
    reason = "max duration"
    with open(args.out, "w", encoding="utf-8") as f, open(apps_path, "w", encoding="utf-8") as fa:
        f.write("# " + ",".join(fields) + "\n")
        f.write(f"# recorder pid {os.getpid()} started {datetime.datetime.now().isoformat()}\n")
        f.flush()
        while time.time() < t_end:
            t0 = time.time()
            code, out = smi(f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", timeout=6)
            line = out.strip().replace("\n", " ") if code == 0 else f"# nvidia-smi failed ({code}): {out.strip()[:120]}"
            f.write(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} | {line}\n")
            f.flush()
            os.fsync(f.fileno())

            if n % 5 == 0:
                _, apps = smi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader")
                fa.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + apps.strip().replace("\n", " ; ") + "\n")
                fa.flush()
            n += 1

            if args.marker and n % 2 == 0:
                st = marker_status(args.marker)
                if st is not None and st != "in_progress":
                    ended_for += 2
                    if ended_for >= 6:       # build ended and has stayed ended
                        reason = f"marker status = {st}"
                        break
                else:
                    ended_for = 0
            time.sleep(max(0.0, args.interval - (time.time() - t0)))
        f.write(f"# recorder stopped: {reason} after {time.time() - t_start:.0f} s, {n} samples\n")
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    sys.exit(main())
