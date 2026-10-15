"""Collect everything relevant to an engine-build incident into one report.

Run it after a GPU reset / crash (Unity can be closed or reopened):

    <venv>\\python.exe tools\\collect_diagnostics.py --base E:\\AI\\SDUnity

It reads <base>\\logs\\diagnostics\\build-session.json (written by the app when a build starts)
and gathers: the session marker, the GPU recorder CSV around the end, Windows GPU / driver /
crash events since the build started, GPU watchdog dumps, the server log tail, the engine
cache and build log, the Unity Editor log tail, and the current GPU state. Output:
<base>\\logs\\diagnostics\\report-<time>.md (also printed).
"""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys


def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:
        return f"(failed: {type(exc).__name__}: {exc})"


def tail(path, n=60):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception as exc:
        return f"(cannot read {path}: {exc})"


def fence(text, lang=""):
    return "```" + lang + "\n" + text.rstrip() + "\n```\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="install base folder (contains logs/ and StreamDiffusion/)")
    ap.add_argument("--hours", type=float, default=0, help="if there is no marker, look back this many hours")
    args = ap.parse_args()

    diag = os.path.join(args.base, "logs", "diagnostics")
    marker_path = os.path.join(diag, "build-session.json")
    marker = {}
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)
        except Exception as exc:
            print(f"warning: marker unreadable ({exc}); continuing without it")

    if marker.get("startedUtc"):
        start_utc = datetime.datetime.fromisoformat(marker["startedUtc"].replace("Z", "+00:00"))
    else:
        start_utc = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=args.hours or 3)
    start_local = start_utc.astimezone()
    ps_start = start_local.strftime("%Y-%m-%d %H:%M:%S")

    out = []
    out.append(f"# Engine build diagnostics report\n\nGenerated {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    out.append("## Session marker\n" + fence(json.dumps(marker, indent=2) if marker else "(no marker found)", "json"))

    # ---- GPU recorder
    csv_path = marker.get("recorderCsv") or ""
    if csv_path and os.path.exists(csv_path):
        out.append("## GPU recorder\n")
        out.append(f"File: `{csv_path}`\n")
        lines = open(csv_path, "r", encoding="utf-8", errors="replace").read().splitlines()
        out.append(f"Samples: {len([l for l in lines if not l.startswith('#')])}\n")
        out.append("First lines:\n" + fence("\n".join(lines[:4])))
        out.append("Last 40 lines (the end of the recording):\n" + fence("\n".join(lines[-40:])))
        snap = csv_path + ".snapshot.txt"
        if os.path.exists(snap):
            head = open(snap, "r", encoding="utf-8", errors="replace").read().splitlines()[:30]
            out.append("Start snapshot (driver, display mode, TDR settings):\n" + fence("\n".join(head)))
        apps = csv_path + ".apps.txt"
        if os.path.exists(apps):
            out.append("GPU processes (last entries):\n" + fence(tail(apps, 6)))
    else:
        out.append("## GPU recorder\n(no recorder CSV recorded in the marker)\n")

    # ---- Windows events
    ps = (
        f"$s=[datetime]'{ps_start}'; "
        "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$s} -ErrorAction SilentlyContinue | "
        "Where-Object { $_.ProviderName -match 'nvlddmkm|Display|dxgkrnl|WHEA|Kernel-Power|nvstor|volmgr' } | "
        "Select-Object -First 60 TimeCreated,ProviderName,Id,LevelDisplayName,Message | Format-List | Out-String -Width 220"
    )
    out.append(f"## Windows System events since {ps_start} (GPU / driver / power)\n"
               + fence(run(["powershell", "-NoProfile", "-Command", ps]) or "(none)"))
    ps2 = (
        f"$s=[datetime]'{ps_start}'; "
        "Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=$s; Level=1,2} -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Message -match 'Unity|python|nvidia|StreamDiffusion' } | "
        "Select-Object -First 20 TimeCreated,ProviderName,Id,Message | Format-List | Out-String -Width 220"
    )
    out.append("## Windows Application errors mentioning Unity / python / NVIDIA\n"
               + fence(run(["powershell", "-NoProfile", "-Command", ps2]) or "(none)"))

    # ---- watchdog dumps
    dumps = []
    for pat in (r"C:\Windows\LiveKernelReports\WATCHDOG\*", r"C:\Windows\LiveKernelReports\*.dmp",
                r"C:\Windows\Minidump\*.dmp"):
        for p in glob.glob(pat):
            try:
                if os.path.getmtime(p) >= start_utc.timestamp() - 60:
                    dumps.append(f"{datetime.datetime.fromtimestamp(os.path.getmtime(p)):%H:%M:%S}  {p}")
            except OSError:
                pass
    out.append("## Windows GPU watchdog / crash dumps since the build started\n" + fence("\n".join(dumps) or "(none)"))

    # ---- server log
    slog = marker.get("serverLogPath") or ""
    if slog and os.path.exists(slog):
        out.append(f"## Server log tail\nFile: `{slog}`\n" + fence(tail(slog, 80)))
        fault = slog + ".fault"
        if os.path.exists(fault) and os.path.getsize(fault) > 0:
            out.append("## Server faulthandler dump (Python stacks at the time)\n" + fence(tail(fault, 120)))
    else:
        logs = sorted(glob.glob(os.path.join(args.base, "logs", "server-*.log")), key=os.path.getmtime)
        if logs:
            out.append(f"## Server log tail (latest: `{logs[-1]}`)\n" + fence(tail(logs[-1], 80)))

    # ---- engine cache
    eng = os.path.join(args.base, "StreamDiffusion", "engines", "td")
    if os.path.isdir(eng):
        rows = []
        for root, _d, files in os.walk(eng):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    m = os.path.getmtime(p)
                    if m >= start_utc.timestamp() - 60:
                        rows.append((m, os.path.getsize(p), os.path.relpath(p, eng)))
                except OSError:
                    pass
        rows.sort()
        out.append("## Engine cache files written since the build started\n" + fence(
            "\n".join(f"{datetime.datetime.fromtimestamp(m):%H:%M:%S}  {s:>12}  {r}" for m, s, r in rows) or "(none)"))
        out.append("## build_log.jsonl (last 3 records)\n" + fence(tail(os.path.join(eng, "build_log.jsonl"), 3)[:3000]))

    # ---- Unity editor log
    elog = marker.get("editorLogPath") or ""
    proj = marker.get("projectPath") or ""
    for cand in (os.path.join(proj, "Logs", "Editor.log") if proj else "", elog):
        if cand and os.path.exists(cand):
            out.append(f"## Unity Editor log tail\nFile: `{cand}`\n" + fence(tail(cand, 40)))
            break

    # ---- GPU now
    out.append("## GPU now\n" + fence(run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,"
                                           "utilization.gpu,temperature.gpu,pstate,clocks.sm,power.draw",
                                           "--format=csv"])))

    os.makedirs(diag, exist_ok=True)
    report = os.path.join(diag, f"report-{datetime.datetime.now():%Y%m%d-%H%M%S}.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
