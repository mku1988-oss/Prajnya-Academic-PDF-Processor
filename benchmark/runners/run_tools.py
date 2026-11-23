#!/usr/bin/env python
"""
Run every benchmark tool over the corpus, one paper at a time (protocol §6.9).

For biology researchers: this is the overnight batch. For each tool, for each
paper in benchmark/corpus/MANIFEST.tsv, it runs the tool alone on the GPU,
records wall time, peak GPU memory and peak host memory, and writes the
tool's raw output under benchmark/outputs/run<N>/<tool>/. A paper whose
output already exists is skipped, so the batch can be restarted safely.

Start detached (a background shell job can be killed), from src/v6:
    Start-Process <prajnya env>/python.exe -ArgumentList
      "-u benchmark/runners/run_tools.py --run 1 --tools prajnya,grobid,..."

Timings go to benchmark/outputs/run<N>/timings.jsonl, one line per paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

V6_ROOT = Path(__file__).resolve().parents[2]
CORPUS = V6_ROOT / "benchmark" / "corpus"
OUTPUTS = V6_ROOT / "benchmark" / "outputs"
# Where Prajnya's code is run from (run_prajnya.py, run_llama_server.py).
# Default: this checkout. `--prajnya-root` points at another, so the
# benchmark's code and the Prajnya under test can come from different
# branches (amendment A7: the pilot check of main, run "1b").
PRAJNYA_ROOT = V6_ROOT
# Where each tool is installed, read from the environment or from .env (the
# same file Prajnya reads), so no machine's paths live in the code:
#   BENCH_ENV_ROOT     the folder holding one environment per tool
#                      (bench_docling, bench_marker, bench_mineru, and Prajnya's)
#   BENCH_PRAJNYA_ENV  the name of Prajnya's environment in it (default prajnya)
#   BENCH_LLAMA_SERVER the llama-server binary (else PRAJNYA_LLAMA_SERVER,
#                      Prajnya's own setting, else llama-server on the PATH)
#   BENCH_DOCKER       the docker binary (default: docker on the PATH)
# Environment folders use the Windows layout (python.exe, Scripts/*.exe): the
# published run was made on Windows 11.
load_dotenv(V6_ROOT / ".env")
ENV = Path(os.environ.get("BENCH_ENV_ROOT", "envs"))
PRAJNYA_ENV = os.environ.get("BENCH_PRAJNYA_ENV", "prajnya")
LLAMA_SERVER = Path(os.environ.get("BENCH_LLAMA_SERVER")
                    or os.environ.get("PRAJNYA_LLAMA_SERVER")
                    or shutil.which("llama-server") or "llama-server")
DOCKER = Path(os.environ.get("BENCH_DOCKER") or shutil.which("docker") or "docker")
GROBID_URL = "http://localhost:8070"
TOOL_ORDER = ["prajnya", "grobid", "grobid_consolidated", "docling", "mineru",
              "marker", "docling_formula"]
TIMEOUT_S = {"prajnya": 2400, "marker": 1800}


def main() -> int:
    """Run the chosen tools over the chosen papers and log every timing."""
    global PRAJNYA_ROOT
    args = parse_arguments()
    PRAJNYA_ROOT = Path(args.prajnya_root) if args.prajnya_root else V6_ROOT
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # a tool's Ctrl-C is not ours
    run_dir = OUTPUTS / f"run{args.run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    papers = load_papers(args.roles, args.manifest)
    for tool in [t for t in TOOL_ORDER if t in args.tools]:
        log(run_dir, {"event": "tool_start", "tool": tool,
                      "idle_vram_mib": gpu_memory_used()})
        with tool_service(tool):
            for paper in papers:
                run_one(tool, paper, run_dir)
        log(run_dir, {"event": "tool_end", "tool": tool})
    if "prajnya_online" in args.tools:
        recite_online(run_dir)
    log(run_dir, {"event": "all_done"})
    return 0


# ---------------------------------------------------------------------------
# Helpers (after main, per the project rule)
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="1",
                        help="run name: outputs go to outputs/run<NAME>/")
    parser.add_argument("--tools", default=",".join(TOOL_ORDER + ["prajnya_online"]))
    parser.add_argument("--roles", default="pilot,scored")
    parser.add_argument("--manifest", default="MANIFEST.tsv",
                        help="the manifest under corpus/ (A11: MANIFEST_A11.tsv)")
    parser.add_argument("--prajnya-root", default=None,
                        help="the Prajnya checkout to run (default: this one)")
    args = parser.parse_args()
    args.tools = args.tools.split(",")
    args.roles = args.roles.split(",")
    return args


def load_papers(roles: list[str], manifest: str = "MANIFEST.tsv") -> list[dict]:
    """Manifest rows in role order (pilot first), then manifest order."""
    with open(CORPUS / manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r for role in roles for r in rows if r["role"] == role]


def run_one(tool: str, paper: dict, run_dir: Path) -> None:
    """One tool, one paper: skip if done, else run under the resource meter."""
    out_dir = run_dir / tool
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = CORPUS / "cache" / f"{paper['version']}.pdf"
    if output_exists(tool, paper, out_dir):
        return
    meter = ResourceMeter()
    meter.start()
    started = time.perf_counter()
    try:
        status = TOOL_RUNNERS[tool](paper, pdf, out_dir)
    except Exception as error:                       # recorded, never fatal
        status = f"error: {type(error).__name__}: {error}"[:500]
    seconds = time.perf_counter() - started
    meter.stop()
    log(run_dir, {"tool": tool, "pmcid": paper["pmcid"], "role": paper["role"],
                  "pages": int(paper["pages"]), "seconds": round(seconds, 2),
                  "peak_vram_mib": meter.peak_vram, "peak_ram_used_gb": meter.peak_ram,
                  "status": status})


def output_exists(tool: str, paper: dict, out_dir: Path) -> bool:
    return any(out_dir.glob(f"{paper['pmcid']}*.done"))


def mark_done(out_dir: Path, paper: dict) -> None:
    (out_dir / f"{paper['pmcid']}.done").write_text("ok", encoding="utf-8")


# --- one runner per tool ----------------------------------------------------

def run_prajnya(paper: dict, pdf: Path, out_dir: Path) -> str:
    """Prajnya at `local` (the checkout run), with a fresh llama-server for each paper
    (llama-server's glm-ocr child leaks host RAM on long runs)."""
    paper_dir = out_dir / "input" / paper["pmcid"]
    paper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, paper_dir / f"{paper['pmcid']}.pdf")
    env = {**os.environ, "PYTHONUTF8": "1",
           "PRAJNYA_LEDGER_PATH": str(out_dir / "ledger.csv")}
    server = start_llama_server(out_dir)
    try:
        result = run_process(
            [str(ENV / PRAJNYA_ENV / "python.exe"), "-u", "run_prajnya.py",
             "--pdf", str(paper_dir), "--out", str(out_dir / "rundir"),
             "--noninteractive", "--force", "--citations", "local"],
            cwd=PRAJNYA_ROOT, env=env, log_path=out_dir / f"{paper['pmcid']}.log",
            timeout=TIMEOUT_S["prajnya"])
    finally:
        stop_process_tree(server)
    if result == 0:
        mark_done(out_dir, paper)
    return f"exit {result}"


def run_grobid(paper: dict, pdf: Path, out_dir: Path, consolidate: int = 0) -> str:
    out_file = out_dir / f"{paper['pmcid']}.tei.xml"
    code = run_process(
        ["curl", "-s", "-m", "900", "-o", str(out_file), "-w", "%{http_code}",
         "--form", f"input=@{pdf}",
         "--form-string", f"consolidateHeader={consolidate}",
         "--form-string", f"consolidateCitations={consolidate}",
         "--form-string", "teiCoordinates=figure",
         f"{GROBID_URL}/api/processFulltextDocument"],
        cwd=V6_ROOT, env=dict(os.environ), log_path=out_dir / f"{paper['pmcid']}.log",
        timeout=1000)
    http = (out_dir / f"{paper['pmcid']}.log").read_text(encoding="utf-8").strip()
    if code == 0 and http == "200":
        mark_done(out_dir, paper)
    return f"exit {code} http {http}"


def run_grobid_consolidated(paper: dict, pdf: Path, out_dir: Path) -> str:
    return run_grobid(paper, pdf, out_dir, consolidate=1)


def run_docling(paper: dict, pdf: Path, out_dir: Path, extra: tuple = ()) -> str:
    code = run_process(
        [str(ENV / "bench_docling" / "Scripts" / "docling.exe"), "convert", str(pdf),
         "--to", "json", "--output", str(out_dir), *extra],
        cwd=V6_ROOT, env={**os.environ, "PYTHONUTF8": "1"},
        log_path=out_dir / f"{paper['pmcid']}.log", timeout=1800)
    if code == 0:
        mark_done(out_dir, paper)
    return f"exit {code}"


def run_mineru(paper: dict, pdf: Path, out_dir: Path) -> str:
    code = run_process(
        [str(ENV / "bench_mineru" / "python.exe"), "benchmark/runners/run_mineru.py",
         str(pdf), str(out_dir)],
        cwd=V6_ROOT, env={**os.environ, "PYTHONUTF8": "1",
                          "MINERU_TELEMETRY_DISABLED": "1"},
        log_path=out_dir / f"{paper['pmcid']}.log", timeout=1800)
    if code == 0:
        mark_done(out_dir, paper)
    return f"exit {code}"


def run_marker(paper: dict, pdf: Path, out_dir: Path) -> str:
    code = run_process(
        [str(ENV / "bench_marker" / "Scripts" / "marker_single.exe"), str(pdf),
         "--output_format", "json", "--output_dir", str(out_dir)],
        cwd=V6_ROOT, env={**os.environ, "PYTHONUTF8": "1",
                          "SURYA_INFERENCE_BACKEND": "llamacpp",
                          "LLAMA_CPP_BINARY": str(LLAMA_SERVER)},
        log_path=out_dir / f"{paper['pmcid']}.log", timeout=TIMEOUT_S["marker"])
    if code == 0:
        mark_done(out_dir, paper)
    return f"exit {code}"


def run_docling_formula(paper: dict, pdf: Path, out_dir: Path) -> str:
    """Amendment A12: Docling with its formula reader on (off by default)."""
    return run_docling(paper, pdf, out_dir, extra=("--enrich-formula",))


TOOL_RUNNERS = {"prajnya": run_prajnya, "grobid": run_grobid,
                "grobid_consolidated": run_grobid_consolidated,
                "docling": run_docling, "docling_formula": run_docling_formula, "mineru": run_mineru, "marker": run_marker}


def recite_online(run_dir: Path) -> None:
    """Prajnya `online` = the same reads, re-cited at `online` (`--recite`
    reads no page and calls no model; it only adds the Crossref/OpenAlex
    lookups). Works on a copy so the `local` outputs stay untouched."""
    source, target = run_dir / "prajnya", run_dir / "prajnya_online"
    if (target / "recite.done").exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source / "rundir", target / "rundir")
    shutil.copy2(source / "ledger.csv", target / "ledger.csv")
    started = time.perf_counter()
    code = run_process(
        [str(ENV / PRAJNYA_ENV / "python.exe"), "-u", "run_prajnya.py",
         "--recite", "--citations", "online", "--out", str(target / "rundir"),
         "--noninteractive"],
        cwd=PRAJNYA_ROOT, env={**os.environ, "PYTHONUTF8": "1",
                          "PRAJNYA_LEDGER_PATH": str(target / "ledger.csv"),
                          "PRAJNYA_CITATION_CACHE_PATH": str(target / "citation_cache")},
        log_path=target / "recite.log", timeout=7200)
    log(run_dir, {"tool": "prajnya_online", "event": "recite", "exit": code,
                  "seconds": round(time.perf_counter() - started, 2)})
    if code == 0:
        (target / "recite.done").write_text("ok", encoding="utf-8")


# --- services ---------------------------------------------------------------

class tool_service:
    """Start what a tool needs for the whole batch (GROBID's container) and
    stop it after, so nothing else holds RAM or GPU while other tools run."""

    def __init__(self, tool: str):
        self.tool = tool

    def __enter__(self):
        if self.tool.startswith("grobid"):
            subprocess.run([str(DOCKER), "start", "grobid_bench"], check=False,
                           capture_output=True)
            wait_for_url(f"{GROBID_URL}/api/isalive", timeout=180)
        return self

    def __exit__(self, *exc):
        if self.tool.startswith("grobid"):
            subprocess.run([str(DOCKER), "stop", "grobid_bench"], check=False,
                           capture_output=True)
        return False


def start_llama_server(out_dir: Path) -> subprocess.Popen:
    """Prajnya's own launcher, port 9931, --models-max 1."""
    log_file = open(out_dir / "llama_server.log", "a", encoding="utf-8")
    server = subprocess.Popen(
        [str(ENV / PRAJNYA_ENV / "python.exe"), "run_llama_server.py",
         "--llama-server", str(LLAMA_SERVER), "--models-max", "1"],
        cwd=PRAJNYA_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    wait_for_url("http://127.0.0.1:9931/health", timeout=120)
    return server


def wait_for_url(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{url} did not answer within {timeout}s")


def run_process(command: list[str], cwd: Path, env: dict, log_path: Path,
                timeout: int) -> int:
    """Run a command with its output in a log file; kill its tree on timeout."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        # Own (hidden) console: Marker stops its llama-server with a console
        # Ctrl-C, which reaches every process on the same console, this
        # driver included.
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0                       # SW_HIDE
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log_file,
                                   stderr=subprocess.STDOUT, startupinfo=startup,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process_tree(process)
            return -9


def stop_process_tree(process: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                   capture_output=True)
    # llama-server can outlive its launcher; make sure port 9931 is free.
    subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
    time.sleep(3)


class ResourceMeter:
    """Samples GPU memory (nvidia-smi) and host RAM in use once a second."""

    def __init__(self):
        self.peak_vram = 0
        self.peak_ram = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _sample(self):
        while not self._stop.is_set():
            self.peak_vram = max(self.peak_vram, gpu_memory_used())
            self.peak_ram = max(self.peak_ram, host_ram_used_gb())
            self._stop.wait(1.0)


def gpu_memory_used() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def host_ram_used_gb() -> float:
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return round((status.ullTotalPhys - status.ullAvailPhys) / 2**30, 2)


def log(run_dir: Path, record: dict) -> None:
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(run_dir / "timings.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
