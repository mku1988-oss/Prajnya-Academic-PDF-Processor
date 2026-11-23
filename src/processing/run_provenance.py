"""
What produced a run: the model files, the runtimes and the machine's GPU (audit D8 + L5).

For biology researchers: two runs of the same Prajnya release can still
differ if a model file, its quantisation, the llama.cpp build or a library
version changed. Without a record of those, two output files cannot be told
apart and a number cannot be traced back to what measured it. This collects
that record once per process; it is written into every output file as
`document_info.provenance` (contract 1.6).

Only FILE NAMES are recorded, never full paths (an output file is meant to
be shareable, audit L1). Every part is best-effort: a missing `nvidia-smi`,
a llama-server that is not running, or an unreadable config gives `null` for
that part and never fails a run.
"""
from __future__ import annotations

import configparser
import json
import logging
import platform
import re
import subprocess
import urllib.request
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

#: The libraries whose version changes what Prajnya reads from a PDF.
PACKAGES = ("pypdfium2", "onnxruntime", "numpy", "Pillow", "opencv-python-headless")

#: The roles whose model reads or judges the document (profile_registry).
MODEL_ROLES = ("default", "title_detection", "extraction", "understanding", "figure_escalation")

#: How long the optional probes may take before they give up.
SERVER_TIMEOUT_SECONDS = 0.5
GPU_TIMEOUT_SECONDS = 5.0

_QUANTISATION = re.compile(r"(?i)(?<![A-Za-z0-9])((?:IQ|Q)\d(?:_[A-Z0-9]+)*|BF16|F16|F32)(?=[.\-_]|$)")


def run_provenance(config_root: Path | str) -> Dict[str, Any]:
    """
    The provenance record for runs made with the config tree at `config_root`.

    Cached per process and config tree: the record describes the run, not
    the document, and probing a GPU or a server per document would cost time
    for nothing.

    Args:
        config_root: the config directory the run uses (holds
            `research_article_strategy.yaml` and `vlm_profiles/`).

    Returns:
        `{"python", "os", "packages", "pdfium", "layout_model", "models",
        "llama_server", "gpu"}`; a part that could not be read is None.
    """
    return dict(_cached_provenance(str(Path(config_root).resolve())))


@lru_cache(maxsize=4)
def _cached_provenance(config_root: str) -> Dict[str, Any]:
    root = Path(config_root)
    strategy = _safely(lambda: load_strategy(root), {})
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.version()}",
        "packages": package_versions(),
        "pdfium": _safely(pdfium_build),
        "layout_model": _safely(lambda: layout_model_record(root)),
        "models": _safely(lambda: model_records(root, strategy), []),
        "llama_server": _safely(lambda: llama_server_record(strategy)),
        "gpu": _safely(gpu_record),
    }


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def package_versions() -> Dict[str, Optional[str]]:
    """{package: installed version or None}."""
    found: Dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def pdfium_build() -> Optional[str]:
    """The PDFium build pypdfium2 bundles ("159.0.8149.0@sourcebuild-...")."""
    import pypdfium2
    info = getattr(pypdfium2, "PDFIUM_INFO", None)
    return str(info) if info is not None else None


def layout_model_record(config_root: Path) -> Optional[Dict[str, Any]]:
    """The ONNX layout model's file name and size in bytes.

    The size, not a hash: every document runs in its own worker process, and
    hashing the model took 1.2 s each time. Different builds of a model file
    differ in size.
    """
    from .strategies.research_article.body_pipeline import DEFAULT_ONNX_MODEL_PATH
    path = Path(DEFAULT_ONNX_MODEL_PATH)
    if not path.is_file():
        return None
    return {"file": path.name, "bytes": path.stat().st_size}


def model_records(config_root: Path, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One record per model role: the model id, and for a local model its file and quantisation."""
    registry = strategy.get("profile_registry") or {}
    presets = read_model_presets(config_root, strategy)
    records = []
    for role in MODEL_ROLES:
        profile = registry.get(role)
        if not isinstance(profile, str):
            continue
        model_name = profile_model_name(config_root, profile) or profile
        record: Dict[str, Any] = {"role": role, "model": model_name}
        model_file = (presets.get(model_name) or {}).get("model")
        if model_file:
            file_name = PureWindowsPath(model_file).name
            record.update({"file": file_name, "quantisation": quantisation_of(file_name)})
        records.append(record)
    return records


def profile_model_name(config_root: Path, profile: str) -> Optional[str]:
    """`model_name` of `vlm_profiles/<profile>.yaml`, or None."""
    import yaml
    path = config_root / "vlm_profiles" / f"{profile}.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("model_name")


def read_model_presets(config_root: Path, strategy: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """The sections of llama-server's `model.ini`, as {model id: {key: value}}."""
    server = ((strategy.get("vlm_provider") or {}).get("local_server") or {})
    named = server.get("model_profiles")
    candidates = [config_root.parent / named] if named else []
    candidates.append(config_root / "vlm_profiles" / "model.ini")
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {}
    parser = configparser.ConfigParser(strict=False, inline_comment_prefixes=(";", "#"),
                                       interpolation=None)
    parser.read(path, encoding="utf-8")
    return {section: dict(parser[section]) for section in parser.sections()}


def quantisation_of(file_name: str) -> Optional[str]:
    """"olmOCR-2-7B-1025.Q5_K_M.gguf" -> "Q5_K_M"; None when the name does not say."""
    stem = file_name[:-len(".gguf")] if file_name.lower().endswith(".gguf") else file_name
    found = _QUANTISATION.findall(stem)
    return found[-1].upper() if found else None


def llama_server_record(strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """llama-server's build, from its `/props`; None when it does not answer."""
    server = ((strategy.get("vlm_provider") or {}).get("local_server") or {})
    if not server.get("enabled"):
        return None
    host = server.get("host", "localhost")
    host = "127.0.0.1" if host == "localhost" else host    # skip a slow IPv6 try when it is down
    url = f"http://{host}:{server.get('port', 9931)}/props"
    with urllib.request.urlopen(url, timeout=SERVER_TIMEOUT_SECONDS) as response:
        props = json.loads(response.read().decode("utf-8"))
    return {"build": props.get("build_info")}


def gpu_record() -> Optional[Dict[str, Optional[str]]]:
    """The GPU's name, driver and CUDA version, from `nvidia-smi`; None without one."""
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=GPU_TIMEOUT_SECONDS)
    if query.returncode != 0 or not query.stdout.strip():
        return None
    name, _, driver = query.stdout.strip().splitlines()[0].partition(",")
    banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=GPU_TIMEOUT_SECONDS)
    cuda = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", banner.stdout or "")
    return {"name": name.strip(), "driver": driver.strip(), "cuda": cuda.group(1) if cuda else None}


def load_strategy(config_root: Path) -> Dict[str, Any]:
    """The research-article strategy config, references resolved."""
    from .config.loader import ConfigLoader
    return ConfigLoader(config_root).strategy()


def _safely(read, fallback: Any = None) -> Any:
    """Run one probe; anything it raises gives `fallback` (provenance never fails a run)."""
    try:
        return read()
    except Exception as error:  # noqa: BLE001 - a missing GPU or server is normal
        LOG.debug("Provenance probe %s failed: %s", getattr(read, "__name__", read), error)
        return fallback
