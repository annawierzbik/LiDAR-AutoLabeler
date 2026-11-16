"""Git state, library versions and hardware, recorded in every result file."""
from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path

from experiments.lib.paths import CHECKPOINTS_DIR, OPENPCDET_ROOT, PROJECT_ROOT

LIBRARIES = ("numpy", "scipy", "torch", "spconv", "cumm", "numba", "pcdet", "open3d",
             "matplotlib", "pandas", "yaml", "nuscenes", "motmetrics")


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                             timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo: Path = PROJECT_ROOT) -> dict:
    """{"commit", "dirty", "dirty_files"}; commit is None when `repo` is not a git checkout.

    Untracked files count as dirty only if they are not ignored (git status --porcelain)."""
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    lines = status.splitlines() if status else []
    return {"commit": commit, "dirty": bool(lines) if status is not None else None,
            "dirty_files": [ln[3:] for ln in lines][:50]}


def library_versions(names=LIBRARIES) -> dict:
    """Version string per library, or None when it is not installed. torch is imported first
    because the compiled pcdet ops need its DLLs on Windows."""
    out = {}
    for name in names:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001  (broken optional installs must not stop a run)
            out[name] = None
            continue
        out[name] = str(getattr(mod, "__version__", "unknown"))
    return out


def cuda_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False, "torch_cuda": torch.version.cuda}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": props.name,
        "total_memory_mib": props.total_memory // 2**20,
        "capability": f"{props.major}.{props.minor}",
    }


def environment() -> dict:
    """Everything except the experiment-specific part of a result file's metadata."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_info(PROJECT_ROOT),
        "openpcdet_git": git_info(OPENPCDET_ROOT),
        "libraries": library_versions(),
        "cuda": cuda_info(),
    }


def available_weights(cfg_dir: Path = OPENPCDET_ROOT / "tools" / "cfgs" / "kitti_models",
                      ckpt_dir: Path = CHECKPOINTS_DIR) -> dict:
    """KITTI model configs and the local checkpoint files whose name starts with the config name
    (e.g. pointpillar.yaml -> pointpillar_7728.pth). The name match is a convention, not a check
    that the weights belong to that config; loading them is the real check."""
    ckpts = sorted(p.name for p in ckpt_dir.glob("*.pth")) if ckpt_dir.exists() else []
    out = {}
    for cfg in sorted(cfg_dir.glob("*.yaml")):
        out[cfg.stem] = [c for c in ckpts if c.rsplit("_", 1)[0] == cfg.stem]
    return out
