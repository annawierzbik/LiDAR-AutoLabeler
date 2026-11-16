"""Settings stored with cached detection/tracking results, so a cache made with other settings is
never reused silently.

A cache file (results/<name>.json, results/<name>_f<start>-<stop>.json, results/<seq>_detections.json)
gets a "meta" entry when it is written. A reader compares it with the current settings:
    same settings      the file is reused;
    other settings     a warning lists what differs and the results are recomputed; the old file
                       is kept and the new one gets the configuration key in its name
                       (<stem>_c<key>.json, see `output_path`);
    no "meta" (files written before cache format 2)
                       still reused with a warning printed once per file, but only for the KITTI
                       split "training" (or a folder) and FOV crop "none": they were made before the
                       split was recorded and before the crop existed.
Compared: cache format, detector config (the YAML and its _BASE_CONFIG_ files), checkpoint
fingerprint, score threshold, point-cloud FOV crop, tracker parameters and the KITTI split. The paths
and the git commit are stored for information only (a new commit must not throw away hours of
detections). A checkpoint missing on disk is reported as such; with allow_missing_weights a cache
that matches in everything else is still used.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import detection.paths

CACHE_FORMAT_VERSION = 2   # 1 = the format without "meta"
HASH_BYTES = 1 << 20
HASH_METHOD = "sha256 of the first 1 MiB + file size"
TRACKER_PARAMS = ("iou_threshold", "max_age", "min_hits")
COMPARED = ("format_version", "detector_config_hash", "checkpoint_hash", "score_threshold", "fov",
            "tracker", "split")
LEGACY_SPLITS = ("training", None)   # None = a folder of .bin files (no KITTI split)
LEGACY_FOV = "none"                  # files without metadata were made without a point crop
_BASE_RE = re.compile(r"^\s*_BASE_CONFIG_\s*:\s*(\S+)", re.MULTILINE)

_warned: set[str] = set()


def file_hash(path, nbytes: int = HASH_BYTES) -> str | None:
    """sha256 of the first `nbytes` bytes and the size of the file (16 hex digits); None if missing.

    Not a full-file hash, to stay fast on large checkpoints: two files of the same size that differ
    only after the first MiB get the same value. Different checkpoints of PointPillars differ in the
    first MiB already (it holds weight tensors)."""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]


def config_files(cfg_file) -> list[Path]:
    """The config and, recursively, every _BASE_CONFIG_ it names. OpenPCDet resolves those relative
    to tools/ (cfg_file.parents[2], as detection.detector does)."""
    cfg_file = Path(cfg_file)
    tools = cfg_file.parents[2] if len(cfg_file.parents) > 2 else cfg_file.parent
    out, todo = [], [cfg_file]
    while todo:
        f = todo.pop(0)
        if f in out:
            continue
        out.append(f)
        if f.is_file():
            todo += [tools / m for m in _BASE_RE.findall(f.read_text(encoding="utf-8"))]
    return out


def config_hash(cfg_file) -> str | None:
    """sha256 over the config and its base configs (names and contents); None if the config is missing."""
    files = config_files(cfg_file)
    if not files[0].is_file():
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], check=False, cwd=detection.paths.PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def tracker_params(tracker) -> dict:
    return {k: getattr(tracker, k) for k in TRACKER_PARAMS}


def make_meta(score_threshold: float, tracker: dict | None = None, cfg_file=None, checkpoint=None,
              fov: str = "none", split: str | None = None) -> dict:
    """Settings of one run. `tracker` = tracker_params(...) or None for detections only; `fov` = the
    resolved crop (detection.fov.describe); `split` = KITTI split, None for a folder. Without
    `cfg_file`/`checkpoint` the detector defaults (detection.paths) are used."""
    cfg_file = Path(cfg_file or detection.paths.DEFAULT_CFG)
    checkpoint = Path(checkpoint or detection.paths.DEFAULT_CKPT)
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "detector_config": str(cfg_file),
        "detector_config_files": [str(f) for f in config_files(cfg_file)],
        "detector_config_hash": config_hash(cfg_file),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": file_hash(checkpoint),
        "hash_method": HASH_METHOD,
        "score_threshold": float(score_threshold),
        "fov": fov,
        "tracker": dict(tracker) if tracker is not None else None,
        "split": split,
        "git_commit": git_commit(),
    }


def config_key(meta: dict) -> str:
    """8 hex digits identifying the compared settings (used in file names)."""
    return hashlib.sha1(json.dumps({k: meta.get(k) for k in COMPARED}, sort_keys=True).encode()).hexdigest()[:8]


def differences(stored: dict, current: dict, ignore=()) -> list[str]:
    return [f"{k}: {stored.get(k)!r} -> {current.get(k)!r}" for k in COMPARED
            if k not in ignore and stored.get(k) != current.get(k)]


def weights_missing(current: dict) -> bool:
    return current.get("checkpoint_hash") is None


def warn_once(path: Path, text: str) -> None:
    key = f"{Path(path).resolve()}|{text}"
    if key not in _warned:
        _warned.add(key)
        print(f"WARNING: {text}")


def warn_legacy(path: Path) -> None:
    warn_once(path, f"{Path(path).name} has no settings metadata (written before cache format "
                    f"{CACHE_FORMAT_VERSION}); reusing it, but it may come from another detector, "
                    "checkpoint, threshold, FOV crop or tracker. Delete it to recompute.")


def warn_mismatch(path: Path, stored: dict, current: dict) -> None:
    warn_once(path, f"{Path(path).name} was made with other settings ({'; '.join(differences(stored, current))}); "
                    "not reused, recomputing.")


def choose(candidates, current: dict, usable=lambda data: True,
           allow_missing_weights: bool = False) -> tuple[Path, dict] | None:
    """(path, loaded JSON) of the cache file to reuse, or None.

    The first usable candidate made with the `current` settings wins; otherwise the first usable
    file without metadata (with a warning; only for the split "training" or a folder); otherwise
    None, after a warning for every usable file made with other settings.
    If the checkpoint is missing on disk its fingerprint cannot be compared: that is said explicitly,
    and with `allow_missing_weights` a file that matches in everything else is used."""
    no_weights = weights_missing(current)
    ignore = ("checkpoint_hash",) if no_weights and allow_missing_weights else ()
    legacy, mismatched, unchecked, skipped = None, [], [], []
    for path in candidates:
        path = Path(path)
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if not usable(data):
            continue
        stored = data.get("meta")
        if stored is None:
            if current.get("split") not in LEGACY_SPLITS:
                text = (f"{path.name} has no settings metadata; such files are only used for "
                        f"the split 'training', not '{current.get('split')}'.")
                skipped.append((path, text))
            elif current.get("fov") != LEGACY_FOV:
                text = (f"{path.name} has no settings metadata, so it was made without a FOV "
                        f"crop; not used with the crop {current.get('fov')!r} (--fov none reuses it).")
                skipped.append((path, text))
            else:
                legacy = legacy or (path, data)
        elif differences(stored, current, ignore):
            if no_weights and not differences(stored, current, ("checkpoint_hash",)):
                unchecked.append(path)
            else:
                mismatched.append((path, stored))
        else:
            if ignore:
                warn_once(path, f"checkpoint {current['checkpoint']} is missing on disk; using {path.name} "
                                "without checking that it was made with these weights "
                                "(--use-cache-without-weights).")
            return path, data
    if legacy is not None:
        warn_legacy(legacy[0])
        return legacy
    for path, text in skipped:
        warn_once(path, text)
    for path in unchecked:
        warn_once(path, f"checkpoint {current['checkpoint']} is missing on disk, so {path.name} cannot be "
                        "checked against it (everything else matches). Restore the checkpoint, or pass "
                        "--use-cache-without-weights to use the cache anyway.")
    for path, stored in mismatched:
        warn_mismatch(path, stored, current)
    return None


def output_path(path: Path, current: dict) -> Path:
    """Where to write results made with `current`: `path`, unless a file with other or unknown
    settings is already there; then <stem>_c<key><suffix> next to it (the old file is kept)."""
    path = Path(path)
    if path.exists():
        try:
            stored = json.loads(path.read_text()).get("meta")
        except (OSError, ValueError):
            stored = None
        if stored is None or differences(stored, current):
            return keyed_path(path, current)
    return path


def keyed_path(path: Path, current: dict) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_c{config_key(current)}{path.suffix}")


def is_cache_name(stem: str, name: str) -> bool:
    """`stem` is <name>, <name>_f<a>-<b>, optionally followed by _c<key> (and nothing else)."""
    return re.fullmatch(rf"{re.escape(name)}(_f\d+-\d+)?(_c[0-9a-f]{{8}})?", stem) is not None
