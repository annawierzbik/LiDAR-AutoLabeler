"""YAML experiment configs: plain dicts, with `--set key.sub=value` overrides from the command line."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)   # "0.5" -> 0.5, "[1,2]" -> list, "true" -> True
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML file in experiments/configs/")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set seed=1 tracker.max_age=5")
    return parser
