"""Thesis output: figures as PDF + PNG and booktabs LaTeX tables, labels in Polish by default.

    save_figure(fig, "e0_crosscheck")                     -> outputs/figures/e0_crosscheck.{pdf,png}
    write_latex_table(rows, columns, "e0_crosscheck", caption=..., label=...)  -> outputs/tables/*.tex
    t("ap3d")                                             -> "AP 3D" / label in the chosen language
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from experiments.lib.paths import FIGURES_DIR, TABLES_DIR

LABELS = {
    "pl": {
        "class": "Klasa", "Car": "Samochód", "Pedestrian": "Pieszy", "Cyclist": "Rowerzysta",
        "easy": "łatwe", "moderate": "średnie", "hard": "trudne", "difficulty": "Trudność",
        "ap3d": "AP 3D", "apbev": "AP BEV", "ap40": "AP$_{40}$ [\\%]", "diff_pp": "Różnica [p.p.]",
        "repo": "Metryka repozytorium", "openpcdet": "Ewaluator OpenPCDet", "variant": "Wariant",
        "sequence": "Sekwencja", "frames": "Klatki", "split": "Podział", "dev": "dev", "test": "test",
        "excluded": "wykluczona", "object_split": "Podział zbioru object", "drive": "Przejazd (raw)",
        "num_gt": "Liczba GT",
    },
    "en": {
        "class": "Class", "Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Cyclist",
        "easy": "easy", "moderate": "moderate", "hard": "hard", "difficulty": "Difficulty",
        "ap3d": "3D AP", "apbev": "BEV AP", "ap40": "AP$_{40}$ [\\%]", "diff_pp": "Difference [pp]",
        "repo": "Repository metric", "openpcdet": "OpenPCDet evaluator", "variant": "Variant",
        "sequence": "Sequence", "frames": "Frames", "split": "Split", "dev": "dev", "test": "test",
        "excluded": "excluded", "object_split": "Object split", "drive": "Raw drive",
        "num_gt": "GT objects",
    },
}


def t(key: str, lang: str = "pl") -> str:
    """Label for `key` in `lang`; unknown keys are returned unchanged."""
    return LABELS[lang].get(key, key)


def save_figure(fig, name: str, folder: Path = FIGURES_DIR, dpi: int = 200) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}.pdf", folder / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight", dpi=dpi)
    return paths


def _cell(v, decimals: int) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.{decimals}f}".replace(".", "{,}") if decimals >= 0 else str(v)
    return str(v)


def latex_table(rows: Sequence[Sequence], columns: Sequence[str], caption: str, label: str,
                decimals: int = 2, align: str | None = None, decimal_comma: bool = True) -> str:
    """booktabs table as a string. With decimal_comma (Polish convention) 1.5 is written 1{,}5."""
    align = align or "l" + "r" * (len(columns) - 1)
    fmt = (lambda v: _cell(v, decimals)) if decimal_comma else (
        lambda v: "--" if v is None else (f"{v:.{decimals}f}" if isinstance(v, float) else str(v)))
    lines = [
        "\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(columns) + " \\\\", "\\midrule",
        *(" & ".join(fmt(v) for v in row) + " \\\\" for row in rows),
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
    ]
    return "\n".join(lines)


def write_latex_table(rows, columns, name: str, caption: str, label: str, folder: Path = TABLES_DIR,
                      **kwargs) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.tex"
    path.write_text(latex_table(rows, columns, caption, label, **kwargs), encoding="utf-8")
    return path
