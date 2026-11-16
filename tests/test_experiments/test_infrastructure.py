"""Result files, configs, seeds and timing helpers."""
import json
import math
import random

import numpy as np
import pytest

from experiments.lib.config import load_config
from experiments.lib.results_io import (
    REQUIRED_KEYS,
    finish_run,
    load_results,
    start_run,
)
from experiments.lib.seeds import set_seeds
from experiments.lib.timing import summarize, time_calls


def test_result_file_has_required_metadata(tmp_path):
    run = start_run("EX", {"a": 1}, seed=3, tag="t", dataset="d", split="dev")
    path = finish_run(run, metrics={"ap": np.float32(0.5), "bad": math.nan}, timing={"s": 1.0},
                      notes="n", root=tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent.name == "EX" and path.stem == data["run_id"]
    assert all(k in data for k in REQUIRED_KEYS)
    assert data["metrics"] == {"ap": 0.5, "bad": None}
    assert data["seeds"]["seed"] == 3
    assert "commit" in data["git"] and "dirty" in data["git"]
    assert load_results("EX", root=tmp_path)[0]["run_id"] == data["run_id"]


def test_result_file_records_the_detector_fov(tmp_path):
    run = start_run("EX", {}, detector_fov="camera")
    data = json.loads(finish_run(run, root=tmp_path).read_text(encoding="utf-8"))
    assert "detector_fov" in REQUIRED_KEYS and data["detector_fov"] == "camera"
    assert start_run("EX", {})["detector_fov"] is None          # no detector inference in the run


def test_detection_cache_is_separate_per_fov_mode():
    from experiments.lib.tracking_eval import detection_cache

    none, cam, az = (detection_cache("E0", "zoo", f, "0001") for f in ("none", "camera", "azimuth:39.75"))
    assert none.parent.name == "zoo"                             # where the runs before the crop cached
    assert cam.parent.name == "fov_camera" and az.parent.name == "fov_azimuth_39.75"
    assert len({none, cam, az}) == 3


def test_configs_that_run_the_detector_record_its_fov():
    from experiments.lib.paths import PROJECT_ROOT

    cfgs = PROJECT_ROOT / "experiments" / "configs"
    assert load_config(cfgs / "E0_leakage_tracking.yaml")["detector_fov"] == "auto"
    assert load_config(cfgs / "E0_metric_crosscheck.yaml")["tracking_dev"]["detector_fov"] == "auto"
    assert load_config(cfgs / "E0_leakage_val.yaml")["detector_fov"] == "openpcdet_fov_points_only"


def test_seeds_make_numpy_and_python_repeatable():
    set_seeds(7)
    a = (random.random(), np.random.rand())
    set_seeds(7)
    assert (random.random(), np.random.rand()) == a


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 0\ntracker:\n  max_age: 3\n")
    cfg = load_config(p, ["seed=2", "tracker.max_age=5", "new.list=[1, 2]"])
    assert cfg == {"seed": 2, "tracker": {"max_age": 5}, "new": {"list": [1, 2]}}
    with pytest.raises(ValueError):
        load_config(p, ["no_equals_sign"])


def test_summarize():
    s = summarize([0.001, 0.002, 0.003, 0.004, 0.005])
    assert s["n"] == 5 and s["median_ms"] == pytest.approx(3.0) and s["iqr_ms"] == pytest.approx(2.0)
    assert summarize([])["n"] == 0


def test_time_calls_skips_warmup():
    out, stats = time_calls(lambda x: x * 2, range(12), warmup=10, sync=False)
    assert out == [x * 2 for x in range(12)]
    assert stats["n"] == 2 and stats["warmup"] == 10


def test_latex_table_booktabs_and_decimal_comma():
    from experiments.lib.report import latex_table, t
    tex = latex_table([["Car", 78.386, None]], [t("class"), t("ap40"), t("diff_pp")], "Podpis", "tab:x")
    assert r"\toprule" in tex and r"\bottomrule" in tex and r"\label{tab:x}" in tex
    assert "78{,}39" in tex and "--" in tex
    assert t("Car") == "Samochód" and t("Car", "en") == "Car" and t("unknown") == "unknown"
