import json

import pytest

from experiments.lib.splits import (
    SplitAccessError,
    check_split_access,
    fraction_summary,
    make_dev_test_split,
)

SEQS = ["0001", "0004", "0007", "0008", "0015", "0019", "0020"]


def test_split_is_deterministic_and_order_independent():
    a = make_dev_test_split(SEQS, 0.5, seed=0)
    b = make_dev_test_split(list(reversed(SEQS)), 0.5, seed=0)
    assert a == b


def test_split_is_a_partition_by_whole_sequences():
    s = make_dev_test_split(SEQS, 0.5, seed=3)
    assert set(s["dev"]) | set(s["test"]) == set(SEQS)
    assert not set(s["dev"]) & set(s["test"])
    assert len(s["test"]) == round(len(SEQS) * 0.5)


def test_different_seeds_give_different_splits():
    assert len({tuple(make_dev_test_split(SEQS, 0.5, seed=k)["test"]) for k in range(10)}) > 1


def test_both_sides_non_empty():
    s = make_dev_test_split(["a", "b"], 0.9, seed=0)
    assert len(s["dev"]) == 1 and len(s["test"]) == 1


def test_bad_fraction():
    with pytest.raises(ValueError):
        make_dev_test_split(SEQS, 1.0, seed=0)


def test_fraction_summary():
    s = {"dev": ["a"], "test": ["b", "c"]}
    out = fraction_summary(s, {"a": 10, "b": 20, "c": 10})
    assert out["dev"]["frames"] == 10 and out["test"]["frame_fraction"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------- guard
def _write_run(root, experiment, run_id, split):
    (root / experiment).mkdir(parents=True, exist_ok=True)
    (root / experiment / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "split": split}))


def test_dev_is_always_allowed(tmp_path):
    assert check_split_access("dev", results_root=tmp_path) == {"split": "dev", "final": False, "frozen_from": None}


def test_test_split_needs_final(tmp_path):
    with pytest.raises(SplitAccessError):
        check_split_access("test", results_root=tmp_path)


def test_final_needs_a_dev_run_id(tmp_path):
    with pytest.raises(SplitAccessError):
        check_split_access("test", final=True, results_root=tmp_path)
    with pytest.raises(SplitAccessError):
        check_split_access("test", final=True, frozen_from="missing", results_root=tmp_path)


def test_final_rejects_parameters_frozen_from_a_test_run(tmp_path):
    _write_run(tmp_path, "E2", "r_test", "test")
    with pytest.raises(SplitAccessError):
        check_split_access("test", final=True, frozen_from="r_test", results_root=tmp_path)


def test_final_with_dev_run_is_allowed(tmp_path):
    _write_run(tmp_path, "E2", "r_dev", "dev")
    assert check_split_access("test", final=True, frozen_from="r_dev", results_root=tmp_path) == \
        {"split": "test", "final": True, "frozen_from": "r_dev"}


def test_final_on_dev_is_an_error(tmp_path):
    with pytest.raises(SplitAccessError):
        check_split_access("dev", final=True, results_root=tmp_path)
