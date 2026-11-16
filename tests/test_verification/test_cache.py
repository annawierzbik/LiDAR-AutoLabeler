"""Cached detection/tracking results carry their settings and are not reused with other settings."""
import json
from argparse import Namespace

import numpy as np
import pytest

import detection.paths
import evaluation.runner
import scripts.track_sequence
import scripts.verify
import verification.paths
from common.boxes import BoundingBox3D
from detection.fov import describe
from ingestion.folder import FolderSequence, folder_key
from scripts.track_sequence import current_meta, save_results
from scripts.verify import build_session
from verification import cache

AZ = describe("azimuth")    # the mode "auto" resolves to for a folder (no calibration)


def box(x, score=0.9, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", score, tid)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """10 tiny frames, a temp results dir, a fake config (with a base config) + checkpoint and a fake
    detector + tracker that records its calls and returns one box per frame."""
    folder = tmp_path / "clouds"
    folder.mkdir()
    for i in range(10):
        np.zeros((5, 4), np.float32).tofile(folder / f"{i:06d}.bin")
    results = tmp_path / "results"
    monkeypatch.setattr(verification.paths, "RESULTS_DIR", results)
    monkeypatch.setattr(scripts.verify, "RESULTS_DIR", results)
    tools = tmp_path / "tools"
    (tools / "cfgs" / "models").mkdir(parents=True)
    (tools / "cfgs" / "base.yaml").write_text("POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]\n")
    cfg, ckpt = tools / "cfgs" / "models" / "model.yaml", tmp_path / "model.pth"
    cfg.write_text("DATA_CONFIG:\n    _BASE_CONFIG_: cfgs/base.yaml\nMODEL: x\n")
    ckpt.write_bytes(b"weights-a" * 100)
    monkeypatch.setattr(detection.paths, "DEFAULT_CFG", cfg)
    monkeypatch.setattr(detection.paths, "DEFAULT_CKPT", ckpt)
    monkeypatch.setattr(cache, "_warned", set())

    calls = []

    def fake_track(seq, indices, threshold, fov="auto"):
        calls.append(threshold)
        return [[box(100 + i, score=0.95)] for i in indices]

    monkeypatch.setattr(scripts.track_sequence, "track_sequence", fake_track)
    return Namespace(folder=folder, results=results, ckpt=ckpt, cfg=cfg, base=tools / "cfgs" / "base.yaml",
                     calls=calls, key=folder_key(folder))


def args(folder, **kw):
    base = {"folder": str(folder), "data_root": "data/raw", "split": "training", "sequence": "0000",
            "mode": "assisted", "start": 0, "stop": None, "threshold": 0.5, "fps": 10.0, "fresh": True,
            "fov": "auto", "use_cache_without_weights": False}
    return Namespace(**{**base, **kw})


def first_x(session):
    return session.annotations(0)[0].box.x


def legacy(env, x0=10):
    save_results(env.results / f"{env.key}.json", range(10), [[box(x0 + i)] for i in range(10)])


# ------------------------------------------------------------------ B1 basics
def test_written_cache_has_metadata(env):
    build_session(args(env.folder))
    meta = json.loads((env.results / f"{env.key}.json").read_text())["meta"]
    assert meta["format_version"] == cache.CACHE_FORMAT_VERSION
    assert meta["score_threshold"] == 0.5 and meta["fov"] == AZ and meta["split"] is None
    assert meta["tracker"] == {"iou_threshold": 0.1, "max_age": 3, "min_hits": 3}
    assert meta["checkpoint"] == str(env.ckpt) and meta["checkpoint_hash"] == cache.file_hash(env.ckpt)
    assert meta["detector_config_files"] == [str(env.cfg), str(env.base)]
    assert meta["detector_config_hash"] is not None and "git_commit" in meta


def test_matching_metadata_reuses_the_cache(env, capsys):
    build_session(args(env.folder))
    build_session(args(env.folder, start=2, stop=6))
    assert env.calls == [0.5]                                   # detector ran once
    assert "WARNING" not in capsys.readouterr().out


def test_changed_threshold_recomputes_and_keeps_the_old_file(env, capsys):
    build_session(args(env.folder))
    old = (env.results / f"{env.key}.json").read_text()
    build_session(args(env.folder, threshold=0.7))
    assert env.calls == [0.5, 0.7]
    out = capsys.readouterr().out
    assert "other settings" in out and "score_threshold: 0.5 -> 0.7" in out
    assert (env.results / f"{env.key}.json").read_text() == old
    keyed = env.results / f"{env.key}_c{cache.config_key(current_meta(0.7, AZ, None))}.json"
    assert json.loads(keyed.read_text())["meta"]["score_threshold"] == 0.7
    build_session(args(env.folder, threshold=0.7))              # now found under the keyed name
    build_session(args(env.folder, threshold=0.5))              # and the old one still works
    assert env.calls == [0.5, 0.7]


def test_changed_weights_recompute(env, capsys):
    build_session(args(env.folder))
    env.ckpt.write_bytes(b"weights-b" * 100)
    build_session(args(env.folder))
    assert env.calls == [0.5, 0.5]
    assert "checkpoint_hash" in capsys.readouterr().out


def test_changed_tracker_parameters_recompute(env, monkeypatch):
    build_session(args(env.folder))
    monkeypatch.setattr(cache, "tracker_params", lambda t: {"iou_threshold": 0.3, "max_age": 3, "min_hits": 3})
    build_session(args(env.folder))
    assert env.calls == [0.5, 0.5]


def test_changed_fov_mode_recomputes(env, capsys):
    build_session(args(env.folder))
    build_session(args(env.folder, fov="none"))
    assert env.calls == [0.5, 0.5]
    assert f"fov: '{AZ}' -> 'none'" in capsys.readouterr().out


def test_legacy_file_without_metadata_is_reused_with_one_warning(env, capsys):
    legacy(env)
    s1, _, _ = build_session(args(env.folder, fov="none"))
    s2, _, _ = build_session(args(env.folder, fov="none", start=3, stop=7))
    assert env.calls == []
    assert first_x(s1) == 10 and first_x(s2) == 13
    assert capsys.readouterr().out.count("has no settings metadata") == 1


def test_legacy_file_is_not_used_with_a_fov_crop(env, capsys):
    legacy(env)
    build_session(args(env.folder))                              # auto -> azimuth for a folder
    assert env.calls == [0.5]
    assert "made without a FOV crop" in capsys.readouterr().out
    assert (env.results / f"{env.key}.json").exists()           # kept


def test_matching_file_is_preferred_over_a_legacy_one(env, capsys):
    legacy(env)
    keyed = env.results / f"{env.key}_c{cache.config_key(current_meta(0.5, AZ, None))}.json"
    save_results(keyed, range(10), [[box(50 + i)] for i in range(10)], current_meta(0.5, AZ, None))
    s, _, _ = build_session(args(env.folder))
    assert first_x(s) == 50 and "WARNING" not in capsys.readouterr().out


def test_partial_cache_with_other_settings_is_not_reused(env):
    build_session(args(env.folder, start=0, stop=5))
    assert (env.results / f"{env.key}_f0-5.json").exists()
    build_session(args(env.folder, start=1, stop=4, threshold=0.9))
    assert env.calls == [0.5, 0.9]


def test_files_of_other_names_are_never_candidates():
    assert cache.is_cache_name("0000", "0000")
    assert cache.is_cache_name("0000_f3-9", "0000")
    assert cache.is_cache_name("0000_f3-9_c0123abcd", "0000")
    assert cache.is_cache_name("0000_c0123abcd", "0000")
    assert not cache.is_cache_name("0000_detections", "0000")
    assert not cache.is_cache_name("0000_cx_abcdef", "0000")    # folder key of a folder called "0000_cx"


def test_file_hash_reads_the_first_mib_and_the_size(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    head = b"\0" * cache.HASH_BYTES
    a.write_bytes(head + b"1")
    b.write_bytes(head + b"2")             # differs only after the first MiB, same size
    c.write_bytes(head + b"12")
    assert cache.file_hash(a) == cache.file_hash(b) != cache.file_hash(c)
    assert cache.file_hash(tmp_path / "missing") is None


# ------------------------------------------------------ 7a: base config hash
def test_changed_base_config_recomputes(env, capsys):
    build_session(args(env.folder))
    env.base.write_text("POINT_CLOUD_RANGE: [0, -39.68, -3, 69.12, 39.68, 1]\n")
    build_session(args(env.folder))
    assert env.calls == [0.5, 0.5]
    assert "detector_config_hash" in capsys.readouterr().out


def test_config_hash_follows_nested_base_configs(tmp_path):
    tools = tmp_path / "tools"
    (tools / "cfgs" / "m").mkdir(parents=True)
    (tools / "cfgs" / "a.yaml").write_text("_BASE_CONFIG_: cfgs/b.yaml\n")
    (tools / "cfgs" / "b.yaml").write_text("X: 1\n")
    top = tools / "cfgs" / "m" / "top.yaml"
    top.write_text("D:\n  _BASE_CONFIG_: cfgs/a.yaml\n")
    assert [f.name for f in cache.config_files(top)] == ["top.yaml", "a.yaml", "b.yaml"]
    before = cache.config_hash(top)
    (tools / "cfgs" / "b.yaml").write_text("X: 2\n")
    assert cache.config_hash(top) != before


# ------------------------------------------------------------- 7b: KITTI split
def test_split_is_part_of_the_key(env):
    meta = {**current_meta(0.5, "camera", "training")}
    other = {**meta, "split": "testing"}
    assert cache.config_key(meta) != cache.config_key(other)
    assert cache.differences(meta, other) == ["split: 'training' -> 'testing'"]


def test_legacy_kitti_file_only_for_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cache, "_warned", set())
    path = tmp_path / "0000.json"
    save_results(path, range(2), [[box(1)], [box(2)]])
    train = current_meta(0.5, "none", "training")
    test = current_meta(0.5, "none", "testing")
    assert cache.choose([path], train) is not None
    assert cache.choose([path], test) is None
    out = capsys.readouterr().out
    assert "reusing it" in out
    assert "only used for the split 'training', not 'testing'" in out


def test_testing_split_does_not_reuse_a_training_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_warned", set())
    path = tmp_path / "0000.json"
    save_results(path, range(2), [[], []], current_meta(0.5, "camera", "training"))
    test = current_meta(0.5, "camera", "testing")
    assert cache.choose([path], test) is None
    assert cache.output_path(path, test).name == f"0000_c{cache.config_key(test)}.json"


# ---------------------------------------------------- 7c: missing checkpoint
def test_missing_checkpoint_is_reported_and_not_used_by_default(env, capsys):
    build_session(args(env.folder))
    env.ckpt.unlink()
    build_session(args(env.folder))                          # would recompute (fake detector here)
    out = capsys.readouterr().out
    assert "is missing on disk" in out and "--use-cache-without-weights" in out
    assert "other settings" not in out
    assert env.calls == [0.5, 0.5]


def test_missing_checkpoint_with_the_flag_uses_a_matching_cache(env, capsys):
    build_session(args(env.folder))
    env.ckpt.unlink()
    s, _, _ = build_session(args(env.folder, use_cache_without_weights=True))
    assert env.calls == [0.5] and first_x(s) == 100
    assert "without checking that it was made with these weights" in capsys.readouterr().out


def test_missing_checkpoint_flag_does_not_hide_other_differences(env):
    build_session(args(env.folder))
    env.ckpt.unlink()
    build_session(args(env.folder, threshold=0.6, use_cache_without_weights=True))
    assert env.calls == [0.5, 0.6]


# ------------------------------------------------------ evaluation runner cache
def test_evaluation_cache_recomputes_for_other_weights(env, monkeypatch, capsys):
    runs = []
    monkeypatch.setattr(evaluation.runner, "run_detector",
                        lambda seq, idx, fov="auto": runs.append(fov) or [[box(1.0)] for _ in idx])
    seq = FolderSequence(env.folder)
    path = env.results / "0000_detections.json"
    evaluation.runner.load_or_detect(seq, range(5), path, True, split=None)
    evaluation.runner.load_or_detect(seq, range(5), path, True, split=None)
    assert len(runs) == 1
    env.ckpt.write_bytes(b"weights-b" * 100)
    evaluation.runner.load_or_detect(seq, range(5), path, True, split=None)
    assert len(runs) == 2 and "checkpoint_hash" in capsys.readouterr().out
    assert len(list(env.results.glob("0000_detections*.json"))) == 2


def test_evaluation_cache_without_metadata_still_loads_with_fov_none(env, capsys):
    path = env.results / "0000_detections.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"start": 0, "stop": 2, "frames": [[], []]}))
    seq = FolderSequence(env.folder)
    assert evaluation.runner.load_or_detect(seq, range(2), path, True, split="training", fov="none") == [[], []]
    assert "has no settings metadata" in capsys.readouterr().out
