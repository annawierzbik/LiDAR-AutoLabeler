from experiments.run_train_detector import derive_config, set_path


def test_set_path_handles_dicts_and_list_indices():
    cfg = {"A": {"L": [{"x": 1}, {"x": 2}]}}
    set_path(cfg, "A.L.1.x", 5)
    set_path(cfg, "A.new", {"k": 1})
    assert cfg == {"A": {"L": [{"x": 1}, {"x": 5}], "new": {"k": 1}}}


def test_clean_config_changes_only_the_listed_keys():
    overrides = {
        "DATA_CONFIG.DATA_SPLIT": {"train": "train_clean", "test": "val"},
        "DATA_CONFIG.AUG_DB": None,
    }
    base = derive_config("cfgs/kitti_models/pointpillar.yaml", {})
    derived = derive_config("cfgs/kitti_models/pointpillar.yaml", overrides)
    assert derived["DATA_CONFIG"]["DATA_SPLIT"]["train"] == "train_clean"
    for key in ("MODEL", "OPTIMIZATION", "CLASS_NAMES"):
        assert derived[key] == base[key]
    assert derived["DATA_CONFIG"]["_BASE_CONFIG_"] == base["DATA_CONFIG"]["_BASE_CONFIG_"]


def test_loss_curve_parses_train_log(tmp_path):
    from experiments.run_train_detector import loss_curve
    log = tmp_path / "train.log"
    log.write_text(
        "2026-10-28 17:24:50,002   INFO  Train:    1/2 ( 50%) [   0/564 (  0%)]  Loss: 3.357 (3.36)  LR: 3.000e-04  Time\n"
        "2026-10-28 17:24:58,036   INFO  Train:    1/2 ( 50%) [  49/564 (  9%)]  Loss: 1.861 (2.05)  LR: 3.779e-04  Time\n"
        "2026-10-28 17:28:14,457   INFO  Train:    2/2 (100%) [ 563/564 (100%)]  Loss: 0.8946 (0.941)  LR: 4.615e-08  Time\n")
    c = loss_curve(log)
    assert c[1]["n_logged"] == 2 and abs(c[1]["mean_logged_loss"] - (3.357 + 1.861) / 2) < 1e-9
    assert c[2]["running_avg_end"] == 0.941 and c[2]["lr_end"] == 4.615e-08


def test_resolve_sequences(tmp_path, monkeypatch):
    import json

    import experiments.run_model_tracking_eval as m
    (tmp_path / "s.json").write_text(json.dumps({"dev": ["0001"], "test": ["0002"], "excluded": ["0000"]}))
    monkeypatch.setattr(m, "SPLITS_DIR", tmp_path)
    assert m.resolve_sequences("s.json:excluded") == ["0000"]
    assert m.resolve_sequences("s.json:all") == ["0000", "0001", "0002"]
    assert m.resolve_sequences([3, "7"]) == ["0003", "0007"]
