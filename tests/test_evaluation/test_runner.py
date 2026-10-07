from evaluation.runner import available_sequences, parse_sequence_spec

AVAILABLE = [f"{i:04d}" for i in range(21)]


def test_all():
    assert parse_sequence_spec("all", AVAILABLE) == AVAILABLE


def test_single_list_and_range():
    assert parse_sequence_spec("0,3,7", AVAILABLE) == ["0000", "0003", "0007"]
    assert parse_sequence_spec("2-4", AVAILABLE) == ["0002", "0003", "0004"]
    assert parse_sequence_spec("0-1,5", AVAILABLE) == ["0000", "0001", "0005"]


def test_unknown_sequences_are_skipped(capsys):
    assert parse_sequence_spec("19-22", AVAILABLE) == ["0019", "0020"]
    assert "not found" in capsys.readouterr().out


def test_available_sequences_lists_dirs(tmp_path):
    for name in ("0001", "0000"):
        (tmp_path / "training" / "velodyne" / name).mkdir(parents=True)
    (tmp_path / "training" / "velodyne" / "readme.txt").write_text("x")
    assert available_sequences(tmp_path, "training") == ["0000", "0001"]
    assert available_sequences(tmp_path, "missing") == []