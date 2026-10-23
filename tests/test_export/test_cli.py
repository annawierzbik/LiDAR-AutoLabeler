import numpy as np

from scripts.export_labels import main
from tests.test_export.test_formats import accept_frame, box, make_session
from verification.session import AnnotationSession


def make_folder(tmp_path, n):
    folder = tmp_path / "clouds"
    folder.mkdir()
    for i in range(n):
        np.zeros((5, 4), "<f4").tofile(folder / f"{i:06d}.bin")
    return folder


def test_cli_writes_the_labels_of_a_saved_session(tmp_path, capsys):
    session = make_session()
    accept_frame(session, 0)
    saved = tmp_path / "s.json"
    session.save(saved)
    out = tmp_path / "out"
    code = main(["--session", str(saved), "--folder", str(make_folder(tmp_path, 3)), "--out", str(out)])
    assert code == 0
    assert sorted(p.name for p in (out / "labels").iterdir()) == ["000000.txt"]
    text = capsys.readouterr().out
    assert "Wrote 1 frames, 2 boxes" in text and "2 frames still have boxes to review" in text


def test_cli_reports_a_format_that_is_not_possible(tmp_path, capsys):
    saved = tmp_path / "s.json"
    AnnotationSession.from_tracking("0000", [[box(10, 0)]]).save(saved)
    code = main(["--session", str(saved), "--folder", str(make_folder(tmp_path, 1)), "--format", "kitti",
                 "--out", str(tmp_path / "out")])
    assert code == 1 and "calibration" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
