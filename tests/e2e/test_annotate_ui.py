"""End-to-end tests of the browser UI against the real web app (fake detector, synthetic sequence).

    pip install -r requirements-dev.txt && python -m playwright install chromium
    pytest -m e2e

Assertions use the API state and the visible DOM, never pixels. Box selection is driven through
the keyboard and the box list; picking a box by clicking it in the 3D view is not tested (see
the bottom of this file). Every request outside 127.0.0.1 is blocked and fails the test: the
page must work offline (three.js and the font are vendored in web/static/vendor).
"""
from __future__ import annotations

import json
import re
import time
import zipfile

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import NUM_FRAMES

pytestmark = pytest.mark.e2e

# fake detections (tests/e2e/app_server.py): Car track 0 in all 6 frames, Pedestrian track 1 in frames 0..2
TOTAL_BOXES = NUM_FRAMES + 3


# ------------------------------------------------------------------ helpers
def open_sequence(page, server, folder) -> str:
    page.goto(server.url + "/")
    page.fill("#path", str(folder))
    page.click("#open")
    page.wait_for_url(re.compile(r"/annotate\?sid="))
    expect_frame(page, 1)
    return page.url.split("sid=")[1]


def expect_frame(page, n: int) -> None:
    expect(page.locator("#sub")).to_have_text(re.compile(rf"^Frame {n} of {NUM_FRAMES},"))


def state(page, server, sid, frame=0) -> dict:
    r = page.request.get(f"{server.url}/api/s/{sid}/state?frame={frame}")
    assert r.ok, r.text()
    return r.json()


def statuses(page, server, sid, frame=0) -> list[str]:
    return [a["status"] for a in state(page, server, sid, frame)["annotations"]]


def wait_until(check, timeout=5.0):
    """Poll `check()` until it passes. It may raise AssertionError, or IndexError / KeyError while
    the server has not applied the action yet (e.g. a new box's index does not exist yet)."""
    deadline = time.time() + timeout
    while True:
        try:
            return check()
        except (AssertionError, IndexError, KeyError):
            if time.time() > deadline:
                raise
            time.sleep(0.05)


def selected(page):
    return page.locator("#list .item.sel")


def press(page, *keys):
    for k in keys:
        page.keyboard.press(k)


# ------------------------------------------------------------------ tests
def test_open_folder_shows_the_first_frame(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    expect(page.locator("#list .item")).to_have_count(2)
    expect(page.locator("#progress")).to_have_text(f"Reviewed 0 of {TOTAL_BOXES} boxes (0%)")
    expect(page.locator("#seq")).to_contain_text("drive")
    assert state(page, server, sid)["num_frames"] == NUM_FRAMES


def test_tab_accept_reject_track_and_next_to_review(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    expect(selected(page)).to_have_count(0)
    press(page, "Tab")
    expect(selected(page)).to_contain_text("#0 Car")

    press(page, "a")                                              # accept, then move on to the next box to review
    expect(selected(page)).to_contain_text("#1 Pedestrian")
    wait_until(lambda: statuses(page, server, sid) == ["accepted", "pending"])

    press(page, "r")                                              # reject: hidden, nothing left to select
    wait_until(lambda: statuses(page, server, sid) == ["accepted", "rejected"])
    expect(page.locator("#list .item")).to_have_count(1)
    expect(selected(page)).to_have_count(0)
    expect(page.locator("#inspect")).to_contain_text("Click a box")

    press(page, "n")                                              # next frame with boxes to review
    expect_frame(page, 2)
    press(page, "Tab", "t")                                       # accept the whole Car track
    wait_until(lambda: state(page, server, sid)["stats"]["accepted"] == NUM_FRAMES)
    expect(page.locator("#progress")).to_contain_text(f"Reviewed {NUM_FRAMES + 1} of {TOTAL_BOXES}")


def test_accept_with_nothing_selected_only_selects(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    press(page, "a")
    expect(selected(page)).to_contain_text("#0 Car")
    assert statuses(page, server, sid) == ["pending", "pending"]


def test_arrow_keys_change_the_frame(page, server, sequence_dir, console_errors):
    open_sequence(page, server, sequence_dir)
    press(page, "ArrowRight")
    expect_frame(page, 2)
    press(page, "ArrowRight", "ArrowRight")
    expect_frame(page, 4)
    expect(page.locator("#list .item")).to_have_count(1)          # no Pedestrian after frame 3
    press(page, "ArrowLeft")
    expect_frame(page, 3)
    press(page, "ArrowLeft", "ArrowLeft", "ArrowLeft")             # stops at the first frame
    expect_frame(page, 1)


def test_add_copy_undo_redo(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    press(page, "b")                                              # new Car box in frame 1, selected
    expect(page.locator("#list .item")).to_have_count(3)
    expect(selected(page)).to_contain_text("Car")
    expect(selected(page)).to_contain_text("manual")

    press(page, "c")                                              # copy it to frame 2 and go there
    expect_frame(page, 2)
    expect(page.locator("#list .item")).to_have_count(3)
    expect(selected(page)).to_contain_text("manual")
    wait_until(lambda: statuses(page, server, sid, 1).count("manual") == 1)

    press(page, "z")                                              # undo the copy
    expect(page.locator("#list .item")).to_have_count(2)
    assert "manual" not in statuses(page, server, sid, 1)
    press(page, "y")                                              # and redo it
    expect(page.locator("#list .item")).to_have_count(3)
    assert statuses(page, server, sid, 0).count("manual") == 1


def test_edit_keys_and_class_change(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    press(page, "Tab", "i")                                       # 10 cm forward
    wait_until(lambda: state(page, server, sid)["annotations"][0]["x"] == pytest.approx(10.1))
    assert statuses(page, server, sid)[0] == "edited"
    press(page, "2")                                              # longer by 10 cm
    wait_until(lambda: state(page, server, sid)["annotations"][0]["dx"] == pytest.approx(4.1))

    page.select_option("#cls", "Cyclist")                          # relabels the selected box ...
    wait_until(lambda: state(page, server, sid)["annotations"][0]["label"] == "Cyclist")
    expect(selected(page)).to_contain_text("Cyclist")
    page.locator("#list .item").nth(1).click()                     # (select through the list, by mouse)
    expect(selected(page)).to_contain_text("Pedestrian")
    press(page, "b")                                              # ... and is the class of new boxes
    wait_until(lambda: state(page, server, sid)["annotations"][2]["label"] == "Cyclist")


def test_show_rejected(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    press(page, "Tab", "r")
    wait_until(lambda: statuses(page, server, sid)[0] == "rejected")
    expect(page.locator("#list .item")).to_have_count(1)
    page.check("#rej")
    expect(page.locator("#list .item")).to_have_count(2)
    expect(page.locator("#list .item").first).to_contain_text("rejected")
    page.uncheck("#rej")
    expect(page.locator("#list .item")).to_have_count(1)


def test_playback_logs_start_and_stop_but_not_every_frame(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    press(page, " ")
    expect(page.locator("#play")).to_contain_text("Pause")
    expect_frame(page, NUM_FRAMES)
    expect(page.locator("#play")).to_contain_text("Play")         # stops by itself on the last frame
    path = server.results_dir / "sessions"

    def actions():
        data = _only_session_file(path)
        acts = [e["action"] for e in data["events"]]
        assert "playback_stop" in acts
        return acts

    acts = wait_until(actions)
    assert acts.count("playback_start") == 1 and acts.count("playback_stop") == 1
    assert "frame_change" not in acts                              # the frames played are not logged
    assert _only_session_file(path)["revision"] == 0               # logging never changes the revision
    press(page, "Tab", "a")
    wait_until(lambda: state(page, server, sid, NUM_FRAMES - 1)["annotations"][0]["status"] == "accepted")


def _only_session_file(folder):
    files = list(folder.glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_timeline_click_and_drag(page, server, sequence_dir, console_errors):
    open_sequence(page, server, sequence_dir)
    box = page.locator("#strip").bounding_box()
    x_of = lambda frame: box["x"] + (frame + 0.5) * box["width"] / NUM_FRAMES
    y = box["y"] + box["height"] / 2
    page.mouse.click(x_of(4), y)
    expect_frame(page, 5)
    page.mouse.move(x_of(1), y)
    page.mouse.down()
    page.mouse.move(x_of(2), y, steps=3)
    page.mouse.move(x_of(3), y, steps=3)
    page.mouse.up()
    expect_frame(page, 4)


def test_reload_keeps_the_view_state(page, server, sequence_dir, console_errors):
    open_sequence(page, server, sequence_dir)
    page.select_option("#cls", "Pedestrian")                       # nothing selected: class for new boxes only
    press(page, "ArrowRight", "ArrowRight")
    expect_frame(page, 3)
    press(page, "Tab", "Tab")
    expect(selected(page)).to_contain_text("#1 Pedestrian")
    page.check("#rej")

    page.reload()
    expect_frame(page, 3)
    expect(selected(page)).to_contain_text("#1 Pedestrian")
    expect(page.locator("#rej")).to_be_checked()
    expect(page.locator("#cls")).to_have_value("Pedestrian")


def test_second_tab_is_refused(page, context, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    other = context.new_page()
    other.goto(server.url + "/")
    other.fill("#path", str(sequence_dir))
    other.click("#open")
    expect(other.locator("#openMsg")).to_contain_text("already open in another tab")
    expect(other.locator("#openMsg")).to_contain_text(sid)
    assert "/annotate" not in other.url
    press(page, "Tab", "a")                                       # the first tab keeps working
    wait_until(lambda: statuses(page, server, sid)[0] == "accepted")


def test_finish_then_edits_get_409(page, context, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    stale = context.new_page()                                    # the same session page in another tab
    stale.goto(f"{server.url}/annotate?sid={sid}")
    expect_frame(stale, 1)

    page.once("dialog", lambda d: d.accept())
    page.click("#finish")
    page.wait_for_url(server.url + "/")

    console_errors.expect("409")                                  # Chromium logs the failed request
    stale.keyboard.press("Tab")
    stale.keyboard.press("a")
    expect(stale.locator("#toast")).to_contain_text("finished")
    r = page.request.post(f"{server.url}/api/s/{sid}/action", data={"name": "accept", "frame": 0, "index": 0})
    assert r.status == 409 and "finished" in r.json()["detail"]


def test_export_download(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    press(page, "Tab", "t", "Tab", "t")                           # accept both tracks
    wait_until(lambda: state(page, server, sid)["stats"]["reviewed"] == TOTAL_BOXES)
    page.click("#exp summary")
    expect(page.locator("#expInfo")).to_contain_text(f"{NUM_FRAMES} frames with {TOTAL_BOXES} boxes")
    with page.expect_download() as dl:
        page.click("#expGo")
    with zipfile.ZipFile(dl.value.path()) as z:
        labels = sorted(n for n in z.namelist() if n.startswith("labels/"))
    assert labels == [f"labels/{i:06d}.txt" for i in range(NUM_FRAMES)]


def test_undo_after_server_restart(page, server, sequence_dir, console_errors):
    sid = open_sequence(page, server, sequence_dir)
    press(page, "Tab", "a")
    wait_until(lambda: statuses(page, server, sid)[0] == "accepted")

    server.restart()                                              # a new process: nothing in memory
    new_sid = open_sequence(page, server, sequence_dir)
    assert new_sid != sid
    expect(page.locator("#list .item").first).to_contain_text("accepted")
    press(page, "z")
    expect(page.locator("#list .item").first).to_contain_text("pending")
    assert statuses(page, server, new_sid)[0] == "pending"


@pytest.mark.postgres
def test_review_on_postgres_survives_a_restart(page, tmp_path, sequence_dir, console_errors, pg_engine, pg_url):
    """The whole stack with LAL_SESSION_STORE=postgres: decisions, undo after a server restart.
    The lock is in the database, so it outlives the restart: the session can be opened again only
    once the old tab's lock has expired (here after 3 s instead of the default 120 s)."""
    from tests.e2e.conftest import AppServer

    ttl = 3.0
    srv = AppServer(tmp_path / "results", {"LAL_SESSION_STORE": "postgres", "LAL_DATABASE_URL": pg_url,
                                           "LAL_LOCK_TTL_SECONDS": str(ttl)})
    srv.start()
    try:
        sid = open_sequence(page, srv, sequence_dir)
        press(page, "Tab", "a")
        wait_until(lambda: statuses(page, srv, sid)[0] == "accepted")
        assert not (tmp_path / "results" / "sessions").exists()       # nothing written as files
        srv.restart()
        started = time.time()
        page.goto(srv.url + "/")
        page.fill("#path", str(sequence_dir))
        page.click("#open")
        if time.time() - started < ttl - 0.5:                          # opened again before the lock expired
            expect(page.locator("#openMsg")).to_contain_text("already open in another tab")
            expect(page.locator("#openMsg")).to_contain_text(sid)
        time.sleep(max(0.0, ttl + 0.5 - (time.time() - started)))
        new_sid = open_sequence(page, srv, sequence_dir)
        expect(page.locator("#list .item").first).to_contain_text("accepted")
        press(page, "z")
        expect(page.locator("#list .item").first).to_contain_text("pending")
        assert statuses(page, srv, new_sid)[0] == "pending"
    finally:
        srv.stop()


def test_opening_shows_detection_progress_and_can_be_cancelled(page, tmp_path, sequence_dir, console_errors):
    from tests.e2e.conftest import AppServer

    srv = AppServer(tmp_path / "results", {"LAL_E2E_TRACK_DELAY": "0.5"})
    srv.start()
    try:
        page.goto(srv.url + "/")
        page.fill("#path", str(sequence_dir))
        page.click("#open")
        expect(page.locator("#openMsg")).to_contain_text(re.compile(r"Detecting objects: [1-9]\d*%"))
        expect(page.locator("#cancelOpen")).to_be_visible()
        page.click("#cancelOpen")
        expect(page.locator("#openMsg")).to_have_text("Cancelled.", timeout=10000)
        expect(page.locator("#cancelOpen")).to_be_hidden()
        assert "/annotate" not in page.url
        assert not list((tmp_path / "results").glob("*.json"))       # no partial detection cache
        page.fill("#path", str(sequence_dir))                           # opening again works
        page.click("#open")
        page.wait_for_url(re.compile(r"/annotate\?sid="), timeout=20000)
    finally:
        srv.stop()


# Not covered (needs a human look or pixel-level tests):
# - selecting a box by clicking it in the 3D view (raycast on the WebGL canvas),
# - orbit / pan / zoom with the mouse and the view buttons (Top view, Focus box, Reset view),
# - that boxes and points are drawn at the right place and colour.
