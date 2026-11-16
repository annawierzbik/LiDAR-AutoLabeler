"""End-to-end tests of the browser UI against the real stack (API process, Celery worker with fake
detection, PostgreSQL, Redis; synthetic sequence). Every test logs in first.

    docker compose -f docker-compose.db.yml up -d
    pip install -r requirements-dev.txt && python -m playwright install chromium
    pytest -m e2e

Assertions use the API state and the visible DOM, never pixels. Box selection is driven through
the keyboard and the box list; picking a box by clicking it in the 3D view is not tested (see
the bottom of this file). Every request outside 127.0.0.1 is blocked and fails the test: the
page must work offline (three.js and the font are vendored in web/static/vendor).
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pytest
from playwright.sync_api import expect
from sqlalchemy import text

from tests.e2e.conftest import NUM_FRAMES, PASSWORD, login

pytestmark = [pytest.mark.e2e, pytest.mark.postgres]

# fake detections (tests/e2e/fake_tracks.py): Car track 0 in all 6 frames, Pedestrian track 1 in frames 0..2
TOTAL_BOXES = NUM_FRAMES + 3


# ------------------------------------------------------------------ helpers
def open_sequence(page, stack, name: str = "drive", log_in: bool = True) -> str:
    if log_in:
        login(page, stack)
    else:
        page.goto(stack.url + "/")
    page.get_by_role("button", name=f"Open {name}").click()
    page.wait_for_url(re.compile(r"/annotate\?sid="), timeout=20000)
    expect_frame(page, 1)
    return page.url.split("sid=")[1]


def expect_frame(page, n: int) -> None:
    expect(page.locator("#sub")).to_have_text(re.compile(rf"^Frame {n} of {NUM_FRAMES},"))


def client_id(page) -> str:
    return page.evaluate("() => sessionStorage.getItem('lal:client')")


def csrf(page) -> str:
    return next(c["value"] for c in page.context.cookies() if c["name"] == "lal_csrf")


def state(page, stack, sid, frame=0) -> dict:
    r = page.request.get(f"{stack.url}/api/v1/tabs/{sid}/state?frame={frame}", headers={"X-Client-Id": client_id(page)})
    assert r.ok, r.text()
    return r.json()


def statuses(page, stack, sid, frame=0) -> list[str]:
    return [a["status"] for a in state(page, stack, sid, frame)["annotations"]]


def wait_until(check, timeout=5.0):
    """Poll `check()` until it passes. It may raise AssertionError, or IndexError / KeyError while
    the server has not applied the action yet (e.g. a new box's index does not exist yet)."""
    import time

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


# ------------------------------------------------------------------ login
def test_pages_need_a_login(page, stack, console_errors):
    page.goto(stack.url + "/")
    page.wait_for_url(re.compile(r"/login\?next=%2F$"))
    page.fill("#username", "anna")
    page.fill("#password", "not the password")
    console_errors.expect("401")
    page.click("#login")
    expect(page.locator("#msg")).to_have_text("Wrong user name or password.")
    page.fill("#password", PASSWORD)
    page.click("#login")
    page.wait_for_url(stack.url + "/")
    expect(page.locator("#me")).to_have_text("anna (annotator)")
    page.click("#logout")
    page.wait_for_url(stack.url + "/login")
    page.goto(stack.url + "/")
    page.wait_for_url(re.compile(r"/login"))


# ------------------------------------------------------------------ review
def test_open_sequence_shows_the_first_frame(page, stack, console_errors):
    sid = open_sequence(page, stack)
    expect(page.locator("#list .item")).to_have_count(2)
    expect(page.locator("#progress")).to_have_text(f"Reviewed 0 of {TOTAL_BOXES} boxes (0%)")
    expect(page.locator("#seq")).to_contain_text("drive")
    assert state(page, stack, sid)["num_frames"] == NUM_FRAMES


def test_tab_accept_reject_track_and_next_to_review(page, stack, console_errors):
    sid = open_sequence(page, stack)
    expect(selected(page)).to_have_count(0)
    press(page, "Tab")
    expect(selected(page)).to_contain_text("#0 Car")

    press(page, "a")                                              # accept, then move on to the next box to review
    expect(selected(page)).to_contain_text("#1 Pedestrian")
    wait_until(lambda: statuses(page, stack, sid) == ["accepted", "pending"])

    press(page, "r")                                              # reject: hidden, nothing left to select
    wait_until(lambda: statuses(page, stack, sid) == ["accepted", "rejected"])
    expect(page.locator("#list .item")).to_have_count(1)
    expect(selected(page)).to_have_count(0)
    expect(page.locator("#inspect")).to_contain_text("Click a box")

    press(page, "n")                                              # next frame with boxes to review
    expect_frame(page, 2)
    press(page, "Tab", "t")                                       # accept the whole Car track
    wait_until(lambda: state(page, stack, sid)["stats"]["accepted"] == NUM_FRAMES)
    expect(page.locator("#progress")).to_contain_text(f"Reviewed {NUM_FRAMES + 1} of {TOTAL_BOXES}")


def test_accept_with_nothing_selected_only_selects(page, stack, console_errors):
    sid = open_sequence(page, stack)
    press(page, "a")
    expect(selected(page)).to_contain_text("#0 Car")
    assert statuses(page, stack, sid) == ["pending", "pending"]


def test_arrow_keys_change_the_frame(page, stack, console_errors):
    open_sequence(page, stack)
    press(page, "ArrowRight")
    expect_frame(page, 2)
    press(page, "ArrowRight", "ArrowRight")
    expect_frame(page, 4)
    expect(page.locator("#list .item")).to_have_count(1)          # no Pedestrian after frame 3
    press(page, "ArrowLeft")
    expect_frame(page, 3)
    press(page, "ArrowLeft", "ArrowLeft", "ArrowLeft")             # stops at the first frame
    expect_frame(page, 1)


def test_add_copy_undo_redo(page, stack, console_errors):
    sid = open_sequence(page, stack)
    press(page, "b")                                              # new Car box in frame 1, selected
    expect(page.locator("#list .item")).to_have_count(3)
    expect(selected(page)).to_contain_text("Car")
    expect(selected(page)).to_contain_text("manual")

    press(page, "c")                                              # copy it to frame 2 and go there
    expect_frame(page, 2)
    expect(page.locator("#list .item")).to_have_count(3)
    expect(selected(page)).to_contain_text("manual")
    wait_until(lambda: statuses(page, stack, sid, 1).count("manual") == 1)

    press(page, "z")                                              # undo the copy
    expect(page.locator("#list .item")).to_have_count(2)
    assert "manual" not in statuses(page, stack, sid, 1)
    press(page, "y")                                              # and redo it
    expect(page.locator("#list .item")).to_have_count(3)
    assert statuses(page, stack, sid, 0).count("manual") == 1


def test_edit_keys_and_class_change(page, stack, console_errors):
    sid = open_sequence(page, stack)
    press(page, "Tab", "i")                                       # 10 cm forward
    wait_until(lambda: state(page, stack, sid)["annotations"][0]["x"] == pytest.approx(10.1))
    assert statuses(page, stack, sid)[0] == "edited"
    press(page, "2")                                              # longer by 10 cm
    wait_until(lambda: state(page, stack, sid)["annotations"][0]["dx"] == pytest.approx(4.1))

    page.select_option("#cls", "Cyclist")                          # relabels the selected box ...
    wait_until(lambda: state(page, stack, sid)["annotations"][0]["label"] == "Cyclist")
    expect(selected(page)).to_contain_text("Cyclist")
    page.locator("#list .item").nth(1).click()                     # (select through the list, by mouse)
    expect(selected(page)).to_contain_text("Pedestrian")
    press(page, "b")                                              # ... and is the class of new boxes
    wait_until(lambda: state(page, stack, sid)["annotations"][2]["label"] == "Cyclist")


def test_show_rejected(page, stack, console_errors):
    sid = open_sequence(page, stack)
    press(page, "Tab", "r")
    wait_until(lambda: statuses(page, stack, sid)[0] == "rejected")
    expect(page.locator("#list .item")).to_have_count(1)
    page.check("#rej")
    expect(page.locator("#list .item")).to_have_count(2)
    expect(page.locator("#list .item").first).to_contain_text("rejected")
    page.uncheck("#rej")
    expect(page.locator("#list .item")).to_have_count(1)


def test_playback_logs_start_and_stop_but_not_every_frame(page, stack, console_errors):
    sid = open_sequence(page, stack)
    press(page, " ")
    expect(page.locator("#play")).to_contain_text("Pause")
    expect_frame(page, NUM_FRAMES)
    expect(page.locator("#play")).to_contain_text("Play")         # stops by itself on the last frame

    def stored():
        with stack.pg_engine.connect() as c:
            key, revision = c.execute(text("SELECT key, revision FROM sessions")).one()
            acts = list(c.execute(text("SELECT action FROM session_events WHERE session_key = :k ORDER BY id"),
                                  {"k": key}).scalars())
        assert "playback_stop" in acts
        return acts, revision

    acts, revision = wait_until(stored)
    assert acts.count("playback_start") == 1 and acts.count("playback_stop") == 1
    assert "frame_change" not in acts                              # the frames played are not logged
    assert revision == 0                                           # logging never changes the revision
    press(page, "Tab", "a")
    wait_until(lambda: state(page, stack, sid, NUM_FRAMES - 1)["annotations"][0]["status"] == "accepted")


def test_timeline_click_and_drag(page, stack, console_errors):
    open_sequence(page, stack)
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


def test_reload_keeps_the_view_state(page, stack, console_errors):
    open_sequence(page, stack)
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


# ------------------------------------------------------------------ tabs, locks, restarts
def test_second_tab_is_refused(page, context, stack, console_errors):
    sid = open_sequence(page, stack)
    other = context.new_page()                                    # a new tab: its own client id
    open_sequence_attempt = other.goto(stack.url + "/")
    assert open_sequence_attempt.ok
    other.get_by_role("button", name="Open drive").click()
    expect(other.locator("#openMsg")).to_contain_text("open in another tab")
    assert "/annotate" not in other.url
    press(page, "Tab", "a")                                       # the first tab keeps working
    wait_until(lambda: statuses(page, stack, sid)[0] == "accepted")


def test_finish_then_the_tab_is_closed(page, stack, console_errors):
    sid = open_sequence(page, stack)
    page.once("dialog", lambda d: d.accept())
    page.click("#finish")
    page.wait_for_url(stack.url + "/")
    r = page.request.post(f"{stack.url}/api/v1/tabs/{sid}/action", data={"name": "accept", "frame": 0, "index": 0},
                          headers={"X-Client-Id": client_id(page), "X-CSRF-Token": csrf(page)})
    assert r.status == 409 and r.json()["error"]["code"] == "finished"
    console_errors.expect("409", 2)                               # the page asks twice (frame, then frame 0)
    page.goto(f"{stack.url}/annotate?sid={sid}")                  # back on the finished session's page
    expect(page.locator("#toast")).to_contain_text("finished")


def test_page_keeps_working_across_a_server_restart_with_undo(page, stack, console_errors):
    """The tab binding and the lock are in the database: the page that was open before the restart
    goes on at once (same user, same client id), including undo of a decision made before it."""
    sid = open_sequence(page, stack)
    press(page, "Tab", "a")
    wait_until(lambda: statuses(page, stack, sid)[0] == "accepted")
    assert not (stack.results_dir / "sessions").exists()           # nothing written as session files

    stack.apis[0].restart()                                        # a new process: nothing in memory
    press(page, "z")
    expect(page.locator("#list .item").first).to_contain_text("pending")
    assert statuses(page, stack, sid)[0] == "pending"
    press(page, "y")
    expect(page.locator("#list .item").first).to_contain_text("accepted")
    page.reload()                                                  # and after a reload of the page
    expect(page.locator("#list .item").first).to_contain_text("accepted")


# ------------------------------------------------------------------ export, upload, progress
def test_export_through_the_cpu_queue(page, stack, console_errors):
    sid = open_sequence(page, stack)
    press(page, "Tab", "t", "Tab", "t")                           # accept both tracks
    wait_until(lambda: state(page, stack, sid)["stats"]["reviewed"] == TOTAL_BOXES)
    page.click("#exp summary")
    page.click("#expRun")
    expect(page.locator("#expInfo")).to_contain_text(f"{NUM_FRAMES} frames with {TOTAL_BOXES} boxes", timeout=15000)
    with page.expect_download() as dl:
        page.click("#expGo")
    with zipfile.ZipFile(dl.value.path()) as z:
        labels = sorted(n for n in z.namelist() if n.startswith("labels/"))
    assert labels == [f"labels/{i:06d}.txt" for i in range(NUM_FRAMES)]
    page.select_option("#expFmt", "kitti")                         # other options: the old link goes away
    expect(page.locator("#expGo")).to_have_attribute("aria-disabled", "true")


def test_upload_a_zip_and_annotate_it(page, stack, tmp_path, console_errors):
    rng = np.random.default_rng(3)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(NUM_FRAMES):
            z.writestr(f"scan/{i:04d}.bin", rng.uniform(-30, 30, (500, 4)).astype("<f4").tobytes())
    archive = tmp_path / "scan.zip"
    archive.write_bytes(buf.getvalue())
    login(page, stack)
    page.set_input_files("#upFile", str(archive))
    page.click("#upload")
    expect(page.locator("#upMsg")).to_have_text(f"Uploaded scan: {NUM_FRAMES} frames.")
    sid = open_sequence(page, stack, "scan", log_in=False)
    expect(page.locator("#list .item")).to_have_count(2)
    assert state(page, stack, sid)["sequence"]


def test_opening_shows_detection_progress_and_can_be_cancelled(page, make_stack, console_errors):
    stack = make_stack(env={"LAL_E2E_TRACK_DELAY": "0.5"})
    anna = stack.user("anna")
    stack.sequence(anna)
    login(page, stack)
    page.get_by_role("button", name="Open drive").click()
    expect(page.locator("#openMsg")).to_contain_text(re.compile(r"Detecting objects: [1-9]\d*%"), timeout=20000)
    expect(page.locator("#cancelOpen")).to_be_visible()
    page.click("#cancelOpen")
    expect(page.locator("#openMsg")).to_have_text("Cancelled.", timeout=10000)
    expect(page.locator("#cancelOpen")).to_be_hidden()
    assert "/annotate" not in page.url
    assert not list(stack.results_dir.glob("*.json"))              # no partial detection cache
    page.get_by_role("button", name="Open drive").click()          # opening again works
    page.wait_for_url(re.compile(r"/annotate\?sid="), timeout=30000)


# ------------------------------------------------------------------ users
def test_a_second_user_does_not_see_the_first_users_session(page, stack, browser, console_errors):
    sid = open_sequence(page, stack)
    press(page, "Tab", "a")
    wait_until(lambda: statuses(page, stack, sid)[0] == "accepted")

    stack.user("bob")
    bob_context = browser.new_context()
    console_errors.guard(bob_context)                             # watches bob's pages too
    bob = bob_context.new_page()
    try:
        login(bob, stack, "bob")
        expect(bob.locator("#noSeqs")).to_be_visible()
        expect(bob.locator("#seqs li")).to_have_count(0)
        h = {"X-Client-Id": client_id(bob)}
        for path in (f"/api/v1/tabs/{sid}", f"/api/v1/tabs/{sid}/state", f"/api/v1/tabs/{sid}/points/0",
                     f"/api/v1/sequences/{stack.drive}"):
            assert bob.request.get(stack.url + path, headers=h).status == 404, path
        assert bob.request.get(stack.url + "/api/v1/sessions").json()["total"] == 0
        console_errors.expect("404", 2)                            # the page's own failed requests
        bob.goto(f"{stack.url}/annotate?sid={sid}")
        expect(bob.locator("#toast")).to_contain_text("unknown tab")
    finally:
        bob_context.close()
    assert statuses(page, stack, sid)[0] == "accepted"               # anna's work is untouched


# ------------------------------------------------------------------ statelessness
def test_two_api_processes_behind_a_round_robin_gateway(page, make_stack, console_errors):
    """No per-session state in the API: every request goes to the other of two API processes, and
    the whole scenario works (login, opening, review, undo / redo, export, finish)."""
    stack = make_stack(apis=2, proxy=True)
    anna = stack.user("anna")
    stack.sequence(anna)
    sid = open_sequence(page, stack)
    press(page, "Tab", "a")
    wait_until(lambda: statuses(page, stack, sid) == ["accepted", "pending"])
    press(page, "b")                                               # add a box ...
    expect(page.locator("#list .item")).to_have_count(3)
    press(page, "z")                                               # ... undo it
    expect(page.locator("#list .item")).to_have_count(2)
    press(page, "z")                                               # ... and the accept
    wait_until(lambda: statuses(page, stack, sid) == ["pending", "pending"])
    press(page, "y")
    wait_until(lambda: statuses(page, stack, sid) == ["accepted", "pending"])
    press(page, "ArrowRight")
    expect_frame(page, 2)
    page.click("#exp summary")
    page.check("#expUnrev")
    page.click("#expRun")
    expect(page.locator("#expInfo")).to_contain_text(f"{NUM_FRAMES} frames with", timeout=15000)
    with page.expect_download() as dl:
        page.click("#expGo")
    assert zipfile.ZipFile(dl.value.path()).namelist()
    page.once("dialog", lambda d: d.accept())
    page.click("#finish")
    page.wait_for_url(stack.url + "/")
    hits = stack.proxy_hits()
    assert min(hits) >= 10 and abs(hits[0] - hits[1]) <= 1, hits   # both processes served the session


# Not covered (needs a human look or pixel-level tests):
# - selecting a box by clicking it in the 3D view (raycast on the WebGL canvas),
# - orbit / pan / zoom with the mouse and the view buttons (Top view, Focus box, Reset view),
# - that boxes and points are drawn at the right place and colour.
