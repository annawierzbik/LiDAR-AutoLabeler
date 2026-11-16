"""Both UIs take the box edit steps from verification/edit_steps.py."""
import pytest

import verification.edit_steps as steps
import web.service

NAMES = ("MOVE_STEP", "Z_STEP", "ROT_STEP", "SIZE_STEP")


@pytest.mark.parametrize("name", NAMES)
def test_browser_uses_the_shared_steps(name):
    assert getattr(web.service, name) is getattr(steps, name)


@pytest.mark.parametrize("name", NAMES)
def test_desktop_uses_the_shared_steps(name):
    app = pytest.importorskip("verification.app")   # needs open3d
    assert getattr(app, name) is getattr(steps, name)
