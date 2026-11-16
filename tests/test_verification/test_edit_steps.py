"""The browser's edit actions take their step sizes from verification/edit_steps.py."""
import pytest

import verification.edit_steps as steps
import web.service

NAMES = ("MOVE_STEP", "Z_STEP", "ROT_STEP", "SIZE_STEP")


@pytest.mark.parametrize("name", NAMES)
def test_browser_uses_the_shared_steps(name):
    assert getattr(web.service, name) is getattr(steps, name)


def test_actions_use_exactly_these_steps():
    assert web.service.NUDGES["forward"] == {"dx": steps.MOVE_STEP}
    assert web.service.NUDGES["up"] == {"dz": steps.Z_STEP}
    assert web.service.NUDGES["rotate_left"] == {"dheading": steps.ROT_STEP}
    assert web.service.RESIZES["length_plus"] == {"dlength": steps.SIZE_STEP}
