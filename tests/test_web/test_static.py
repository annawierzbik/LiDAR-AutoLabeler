"""The pages and their assets: served from web/static, no CDN, everything they load tracked in git."""
import re
import shutil
import subprocess

import pytest

from api.app import STATIC

PAGES = ("home.html", "login.html", "annotate.html")


def test_pages_need_no_cdn():
    for page in (*PAGES, "client.js"):
        urls = re.findall(r"https?://[^\s'\"<>)`]+", (STATIC / page).read_text(encoding="utf-8"))
        assert [u for u in urls if not u.startswith("http://www.w3.org/")] == [], page


def test_vendored_assets_exist():
    assert "const REVISION = '160';" in (STATIC / "vendor/three-0.160.0/build/three.module.js").read_text(encoding="utf-8")
    assert (STATIC / "vendor/three-0.160.0/examples/jsm/controls/OrbitControls.js").is_file()
    assert (STATIC / "vendor/schibsted-grotesk/font.css").is_file()


def test_no_token_in_web_storage():
    """Tokens live only in HttpOnly cookies (docs/adr/0016): no script stores one."""
    for page in (*PAGES, "client.js"):
        text = (STATIC / page).read_text(encoding="utf-8")
        for store in re.findall(r"(?:localStorage|sessionStorage)\.setItem\(([^,]+),", text):
            assert "token" not in store.lower() and "jwt" not in store.lower(), (page, store)


def test_vendored_assets_are_in_git():
    """Every asset the pages load must be tracked (a .gitignore rule once kept three.module.js out)."""
    root = STATIC.parents[1]
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = set(subprocess.run(["git", "ls-files", "web/static"], cwd=root, capture_output=True, text=True,
                                 check=True).stdout.split())
    used = {"web/static/vendor/three-0.160.0/examples/jsm/controls/OrbitControls.js"}
    for page in PAGES:
        text = (STATIC / page).read_text(encoding="utf-8")
        used |= {"web" + u for u in re.findall(r"['\"](/static/[^'\"]+\.(?:js|css))['\"]", text)}
    css = (STATIC / "vendor" / "schibsted-grotesk" / "font.css").read_text(encoding="utf-8")
    used |= {f"web/static/vendor/schibsted-grotesk/{f}" for f in re.findall(r"url\(([^)]+\.woff2)\)", css)}
    assert "web/static/client.js" in used and used <= tracked, sorted(used - tracked)
