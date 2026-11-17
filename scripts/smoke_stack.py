"""Bring the deployment stack up, test it through nginx at https://localhost, tear it down.

    python scripts/smoke_stack.py                 # fake detector (tests/smoke/docker-compose.smoke.yml)
    python scripts/smoke_stack.py --keep          # leave the stack running afterwards
    python scripts/smoke_stack.py --gpu           # the real GPU worker (--profile gpu) instead of the
                                                  # fake one, on 6 frames of KITTI tracking 0000
                                                  # (data/raw): only when no experiment uses the GPU

Steps: random secrets in a temporary .env (nothing is written into the repository), a self-signed
certificate for localhost, `docker compose up -d --build` of docker-compose.yml (+ the test-only
override with a fake-detector worker on the gpu queue), wait until api and nginx are healthy,
create the user "admin", run tests/smoke (pytest -m smoke: gateway, the browser scenario of two
users, worker-cpu scaled to 3, the login rate limit), then `docker compose down -v` (also on
failure, unless --keep). Exit code: the tests' exit code (or 1 if the stack did not come up).
Needs Docker with Compose, openssl, and requirements-dev.txt with Chromium for Playwright.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UPLOAD_MAX_BYTES = 1_000_000           # small, so the gateway's 413 is cheap to test
UPLOAD_MAX_BYTES_GPU = 64_000_000      # --gpu uploads 6 real KITTI frames (about 11 MB)


def log(msg: str) -> None:
    print(f"[smoke {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_env(folder: Path, https_port: int, http_port: int, gpu: bool) -> Path:
    env = {
        "LAL_DB_PASSWORD": secrets.token_urlsafe(24),
        "LAL_REDIS_PASSWORD": secrets.token_urlsafe(24),
        "LAL_JWT_SECRET": secrets.token_urlsafe(48),
        "LAL_MINIO_USER": "lal-" + secrets.token_hex(6),
        "LAL_MINIO_PASSWORD": secrets.token_urlsafe(24),
        "LAL_HTTPS_PORT": str(https_port),
        "LAL_HTTP_PORT": str(http_port),
        "LAL_TLS_DIR": str(folder / "certs"),
        "LAL_HSTS_MAX_AGE": "0",
        "LAL_ASSISTED_MODE": "true",
        "LAL_UPLOAD_MAX_BYTES": str(UPLOAD_MAX_BYTES_GPU if gpu else UPLOAD_MAX_BYTES),
        "LAL_IMAGE_TAG": "smoke",
    }
    if gpu:
        env["LAL_MODELS_DIR"] = str(ROOT / "checkpoints")
    path = folder / ".env"
    path.write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
    return path


def wait_healthy(compose: list[str], services: list[str], timeout: float) -> None:
    deadline = time.time() + timeout
    while True:
        out = subprocess.run([*compose, "ps", "--all", "--format", "json"], check=False, capture_output=True,
                             text=True).stdout
        rows = [json.loads(line) for line in out.splitlines() if line.strip().startswith("{")]
        state = {r["Service"]: (r.get("State"), r.get("Health"), r.get("ExitCode")) for r in rows}
        migrate = state.get("migrate")
        if migrate and migrate[0] == "exited" and migrate[2] not in (0, None):
            raise RuntimeError(f"the migration failed: {migrate}")
        if all(state.get(s, (None, None, None))[1] == "healthy" for s in services):
            return
        if time.time() > deadline:
            raise RuntimeError(f"not healthy after {timeout:.0f} s: {state}")
        time.sleep(3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep", action="store_true", help="do not tear the stack down")
    p.add_argument("--gpu", action="store_true", help="real GPU worker instead of the fake detector")
    p.add_argument("--https-port", type=int, default=8443)
    p.add_argument("--http-port", type=int, default=8080)
    p.add_argument("--project", default="lal-smoke")
    p.add_argument("--timeout", type=float, default=900, help="seconds for the stack to become healthy")
    args = p.parse_args(argv)

    from scripts.make_dev_cert import make

    work = Path(tempfile.mkdtemp(prefix="lal-smoke-"))
    env_file = write_env(work, args.https_port, args.http_port, args.gpu)
    make(work / "certs", days=2)
    files = ["-f", str(ROOT / "docker-compose.yml")]
    if args.gpu:
        files = [*files, "--profile", "gpu"]
    else:
        files = [*files, "-f", str(ROOT / "tests" / "smoke" / "docker-compose.smoke.yml")]
    compose = ["docker", "compose", "-p", args.project, "--project-directory", str(ROOT), "--env-file", str(env_file),
               *files]
    code = 1
    try:
        log("building and starting the stack (the first build takes a while)")
        subprocess.run([*compose, "up", "-d", "--build"], check=True)
        wait_healthy(compose, ["api", "nginx", "worker-cpu", "scheduler",
                               "worker-gpu" if args.gpu else "worker-fake-gpu"], args.timeout)
        log("stack healthy; creating the user admin")
        admin_password = secrets.token_urlsafe(18)
        subprocess.run([*compose, "exec", "-T", "api", "python", "-m", "scripts.create_user", "admin", "--role", "admin",
                        "--password-stdin"], input=admin_password + "\n", text=True, check=True)
        test_env = {**os.environ,
                    "LAL_SMOKE_URL": f"https://localhost:{args.https_port}",
                    "LAL_SMOKE_HTTP_URL": f"http://localhost:{args.http_port}",
                    "LAL_SMOKE_ADMIN_PASSWORD": admin_password,
                    "LAL_SMOKE_UPLOAD_MAX_BYTES": str(UPLOAD_MAX_BYTES_GPU if args.gpu else UPLOAD_MAX_BYTES),
                    "LAL_SMOKE_COMPOSE": json.dumps(compose)}
        if args.gpu:                                             # the real model needs real point clouds
            test_env["LAL_SMOKE_FRAMES_DIR"] = str(ROOT / "data" / "raw" / "training" / "velodyne" / "0000")
        log("running tests/smoke through nginx")
        code = subprocess.run([sys.executable, "-m", "pytest", "-m", "smoke", "tests/smoke", "-v", "-s",
                               "-p", "no:cacheprovider", "-p", "no:randomly"],      # the smoke tests run in file order
                              check=False, cwd=ROOT, env=test_env).returncode
        log(f"tests finished with exit code {code}")
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        log(f"the stack did not come up: {exc}")
        subprocess.run([*compose, "ps", "--all"], check=False)
        subprocess.run([*compose, "logs", "--tail", "60"], check=False)
    finally:
        if args.keep:
            log(f"left running: {' '.join(compose)} down -v   # to remove it")
        else:
            log("tearing down (docker compose down -v)")
            subprocess.run([*compose, "down", "-v", "--remove-orphans"], check=False)
    return code


if __name__ == "__main__":
    sys.exit(main())
