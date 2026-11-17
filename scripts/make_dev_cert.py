"""A self-signed TLS certificate for https://localhost (local deployment and the smoke test only).

    python scripts/make_dev_cert.py                   # writes deploy/nginx/certs/tls.crt and tls.key
    python scripts/make_dev_cert.py --out D:/certs --days 7

Uses the `openssl` program (Git for Windows ships one; Linux and macOS have it). Browsers warn
about a self-signed certificate; a real deployment mounts its own certificate and key as tls.crt /
tls.key from LAL_TLS_DIR (docs/adr/0019). The key never goes into git (deploy/nginx/certs is ignored).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make(out: Path, days: int = 30, host: str = "localhost") -> tuple[Path, Path]:
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("the openssl program is not on PATH")
    out.mkdir(parents=True, exist_ok=True)
    crt, key = out / "tls.crt", out / "tls.key"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes", "-days", str(days),
                    "-keyout", str(key), "-out", str(crt), "-subj", f"/CN={host}",
                    "-addext", f"subjectAltName=DNS:{host},IP:127.0.0.1"],
                   check=True, capture_output=True, text=True)
    return crt, key


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "deploy" / "nginx" / "certs")
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args(argv)
    try:
        crt, key = make(args.out, args.days)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(getattr(exc, "stderr", None) or exc, file=sys.stderr)
        return 1
    print(f"wrote {crt} and {key} (self-signed, {args.days} days)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
