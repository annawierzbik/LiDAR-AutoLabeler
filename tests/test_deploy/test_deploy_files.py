"""The deployment files (docs/adr/0019): no default secrets, every variable documented, the nginx
upload limit is the API's, and nothing secret is copied into an image."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
TEMPLATE = (ROOT / "deploy" / "nginx" / "templates" / "default.conf.template").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")
SECRETS = ("LAL_DB_PASSWORD", "LAL_REDIS_PASSWORD", "LAL_JWT_SECRET", "LAL_MINIO_USER", "LAL_MINIO_PASSWORD")


def compose_vars() -> dict[str, list[str]]:
    """{name: [the text after the name in each ${...}]}"""
    out: dict[str, list[str]] = {}
    for name, rest in re.findall(r"\$\{([A-Z0-9_]+)([^}]*)\}", COMPOSE):
        out.setdefault(name, []).append(rest)
    return out


def test_secrets_have_no_default_and_are_required():
    found = compose_vars()
    for name in SECRETS:
        assert found.get(name), name
        assert all(rest.startswith(":?") for rest in found[name]), (name, found[name])
    for word in ("password", "secret"):                                # no literal credentials either
        for line in COMPOSE.splitlines():
            if word in line.lower() and ":" in line and "${" not in line and not line.strip().startswith("#"):
                assert "$$" in line or line.strip().endswith(":"), line


def test_every_variable_is_documented_in_env_example():
    documented = set(re.findall(r"^([A-Z0-9_]+)=", ENV_EXAMPLE, re.MULTILINE))
    assert set(compose_vars()) <= documented, sorted(set(compose_vars()) - documented)
    for name in SECRETS:
        assert re.search(rf"^{name}=$", ENV_EXAMPLE, re.MULTILINE), name    # empty: no example secret


def test_nginx_gets_every_template_variable_and_the_api_upload_limit():
    used = set(re.findall(r"\$\{(LAL_[A-Z0-9_]+)\}", TEMPLATE))
    nginx = COMPOSE[COMPOSE.index("  nginx:"):]
    given = set(re.findall(r"^\s+(LAL_[A-Z0-9_]+):", nginx, re.MULTILINE))
    assert used <= given, sorted(used - given)
    assert "client_max_body_size ${LAL_UPLOAD_MAX_BYTES};" in TEMPLATE
    app_default = re.search(r"LAL_UPLOAD_MAX_BYTES: \$\{LAL_UPLOAD_MAX_BYTES:-(\d+)\}", COMPOSE).group(1)
    nginx_default = re.search(r"LAL_UPLOAD_MAX_BYTES: \$\{LAL_UPLOAD_MAX_BYTES:-(\d+)\}", nginx).group(1)
    from infra.settings import Settings

    assert int(app_default) == int(nginx_default) == Settings.model_fields["upload_max_bytes"].default


def test_only_nginx_publishes_ports():
    services = re.split(r"^  (?=[a-z][a-z0-9-]*:\n)", COMPOSE.split("\nservices:\n", 1)[1].split("\nvolumes:\n")[0],
                        flags=re.MULTILINE)
    with_ports = [s.split(":", 1)[0] for s in services if "\n    ports:" in s]
    assert with_ports == ["nginx"]


def test_the_image_context_leaves_out_data_weights_and_secrets():
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert next(line for line in ignore if line.strip() and not line.startswith("#")) == "*"   # allow-list
    allowed = [line[1:] for line in ignore if line.startswith("!")]
    for path in allowed:
        assert not re.search(r"(^|/)(\.env|data|checkpoints|results|venv|OpenPCDet)(/|$)", path), path
        assert not path.endswith((".pth", ".pt", ".key")), path


def test_images_put_the_project_on_the_python_path():
    """The `celery` console script adds the working directory to sys.path only while it imports
    -A infra.celery_app; later imports of other project packages (detection, pipeline, ...) need
    PYTHONPATH (found by the smoke test: "No module named 'detection'" in a worker container)."""
    for name in ("Dockerfile", "Dockerfile.gpu"):
        text = (ROOT / "deploy" / name).read_text(encoding="utf-8")
        assert re.search(r"PYTHONPATH=/app\b", text), name
