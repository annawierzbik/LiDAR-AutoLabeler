# 0019 Containers and gateway: images, compose services, nginx, secrets, smoke test

Status: accepted (phase 5)

## Decision
- **Images** (`deploy/`):
  - `deploy/Dockerfile` (app): `python:3.11.9-slim-bookworm`, `deploy/requirements-app.txt` (the
    pins of `requirements.txt` without torch / OpenPCDet; `celery` without the `[redis]` extra,
    which pins an older redis-py than kombu 5.7 needs; the same fix is in `requirements.txt`),
    non-root user. Runs API, cpu worker, scheduler and the migration.
  - `deploy/Dockerfile.gpu` (worker-gpu): build stage `nvidia/cuda:12.4.1-devel-ubuntu22.04`,
    Python 3.11 (deadsnakes), torch 2.6.0+cu124 from the PyTorch index, `spconv-cu120` 2.3.6 and the
    other versions installed in the Windows venv (`deploy/requirements-gpu.txt`), OpenPCDet at
    commit 233f849829b6ac19afb8af8837a0246890908755 with the project's
    `openpcdet_windows_fixes.patch` (portable C++ / CUDA fixes: `int64_t`, `std::vector`,
    `data_ptr`; optional imports), built with `setup.py develop` in `/app/OpenPCDet` for
    `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9`; runtime stage
    `nvidia/cuda:12.4.1-runtime-ubuntu22.04`. `opencv-python` is left out (decided with the user):
    5.0.0.93 needs numpy >= 2, numpy 1.26.4 is pinned (numpy 2 crashes spconv/cumm), and inference
    never imports cv2. The weights are mounted read-only at `/app/checkpoints` (where
    `detection/paths.py` expects them), never baked in.
  - `deploy/minio/Dockerfile`: the MinIO server built from its official source at
    `RELEASE.2025-10-15T17-29-55Z` (`golang:1.24.9-alpine3.22`, runtime `alpine:3.22.2`). The
    project publishes no pullable image any more (`docker pull minio/minio:<tag>`: "repository does
    not exist"; decided with the user).
  - `.dockerignore` is an allow-list: data, weights, results, venv, `.env` and keys never reach a
    build context.
- **Services** (`docker-compose.yml`): `nginx` (the only published ports), `api` (scalable:
  `--scale api=N`), `worker-cpu` (scalable), `worker-gpu` (profile `gpu`, one NVIDIA GPU
  reserved), `scheduler` (celery beat, exactly one), `postgres`, `redis` (password, no
  persistence), `minio`, and the one-shot `migrate` (`python -m infra.migrate` and
  `python -m infra.storage --ensure-bucket`), on which every app service waits
  (`service_completed_successfully`). Without `--profile gpu` the stack is cpu-only and
  `LAL_ASSISTED_MODE=false` refuses assisted openings (ADR 0020).
- **Healthchecks**: `api` runs `python -m infra.healthcheck http http://127.0.0.1:8000/readyz`
  (database, Redis, blob areas); `nginx` checks `/healthz` through itself (an internal listener on
  127.0.0.1:8081) and starts only after `api` is healthy; workers and the scheduler run
  `python -m infra.healthcheck worker` (database, Redis, blob store) because a solo-pool worker
  answers `celery inspect ping` only between jobs; postgres `pg_isready`, redis `PING`, minio
  `/minio/health/live`.
- **Secrets**: none in the compose file. `LAL_DB_PASSWORD`, `LAL_REDIS_PASSWORD`,
  `LAL_JWT_SECRET`, `LAL_MINIO_USER`, `LAL_MINIO_PASSWORD` use `${VAR:?message}`, so
  `docker compose` refuses every command while one is missing. `.env.example` documents every
  variable the compose file reads (tested); `.env` and `deploy/nginx/certs/*` are git-ignored.
- **nginx** (`deploy/nginx/templates/default.conf.template`, filled from `LAL_*` variables):
  - TLS 1.2 / 1.3 with the certificate and key from `LAL_TLS_DIR`
    (`scripts/make_dev_cert.py` makes a self-signed pair for localhost); port 80 only redirects.
  - Upload: `client_max_body_size ${LAL_UPLOAD_MAX_BYTES}`, the variable the API also reads (one
    value for both, tested), streamed (`proxy_request_buffering off`) with 300 s / 900 s timeouts;
    1 MiB and 60 s everywhere else; 300 s read timeout for export downloads.
  - `limit_req` on `/api/v1/auth/login`: `LAL_LOGIN_RATE` (10r/m) per client IP with
    `LAL_LOGIN_BURST` (5); above it 429 `rate_limited` with `Retry-After: 60`.
  - Headers on every answer: HSTS (`LAL_HSTS_MAX_AGE`; 0 on localhost), `nosniff`,
    `X-Frame-Options: DENY`, `Referrer-Policy: same-origin`, `Permissions-Policy`,
    `Cross-Origin-Opener-Policy` and `-Resource-Policy: same-origin`; `server_tokens off`. The
    pages' Content-Security-Policy is set by the API (`api/app.py: page_csp`): scripts only from
    the site plus the SHA-256 of each inline script of that page, computed from the file; no
    eval, `object-src 'none'`, `frame-ancestors 'none'`. The e2e and smoke tests fail on any
    console error, which includes CSP violations.
  - Errors of the gateway (413, 429, 502-504) use the API's error body (ADR 0018).
  - The API address is resolved per request through Docker's DNS, so API replicas started later
    are used too.
- **Smoke test**: `python scripts/smoke_stack.py` writes random secrets into a temporary `.env`,
  makes a certificate, starts `docker-compose.yml` plus the test-only
  `tests/smoke/docker-compose.smoke.yml` (a gpu-queue worker from the app image with the fake
  detector of `tests/fakes.py`), waits for health, creates the admin, runs `tests/smoke` through
  `https://localhost:8443` (gateway, a two-user browser scenario, worker-cpu scaled to 3, login rate
  limit) and removes everything (`down -v`). `--gpu` uses the real GPU worker instead; it is run
  only when the user says the GPU is free.

## Consequences
- The app uses the MinIO root credentials as its S3 keys; a separate least-privilege MinIO user
  would need the `mc` admin tool (not done).
- Redis gets its password on the command line of its own container (visible to `ps` inside it).
- Database and Redis passwords become parts of URLs: they must be URL-safe (`.env.example` says
  how to make them).
- Upgrading: `docker compose build`, then `docker compose run --rm migrate`, then
  `docker compose up -d` (README).
