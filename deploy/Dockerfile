# The app image: API, cpu worker, scheduler and the one-shot migration (docker-compose.yml).
# No torch / OpenPCDet: the detector runs only in the worker-gpu image (deploy/Dockerfile.gpu).
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LAL_RESULTS_DIR=/tmp/lal-results \
    PYTHONPATH=/app

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin lal
WORKDIR /app
COPY deploy/requirements-app.txt /tmp/requirements-app.txt
RUN pip install -r /tmp/requirements-app.txt
COPY --chown=lal:lal . /app

USER lal
EXPOSE 8000
CMD ["uvicorn", "--factory", "api.app:app_factory", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
