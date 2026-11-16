# MinIO server built from its official source: the project no longer publishes images that can be
# pulled (docker pull minio/minio answers "repository does not exist"), see docs/adr/0019.
ARG MINIO_RELEASE=RELEASE.2025-10-15T17-29-55Z

FROM golang:1.24.9-alpine3.22 AS build
ARG MINIO_RELEASE
RUN apk add --no-cache git
RUN git clone --depth 1 --branch "${MINIO_RELEASE}" https://github.com/minio/minio.git /src
WORKDIR /src
RUN CGO_ENABLED=0 go build -trimpath -tags kqueue -ldflags "-s -w" -o /out/minio .

FROM alpine:3.22.2
ARG MINIO_RELEASE
LABEL org.opencontainers.image.source="https://github.com/minio/minio" \
      org.opencontainers.image.version="${MINIO_RELEASE}"
RUN apk add --no-cache ca-certificates curl \
 && adduser -D -H -u 1000 minio \
 && mkdir -p /data && chown minio:minio /data
COPY --from=build /out/minio /usr/bin/minio
USER minio
VOLUME ["/data"]
EXPOSE 9000 9001
ENTRYPOINT ["/usr/bin/minio"]
CMD ["server", "/data", "--console-address", ":9001"]
