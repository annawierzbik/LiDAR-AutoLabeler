# MinIO server built from its official source (docs/adr/0022): the project publishes no pullable
# image any more and its community repository is archived. Pinned to the last source release,
# which contains the fix for CVE-2025-62506 / GHSA-jjjj-jwhf-8rgr (PR #21642, merge commit
# c1a49490c78e9c3ebcad86ba0662319138ace190, one commit before this tag). The build fails if the
# tag does not point at the expected commit.
ARG MINIO_RELEASE=RELEASE.2025-10-15T17-29-55Z
ARG MINIO_COMMIT=9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a

FROM golang:1.24.9-alpine3.22 AS build
ARG MINIO_RELEASE
ARG MINIO_COMMIT
RUN apk add --no-cache git
RUN git clone --depth 1 --branch "${MINIO_RELEASE}" https://github.com/minio/minio.git /src \
 && test "$(git -C /src rev-parse HEAD)" = "${MINIO_COMMIT}"
WORKDIR /src
RUN CGO_ENABLED=0 go build -trimpath -tags kqueue -ldflags "-s -w" -o /out/minio .

FROM alpine:3.22.2
ARG MINIO_RELEASE
ARG MINIO_COMMIT
LABEL org.opencontainers.image.source="https://github.com/minio/minio" \
      org.opencontainers.image.version="${MINIO_RELEASE}" \
      org.opencontainers.image.revision="${MINIO_COMMIT}"
RUN apk add --no-cache ca-certificates curl \
 && adduser -D -H -u 1000 minio \
 && mkdir -p /data && chown minio:minio /data
COPY --from=build /out/minio /usr/bin/minio
USER minio
VOLUME ["/data"]
EXPOSE 9000 9001
ENTRYPOINT ["/usr/bin/minio"]
CMD ["server", "/data", "--console-address", ":9001"]
