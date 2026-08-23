# syntax=docker/dockerfile:1

# Rebuild the browser artifacts from the canonical C sources. The final image
# contains no compiler or build tooling.
FROM emscripten/emsdk:3.1.69 AS wasm-build

USER root
WORKDIR /src
COPY Makefile ./
COPY src ./src
COPY tests/test_wasm.mjs ./tests/test_wasm.mjs
COPY datasets/optdigits.tra datasets/optdigits.tes ./datasets/
COPY webapp/index.html webapp/styles.css webapp/app.js ./webapp/
RUN mkdir -p webapp/assets \
    && cp datasets/optdigits.tra webapp/assets/optdigits.tra \
    && cp datasets/optdigits.tes webapp/assets/optdigits.tes \
    && make NODE="$(command -v node)" web-check


# Nginx runs as an unprivileged user and listens above the privileged port
# range, which also lets the Compose service drop every Linux capability.
FROM nginxinc/nginx-unprivileged:alpine AS runtime

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=wasm-build --chown=101:101 /src/webapp/index.html /usr/share/nginx/html/index.html
COPY --from=wasm-build --chown=101:101 /src/webapp/styles.css /usr/share/nginx/html/styles.css
COPY --from=wasm-build --chown=101:101 /src/webapp/app.js /usr/share/nginx/html/app.js
COPY --from=wasm-build --chown=101:101 /src/webapp/enen.js /usr/share/nginx/html/enen.js
COPY --from=wasm-build --chown=101:101 /src/webapp/enen.wasm /usr/share/nginx/html/enen.wasm
COPY --from=wasm-build --chown=101:101 /src/webapp/assets/nn.c /usr/share/nginx/html/assets/nn.c
COPY --from=wasm-build --chown=101:101 /src/webapp/assets/optdigits.tra /usr/share/nginx/html/assets/optdigits.tra
COPY --from=wasm-build --chown=101:101 /src/webapp/assets/optdigits.tes /usr/share/nginx/html/assets/optdigits.tes

RUN nginx -t

USER 101:101

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
