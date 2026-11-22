FROM node:24-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=secret,id=build_ca \
    if [ -s /run/secrets/build_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/build_ca; fi; npm ci
COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts ./
COPY frontend/src/ src/
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git libgl1 libglib2.0-0 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=build_ca \
    if [ -s /run/secrets/build_ca ]; then export GIT_SSL_CAINFO=/run/secrets/build_ca; fi; \
    git clone https://github.com/facebookresearch/dinov2.git /opt/dinov2 \
    && git -C /opt/dinov2 checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
WORKDIR /app
COPY requirements-lock.txt requirements-api.txt requirements-space.txt pyproject.toml ./
COPY src/ src/
RUN --mount=type=secret,id=build_ca \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca GIT_SSL_CAINFO=/run/secrets/build_ca; fi; \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
      'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu' \
    && python -m pip install -c requirements-lock.txt . \
    && python -m pip install -r requirements-api.txt -r requirements-space.txt \
    && python -m pip check
COPY --from=frontend /build/frontend/dist /app/frontend/dist
COPY --chmod=0644 deploy/space-start.sh deploy/stage_models.py /app/deploy/
RUN chmod 0755 /app /app/deploy && useradd --create-home --uid 1000 user && chown -R user:user /opt/dinov2
USER user
EXPOSE 7860
CMD ["sh", "/app/deploy/space-start.sh"]
