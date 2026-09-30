# edge_signals - runtime image (default target) and test image (--target test). CPU or GPU (--gpus all).
# Contains NO weights (.pt), data, frames, labels or videos: fetch-assets downloads them into the /assets and /data volumes at run time.
#
# Environment = the YOLOv12 fork's own: Python 3.11 (conda, as its README), torch 2.2.2 / torchvision 0.17.2 from its requirements.txt (PyPI wheels, CUDA 12.1
# libraries bundled) and its flash-attn 2.7.3 wheel, which is built for CUDA 11 and needs libcudart.so.11.0: the CUDA 11.8 runtime base image provides it.
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base
LABEL org.opencontainers.image.source="https://github.com/commin/edge-signals" \
      org.opencontainers.image.title="edge_signals" \
      org.opencontainers.image.description="Label-free local signals, adaptation requests and retraining stages for a camera stream (no weights or data inside)" \
      org.opencontainers.image.licenses="GPL-3.0-only"

# Optional: --build-arg DNS=<address> sets the resolver for the build steps that need the network (only for networks where the Docker daemon's DNS is unreachable;
# then build with DOCKER_BUILDKIT=0)
ARG DNS=""
ARG MINIFORGE_VERSION=24.11.3-2
ARG MINIFORGE_SHA256=65af53dad30b3fcbd1cb1d4ad62fd3a86221464754844544558aae3a28795189
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN if [ -n "$DNS" ]; then echo "nameserver $DNS" > /etc/resolv.conf; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN if [ -n "$DNS" ]; then echo "nameserver $DNS" > /etc/resolv.conf; fi \
 && curl -fsSL -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
 && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum -c - \
 && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
 && /opt/conda/bin/conda create -y -n yolov12 python=3.11 && /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/envs/yolov12/bin:$PATH PYTHON=/opt/conda/envs/yolov12/bin/python

COPY docker/install_fork.sh docker/constraints.txt /opt/install/
RUN if [ -n "$DNS" ]; then echo "nameserver $DNS" > /etc/resolv.conf; fi \
 && bash /opt/install/install_fork.sh \
 && apt-get purge -y --auto-remove git curl && rm -rf /var/lib/apt/lists/* /opt/install /tmp/*

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /assets /data /outputs /home/app/.config/Ultralytics \
 && chown -R app:app /assets /data /outputs /home/app
ENV EDGE_ASSETS=/assets EDGE_DATA=/data PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 YOLO_OFFLINE=True HOME=/tmp YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/matplotlib NO_ALBUMENTATIONS_UPDATE=1

# ---- code (no weights, no data: see .dockerignore)
FROM base AS code
COPY . /app
RUN chmod +x /app/run.sh /app/docker/entrypoint.sh
WORKDIR /app

# ---- test image: the runtime image plus pytest (test-only package)
FROM code AS test
ARG DNS=""
USER root
COPY requirements-test.txt /tmp/requirements-test.txt
RUN if [ -n "$DNS" ]; then echo "nameserver $DNS" > /etc/resolv.conf; fi && python -m pip install -r /tmp/requirements-test.txt && rm /tmp/requirements-test.txt
USER app
VOLUME ["/assets", "/data", "/outputs"]
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["test"]

# ---- runtime image (default: the last stage)
FROM code AS runtime
USER app
VOLUME ["/assets", "/data", "/outputs"]
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["help"]
