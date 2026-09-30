#!/usr/bin/env bash
# Install the YOLOv12 fork EXACTLY as its own requirements specify, at the pinned commit. Used by the Dockerfile; also usable on a host:
#     PYTHON=/path/to/python3.11 bash docker/install_fork.sh
# What the fork specifies (README + requirements.txt at the pinned commit): Python 3.11; pip install -r requirements.txt (torch 2.2.2, torchvision 0.17.2,
# a flash-attn 2.7.3 wheel built for CUDA 11 / torch 2.2, timm, albumentations, onnx*, gradio, supervision, ...); pip install the fork.
# Two mechanical adaptations, nothing else: (1) the fork's requirements.txt names the flash-attn wheel by file name only, so it is downloaded (SHA256-verified)
# and the line points at the local file; (2) the fork is installed with `pip install "ultralytics @ git+URL@COMMIT"` so that pip records the commit
# (the environment guard reads it). Optional: docker/constraints.txt pins the versions that the fork's own pyproject leaves open (matplotlib, pillow,
# requests, ...) to the versions this delivery was tested with; it adds no package.
set -euo pipefail
: "${FORK_URL:=https://github.com/sunsmarterjie/yolov12.git}"
: "${FORK_COMMIT:=2abab7153a065fb2925e8088e9ca2b19016ab7d6}"
: "${FLASH_WHEEL:=flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl}"
: "${FLASH_URL:=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl}"
: "${FLASH_SHA256:=9f28de4a263312786c4bc0444ecdf8397c24e46100fa24a9b92b036a62d1940d}"
: "${PYTHON:=python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$(mktemp -d)}"
git clone --quiet "$FORK_URL" "$WORK/src"
git -C "$WORK/src" checkout --quiet "$FORK_COMMIT"
[ "$(git -C "$WORK/src" rev-parse HEAD)" = "$FORK_COMMIT" ] || { echo "fork commit mismatch" >&2; exit 1; }
curl -fsSL -o "$WORK/$FLASH_WHEEL" "$FLASH_URL"
echo "$FLASH_SHA256  $WORK/$FLASH_WHEEL" | sha256sum -c -
sed "s#^flash_attn-.*#$WORK/$FLASH_WHEEL#" "$WORK/src/requirements.txt" > "$WORK/requirements.txt"
C=(); [ -f "$HERE/constraints.txt" ] && C=(-c "$HERE/constraints.txt")
"$PYTHON" -m pip install "${C[@]}" -r "$WORK/requirements.txt"
"$PYTHON" -m pip install "${C[@]}" "ultralytics @ git+${FORK_URL}@${FORK_COMMIT}"
"$PYTHON" -m pip check
