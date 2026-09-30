#!/usr/bin/env bash
# run.sh - single entry point.
#   ./run.sh test                                   unit tests + the released package's reference test
#   ./run.sh fit-reference --list L --images-root R  fit the feature_drift reference (writes references/)
#   ./run.sh stream --list L --images-root R [--out DIR]   run the signal stage on a frame stream (default ./outputs/)
#   ./run.sh collect|annotate|retrain|evaluate|register --out RUN ...   adaptation stages (configs/adaptation.yaml)
#   ./run.sh fetch-assets [--url NAME=URL ...]              download weights / data / videos (SHA256-verified) into $EDGE_ASSETS / $EDGE_DATA
#   ./run.sh check-alignment --videos-dir D --frames-dir F   decoded video frames vs labelled images
#   ./run.sh compare-expected --mode images|video --out RUN  requests of a stream run vs configs/expected_results.yaml
#   ./run.sh prepare-data --config configs/data.yaml       download / lay out the dataset
#   ./run.sh synthetic-frames --out DIR                    tiny synthetic frames for a smoke test (no dataset needed)
# Environment: the YOLOv12 fork's own requirements at the pinned commit (docker/install_fork.sh, or the Docker image); PYTHON = the interpreter of that environment;
# optional YOLOV12_DIR = a local checkout of the fork instead of the pip-installed one; EDGE_ASSETS / EDGE_DATA = the asset volumes;
# EXTRA_SITE = extra site-packages dir appended AFTER the fork (optional).
set -euo pipefail
DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PYTHON:=python3}"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg YOLO_OFFLINE="${YOLO_OFFLINE:-True}"   # nothing is downloaded at run time
export EDGE_ASSETS="${EDGE_ASSETS:-$DEMO/assets}" EDGE_DATA="${EDGE_DATA:-$DEMO/data}"   # weights / data + videos volumes (/assets and /data in the container)
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${TMPDIR:-/tmp}/edge_signals_yolo_config}"   # ultralytics writes its settings here (never into the package)
export PYTHONPATH="${YOLOV12_DIR:+$YOLOV12_DIR:}$DEMO:$DEMO/third_party/consistency${EXTRA_SITE:+:$EXTRA_SITE}"
cd "$DEMO"
cmd="${1:-}"; [ -n "$cmd" ] && shift || true
case "$cmd" in
  test)          exec "$PYTHON" -m pytest -q -p no:cacheprovider tests third_party/consistency/tests "$@" ;;
  stream|fit-reference|collect|annotate|retrain|evaluate|register|fetch-assets)
                 exec "$PYTHON" -m edge_signals.cli "$cmd" "$@" ;;
  synthetic-frames) exec "$PYTHON" scripts/make_synthetic_frames.py "$@" ;;
  check-alignment) exec "$PYTHON" scripts/check_alignment.py "$@" ;;
  compare-expected) exec "$PYTHON" scripts/compare_expected.py "$@" ;;
  prepare-data)  exec "$PYTHON" scripts/prepare_data.py "$@" ;;
  *)             sed -n '2,16p' "$0"; exit 2 ;;
esac
