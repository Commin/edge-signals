#!/usr/bin/env bash
# Container entrypoint: docker run [--gpus all] [-v DATA:/data:ro] [-v MODELS:/models:ro] [-v OUT:/outputs] IMAGE <command> [args]
# commands: fetch-assets | stream | collect | annotate | retrain | evaluate | register | check-alignment | synthetic-frames | prepare-data | test   (same as ./run.sh;
# `test` exists in the test image only)
set -euo pipefail
cd /app
usage() { sed -n '2,3p' "$0"; echo "Write outputs under /outputs (e.g. --out /outputs/run1)."; }
case "${1:-}" in
  test|stream|collect|annotate|retrain|evaluate|register|fetch-assets|check-alignment|prepare-data|synthetic-frames) exec ./run.sh "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
