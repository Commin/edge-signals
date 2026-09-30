#!/usr/bin/env bash
# run_all.sh - run edge_signals end to end with the published container images, in one command.
#
#   scripts/run_all.sh [--work-dir DIR] [--videos-dir DIR] [options]
#
# Steps, in order (the script stops at the first failure):
#   pull images -> fetch-assets -> SHA256 check + second fetch-assets (idempotent) -> test suite (test image)
#   -> stream drift_calib on labelled images -> stream drift_calib on raw video (needs --videos-dir)
#   -> compare the requests with configs/expected_results.yaml -> retraining chain on the first request
#
# Before every step the exact docker command is printed, so the output doubles as documentation. Output is shown live and also written to <work-dir>/logs/<step>.log.
# Needs on the host: bash, docker (with the NVIDIA Container Toolkit for a GPU), sed, grep, id. Nothing else is installed.
#
# Options:
#   --work-dir DIR        where assets, data, outputs and logs go (default ./edge-signals-run); re-running resumes (downloaded files are kept)
#   --tag TAG             image tag (default: the version of this release)
#   --repo NAME           image repository (default ghcr.io/commin/edge-signals)
#   --videos-dir DIR      the original SMD videos, any folder layout; without it the raw-video mode is skipped
#   --dns IP              DNS resolver for the fetch-assets container only (for hosts where container DNS does not work)
#   --network-host        run the fetch-assets container with --network host (alternative to --dns)
#   --gpus VALUE          docker --gpus value (default all; "none" = no GPU: the retraining chain is then skipped)
#   --no-pull             use the images that are already on this machine (no docker pull)
#   --local-assets DIR    use asset files you already downloaded instead of the URLs in configs/assets.yaml (repeatable). Files are matched by asset name:
#                         DIR/<name>.pt | DIR/weights/<name>.pt (weights), DIR/<name>.tar | DIR/data/<name>.tar (data archives, e.g. data_clear_val.tar).
#                         Assets not found in any DIR are still downloaded from their URL. SHA256 is verified either way.
#   --skip-tests          skip the test suite
#   --skip-retrain        skip the retraining chain
#   -h, --help            this text
#
# Every container runs as your user with --shm-size=2g and -e PYTHONUNBUFFERED=1; everything except fetch-assets runs with --network none.
# Exit status: 0 if no step failed (differences from the expected requests are only a WARNING), 1 otherwise.
set -uo pipefail

VERSION="v0.5.2"
REPO="ghcr.io/commin/edge-signals"
TAG="$VERSION"
WORK="./edge-signals-run"
VIDEOS=""
DNS=""
NETHOST=0
GPUS="all"
NOPULL=0
SKIP_TESTS=0
SKIP_RETRAIN=0
LOCAL_ASSETS=()

usage() { sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "run_all.sh: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --work-dir)      [ $# -ge 2 ] || die "--work-dir needs a value"; WORK="$2"; shift 2;;
    --tag)           [ $# -ge 2 ] || die "--tag needs a value"; TAG="$2"; shift 2;;
    --repo)          [ $# -ge 2 ] || die "--repo needs a value"; REPO="$2"; shift 2;;
    --videos-dir)    [ $# -ge 2 ] || die "--videos-dir needs a value"; VIDEOS="$2"; shift 2;;
    --dns)           [ $# -ge 2 ] || die "--dns needs a value"; DNS="$2"; shift 2;;
    --network-host)  NETHOST=1; shift;;
    --gpus)          [ $# -ge 2 ] || die "--gpus needs a value"; GPUS="$2"; shift 2;;
    --no-pull)       NOPULL=1; shift;;
    --local-assets)  [ $# -ge 2 ] || die "--local-assets needs a value"; LOCAL_ASSETS+=("$2"); shift 2;;
    --skip-tests)    SKIP_TESTS=1; shift;;
    --skip-retrain)  SKIP_RETRAIN=1; shift;;
    -h|--help)       usage; exit 0;;
    *)               echo "run_all.sh: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
[ -z "$DNS" ] || [ "$NETHOST" = "0" ] || die "use either --dns or --network-host, not both"
command -v docker >/dev/null 2>&1 || die "docker not found"
if [ -n "$VIDEOS" ]; then [ -d "$VIDEOS" ] || die "--videos-dir is not a directory: $VIDEOS"; VIDEOS="$(cd "$VIDEOS" && pwd)"; fi
for d in "${LOCAL_ASSETS[@]+"${LOCAL_ASSETS[@]}"}"; do [ -d "$d" ] || die "--local-assets is not a directory: $d"; done

mkdir -p "$WORK"/assets "$WORK"/data "$WORK"/outputs "$WORK"/logs || die "cannot create $WORK"
WORK="$(cd "$WORK" && pwd)"
IMG="$REPO:$TAG"
TIMG="$REPO:$TAG-test"
UIDGID="$(id -u):$(id -g)"
GPU_ARGS=()
[ "$GPUS" = "none" ] || GPU_ARGS=(--gpus "$GPUS")
NET_FETCH=()
[ -z "$DNS" ] || NET_FETCH=(--dns "$DNS")
[ "$NETHOST" = "0" ] || NET_FETCH=(--network host)

# ---------------------------------------------------------------- bookkeeping
T_START=$SECONDS
STEPS=()      # "name|STATUS|detail"
record() { STEPS+=("$1|$2|$3"); }

summary() {
  echo
  echo "================================================================ summary"
  local s name status detail
  for s in "${STEPS[@]+"${STEPS[@]}"}"; do
    IFS='|' read -r name status detail <<< "$s"
    printf '%-6s %-22s %s\n' "$status" "$name" "$detail"
  done
  echo "outputs : $WORK/outputs   (requests.jsonl, windows.jsonl, summary.json, resolution.json per run; adapt/ = retraining run)"
  echo "assets  : $WORK/assets (weights)   data: $WORK/data (labelled frames and labels)   logs: $WORK/logs"
  echo "total time: $((SECONDS - T_START)) s"
}
fail() {                     # fail NAME DETAIL: record, print the summary, stop
  record "$1" FAIL "$2"
  echo
  echo "run_all.sh: step '$1' FAILED: $2 (log: $WORK/logs/$1.log)" >&2
  summary
  exit 1
}

# print the command, run it, show the output live and keep it in the log (tee); sets RC
RC=0
step_run() {                 # step_run LOGNAME cmd...
  local name="$1"; shift
  local line
  line="$(printf '%q ' "$@")"
  echo
  echo "---------------------------------------------------------------- $name"
  echo "+ $line"
  { echo "+ $line"; } > "$WORK/logs/$name.log"
  "$@" 2>&1 | tee -a "$WORK/logs/$name.log"
  RC=${PIPESTATUS[0]}
}

# every container: your uid, --shm-size=2g, unbuffered python (the printed commands show exactly this prefix)
DK=(docker run --rm --user "$UIDGID" --shm-size=2g -e PYTHONUNBUFFERED=1)
OFFLINE=(--network none)
VOL_ASSETS=(-v "$WORK/assets:/assets")
VOL_DATA=(-v "$WORK/data:/data")
VOL_OUT=(-v "$WORK/outputs:/outputs")
VOL_ASSETS_RO=(-v "$WORK/assets:/assets:ro")
VOL_DATA_RO=(-v "$WORK/data:/data:ro")
VOL_OUT_RO=(-v "$WORK/outputs:/outputs:ro")

echo "edge_signals run_all.sh ($VERSION): image $IMG"
echo "work dir $WORK | videos: ${VIDEOS:-none (raw-video mode will be skipped)} | gpus: $GPUS"

# ---------------------------------------------------------------- 1. pull
if [ "$NOPULL" = "1" ]; then
  echo; echo "---------------------------------------------------------------- pull (skipped: --no-pull)"
  docker image inspect "$IMG" >/dev/null 2>&1 || fail pull "image $IMG is not on this machine (drop --no-pull to pull it)"
  if [ "$SKIP_TESTS" = "0" ]; then docker image inspect "$TIMG" >/dev/null 2>&1 || fail pull "image $TIMG is not on this machine"; fi
  record pull SKIP "--no-pull: using the local images"
else
  step_run pull docker pull "$IMG"
  [ "$RC" = "0" ] || fail pull "cannot pull $IMG (wrong --tag? no network? for a private registry log in first)"
  if [ "$SKIP_TESTS" = "0" ]; then
    step_run pull_test docker pull "$TIMG"
    [ "$RC" = "0" ] || fail pull_test "cannot pull $TIMG"
  fi
  record pull PASS "$IMG$([ "$SKIP_TESTS" = "1" ] || echo " and $TIMG")"
fi

# ---------------------------------------------------------------- 2. fetch-assets
FETCH_CFG=()
FETCH_MOUNT=()
if [ "${#LOCAL_ASSETS[@]}" -gt 0 ]; then
  # write a config that points to your local files (the container's python fills in file:// urls for the assets it finds)
  LA_MOUNT=()
  i=0
  for d in "${LOCAL_ASSETS[@]}"; do LA_MOUNT+=(-v "$(cd "$d" && pwd):/local_assets/$i:ro"); i=$((i + 1)); done
  step_run local_assets_config "${DK[@]}" "${OFFLINE[@]}" -i "${LA_MOUNT[@]}" -v "$WORK:/work" --entrypoint python "$IMG" - "$i" <<'PY'
import os, sys, yaml
n = int(sys.argv[1])
cfg = yaml.safe_load(open("/app/configs/assets.yaml"))
used = []
for a in cfg["assets"]:
    ext = ".pt" if a["kind"] == "weights" else ".tar"
    sub = "weights" if a["kind"] == "weights" else "data"
    for k in range(n):
        for rel in (f"{a['name']}{ext}", f"{sub}/{a['name']}{ext}"):
            p = f"/local_assets/{k}/{rel}"
            if os.path.isfile(p):
                a["url"] = "file://" + p
                used.append(a["name"])
                break
        if a["name"] in used:
            break
yaml.safe_dump(cfg, open("/work/assets_local.yaml", "w"), sort_keys=False, width=220)
print("local files used for:", ", ".join(used) or "none", "| downloaded from their URL:", ", ".join(a["name"] for a in cfg["assets"] if a["name"] not in used) or "none")
PY
  [ "$RC" = "0" ] || fail local_assets_config "cannot build the config for --local-assets"
  FETCH_CFG=(--assets-config /work/assets_local.yaml)
  FETCH_MOUNT=("${LA_MOUNT[@]}" -v "$WORK/assets_local.yaml:/work/assets_local.yaml:ro")
fi

step_run fetch_assets "${DK[@]}" "${NET_FETCH[@]+"${NET_FETCH[@]}"}" "${FETCH_MOUNT[@]+"${FETCH_MOUNT[@]}"}" "${VOL_ASSETS[@]}" "${VOL_DATA[@]}" "$IMG" fetch-assets "${FETCH_CFG[@]+"${FETCH_CFG[@]}"}"
[ "$RC" = "0" ] || fail fetch_assets "fetch-assets failed. Network trouble inside containers: retry with --network-host or --dns <resolver>; a Google Drive quota / permission page is reported with the file id"
record fetch_assets PASS "weights and labelled frames downloaded and SHA256-verified"

step_run fetch_assets_again "${DK[@]}" "${OFFLINE[@]}" "${FETCH_MOUNT[@]+"${FETCH_MOUNT[@]}"}" "${VOL_ASSETS[@]}" "${VOL_DATA[@]}" "$IMG" fetch-assets "${FETCH_CFG[@]+"${FETCH_CFG[@]}"}"
if [ "$RC" = "0" ] && ! grep -Eq '"status": "(ok|failed|skipped)"' "$WORK/logs/fetch_assets_again.log" && grep -q '"up_to_date"' "$WORK/logs/fetch_assets_again.log"; then
  record fetch_assets_again PASS "second run: every asset up_to_date, nothing downloaded"
else
  fail fetch_assets_again "the second fetch-assets run is not idempotent"
fi

step_run verify_sha256 "${DK[@]}" "${OFFLINE[@]}" -i --entrypoint python "${VOL_ASSETS_RO[@]}" "$IMG" - <<'PY'
import hashlib, json, sys, yaml
cfg = yaml.safe_load(open("/app/configs/assets.yaml"))["assets"]
rec = json.load(open("/assets/.fetched.json"))
bad = []
for a in cfg:
    n = a["name"]
    if rec.get(n, {}).get("sha256") != a["sha256"]:
        bad.append((n, "recorded SHA256 differs", rec.get(n)))
    if a["kind"] == "weights":
        h = hashlib.sha256(open(f"/assets/weights/{n}.pt", "rb").read()).hexdigest()
        if h != a["sha256"]:
            bad.append((n, "file SHA256 differs", h))
print(f"{len(cfg)} assets checked against configs/assets.yaml; mismatches:", bad or "none")
sys.exit(1 if bad else 0)
PY
[ "$RC" = "0" ] || fail verify_sha256 "a downloaded file does not match the SHA256 in configs/assets.yaml"
record verify_sha256 PASS "every asset matches configs/assets.yaml (weights re-hashed)"

# ---------------------------------------------------------------- 3. tests
if [ "$SKIP_TESTS" = "1" ]; then
  record tests SKIP "--skip-tests"
else
  step_run tests "${DK[@]}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" "${OFFLINE[@]}" "${VOL_ASSETS_RO[@]}" "${VOL_DATA_RO[@]}" "$TIMG" test
  [ "$RC" = "0" ] || fail tests "the test suite failed"
  record tests PASS "$(grep -E '^[0-9]+ passed' "$WORK/logs/tests.log" | tail -n 1)"
fi

# ---------------------------------------------------------------- 4. stream: labelled images
step_run stream_images "${DK[@]}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" "${OFFLINE[@]}" "${VOL_ASSETS_RO[@]}" "${VOL_DATA_RO[@]}" "${VOL_OUT[@]}" "$IMG" \
  stream --images-dir /data/frames --stream drift_calib --out /outputs/drift_calib_images
[ "$RC" = "0" ] || fail stream_images "stream on labelled images failed"
record stream_images PASS "outputs/drift_calib_images"

# ---------------------------------------------------------------- 5. stream: raw video
RAN_VIDEO=0
if [ -z "$VIDEOS" ]; then
  echo; echo "---------------------------------------------------------------- stream_video (skipped)"
  echo "NOTICE: no --videos-dir: the raw-video mode is skipped. Get the original SMD videos from the dataset page (see NOTICE / README) and pass their folder with --videos-dir."
  record stream_video SKIP "no --videos-dir"
else
  step_run stream_video "${DK[@]}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" "${OFFLINE[@]}" "${VOL_ASSETS_RO[@]}" "${VOL_DATA_RO[@]}" -v "$VIDEOS:/videos:ro" "${VOL_OUT[@]}" "$IMG" \
    stream --videos-dir /videos --stream drift_calib --out /outputs/drift_calib_video
  [ "$RC" = "0" ] || fail stream_video "stream on raw video failed (see resolution.json in the output for videos that were not found)"
  RAN_VIDEO=1
  record stream_video PASS "outputs/drift_calib_video"
fi

# ---------------------------------------------------------------- 6. compare with the expected requests (a difference is a WARNING)
compare() {                  # compare MODE OUTDIR
  step_run "compare_$1" "${DK[@]}" "${OFFLINE[@]}" "${VOL_OUT_RO[@]}" "$IMG" compare-expected --mode "$1" --out "/outputs/$2"
  case "$RC" in
    0) record "compare_$1" PASS "requests are exactly the expected ones (configs/expected_results.yaml)";;
    1) if grep -q '^\[compare-expected\] WARNING' "$WORK/logs/compare_$1.log"; then
         record "compare_$1" WARN "differs from configs/expected_results.yaml (windows near the threshold can flip across input modes and GPUs; see the log)"
       else
         fail "compare_$1" "compare-expected failed unexpectedly"
       fi;;
    *) fail "compare_$1" "cannot compare with configs/expected_results.yaml";;
  esac
}
compare images drift_calib_images
if [ "$RAN_VIDEO" = "1" ]; then compare video drift_calib_video; else record compare_video SKIP "no raw-video run"; fi

# ---------------------------------------------------------------- 7. retraining chain on the first request
if [ "$SKIP_RETRAIN" = "1" ]; then
  record retrain_chain SKIP "--skip-retrain"
elif [ "$GPUS" = "none" ]; then
  record retrain_chain SKIP "retraining needs a GPU (--gpus none)"
else
  if [ "$RAN_VIDEO" = "1" ]; then SRC="drift_calib_video"; else SRC="drift_calib_images"; fi
  RID="$(sed -n '1s/.*"request_id": "\([^"]*\)".*/\1/p' "$WORK/outputs/$SRC/requests.jsonl" 2>/dev/null)"
  if [ -z "$RID" ]; then
    record retrain_chain SKIP "no request in $SRC"
  else
    O="/outputs/adapt/$RID"
    rm -rf "${WORK:?}/outputs/adapt/$RID"
    RUN=("${DK[@]}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" "${OFFLINE[@]}" "${VOL_ASSETS_RO[@]}" "${VOL_DATA_RO[@]}" "${VOL_OUT[@]}")
    EVAL_IN_STREAM=()
    if [ "$RAN_VIDEO" = "1" ]; then
      # same-video retraining: the labelled frames of the video(s) seen from the stream start up to the request (video runs carry the original frame index)
      COLLECT=(collect --out "$O" --stream-out "/outputs/$SRC" --request-id "$RID" --images-root /data/frames)
      EVAL_IN_STREAM=(--in-stream "/outputs/$SRC" --request-id "$RID")
    else
      # image run: the frame list the stream ran on defines the stream time
      step_run stream_list "${RUN[@]}" --entrypoint python "$IMG" -c \
        "import json; print(''.join(json.loads(l)['frame_id'] + '\n' for l in open('/outputs/$SRC/frames.jsonl')), end='', file=open('/outputs/$SRC/stream_list.txt', 'w'))"
      [ "$RC" = "0" ] || fail stream_list "cannot write the stream list of the image run"
      COLLECT=(collect --out "$O" --requests "/outputs/$SRC/requests.jsonl" --request-id "$RID" --stream-list "/outputs/$SRC/stream_list.txt" --images-root /data/frames)
    fi
    echo; echo "retraining chain on request $RID of $SRC (default recipe: provided labels = simulated annotation, replay 200 %)"
    step_run chain_collect  "${RUN[@]}" "$IMG" "${COLLECT[@]}";                                                                         [ "$RC" = "0" ] || fail chain_collect  "collect failed (exit $RC; 3 = insufficient data: fewer new frames than the minimum)"
    step_run chain_annotate "${RUN[@]}" "$IMG" annotate --out "$O" --images-root /data/frames;                                         [ "$RC" = "0" ] || fail chain_annotate "annotate failed"
    step_run chain_retrain  "${RUN[@]}" "$IMG" retrain --out "$O" --images-root /data/frames --seed 0;                                 [ "$RC" = "0" ] || fail chain_retrain  "retrain failed (needs a GPU and --shm-size=2g)"
    step_run chain_register "${RUN[@]}" "$IMG" register --out "$O" --parent maritime_s_base.v0 --request-id "$RID";                    [ "$RC" = "0" ] || fail chain_register "register failed"
    step_run chain_evaluate "${RUN[@]}" "$IMG" evaluate --out "$O" --weights "$O/models/maritime_s_base.v1.pt" --images-root /data/frames --labels-dir /data/labels \
      --eval nir_eval=/data/lists/nir_eval.txt "${EVAL_IN_STREAM[@]+"${EVAL_IN_STREAM[@]}"}" --tag v1;                                    [ "$RC" = "0" ] || fail chain_evaluate "evaluate failed"
    record retrain_chain PASS "request $RID: collect, annotate, retrain, register, evaluate (outputs/adapt/$RID)"
  fi
fi

summary
if printf '%s\n' "${STEPS[@]}" | grep -q '|FAIL|'; then exit 1; fi
echo "run_all.sh: finished (no step failed)."
exit 0
