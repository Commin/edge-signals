# edge_signals - label-free local signals and a drift trigger for a camera stream

This package watches the output of a ship-camera object detector and reports how healthy it looks, **without any ground-truth labels**.
It computes a small set of *signals* per frame or per pair of frames, summarises them over windows, compares each window with a
reference period, and can emit an *adaptation request* when a chosen signal gets worse than usual. A retraining stage can then adapt the
detector on the labelled frames of the same video.

**Nothing heavy is in this repository or in the container image**: no model weights (`.pt`), no dataset, frames, labels or videos. The **weights and the labelled frames and labels are
downloaded from Google Drive** at run time by `fetch-assets` (SHA256-verified, links in `configs/assets.yaml`) into the `/assets` and `/data` volumes; `yolov12n.pt` comes from the official YOLOv12
release. After that every command runs offline (`--network none`). The original videos are not distributed: see "Videos" below.

Licence: GPL-3.0-only (`LICENSE`). Third-party components and their licences: `NOTICE` (the YOLOv12 fork installed in the image is AGPL-3.0, unmodified).

**Dataset.** The frames, labels and videos derive from the Singapore Maritime Dataset (SMD): https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset. If you use them, cite:
D. K. Prasad, D. Rajan, L. Rachmawati, E. Rajabally, and C. Quek, "Video Processing From Electro-Optical Sensors for Object Detection and Tracking in a Maritime Environment: A Survey," *IEEE Transactions on Intelligent Transportation Systems*, vol. 18, no. 8, pp. 1993–2016, 2017, doi: 10.1109/TITS.2016.2634580.

* `edge_signals/` - the framework (plugin interface, registry, windows, trigger, engine, video input, inference with feature capture, adaptation stages, `assets.py`).
* `third_party/consistency/` - the released consistency package, vendored unchanged (see `VENDORED.txt` for the commit; `FILES.sha256`).
* `configs/` - all parameters (YAML), including `assets.yaml`. `schemas/local_signal_event.json` - the log format. `references/` - the small derived reference bundles (`.npz`).
* `data/mapping/frame_map.csv` - metadata only: (video, original frame index) <-> labelled image name.
* `docs/HANDOFF.md` - step-by-step handoff for the integration (commands verified in fresh containers).
* `docker/`, `Dockerfile` - the environment of the YOLOv12 fork, exactly as the fork specifies. `tests/` - unit tests on tiny synthetic fixtures.

## Requirements

x86-64 Linux host with Docker, an NVIDIA GPU and the NVIDIA Container Toolkit (CPU works for tests and replay, retraining needs a GPU). About 11 GB for the image, 1.1 GB for assets and labelled frames, 5.4 GB for the original
videos. Only `fetch-assets` (and the first `docker pull`) needs the network: **if container networking is restricted or the container cannot resolve names, run `fetch-assets` with `--network host` or with
`--dns <resolver>`**; every other command runs with `--network none`. `fetch-assets` prints a progress line at least every 10 s per file, times out after 15 s (connect) / 60 s (no data), and stops with the file id if Google
Drive answers with a web page (quota, permission) instead of the file.

## Quick start

### Run everything with one script

`scripts/run_all.sh` pulls the images, downloads and verifies the assets, runs the test suite, replays the stream `drift_calib` on the labelled images and (if you give it the original videos) on raw video,
compares the requests with `configs/expected_results.yaml`, and runs the retraining chain on the first request. It prints the exact docker command before every step, shows the output live, keeps it in `<work-dir>/logs/`, and ends
with a PASS / WARN / SKIP / FAIL line per step. It needs only bash and docker on the host.

```bash
git clone https://github.com/commin/edge-signals && cd edge-signals          # or just download scripts/run_all.sh
scripts/run_all.sh --work-dir ./edge-signals-run --videos-dir /path/to/original/videos
```

Useful options: `--tag v0.5.2` (image version; see below), `--dns <resolver>` or `--network-host` (only the `fetch-assets` container; for hosts where container networking is restricted),
`--gpus <value>`, `--skip-tests`, `--skip-retrain`, `--no-pull`, `--local-assets DIR` (use asset files you already downloaded instead of the download links). Without `--videos-dir` the raw-video mode is skipped with a notice.
A difference from the expected requests is only a warning: a window whose drop sits right at the threshold can flip between input modes and between GPUs.

**Image tags.** `:latest` (and `:latest-test`) always point to the newest release. For experiments pin a version tag (`:v0.5.2`), so that the code, the expected results and the assets stay together.

### Run step by step

```bash
# 1. pull the image (runtime and test targets; no weights or data inside) and prepare three host directories
docker pull ghcr.io/commin/edge-signals:v0.5.2
mkdir -p assets data outputs && chmod 777 outputs            # the container runs as uid 10001
RUN="docker run --rm --gpus all --shm-size=2g -v $PWD/assets:/assets -v $PWD/data:/data -v $PWD/outputs:/outputs ghcr.io/commin/edge-signals:v0.5.2"

# 2. download weights and labelled frames (needs network; resumable and idempotent; SHA256-verified). The VIDEOS are not downloaded: see "Videos" below
$RUN fetch-assets                                            # or: --url NAME=URL to change a link, --assets-config /assets/my_assets.yaml, --only NAME
                                                             # restricted container network: docker run --network host ... fetch-assets   (or --dns <resolver>)
# 3. from here on no network is needed: add --network none to every command below

# 4. tests (the test image adds pytest; it has no weights either, so the tests read /assets)
docker run --rm --gpus all --shm-size=2g --network none -v $PWD/assets:/assets ghcr.io/commin/edge-signals:v0.5.2-test test

# 5. stream on the original videos (your own copy, mounted read-only anywhere): camera replay -> inference -> signals -> adaptation requests
#    which videos, in which order: configs/replay_order.yaml (default stream drift_calib; or --stream NAME / --split NAME)
docker run --rm --gpus all --shm-size=2g --network none --user "$(id -u):$(id -g)" -v $PWD/assets:/assets:ro -v $PWD/data:/data:ro -v $PWD/outputs:/outputs \
     -v /path/to/my/videos:/videos:ro ghcr.io/commin/edge-signals:v0.5.2 stream --videos-dir /videos --stream drift_calib --out /outputs/stream
# 6. retrain on the labelled frames of the SAME video from before a request, then evaluate and register
$RUN --network none collect  --out /outputs/run1 --stream-out /outputs/stream --request-id <request id> --images-root /data/frames
$RUN --network none annotate --out /outputs/run1 --images-root /data/frames
$RUN --network none retrain  --out /outputs/run1 --images-root /data/frames --seed 0
$RUN --network none register --out /outputs/run1 --parent maritime_s_base.v0 --request-id <request id>
$RUN --network none evaluate --out /outputs/run1 --weights /outputs/run1/models/maritime_s_base.v1.pt --images-root /data/frames --labels-dir /data/labels \
     --eval nir_eval=/data/lists/nir_eval.txt --in-stream /outputs/stream --request-id <request id> --tag v1
```

The single class of the detector is called `vessel`. **It merges all original SMD object classes**: Boat, Ferry, Kayak, Sail boat, Speed boat, Vessel-ship *and also* Buoy, Flying bird-plane and Other,
so it is a "maritime object" class rather than a ship class; the name is kept for compatibility with the released weights.

The three volumes: `/assets` (weights: `maritime_s_base`, `maritime_x_base` = the teacher of the pseudo-label recipe, `yolov12n` = the trainer's AMP self-check file), `/data` (labelled
frames, labels and frame lists per split, `videos/`), `/outputs` (everything the stages write). Always pass `--gpus all --shm-size=2g` for `retrain` (DataLoader workers need shared memory);
`stream` and `evaluate` also use the GPU (CPU works, slowly: the device is `auto`). The host directory mounted at `/outputs` must be writable by uid 10001.

Commands (same as `./run.sh` outside a container): `fetch-assets`, `stream`, `collect`, `annotate`, `retrain`, `evaluate`, `register`, `check-alignment`, `synthetic-frames`,
`fit-reference`, `prepare-data`, `test` (test image only).

## Environment

The environment is **the YOLOv12 fork's own**, at the pinned commit `2abab7153a065fb2925e8088e9ca2b19016ab7d6`, installed by `docker/install_fork.sh` exactly as the fork specifies:
Python 3.11 (conda, as its README), `pip install -r requirements.txt` of the fork (torch 2.2.2, torchvision 0.17.2, flash-attn 2.7.3, timm, albumentations 2.0.4, onnx, gradio, supervision, ...),
then the fork itself. This package adds **no runtime dependency** beyond that (it imports numpy, scipy, PyYAML, requests, OpenCV, PyTorch and the fork itself; all are in the fork's requirements
or `pyproject.toml`). Two mechanical adaptations are documented in the script: the fork's `requirements.txt` names its flash-attn wheel by file name only (the wheel is downloaded and
SHA256-checked), and the fork is installed from its git URL so that pip records the commit, which the environment guard checks. `docker/constraints.txt` pins the versions that the fork's own
`pyproject.toml` leaves open to the versions this delivery was tested with (it adds no package).

The flash-attn wheel is built for CUDA 11 while torch 2.2.2 is the PyPI wheel (its CUDA 12.1 libraries are bundled); the wheel needs `libcudart.so.11.0`, which the CUDA 11.8 runtime base image
provides, so flash-attn is active on Ampere GPUs as the fork intends. Without it the fork falls back to `scaled_dot_product_attention` (boxes and features change in the last digits). The test
image (`--target test`) adds only `pytest`.

## The signals

| signal | family | granularity | needs | what it measures | higher is |
|---|---|---|---|---|---|
| `consistency` | quality | pair | predictions | how well the boxes of two consecutive frames agree after the global camera shift is removed (released package, locked cell `Gmc-full-1ch-incl`). Value in [0, 1]; also logged per pair: `alarm`, `status`, `G`, `G_mc`, `dx`, `dy`, `K`, `N_t`, `N_t1`. `EMPTY_PAIR` (no boxes in both frames) is *invalid*; `NO_MATCH` / `ONE_SIDED` count as consistency 0 | healthier |
| `confidence` | quality | frame | predictions | mean confidence of the kept boxes (invalid if a frame has no boxes) | healthier |
| `platform_motion` | context | pair | predictions | size of the global image shift, from the consistency output (invalid when no boxes were matched) | more disturbance |
| `feature_drift` | drift | frame | backbone features | Mahalanobis distance of the frame's pooled backbone feature to the training data (see below) | further from training |

A signal is one module in `edge_signals/builtin/` that declares `name`, `family` (quality | drift | context), `granularity` (frame | pair | window),
`requires` (preds | features | image), `higher_is_healthier`, `label_free`, implements `update(record) -> SignalValue` and optionally
`fit_reference(records) -> reference bundle`. A `SignalValue` carries `signal, version, value, valid, invalid_reason, t, device_id,
model_version, reference_id`. **An invalid value is never replaced by a number**: `value` is `null`, `valid` is `false`, and the reason is logged.
To add a signal, write a module, decorate the class with `@register`, import it in `edge_signals/builtin/__init__.py` and add it to `configs/signals.yaml`.

### feature_drift
The edge model's last backbone layer (index and per-frame output shape are stored in the reference metadata) is hooked in the **same forward
pass** that produces the detections (no second pass) and globally average-pooled to one vector per frame. A PCA (32 dimensions, configurable) and a
Ledoit-Wolf shrunk covariance are fitted on the model's training frames (the `clear_train` and `clear_val` frame lists of the data assets); the per-frame value is the
Mahalanobis distance in PCA space. The fitted **reference bundle** is a file in `references/` whose name carries its content hash (`reference_id`).
The backbone is frozen during adaptation, so the reference applies to every model version with the same learnable backbone weights
(`backbone_params_id`); a version with a different backbone gets *invalid* values (`backbone_mismatch`). Caveat: fine-tuning still updates the
BatchNorm running statistics, so a new version has the same `backbone_params_id` but a different `backbone_state_id` and its features are shifted slightly.
This is logged per value (`bn_stats_match`); set `params.backbone_match: state` to make the check strict. The reference frames are in-sample for the model.

## Two switches per signal, and the window statistic (`configs/signals.yaml`)

```yaml
signals:
  consistency:
    measure: true      # switch 1: compute and log it
    trigger: true      # switch 2: take part in the trigger decision (requires measure: true; checked when the file is loaded)
    window_transitions: 30   # or window_seconds: 5 (converted with the stream frame rate)
    aggregator: median       # mean | median | worst10 (mean of the worst 10 % of the window's valid values)
    comparator: previous     # previous | anchor | rolling | reference
    rolling_windows: 3       # only for comparator: rolling
    change: difference       # difference | ratio
```

* A **window** holds `window_transitions` consecutive frame pairs (frame signals use the same frames). Windows never cross a video / clip change in
  replay (an incomplete window at the end of a clip is dropped); the **comparison state does cross it** (one continuous stream), and every scene cut is
  logged as a context event (`context.jsonl`). No frame pair is formed across a cut.
* The **statistic** of a window is the aggregate of its *valid* values; a window with fewer than `min_valid_fraction` valid values is itself invalid.
* The **comparator** picks the reference period: `previous` = the window before, `anchor` = the first valid window after the start or after the rule's last fire,
  `rolling` = the mean of the last `rolling_windows` windows, `reference` = the level under the training reference (drift signals; an absolute distance).
* The **change** is reported as a degradation, positive = worse than the reference: `difference` = reference - statistic for a "higher is healthier"
  signal (statistic - reference otherwise); `ratio` = 1 - statistic / reference (statistic / reference - 1 otherwise).

## Adaptation stages (`configs/adaptation.yaml`)

Five commands turn an adaptation request into a registered model version; each is config-driven and writes **only** to its `--out` run directory (`collect/`, `annotate/`, `retrain/`,
`evaluate/`, `models/`, `registry.json`). Values can be overridden with `--set section.key=value`.

| command | what it does |
|---|---|
| `collect` | a request of a **video stream run** (`--stream-out DIR --request-id ID [--since-request-id ID0]`, mapping `data/mapping/frame_map.csv`) -> the labelled frames of the SAME video(s) the stream saw from the last reset up to the request's frame (a concatenated stream contributes each video's segment of that period), plus replay frames at `collect.replay.ratio` x the new frames. Fewer than `collect.min_new_frames` (60) new labelled frames: the request is logged as `insufficient_data` (exit code 3), no annotation, no retraining. A frame list (`--frames-list`) or an image-stream request (`--requests --stream-list`) also work |
| `annotate` | `provided` (default) = existing labels (simulated annotation; optional injected delay, logged, nothing sleeps) or `pseudo` = predictions of a teacher model at `annotate.pseudo.conf` (weights path configurable, mounted at `/models`, **not shipped**). Replay frames keep their existing labels |
| `retrain` | fine-tunes the active model (`retrain.base`) on adaptation + replay frames. The recipe is YAML: `freeze` (9 = backbone), `epochs`, `mosaic`, **`close_mosaic` (must be set explicitly; 0 = mosaic stays on for the whole retrain)**, photometric augmentation (R3), and the albumentations list. The fork's hidden Blur / MedianBlur / ToGray / CLAHE are replaced by the configured list and **verified off** from the dataset's actual transform pipeline (in the log and in `retrain.json`; the stage refuses to train otherwise). Logs wall-clock per phase and peak GPU memory |
| `evaluate` | offline mAP on labelled lists (`--eval NAME=LIST --labels-dir DIR`; **label-required**, marked as such) and the **label-free** consistency proxy on the most recent window. Two views: **(b) held-out splits** (`--eval`; the videos of the collected window are excluded automatically) and **(a) in-stream** (`--in-stream STREAM_OUT --request-id ID`): the later frames of the request's video after the request - expected to be optimistic, because they come from the video that was trained on |
| `register` | appends `registry.json`: version (`<family>.vK`), parent, request id, recipe (and its SHA256), weights SHA256, time, reference ids and `bn_stats_match` |

```bash
R=/outputs/run1
./run.sh collect  --out $R --requests /outputs/stream/requests.jsonl --stream-list frames.txt --images-root /data/frames
./run.sh annotate --out $R --images-root /data/frames                        # default: provided labels (annotate.provided.labels_dir, default /data/labels); --mode pseudo = teacher labels
./run.sh retrain  --out $R --images-root /data/frames --seed 0
./run.sh register --out $R --parent maritime_s_base.v0 --request-id <request id>
./run.sh evaluate --out $R --weights $R/models/maritime_s_base.v1.pt --eval nir_eval=/data/lists/nir_eval.txt --labels-dir /data/labels --images-root /data/frames --tag v1
```

**Default recipe (a demonstration setting): `provided200`.** `configs/adaptation.yaml` ships the ground-truth labels of the collected frames (`annotate.mode: provided` = *simulated annotation*:
in a deployment a person or a service would label them) and replay of 200 % of the adaptation frame count, 10 epochs, backbone frozen (`freeze: 9`), photometric augmentation, mosaic on for the whole
retrain (`close_mosaic: 0`). `pseudo` (labels = predictions of the teacher `maritime_x_base` at confidence 0.5) stays an option: `annotate --mode pseudo`.

*Why provided, not pseudo.* In same-video retraining the teacher can be **weaker than the student on the current scene**. Example (request at `MVI_1526_NIR` frame 450): on the collected frames the teacher's
labels had recall 0.69 (precision 0.77); the student retrained on them found 41 of the 87 objects of the later frames of that video, against 86 with the provided labels and 65 for the base model
(mAP50 0.21 / 0.98 / 0.65). The earlier default (pseudo200) was chosen in a different setting (adaptation set `nir_adapt`, evaluated on `nir_eval`, 3 seeds; gain about +0.016 mAP50 on the new domain,
small loss on clear footage) and does not carry over to retraining on the video that has just been seen.
**Failure condition of pseudo-labels:** they help only when the teacher is stronger than the student on the *current* data. Check the pseudo-label precision / recall that `annotate` reports
(`--check-labels-dir`, needs ground truth) and do not use the mode when the teacher is weak on the scene.

**Held-out means held out.** `evaluate` reads the run's `collect/` and removes every video of the request's collected window (and of its replay frames) from every `--eval` split; the excluded videos are printed
and stored per split in the result (`excluded_collected_videos`, `excluded_videos`, `n_frames_before_exclusion`); a split that is left empty is reported as skipped. `--include-collected` switches this off.

**Switch path: not implemented** (`edge_signals/adapt/switch.py` is a documented stub). Activating a registered version is done by pointing `model.weights` and `model.version` in
`configs/inference.yaml` at its registry entry; deciding *when* and *which* version to switch to is not part of this delivery.

## Adaptation requests (`configs/trigger.yaml`)

The trigger layer turns window statistics into **adaptation requests**: events that a decision layer acts on (or ignores). A request is evidence that the
detector may be degrading; it is **not a retraining decision**. Only signals with `trigger: true` take part. Every rule that fires emits its own request, and
each rule keeps **its own state** (reference period, anchor, counters, flag history, cooldown): a firing rule resets only itself.

| rule | type | condition | stage | v0.2 |
|---|---|---|---|---|
| `step` | `relative_drop`, comparator `previous` | the window's drop against the previous window exceeds `delta` (N = 1) | **screening** | **enabled**, delta 0.080, cooldown 0 |
| `rate` | `rate` | at least `min_flagged` (k = 3) of the last `windows` (M = 12, 1 minute) windows were *flagged*, i.e. met the step rule's condition (its comparator and delta) - the condition, not the fired request | **confirmation** | available, **disabled** |
| `sustained` | `relative_drop`, comparator `anchor` | the level stays more than `delta` (0.079) below the anchor (the window after this rule's last fire, or the stream start) for N = 2 windows in a row | **confirmation** | available, **disabled** |

The four prioritised trigger types (critical, cumulative, preventive, periodic) exist as hooks in `edge_signals/trigger.py`; they are disabled and refuse to run if
enabled. In this delivery only `consistency` takes part; `confidence`, `feature_drift` and `platform_motion` are measure-only.

**What a request contains** (`requests.jsonl`, `schemas/local_signal_event.json`): `request_id`, `stage`, `rule`, the window statistic before (the reference period's level) and after,
the `drop` magnitude, `delta`, the empirical **p-value** of the drop against the healthy change distribution delta was derived from (`references/healthy_changes.*.npz`;
p = (1 + number of healthy drops at least this large) / (n + 1)) with its floor `p_value_floor` = 1 / (n + 1) (a p equal to the floor means the drop is at least as large as every healthy drop; p cannot say by how much), the **robust z** = drop / (1.4826 x MAD of the same healthy distribution), which keeps growing with the drop, the **flag rate** over the last 12 windows (flagged windows, the share expected in healthy footage and the binomial tail
probability), and the latest values of every *other measured* signal (`context`: confidence, feature_drift, platform_motion). The context is for the decision layer only: no rule reads it.

**How `delta` is set.** It is derived without labels: minus the 5th percentile of the window-to-window change of the consistency window median (30 transitions, comparator `previous`;
for the sustained rule, the change against the anchor), measured within videos on held-out healthy footage, i.e. 5 % of healthy windows drop by more than `delta`.
`edge_signals.healthy.derive_changes` computes the distribution from healthy footage and `delta_from` takes the percentile. It depends on the window statistic, the camera and the
model: re-derive `delta` and the healthy distribution when you change any of them. Setting delta from labelled degradations would be a different, supervised choice.

**Window length.** `window_transitions: 30` is one window; alternatively give `window_seconds: 5` and the length is converted with the stream frame rate
(`configs/inference.yaml: stream.fps`, 6 fps in the replay, so 1 window = 30 transitions = 5 s and 1 hour = 720 windows).

### Operating point (a demonstration setting)

On held-out healthy footage of the replayed stream (short clips concatenated; cross-fitted thresholds, 30 random halvings):
* the **screening** (`step`) rule requested about **40 times per hour within a video** (median 43, interquartile range 27-89; about 90 per hour when clips of different videos are concatenated,
  because a scene cut can look like a drop). Against offline quality drops (window F1 lower than the previous window's by at least 0.10; labels were used for evaluation only) about half of its requests
  coincided with such a drop within one window (a random request would coincide in 20-40 % of cases) and about two thirds of such drops had a request within one window.
* the **confirmation (rate) rule is available; it is not evaluable on the short healthy streams used for v0.2 and is to be calibrated on long streams.**

**This is a demonstration setting, not a validated operating point.** The healthy footage available was only a few minutes per half, so these rates rest on a handful of requests and are uncertain
by a factor of two or more; 
Choose delta, N, M, k and the cooldown for your own tolerance for false alarms after collecting healthy footage from the deployed camera.

## Assets (not in the repository, not in the image)

`configs/assets.yaml` lists one entry per asset: `name`, `kind` (weights | reference | data | video), `url`, `sha256`, `size_bytes`, `target` (under `$EDGE_ASSETS` = `/assets` or
`$EDGE_DATA` = `/data`). The weights and the nine data archives are on **Google Drive** (`gdrive:<file id>`, shared with "anyone with the link"); `yolov12n.pt` is the official release file. A link can be changed without rebuilding the image, on the command line
(`fetch-assets --url NAME=URL`) or by mounting your own copy of the file (`--assets-config`). Links can be https URLs, Google Drive share links (including Drive's large-file
confirmation page and resource keys; `gdrive:<id>` also works) or `file://` paths. `fetch-assets` downloads only with `requests`, resumes partial downloads, verifies the SHA256 (and size), extracts archives
safely (no absolute paths, `..` or links) and is idempotent (`/assets/.fetched.json` records what was verified). Assets:

| kind | assets | target |
|---|---|---|
| weights | `maritime_s_base` (the edge detector), `maritime_x_base` (teacher of the pseudo-label recipe), `yolov12n` (official YOLOv12 release file, for the trainer's AMP self-check) | `/assets/weights/<name>.pt` |
| data | `data_<split>` for the 9 splits: labelled frames (`frames/<video>-<idx>.jpg`), single-class labels (`labels/`), the split's frame list (`lists/<split>.txt`) | `/data` |

The small derived references (`references/feature_drift.*.npz`, `references/healthy_changes.*.npz`) stay in the repository. The dataset must be obtained under its own licence (see `NOTICE`).

## Input modes

The stream stage accepts two kinds of input. They use the same detector, signals and trigger.

| | Labelled images | Raw video |
|---|---|---|
| Input | the extracted frames of the labelled dataset (`fetch-assets`) | the original dataset videos (your own copy) |
| Sampling | every labelled frame (one per 5 video frames) | every 5th decoded frame (stride 5 = 6 fps) |
| Matches the threshold calibration | exactly | approximately (see below) |
| Per-window quality check with labels | yes | via the frame mapping (`data/mapping/frame_map.csv`) |
| Use it for | reproducing our numbers | a realistic camera replay |
| Command | `stream --images-dir DIR` or `--list FILE` | `stream --videos-dir DIR --stream NAME` |

**Calibration.** The threshold (delta) and the healthy reference were derived on the extracted JPEG frames. On decoded video, the same window's statistic differs by
about 0.005 (median; 90th percentile 0.039), because the extracted frames went through JPEG compression. Requests whose drop sits close to the threshold can therefore appear in one mode and not the other.

**Expected results** for the stream `drift_calib` (`configs/replay_order.yaml`: clear_calib -> haze_eval -> nir_eval -> onboard_eval; 21 videos):
- labelled images: **8 requests** (frame-based run of the previous release).
- raw video: **6 requests** (2,222 frames, about 70 s on one GPU): `MVI_1448_VIS_Haze` frame 150, `MVI_1526_NIR` 450, `MVI_1528_NIR` 450, `MVI_1545_NIR` 150, `MVI_0790_VIS_OB` 150, `MVI_0801_VIS_OB` 300.
  These are the same as the image run except the first window of `MVI_1617_VIS` and of `MVI_0799_VIS_OB`, whose drops fall just below the threshold on video (for example 0.076 against 0.080).

Note: the image order of `--images-dir` / `--list` is alphabetical by frame name, not the order of the replay file; the request positions above are for the replay order, which only the video mode follows.

## Videos (you provide them) and the replay order

The stream stage reads the **original SMD videos**. They are not part of this repository, the image or `assets.yaml` (nothing downloads them): get the three official archives from the dataset's
page (see `NOTICE`) - **VIS On-Shore**, **VIS On-Board** and **NIR On-Shore** - unpack them anywhere and pass that folder with `--videos-dir`. **Only the files under `Videos/` are used**; the ground-truth
directories (`HorizonGT/`, `ObjectGT/`, `TrackGT/`) are ignored. Mount the folder read-only (e.g. `-v /path/to/videos:/videos:ro`).

```
stream --videos-dir <folder>            REQUIRED, any folder; searched recursively; .avi .mp4 .mov .mkv, any letter case
       [--stream NAME | --split NAME]   default: `default_stream` of the order file (drift_calib)
       [--order FILE]                   default: configs/replay_order.yaml
       [--on-missing skip|error]        default: skip
```

`configs/replay_order.yaml` decides **which videos are replayed and in which order**: `splits` lists the video codes of every split of the labelled dataset (in the order in which the videos appear in that
split's list file); `streams` are named concatenations of splits (`drift_calib` = clear_calib -> haze_eval -> nir_eval -> onboard_eval; `drift_test` = clear_test -> haze_eval; `negative_control` = clear_test).
Videos are read strictly in that order; windows never cross videos; the trigger state continues across a skipped video.

Each code (e.g. `MVI_0788_VIS_OB`) is matched **leniently**: (1) the file stem equals the code (case-insensitive); (2) otherwise a file stem that *contains* the code; (3) if more than one candidate
remains the video is *ambiguous*: the candidates are reported and it is treated as missing (never picked silently). With `--on-missing skip` a missing or ambiguous video is logged as a warning and the
run continues with the next one; with `error` the run stops with the list. Every run writes `resolution.json` (code -> resolved file, or missing / ambiguous with candidates, and the list actually replayed).

## Video input

The stream decodes the videos with OpenCV and keeps every `frame_stride`-th frame (default 5: a 30 fps video becomes the 6 fps stream that `delta`, the
healthy change distribution and the 30-transition window were calibrated on). For every processed frame the video name and its **original frame index** are recorded (`frames.jsonl`, and on every
request). **Another stride is refused** unless `trigger.healthy_changes` points to a healthy change distribution calibrated at that stride (its metadata `frame_stride` must match), and a video whose
fps does not give the calibrated 6 fps at that stride is refused too. `data/mapping/frame_map.csv` maps (video, original frame index) to the labelled image name (the labelled frames are the video frames
0, 5, 10, ...). `./run.sh check-alignment --videos-dir D --frames-dir /data/frames` searches, per video, stride 1-10 and offset 0..stride-1 for the best match between decoded frames and the labelled images (PSNR and
mean absolute difference, margin over the runner-up) so that you can confirm that the mapping is aligned for your copy of the videos.

## What is label-free

Every signal, every window statistic, the reference (training images only, no annotations) and the trigger rule use predictions, features and timestamps only.
No annotations are read anywhere in `edge_signals/`. The threshold `delta` is label-free as derived above.

## Layout

```
run.sh  README.md  docs/HANDOFF.md (integration handoff)  scripts/run_all.sh (run everything)  LICENSE (placeholder)  NOTICE  requirements-test.txt  Dockerfile  .dockerignore  .gitignore
configs/     inference.yaml  signals.yaml  trigger.yaml  adaptation.yaml  assets.yaml  replay_order.yaml  data.yaml
edge_signals/  base registry config windows trigger healthy engine inference streams video replay assets cli  adapt/{collect,annotate,retrain,evaluate,register,streamref,switch,metrics,common}  builtin/{consistency,confidence,platform_motion,feature_drift}
third_party/consistency/   the released package (code, tests, fixtures) + VENDORED.txt + FILES.sha256
references/  feature_drift.<id>.npz / .json, healthy_changes.<id>.npz         data/mapping/frame_map.csv     schemas/local_signal_event.json
docker/      install_fork.sh  constraints.txt  entrypoint.sh              scripts/  prepare_data check_alignment make_synthetic_frames      tests/
```
