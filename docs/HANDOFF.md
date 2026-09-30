# edge_signals — handoff for the QUICK integration

> **Status.** Verified by a fresh test of the **public v0.5.0 images** (anonymous pull, real Google Drive downloads, 119 tests, requests 8 with labelled images and 6 with raw video as in section 7, and the retraining chain).
> Since then only the downloader (timeouts, progress lines, Drive web-page error), the stream progress lines and summary fields, `scripts/run_all.sh`, tests and documentation changed. The same steps were re-run with
> `scripts/run_all.sh` on the v0.5.2 images, once with and once without the original videos (141 tests, requests 8 / 6 exactly as in section 7, all SHA256s, retraining chain).

## 0. One command

`scripts/run_all.sh` does sections 3 to 9 below for you (pull, fetch-assets, tests, both stream modes, comparison with `configs/expected_results.yaml`, retraining chain): `scripts/run_all.sh --work-dir ./edge-signals-run --videos-dir /path/to/videos`.
It prints every docker command before it runs it, so its output is a transcript of this file. The sections below describe the same steps one by one.

## 1. What this is

`edge_signals` runs a maritime object detector on a replayed camera stream and computes **label-free local signals** per time window. When the output consistency drops sharply, it emits an **AdaptationRequest**: evidence for the decision layer that the model may need to change. It also provides the cloud-side steps for retraining on the labelled data of the same video.

| Item | Where |
|---|---|
| Code | `https://github.com/commin/edge-signals`, tag `v0.5.2`; licence GPL-3.0-only |
| Container image | `ghcr.io/commin/edge-signals:v0.5.2` (public); test image `ghcr.io/commin/edge-signals:v0.5.2-test` |
| Model weights, labelled frames and labels | Google Drive, shared "anyone with the link" (ids and SHA256 in `configs/assets.yaml`); `yolov12n.pt` from the official YOLOv12 release |
| Original videos | from the SMD dataset's official page (see section 4) |

Nothing heavy is inside the repository or the image: no weights, no data, no videos.

## 2. Requirements

- **x86-64** Linux host with Docker, an NVIDIA GPU and the NVIDIA Container Toolkit. CPU-only works for tests and replay, but retraining needs a GPU.
- Disk: about 11 GB for the image (10.7 GB), about 1.1 GB for assets and labelled frames after `fetch-assets` (138 MB weights + 982 MB frames and labels; the 1.0 GB of archives is deleted after extraction), about 5.4 GB for all 81 original videos (the 63 used by the replay order are most of it).
- Network for the first `docker pull` and `fetch-assets` only. Everything after that runs offline (`--network none`). **If container networking is restricted (or names cannot be resolved inside containers), run `fetch-assets` with `--network host` or `--dns <resolver>`.**

## 3. Pull the image and fetch the assets

```bash
docker pull ghcr.io/commin/edge-signals:v0.5.2

mkdir -p assets data outputs
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/assets:/assets" -v "$PWD/data:/data" \
  ghcr.io/commin/edge-signals:v0.5.2 fetch-assets
```

`fetch-assets` prints a progress line at least every 10 s per file, gives up after 15 s without a connection or 60 s without data (the error names the host and the likely cause; a partial file is resumed on the next run), and stops with the file id if Google Drive returns a web page (quota, permission) instead of the file. It downloads the two model weights, `yolov12n.pt` (the file the trainer's AMP self-check needs) and the labelled frames and labels (9 archives). Every file is checked against its SHA256 and its size. It is safe to re-run: finished files are skipped (`up_to_date`).

Options: `--only NAME ...`, `--kind weights data`, `--url NAME=URL` (replace one link, https / Google Drive link / `gdrive:<id>` / `file://`), `--assets-config FILE` (a mounted copy of `configs/assets.yaml`).

*Executed (local stand-in):* with a config that points to local `file://` copies of the same files (`--assets-config`) all 12 assets were fetched and verified in 5 s, and a second run reported all 12 `up_to_date`. The Google Drive links in `configs/assets.yaml` are exercised by the mocked downloader tests (confirmation page, resource key, resume, SHA256 mismatch) and, after publication, by `fresh_test.sh`.

## 4. Get the original videos

Download the three archives from the dataset's official page: https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset (see `NOTICE` for the citation).

- VIS On-Shore (about 3.2 GB)
- VIS On-Board (size not verified)
- NIR On-Shore (about 1.5 GB)

Each archive contains `Videos/`, `HorizonGT/`, `ObjectGT/` and `TrackGT/`. **Only the video files are used.** Put them anywhere, for example `./videos/`. Sub-folders are fine.

You do not need every video. `configs/replay_order.yaml` lists the 63 videos used, in order. Missing videos are skipped with a warning.

## 5. Check the installation

```bash
docker run --rm --user "$(id -u):$(id -g)" --gpus all \
  -v "$PWD/assets:/assets" -v "$PWD/data:/data" \
  ghcr.io/commin/edge-signals:v0.5.2-test test
```

Expected: `141 passed` (about one minute on one GPU). The tests need `/assets` (weights); `/data` is optional.

## 6. Replay a stream — two input modes

The same detector, signals and trigger run on either input:

| | Labelled images | Raw video |
|---|---|---|
| Input | extracted frames of the labelled dataset (from `fetch-assets`) | the original videos (section 4) |
| Sampling | every labelled frame (one per 5 video frames) | every 5th decoded frame (stride 5 = 6 fps) |
| Matches the threshold calibration | exactly | approximately (see section 7) |
| Use it for | reproducing our numbers | a realistic camera replay |

### 6a. Labelled images

```bash
docker run --rm --user "$(id -u):$(id -g)" --gpus all --shm-size=2g --network none \
  -v "$PWD/assets:/assets" -v "$PWD/data:/data" -v "$PWD/outputs:/outputs" \
  ghcr.io/commin/edge-signals:v0.5.2 \
  stream --images-dir /data/frames --stream drift_calib --out /outputs/drift_calib_images
```

### 6b. Raw video

```bash
docker run --rm --user "$(id -u):$(id -g)" --gpus all --shm-size=2g --network none \
  -v "$PWD/assets:/assets" -v "$PWD/data:/data" \
  -v "$PWD/videos:/videos:ro" -v "$PWD/outputs:/outputs" \
  ghcr.io/commin/edge-signals:v0.5.2 \
  stream --videos-dir /videos --stream drift_calib --out /outputs/drift_calib_video
```

For both modes:
- `--stream` picks a named sequence from `configs/replay_order.yaml`: `drift_calib`, `drift_test` or `negative_control`. `--split <name>` replays a single split. `--order FILE` uses another order file.
- Videos and frames are read in the order of that file. In image mode the clips are taken from the frame names (`<video>-<index>.jpg`); clips outside the stream are left out; a clip without frames is skipped with a warning (`--on-missing error` stops).

For raw video only:
- Videos are matched by their code (for example `MVI_1448_VIS_Haze`), not by the exact file name: exact stem first, then a stem that contains the code; several candidates = ambiguous = treated as missing.
- Missing videos are skipped with a warning; `--on-missing error` stops instead.
- The stride is fixed at 5 because the threshold was calibrated at that sampling (`--frame-stride` with another value is refused).

### Outputs

| File | Content |
|---|---|
| `resolution.json` | which file was used for each video code (image mode: how many frames per clip); missing or ambiguous codes |
| `signals.jsonl` | one line per signal value (per frame or per frame pair): consistency, confidence, feature drift, platform motion |
| `windows.jsonl` | one line per signal and window: statistic, change against the previous window, validity |
| `requests.jsonl` | one line per AdaptationRequest |
| `frames.jsonl`, `context.jsonl`, `summary.json` | frames processed (video and original frame index in video mode); scene cuts between videos; run summary |

## 7. Expected results (to confirm your setup matches ours)

For `--stream drift_calib` (21 videos, 2,222 frames, about 1 minute on one GPU in either mode):

**Raw video** (full video set):
- **6 requests**, at: `MVI_1448_VIS_Haze` frame 150, `MVI_1526_NIR` 450, `MVI_1528_NIR` 450, `MVI_1545_NIR` 150, `MVI_0790_VIS_OB` 150, `MVI_0801_VIS_OB` 300.

**Labelled images:**
- **8 requests**: the 6 above (as clips `MVI_1448_VIS_Haze`, ...; image-mode requests have `video` and `frame_index` = null, the clip is in `clip_id`), plus the first window of `MVI_1617_VIS` and of `MVI_0799_VIS_OB`.

**Why the two modes differ.** The threshold (delta) and the healthy reference were derived on the extracted JPEG frames. On decoded video, the same window's statistic differs by about 0.005 (median; 90th percentile 0.039, maximum 0.10), because the extracted frames went through JPEG compression. The two extra requests have drops just above the threshold on images and just below it on video (for example 0.076 against 0.080 for `MVI_1617_VIS`; the on-board window is a low-box-count window).

For the same reason, small numeric differences between GPUs can move a window that sits right at the threshold.

> These numbers are from `v0.5.2`. Update them if the threshold is recalibrated.

## 8. Reading an AdaptationRequest

A request is **screening evidence, not a retraining decision**. Its schema is in `schemas/local_signal_event.json`. Main fields:

| Field | Meaning |
|---|---|
| `video`, `frame_index` | where in the stream the request was raised (original video and original frame index; video mode only), `clip_id` and `window_id` in both modes |
| `rule`, `stage` | which rule fired; `screening` in this version |
| `statistic_before`, `statistic_after`, `drop` | window statistic before and after, and the drop (compared with `delta`) |
| `p_value`, `p_value_floor` | how unusual the drop is against healthy footage; the floor is the smallest value this reference can give |
| `robust_z` | drop in units of the healthy variation; keeps growing where `p_value` saturates |
| `flag_rate` | share of recent windows that met the condition (`windows`, `flagged`, `rate`, `healthy_flag_probability`, `binomial_p`) |
| `context` | confidence, feature drift and platform motion at that time; **no rule reads them** |

Deciding what to do — keep, switch, move or retrain — belongs to the decision layer.

## 9. Retraining on a request

These steps run on the cloud side, one command each. They need a **raw-video** stream run (the request must carry the original video and frame index). Example for the first request of section 6b (`step-consistency-9`, read its id in `requests.jsonl`):

```bash
RUN="docker run --rm --user $(id -u):$(id -g) --gpus all --shm-size=2g --network none \
  -v $PWD/assets:/assets -v $PWD/data:/data -v $PWD/outputs:/outputs ghcr.io/commin/edge-signals:v0.5.2"
RID=step-consistency-9; O=/outputs/adapt/$RID

$RUN collect  --out $O --stream-out /outputs/drift_calib_video --request-id $RID --images-root /data/frames
$RUN annotate --out $O --images-root /data/frames
$RUN retrain  --out $O --images-root /data/frames --seed 0
$RUN register --out $O --parent maritime_s_base.v0 --request-id $RID
$RUN evaluate --out $O --weights $O/models/maritime_s_base.v1.pt --images-root /data/frames --labels-dir /data/labels \
      --eval nir_eval=/data/lists/nir_eval.txt --in-stream /outputs/drift_calib_video --request-id $RID --tag v1
```

- `collect`: the labelled frames of the **same video(s) from the last reset to the request** (a stream of several videos contributes each video's segment; `--since-request-id ID` sets the last reset, default the stream start) plus replay frames of earlier clear footage (ratio 2.0). Fewer than `collect.min_new_frames` (60) new frames: it logs "insufficient data", exits with code 3 and nothing is retrained.
- `annotate`: `--mode provided` (default, *simulated annotation*: the ground-truth labels of the collected frames, `annotate.provided.labels_dir` = `/data/labels`); `--mode pseudo` labels them with the larger teacher model — use it only when the teacher is stronger than the student on the current data (see section 11).
- `retrain`: fine-tunes the edge model from `maritime_s_base` (always from the base, never chained), freeze 9, 10 epochs; it needs the GPU and `--shm-size=2g`.
- `register`: writes the new version (`maritime_s_base.v1`) into `registry.json` of the run directory (`--registry` to choose another).
- `evaluate`: offline mAP50 (needs labels) and the label-free consistency proxy on the given splits; `--in-stream` adds the later frames of the request's video (view (a), only a few dozen frames of one video) next to the held-out splits. **Every video of the request's collected window (and of its replay frames) is removed from every held-out split automatically**; the excluded videos are printed and stored per split in the result JSON (`--include-collected` switches this off).

In the run above (default provided labels): 402 new frames + 804 replay frames, annotate 0.2 s (it only copies label files; with `--mode pseudo` about 18 s), retrain 73 s; `evaluate` excluded no video from `nir_eval` (the window held only clear and haze videos). A retrain takes 1–2 minutes on one GPU.

## 10. Configuration

| File | What you can change |
|---|---|
| `configs/signals.yaml` | per signal: `measure` (compute and log) and `trigger` (take part in the decision); window length; window statistic |
| `configs/trigger.yaml` | rules and thresholds; `step` is on; `rate` and `sustained` exist but are off |
| `configs/replay_order.yaml` | video order, splits and streams |
| `configs/adaptation.yaml` | replay ratio, minimum new frames, recipe (default provided labels = simulated annotation, replay 200 %) |
| `configs/assets.yaml` | download URLs and checksums |

The URLs can be overridden at run time (`fetch-assets --url NAME=URL` or a mounted `--assets-config`), so a new link never needs an image rebuild.

## 11. Known limits

- **The threshold is a demonstration setting, not a validated operating point.** It was derived from a few minutes of healthy footage; the request rate is uncertain by a factor of two or more. Setting it for continuous operation needs hours of healthy footage and controlled drift.
- **Feature drift reacts to new scenes**, not only to degradation. It is logged, not used for decisions.
- **Consistency is lower under platform motion** (on-board camera) even when the detector is fine.
- **The single class is called `vessel` but includes every object type** of the original labels, including buoys and flying birds or planes.
- **Pseudo-labels can hurt: failure mode.** `annotate --mode pseudo` helps only when the teacher is stronger than the student on the *current* data. In same-video retraining it can be weaker: for the request at `MVI_1526_NIR` frame 450 the teacher's labels on the collected frames had recall 0.69 (precision 0.77), and the student retrained on them fell from 0.649 to 0.206 mAP50 on the 29 later frames of that video (87 objects; 41 of 87 found against 65 for the base model; three runs: 0.206, 0.233, 0.195), while retraining on the ground-truth labels of the same frames gave 0.983 (86 of 87). This is why the delivered default is `provided` (simulated annotation) and `pseudo` is an option. The drop was **not** caused by too few new frames: the retrain used 687 new + 1374 replay frames; it follows the labels. Held-out evaluation (`clear_test`) stayed stable (0.680-0.689 against 0.672).
- **Retraining is not bit-reproducible.** The same request and seed gave in-stream mAP50 0.740 and 0.718 (request at `MVI_1448_VIS_Haze` frame 150) and 0.206 and 0.233 (`MVI_1526_NIR` frame 450) in two runs.
- **Leakage rule: held-out means held out.** A split is not held out for a request if the request's collected window contains one of its videos (for a request inside the haze or NIR videos, `haze_eval` and `nir_eval` are largely in the training data; with provided labels that gave 0.967 on `haze_eval`). `evaluate` therefore removes every video of the collected window (and of the replay frames) from every `--eval` split automatically, prints the excluded videos and stores them per split in the result (`excluded_collected_videos`); a split left empty is reported as skipped; `--include-collected` switches the rule off. The in-stream view is the later frames of the request's video by definition and is reported separately.
- **The loop is not closed yet.** In our tests, each retrain started from the base model and was evaluated offline; the retrained model was not put back into the stream. Replacing the model during replay, and chaining one retrain on top of the previous one, have not been tested.
- **The threshold was calibrated on extracted JPEG frames** (section 7). In raw-video mode it applies approximately.

## 12. Versions

| Item | Value |
|---|---|
| Code tag | `v0.5.2` |
| Image | `ghcr.io/commin/edge-signals:v0.5.2`; the registry digest exists after the push: `docker inspect --format '{{index .RepoDigests 0}}' ghcr.io/commin/edge-signals:v0.5.2` |
| YOLOv12 fork | pinned commit `2abab7153a065fb2925e8088e9ca2b19016ab7d6` (`docker/install_fork.sh`, `configs/inference.yaml`) |
| Consistency package | vendored at commit `78c4cde` (`third_party/consistency/VENDORED.txt`) |
| Asset checksums | `configs/assets.yaml` |
