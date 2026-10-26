# Dummy signal generator

Cheap, randomised but structured data for testing **model Search / Switch** over several edges (ships) that move through different areas. For every edge and every model it produces the window-level signals and adaptation requests of the real `edge_signals` logs (same files, same keys), without a GPU, a model, images, boxes or video.

> **Synthetic data.** Value ranges and noise are calibrated from real logs; the dynamics are not realistic. **Do not use these data for any claim about detection quality or trigger performance.** The generator contains no model selection or switching logic: the model schedule is an *input*.

## One command

```bash
pip install -r tools/dummy_signals/requirements.txt
python tools/dummy_signals/generate.py --config tools/dummy_signals/default_config.yaml --seed 0 --out run_demo --logs minimal
```

It simulates 24 h on three edges (about a minute) and writes `run_demo/world.json` (the ground truth) and, for every edge, `run_demo/<edge>/`: the stream of the edge's **initial model**. Drop `--logs minimal` to also write the per-pair and per-frame values. The same `--seed` gives byte-identical files. `--config` takes a YAML with any subset of the parameters of `default_config.yaml` (each has a comment there).

> A scripted multi-edge example (three ships that move between areas, with their routes) will follow in a later release; the syntax is described below.

## Edges: random timeline or scripted route

By default (`default_config.yaml`) the edges `e1`, `e2`, `e3` follow a random timeline of areas that moves along the route and reaches each edge with a delay. An edge can instead follow a **scripted route**, and both kinds can be mixed:

```yaml
areas: [A, B, C]                    # or [{name: A, difficulty: 0.2}, ...] to pin the difficulty of an area
models:
  - {id: s_a, size: s, version: a, trained_areas: [A]}      # one entry per model
edges:
  - id: edge1
    models: [s_a, x_a]              # the models stored on this edge
    initial_model: s_a              # the model it runs at t = 0
    route:
      - {stay: A, for: 40m}
      - {move_to: B, over: 20m}     # gradual: quality and signals change linearly over 20 min
      - {stay: B, for: 60m}
random_episodes: off
```

Durations: `30s`, `40m`, `40min`, `2h` (mixed freely; a plain number is seconds). The route steps must add up to `simple.duration`; a route starts with `stay`. Areas, models and the initial model must exist, and the initial model must be stored on the edge. Without `difficulty`, the difficulty of every area is drawn per seed.

## Output files (per edge)

| file | content |
|---|---|
| `windows.jsonl` | one line per 5-s window and signal (`consistency`, `confidence`): `signal, window_id, clip_id, t_start, t_end, n_items, n_valid, statistic, valid, invalid_reason, change, comparisons, model_version, device_id, ...` |
| `requests.jsonl` | adaptation requests of the real trigger: `request_id, rule, signal, window_id, clip_id, t, statistic_before, statistic_after, drop, delta, p_value, robust_z, model_version, device_id, video: null, frame_index: null, ...` |
| `context.jsonl` | a `scene_cut` event at every model switch (empty for a single model) |
| `truth.jsonl` | per frame: `frame_index, t, model_id, area, n_obj, TP, FP, FN` |
| `summary.json` | counts and the model schedule: per entry `t, model, source` (`local` or `fetched`) and `from_edge` |
| `signals.jsonl`, `frames.jsonl` | per-pair / per-frame values (only without `--logs minimal`) |

Every line carries `"synthetic": true`; `device_id` is `synthetic-<edge>`, `model_version` is `synthetic-<model id>`. `world.json` holds every parameter and draw: areas (difficulty, objects per frame), per-edge `mode` (`route` or `random`), `models`, `initial_model` and `route`, the resolved segments, and the model x area matrices `precision`, `recall`, `f1`, `consistency_centre`, `confidence_centre`, `blind_cells`.
Windows, comparison, p-value, `robust_z` and the requests come from the package's own `SignalEngine` / `TriggerEngine` (delta is read from `configs/trigger.yaml`; `--delta` overrides it).

## Use it from your code

```python
from tools.dummy_signals.world import World
w = World("tools/dummy_signals/default_config.yaml", seed=0)   # or a dict, or None for the defaults
w.edge_models("e1")                            # the models stored on e1 (by default all six)
w.area("e1", 3000.0)                           # area of e1 at t = 3000 s (the nearer one during a move)
w.clip_score("e1", "x_c", t_end=3000.0)        # {consistency, confidence, valid_fraction, n_pairs, areas}
w.stream("e1", [(0, "s_a"), (1800, "x_c")], "out", logs="minimal")      # switch to another model at t = 1800 s
```

`clip_score(edge, model, t)` works for **any** model, including models stored on other edges: the clip is always the requesting edge's data (the 30 frame pairs ending at `t`; consistency and confidence are their medians). The model schedule of `stream` may also use any model; `summary.json` records for every entry whether it is `local` or `fetched` and from which edge (`(1800, "x_c", "edge3")` or `1800:x_c@edge3` on the command line names the edge; default: the first edge that stores it; with the default configuration every edge stores every model, so all models are local).
Other commands: `generate.py stream --edge E --schedule 0:s_a,1800:x_c@edge3 --out DIR`, `generate.py clip --edge E --model M --t-end T`, `generate.py world --out DIR`.

## What the data is based on

- Structure: each ship's data depends on where it is (area) and when; each model (backbone size s/x, training version a/b/c trained on area A/B/C) has a quality on each area; quality is generated per frame as TP/FP/FN counts; the two label-free signals (consistency, confidence) are generated from that quality.
- Calibrated on real data:
  - signal levels, noise and the trigger's false-alarm rate (about 5 % of 5-s windows): from the real edge stream (YOLOv12s maritime model, clear SMD videos), passed through the real trigger code of edge-signals;
  - how the signals follow precision/recall (slopes, scatter, frame-level detection of degraded frames): from a per-frame analysis of 63 SMD videos with a YOLOv12x model.
- Assumptions (NOT measured): how much better x is than s, how much better a version is on its own area, how quality changes between areas. These are parameters in the YAML.
- Use it to test systems behaviour (triggering, search, switching, network faults). Do not use it to claim detection or trigger accuracy.

## Details of the model

1. **Areas.** Abstract kinds of data (no weather meaning), each with a difficulty and a mean number of objects per frame.
2. **Edge trajectories.** A route (above), or the random timeline: a global semi-Markov sequence of areas moves along the route at `front_speed_kmh`; an edge at position *p* sees it delayed by `p / front_speed`; changes are ramps of length `ramp_max_s * (1 - steepness)`. Optional random local episodes (`random_episodes: on`) show another area to one edge for a while (abruptly on a route).
3. **Models.** Precision `P[m,a]` and recall `R[m,a]` = base(area difficulty) + size bonus + specialisation if trained / penalty if not + noise, rescaled so that the worst cell has F1 `perf_min` and the best `perf_max`.
   Consistency centre `C = a_c + 0.214 P + 0.125 R + N(0, 0.033)`, confidence centre `Conf = a_f + 0.240 P + 0.183 R + N(0, 0.040)` (slopes and residual sds from the real per-video regression). During a move, `P`, `R`, `C`, `Conf` and the object rate are interpolated linearly and F1 is recomputed from the interpolated `P` and `R`.
   **Blind cells:** with probability 0.05 an untrained cell has its consistency centre raised by 2 x 0.033 (consistency overestimates its quality; confidence is not affected).
4. **Frames** (computed lazily, never pre-generated): the object count depends on the edge and frame only; a frame difficulty is shared by all models of an edge; recall, precision, `TP`, `FP`, consistency per frame pair and confidence per frame follow the centres plus noise and a slow wander. A frame without boxes has an invalid confidence (`no_boxes`); a pair of two empty frames is invalid (`EMPTY_PAIR`); a pair with exactly one empty frame is a valid consistency of 0. All random numbers are counter-based (hash of seed, layer, edge, model, frame): any range is identical whatever the query order.
5. **A model switch is a scene cut:** no pair across it, the partial window is dropped, the trigger state restarts.

**Absolute anchor.** The levels combine the quality of the paper's model (F1 0.94) with the consistency / confidence levels of the edge model (real log A): a healthy cell (P = R = 0.94) has median consistency 0.788 and confidence 0.6625. The collapse of consistency at low quality comes from hard areas (few objects, many empty frames, hence exact zeros and `EMPTY_PAIR`); healthy areas have about no zeros, as in the real clear footage.
Search mistakes in the data come mainly from the cell-level noise (0.033 for consistency, 0.040 for confidence, from the residuals of the real regression) and the blind cells (5 % of the untrained cells).

**Calibration of the defaults.** Healthy area: consistency median 0.79, p5 0.62, p95 0.95, within-window std 0.054, about 5 % of the windows request; confidence median 0.66. The per-frame couplings (`frame`, `consistency_frame`, `confidence_frame`, `calibration` in `default_config.yaml`) come from `python tools/dummy_signals/calibrate.py fit` (differential evolution, then Nelder-Mead on a calibration world with one model, one edge and four areas; `check` evaluates it on fresh seeds); each value has a comment naming the target it serves.

## Limits

* Signals are drawn directly, not computed from detections; the link between quality and signals is a configured correlation, not a measured law.
* Areas are abstract; changes are ramps between constant levels; the statistics inside an area are stationary.
* A few extras of the consistency values are empty (`alarm`, `G`, `G_mc`, `dx`, `dy` are `null`); `status`, `N_t`, `N_t1` follow the generated box counts.
* The pair-level calibration is not perfect: the precision-related event rate and AUROC are within about 1.3 times their tolerance on fresh seeds.
* With the default difficulty draw, an area can be harder than another for every model, so leaving an area does not always lower the F1 of a model; pin the difficulties for a controlled scenario.

Tests: `python -m pytest tools/dummy_signals/tests` (no real data needed).
