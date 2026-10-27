# Dummy signal generator

Cheap, randomised but structured data for testing **model Search / Switch** over several edges (ships) that move through different areas. For every edge and every model it produces the window-level signals and adaptation requests of the real `edge_signals` logs (same files, same keys), without a GPU, a model, images, boxes or video.

> **Synthetic data.** Value ranges and noise are calibrated from real logs; the dynamics are not realistic. **Do not use these data for any claim about detection quality or trigger performance.** The generator contains no model selection or switching logic: the model schedule is an *input*.

## One command

```bash
pip install -r tools/dummy_signals/requirements.txt
python tools/dummy_signals/generate.py --config tools/dummy_signals/default_config.yaml --seed 0 --out run_demo --logs minimal
```

It simulates 24 h on three edges (about a minute) and writes `run_demo/world.json` (the ground truth) and, for every edge, `run_demo/<edge>/`: the stream of the edge's **initial model**. Drop `--logs minimal` to also write the per-pair and per-frame values. The same `--seed` gives byte-identical files. `--config` takes a YAML with any subset of the parameters of `default_config.yaml` (each has a comment there).

## Examples

Three episodes of two hours (route: stay 30 min, move 60 min, stay 30 min; equal area difficulty; the quality of every model on every area is pinned in `matrix.fixed`: 0.86 / 0.58 for an s model on its trained / another area, 0.90 / 0.63 for an x model), written to be copied and adapted (every file is commented). Only the per-frame noise is random:

| file | what it is | run |
|---|---|---|
| `examples/ep1_two_edges.yaml` | two ships swap areas A and B; each stores only the model trained on its start area; transition effect `common: -0.15` (both sides are bad in the mixed zone) | `python tools/dummy_signals/generate.py --config tools/dummy_signals/examples/ep1_two_edges.yaml --seed 0 --out run_ep1 --logs minimal` |
| `examples/ep2_two_edges_cloud.yaml` | the same, plus a cloud node that stores all four models (s and x); plain linear moves (the baseline) | `python tools/dummy_signals/generate.py --config tools/dummy_signals/examples/ep2_two_edges_cloud.yaml --seed 0 --out run_ep2 --logs minimal` |
| `examples/ep3_three_edges_cloud.yaml` | three ships on a cycle A -> B, B -> C, C -> A, each storing the s and x models of its start area, plus a cloud node with all six; transition effect `common: 0.0, spread: 0.10` (some models are better, some worse in the mixed zone) | `python tools/dummy_signals/generate.py --config tools/dummy_signals/examples/ep3_three_edges_cloud.yaml --seed 0 --out run_ep3 --logs minimal` |

Each command writes `world.json` and, for every edge, the stream of its initial model. The examples ship with `signals: {cell_noise: off, blind_cells: off}` (see below); in `default_config.yaml` both switches are on.

## The performance oracle

```bash
python tools/dummy_signals/generate.py oracle --config tools/dummy_signals/examples/ep3_three_edges_cloud.yaml --seed 0 --out oracle_ep3 [--window 5s]
```

For every edge, every window (5 s = 30 frame pairs by default) and **every model of the world, whatever node stores it**, it writes the true F1 (from the TP / FP / FN counts of the window), the clip consistency and confidence medians (the numbers `clip_score` returns for that window) and where the model is stored. For every window it also ranks the models by true F1: the **top 3** (model, location, F1, and the delta to top1) and `best_local`, the best model stored on that edge itself. Only `oracle.csv`, `oracle_summary.csv` and `world.json` are written, into `--out`.

**How to read it.** The oracle is the *unconstrained ideal*: every model anywhere, at no cost, no delay and no failure. It is ground truth about the generated data, not a decision, and the generator never selects or switches a model. An adaptation policy should be judged by how close it gets to `top1` and at what cost (transfers, delay, retraining). `best_local` is the no-transfer reference: the gap between `best_local` and `top1` is the most that fetching a model from another node can gain, and `initial` versus `top1` is the most that switching can gain.

* `oracle.csv`, one row per edge x window x model: `edge, window, t_start, t_end, area, model, size, version, location, location_kind, stored_on, true_f1, consistency, confidence, is_initial, is_top1, initial_model, initial_f1, gap, top1_model, top1_location, top1_f1, top2_model, top2_location, top2_f1, top2_delta, top3_model, top3_location, top3_f1, top3_delta, top3_summary, best_local_model, best_local_f1, best_local_delta`. `location` is `local` (stored on the edge), else the id of the node it would be fetched from: another edge first, otherwise a cloud node; `stored_on` lists every node that stores the model. The columns from `initial_model` on are per window and repeated on every row of the window; `top3_summary` reads like `x_a 0.899 | s_a -0.04 | x_b -0.12` (top1 and its F1, then the deltas to top1); `gap` is top1 minus the initial model.
* `oracle_summary.csv`, one row per edge x route segment: `edge, segment, kind, area, t0, t1, n_windows, initial_model, initial_mean_f1, gap, top1_model, top1_location, top1_mean_f1, top2_…, top3_…, top3_summary, best_local_model, best_local_mean_f1, best_local_delta, share_consistency_first, share_confidence_first, share_both_first`. The top 3 and `best_local` are ranked by the mean F1 over the segment's windows; the three shares are the share of windows in which the model with the highest consistency, confidence, or mean of the two is that window's top1.

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
      - {move_to: B, over: 20m}     # gradual: quality and signals change linearly over 20 min (optional: transition: {common: -0.15, spread: 0.05, models: {s_a: -0.05}})
      - {stay: B, for: 60m}
random_episodes: off
```

**Transition effect** (optional, off by default): `move_to: B, over: 60m, transition: {common: -0.15, spread: 0.05, models: {s_a: -0.05}}` adds a hump to the precision and recall of every model during the move: `w(t) * (common + deviation)`, with `w` a Hann weight (0 at both ends of the move, 1 at its midpoint). `common < 0` makes every model worse in the mixed zone, `common > 0` better; the deviation of each model is drawn from N(0, `spread`) per seed unless it is given under `models`. The consistency and confidence centres follow through their regression on precision and recall, F1 is recomputed, and the object count (a property of the data) is unchanged. `world.json` records the parameters and the drawn deviations under `transitions`. Without `transition` a move is the plain linear one.

Besides the edges, `nodes:` lists nodes that only store models: `nodes: [{id: cloud, kind: cloud, models: [s_a, x_a]}]`. A cloud node has no route, no data and no stream; any node can serve a model to an edge.

Durations: `30s`, `40m`, `40min`, `2h` (mixed freely; a plain number is seconds). The route steps must add up to `simple.duration`; a route starts with `stay`. Areas, models and the initial model must exist, and the initial model must be stored on the edge. Without `difficulty`, the difficulty of every area is drawn per seed.

**Signal switches.** `signals: {cell_noise: on, blind_cells: on}` (on / off / true / false) switch the random parts of the consistency / confidence centres of the (model, area) cells: `cell_noise` the random cell offsets (off = the centres follow P and R exactly), `blind_cells` the cells where consistency overstates the quality of an untrained model. Both are on in `default_config.yaml` (realistic, for experiments); the examples set both off.

## Output files (per edge)

| file | content |
|---|---|
| `windows.jsonl` | one line per 5-s window and signal (`consistency`, `confidence`): `signal, window_id, clip_id, t_start, t_end, n_items, n_valid, statistic, valid, invalid_reason, change, comparisons, model_version, device_id, ...` |
| `requests.jsonl` | adaptation requests of the real trigger: `request_id, rule, signal, window_id, clip_id, t, statistic_before, statistic_after, drop, delta, p_value, robust_z, model_version, device_id, video: null, frame_index: null, ...` |
| `context.jsonl` | a `scene_cut` event at every model switch (empty for a single model) |
| `truth.jsonl` | per frame: `frame_index, t, model_id, area, n_obj, TP, FP, FN` |
| `summary.json` | counts and the model schedule: per entry `t, model, source` (`local` or `fetched`) and `from_node` |
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

`clip_score(edge, model, t)` works for **any** model, including models stored on other edges or on a cloud node: the clip is always the requesting edge's data (the 30 frame pairs ending at `t`; consistency and confidence are their medians). The model schedule of `stream` may also use any model; `summary.json` records for every entry whether it is `local` or `fetched` and from which node (`(1800, "x_c", "cloud")` or `1800:x_c@cloud` on the command line names the node; default: the first other edge that stores it, else a cloud node; with the default configuration every edge stores every model, so all models are local).
Other commands: `generate.py oracle` (above), `generate.py stream --edge E --schedule 0:s_a,1800:x_c@cloud --out DIR`, `generate.py clip --edge E --model M --t-end T`, `generate.py world --out DIR`.

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

## Changes

* The field `from_edge` of the model schedule in `summary.json` (and the key `from_edge` of a schedule entry) is now `from_node`, because a model can also be fetched from a cloud node. There is no alias for the old name.
* New: cloud nodes (`nodes:`), the optional `transition` effect of a move, the `signals` switches, the `oracle` command (top 3 and `best_local` per window and segment) and the three episodes in `examples/`.

## Limits

* Signals are drawn directly, not computed from detections; the link between quality and signals is a configured correlation, not a measured law.
* Areas are abstract; changes are ramps between constant levels; the statistics inside an area are stationary.
* A few extras of the consistency values are empty (`alarm`, `G`, `G_mc`, `dx`, `dy` are `null`); `status`, `N_t`, `N_t1` follow the generated box counts.
* The pair-level calibration is not perfect: the precision-related event rate and AUROC are within about 1.3 times their tolerance on fresh seeds.
* With the default difficulty draw, an area can be harder than another for every model, so leaving an area does not always lower the F1 of a model; pin the difficulties for a controlled scenario.

Tests: `python -m pytest tools/dummy_signals/tests` (no real data needed).
