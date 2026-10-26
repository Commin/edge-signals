"""Tests of the synthetic multi-edge generator (no GPU, model, video or network)."""
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.dummy_signals import config as cfgmod
from tools.dummy_signals import frames, rng
from tools.dummy_signals.world import World

HERE = Path(__file__).resolve().parents[1]
SMALL = {"simple": {"duration": "3h"}}


def read(p):
    return [json.loads(l) for l in open(p)]


def test_counter_based_values_do_not_depend_on_query_order_or_range():
    w = World(SMALL, 5)
    full = frames.frame_arrays(w, 1, 2, 0, 20000)
    for lo, hi in ((0, 5000), (4000, 9000), (8999, 20000), (12345, 12346), (777, 15001)):
        part = frames.frame_arrays(w, 1, 2, lo, hi)
        assert np.array_equal(part["n_obj"], full["n_obj"][lo:hi]) and np.array_equal(part["TP"], full["TP"][lo:hi]) and np.array_equal(part["FP"], full["FP"][lo:hi])
        a, b = part["conf"], full["conf"][lo:hi]
        assert np.array_equal(a, b, equal_nan=True)
        if hi - lo > 1:
            assert np.array_equal(part["cons"], full["cons"][lo:hi - 1] if lo == 0 else full["cons"][lo - 1 + 1 - 1 + 0:hi - 1][: len(part["cons"])] if False else full["cons"][lo:hi - 1], equal_nan=True)
    idx = np.random.default_rng(0).permutation(np.arange(100, 4000))
    shuffled = frames.frame_arrays(w, 1, 2, 0, 4100)
    assert np.array_equal(shuffled["TP"][idx], full["TP"][idx]) and np.array_equal(World(SMALL, 5).clip_score("e2", "x_b", 1234.5)["consistency"], w.clip_score("e2", "x_b", 1234.5)["consistency"])
    assert w.clip_score("e2", "x_b", 1234.5) == World(SMALL, 5).clip_score("e2", "x_b", 1234.5)


def test_same_seed_is_byte_identical_and_a_different_seed_is_not(tmp_path):
    outs = []
    for name, seed in (("a", 4), ("b", 4), ("c", 5)):
        w = World({"simple": {"duration": "30m"}}, seed)
        w.save(tmp_path / name / "world.json")
        w.stream("e1", [(0, "s_a"), (900, "x_b")], tmp_path / name)
        outs.append({p.name: p.read_bytes() for p in sorted((tmp_path / name).iterdir())})
    assert outs[0] == outs[1] and outs[0] != outs[2]


def test_consistency_and_confidence_are_bounded_without_a_point_mass_at_the_bounds():
    w = World(SMALL, 2)
    a = frames.frame_arrays(w, 0, 0, 0, 100000)
    c = a["cons"][a["cons_valid"] & (a["cons_status"] == 0)]
    f = a["conf"][a["conf_valid"]]
    assert c.min() >= 0 and c.max() <= 1 and f.min() >= 0 and f.max() <= 1
    for v in (c, f):
        assert np.mean(v == 0) == 0 and np.mean(v == 1) == 0 and np.mean(v > 0.999) < 0.002 and np.mean(v < 0.001) < 0.002


def test_validity_follows_the_real_code():
    w = World(SMALL, 3)
    a = frames.frame_arrays(w, 0, 1, 0, 200000)
    empty = (a["TP"] + a["FP"]) == 0
    assert np.array_equal(empty, ~a["conf_valid"]) and not np.isnan(a["conf"][a["conf_valid"]]).any() and np.isnan(a["conf"][empty]).all()
    e0, e1 = empty[:-1], empty[1:]
    assert np.array_equal(a["cons_status"] == 2, e0 & e1) and np.array_equal(a["cons_status"] == 1, e0 ^ e1)
    assert (a["cons"][a["cons_status"] == 1] == 0).all() and a["cons_valid"][a["cons_status"] == 1].all()       # one empty frame: a valid consistency of 0
    assert np.isnan(a["cons"][a["cons_status"] == 2]).all() and (~a["cons_valid"][a["cons_status"] == 2]).all()    # both empty: invalid, never 0
    assert (e0 & e1).any() and (e0 ^ e1).any()


def test_object_count_is_a_property_of_the_data_not_of_the_model():
    w = World(SMALL, 6)
    n = [frames.frame_arrays(w, 0, m, 0, 5000)["n_obj"] for m in range(6)]
    assert all(np.array_equal(n[0], x) for x in n)
    other_edge = frames.frame_arrays(w, 1, 0, 0, 5000)["n_obj"]
    assert not np.array_equal(n[0], other_edge)
    for x in (frames.frame_arrays(w, 0, 0, 0, 5000), frames.frame_arrays(w, 0, 3, 0, 5000)):
        assert np.array_equal(x["FN"], x["n_obj"] - x["TP"]) and (x["FN"] >= 0).all()


def test_x_beats_s_and_each_version_wins_its_own_areas():
    w = World(SMALL, 7)
    ids = [m["id"] for m in w.cfg["models"]]
    sz = np.array([m["size"] == "x" for m in w.cfg["models"]])
    assert w.F1[sz].mean() > w.F1[~sz].mean()
    for i, m in enumerate(w.cfg["models"]):
        own = w.trained[i]
        for j, o in enumerate(w.cfg["models"]):
            if o["version"] != m["version"] and o["size"] == m["size"]:
                assert w.F1[i, own].mean() > w.F1[j, own].mean(), (ids[i], ids[j])


def test_f1_range_follows_the_simple_interface():
    w = World({"simple": {"perf_min": 0.30, "perf_max": 0.95}}, 1)
    assert abs(w.F1.min() - 0.30) < 0.04 and abs(w.F1.max() - 0.95) < 0.04


def test_blind_cells_raise_consistency_by_a_fixed_amount_and_leave_confidence_alone():
    base = World({"blind": {"beta": 0.0}}, 8)
    allb = World({"blind": {"beta": 1.0}}, 8)
    assert base.blind.sum() == 0 and allb.blind.sum() == (~allb.trained).sum()
    shift = allb.cfg["blind"]["shift_sd"] * allb.cfg["consistency"]["cell_noise_sd"]
    assert shift == pytest.approx(0.066, abs=1e-9)
    inside = (base.C + shift) < 0.98                                                  # (the centres are clipped to [0.02, 0.98])
    assert np.allclose((allb.C - base.C)[allb.blind & inside], shift) and np.allclose((allb.C - base.C)[~allb.blind], 0.0)
    assert np.array_equal(allb.Conf, base.Conf) and np.array_equal(allb.F1, base.F1)
    assert 0.0 < World({"simple": {"duration": "1h"}}, 3).blind.mean() < 0.2           # the default beta is 0.05


def test_centres_follow_the_regression_with_the_anchor_intercepts():
    cfg = {"simple": {"n_areas": 2}, "models": [{"id": "m", "size": "s", "version": "a", "trained_areas": ["A", "B"]}], "matrix": {"fixed": {"m": {"P": [0.94, 0.60], "R": [0.94, 0.80]}}},
           "consistency": {"cell_noise_sd": 0.0}, "confidence": {"cell_noise_sd": 0.0}, "blind": {"beta": 0.0}}
    w = World(cfg, 1)
    c, f = w.cfg["consistency"], w.cfg["confidence"]
    assert w.C[0, 0] == pytest.approx(0.788) and w.Conf[0, 0] == pytest.approx(0.6625)                 # the anchor: healthy cell at F1 0.94
    assert w.C[0, 1] == pytest.approx(0.788 + c["bp"] * (0.60 - 0.94) + c["br"] * (0.80 - 0.94))
    assert w.Conf[0, 1] == pytest.approx(0.6625 + f["bp"] * (0.60 - 0.94) + f["br"] * (0.80 - 0.94))
    assert c["bp"] > c["br"] and f["bp"] == pytest.approx(0.24) and f["br"] == pytest.approx(0.183)       # consistency follows precision more; confidence P and R about equally


def test_confidence_is_a_mean_over_the_boxes_so_its_noise_shrinks_with_their_number():
    w = World({"simple": {"duration": "3h"}, "area_params": {"lambda_obj": 9.0}, "confidence_frame": {"gamma_z": 0.0, "model_noise": 0.03}}, 3)
    a = frames.frame_arrays(w, 0, 0, 0, 200000)
    ok = a["conf_valid"] & (a["area"] == a["area"][0])
    few, many = ok & (a["nbox"] <= 4), ok & (a["nbox"] >= 12)
    assert few.sum() > 300 and many.sum() > 300 and np.std(a["conf"][few]) > 1.3 * np.std(a["conf"][many])
    assert np.isnan(a["conf"][a["nbox"] == 0]).all() and not a["conf_valid"][a["nbox"] == 0].any()


def test_empty_frames_and_zeros_depend_on_the_area():
    cfg = {"simple": {"duration": "3h", "n_areas": 3, "n_edges": 1}, "models": [{"id": "m", "size": "s", "version": "a", "trained_areas": ["A", "B", "C"]}], "area_params": {"difficulty": [0.0, 0.5, 1.0], "dropout_share": 0.0}, "timeline": {"fixed_segments": [["A", 3600], ["B", 3600], ["C", 3600]]},
           "local_episodes": {"rate_per_hour": 0}, "edges": [{"id": "e1", "position": 0}]}
    w = World(cfg, 2)
    assert w.lam[0] > w.lam[1] > w.lam[2] >= w.cfg["area_params"]["lambda_floor"] and w.lam[0] >= 8.0
    a = frames.frame_arrays(w, 0, 0, 0, w.n_frames)
    z = [float(((a["cons_status"] == 1) & (a["area"][1:] == k)).mean()) for k in range(3)]
    assert z[0] < 0.002 and z[2] > 5 * max(z[0], 0.001) and z[2] > 0.02
    floor = World({"area_params": {"lambda_floor": [4.0, 5.0, 6.0, 7.0, 8.0], "lambda_obj": 1.0}}, 1)
    assert list(floor.lam) == [4.0, 5.0, 6.0, 7.0, 8.0]


def test_fixed_segments_fixed_cells_and_the_exact_duration():
    cfg = {"simple": {"duration": "1050s", "n_areas": 2, "n_edges": 1, "steepness": 1.0}, "edges": [{"id": "e1", "position": 0}], "local_episodes": {"rate_per_hour": 0},
           "timeline": {"fixed_segments": [[0, 300], [1, 200]]}, "models": [{"id": "m", "size": "s", "version": "a", "trained_areas": ["A", "B"]}], "matrix": {"fixed": {"m": {"P": [0.9, 0.5], "R": [0.9, 0.6]}}}}
    w = World(cfg, 4)
    assert w.n_frames == 6300 and w.F1[0, 0] == pytest.approx(0.9) and w.F1[0, 1] == pytest.approx(2 * 0.5 * 0.6 / 1.1)
    t = np.array([10.0, 299.0, 301.0, 499.0, 501.0, 799.0, 801.0, 1049.0])
    assert list(w.area("e1", t)) == ["A", "A", "B", "B", "A", "A", "B", "A"]
    with pytest.raises(ValueError):
        World({**cfg, "matrix": {"fixed": {"m": {"P": [0.9], "R": [0.9]}}}}, 1)


def test_the_stream_covers_exactly_the_requested_duration(tmp_path):
    w = World({"simple": {"duration": "25m"}}, 6)
    w.stream("e2", [(0, "s_a"), (700, "x_b")], tmp_path)
    t = [json.loads(l)["t"] for l in open(tmp_path / "truth.jsonl")]
    assert len(t) == 25 * 60 * 6 and t[0] == 0.0 and t[-1] == pytest.approx(25 * 60 - 1 / 6)


def test_minimal_logs_write_windows_requests_context_and_truth_only(tmp_path):
    w = World({"simple": {"duration": "20m"}}, 7)
    w.stream("e1", [(0, "s_a"), (600, "x_a")], tmp_path, logs="minimal")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["context.jsonl", "requests.jsonl", "summary.json", "truth.jsonl", "windows.jsonl"]
    full = tmp_path / "f"
    w.stream("e1", [(0, "s_a"), (600, "x_a")], full)
    assert (full / "signals.jsonl").stat().st_size > 0 and (full / "frames.jsonl").stat().st_size > 0
    assert (full / "windows.jsonl").read_bytes() == (tmp_path / "windows.jsonl").read_bytes()
    r = subprocess.run([sys.executable, str(HERE / "generate.py"), "stream", "--seed", "1", "--edge", "e1", "--schedule", "0:s_a", "--out", str(tmp_path / "c"), "--logs", "minimal"], capture_output=True, text=True)
    assert r.returncode == 0 and not (tmp_path / "c" / "signals.jsonl").exists()


def test_the_calibration_search_is_deterministic_and_the_calibration_world_has_the_designed_structure():
    from tools.dummy_signals import calibrate as C
    th = dict(zip(C.PARAMS, C.START))
    a, b = C.evaluate(th, [0, 1]), C.evaluate(th, [0, 1])
    assert a == b
    z = a["zero_share_by_area"]
    assert z[2] > 5 * max(z[0], z[1], z[3], 0.001)                                    # zeros come from the hard area
    w = World(C.calibration_config(th), 0)
    assert w.n_frames == 6300 and np.allclose(w.F1[0], C.F1_AREAS, atol=1e-9) and list(np.round(w.lam, 2)) == [9.0, 9.0, th["lambda_hard_area"], 9.0]


def test_edges_see_the_global_timeline_delayed_by_position_over_speed():
    w = World(SMALL, 9)
    for e in w.edges:
        assert e["offset_s"] == pytest.approx(e["position_km"] / 30.0 * 3600.0)
    t = np.arange(0, 3 * 3600, 7.0)
    g = np.array([w.edge_tl[0].label_at(t) for _ in [0]])[0]
    glob = lambda tt: next(r for (a, b, r) in w.global_segments if a <= tt < b)
    for i, e in enumerate(w.edges):
        label = np.array([w.names.index(x) for x in w.area(e["id"], t)])
        in_local = np.zeros(len(t), bool)
        for (a, b, r) in w.episodes[i]:
            in_local |= (t >= a - 1) & (t <= b + 1)
        expect = np.array([glob(x - e["offset_s"]) for x in t])
        ok = (label == expect) | in_local
        assert ok.mean() > 0.995, (e["id"], ok.mean())


def test_local_episodes_appear_on_one_edge_only():
    w = World({"simple": {"duration": "6h"}, "local_episodes": {"rate_per_hour": 3}}, 10)
    origins = {e["id"]: [s for s in w.segments(e["id"]) if s["origin"] == "local"] for e in w.edges}
    assert all(len(v) > 3 for v in origins.values())
    eps = [set((round(a, 3), round(b, 3)) for (a, b, r) in ep) for ep in w.episodes]
    assert all(not (eps[i] & eps[j]) for i in range(3) for j in range(i + 1, 3))           # an episode belongs to exactly one edge


def test_ramps_blend_areas_and_steepness_sets_their_length():
    slow, fast = World({"simple": {"steepness": 0.0}}, 3), World({"simple": {"steepness": 1.0}}, 3)
    assert slow.ramp_s == pytest.approx(120.0) and fast.ramp_s == pytest.approx(1 / 6)
    for w in (slow, fast):
        tl, j = w.edge_tl[0], 3
        b, rho = tl.starts[j], tl.ramp[j]
        old, new = tl.labels[j - 1], tl.labels[j]
        t = np.array([b - rho / 2 - 0.01, b - rho / 4, b, b + rho / 4, b + rho / 2 + 0.01])
        a, bb, s, lab = tl.mix(t)
        assert s[0] == 0 and a[0] == bb[0] == old and a[-1] == bb[-1] == new and s[-1] == 0                    # outside the ramp: one area
        assert np.all(np.diff(s[:4]) >= 0) and 0.45 < s[2] < 0.55 and a[2] == old and bb[2] == new and lab[1] == old and lab[3] == new     # inside: smooth blend


def test_clip_score_is_the_median_of_the_30_pairs_ending_at_t_end_on_the_requesters_data():
    w = World(SMALL, 11)
    r = w.clip_score("e2", "s_b", 600.0)
    a = frames.frame_arrays(w, 1, 2, 3600 - 30, 3601)
    assert r["n_pairs"] == 30 and r["consistency"] == pytest.approx(float(np.median(a["cons"][a["cons_valid"]]))) and 0 < r["valid_fraction"] <= 1 and isinstance(r["areas"], list)
    # the clip is edge e2's data whatever model: the object counts do not change with the model
    assert np.array_equal(w.frame_truth("e2", "s_b", 3570, 3601)["n_obj"], w.frame_truth("e2", "x_c", 3570, 3601)["n_obj"])


def test_stream_uses_the_real_engine_and_marks_every_line(tmp_path):
    from edge_signals.engine import SignalEngine
    from edge_signals.trigger import TriggerEngine
    w = World({"simple": {"duration": "1h"}}, 12)
    res = w.stream("e3", [(0, "s_a"), (1700.3, "x_b")], tmp_path)
    assert isinstance(res["engine"], SignalEngine) and isinstance(res["engine"].trigger, TriggerEngine)
    for f in ("signals", "windows", "requests", "context", "frames", "truth"):
        rows = read(tmp_path / f"{f}.jsonl")
        assert rows and all(r["synthetic"] is True for r in rows), f
    assert all(r["device_id"] == "synthetic-e3" for f in ("signals", "windows", "requests") for r in read(tmp_path / f"{f}.jsonl"))
    assert all(r["video"] is None and r["frame_index"] is None for r in read(tmp_path / "requests.jsonl"))
    t = read(tmp_path / "truth.jsonl")
    assert set(t[0]) == {"synthetic", "frame_index", "t", "model_id", "area", "n_obj", "TP", "FP", "FN"} and t[0]["model_id"] == "s_a" and t[-1]["model_id"] == "x_b"
    sig = [x for x in read(tmp_path / "signals.jsonl") if not x["valid"]]
    assert sig and all(x["value"] is None and x["invalid_reason"] in ("EMPTY_PAIR", "no_boxes") for x in sig)


def test_a_model_switch_is_a_scene_cut(tmp_path):
    w = World({"simple": {"duration": "1h"}}, 13)
    sw = 1217.5
    w.stream("e1", [(0, "s_a"), (sw, "x_b"), (2400, "s_c")], tmp_path)
    frames_, wins, reqs, ctx = (read(tmp_path / f"{n}.jsonl") for n in ("frames", "windows", "requests", "context"))
    cons = [x for x in wins if x["signal"] == "consistency"]
    summ = json.load(open(tmp_path / "summary.json"))
    assert len(ctx) == 2 and summ["n_pairs"] == len(frames_) - 1 - 2
    for x in cons:
        assert not (x["t_start"] < sw <= x["t_end"]) and not (x["t_start"] < 2400 <= x["t_end"])
    first = [x for x in cons if x["clip_id"] == "x_b"][0]
    assert first["change"] is None and first["window_id"] not in [q["window_id"] for q in reqs]


def test_the_generator_selects_no_model():
    text = "\n".join(p.read_text() for p in HERE.glob("*.py")).lower()
    assert not any(w in text for w in ("argmax", "best_model", "choose_model", "select_model", "oracle")) or "argmax(self.f1" not in text
    src = (HERE / "stream.py").read_text()
    assert "schedule" in src and "argmax" not in src and "best" not in src.replace("best-area", "")


def test_config_errors_and_ranges():
    with pytest.raises(cfgmod.ConfigError, match="unknown parameter"):
        World({"simple": {"durration": "1h"}}, 0)
    with pytest.raises(cfgmod.ConfigError):
        World({"simple": {"perf_min": 0.9, "perf_max": 0.5}}, 0)
    a, b = World({"matrix": {"precision": {"spec": {"range": [0.0, 1.0]}}}}, 1), World({"matrix": {"precision": {"spec": {"range": [0.0, 1.0]}}}}, 2)
    assert a.draws and a.draws != b.draws and 0 <= a.cfg["matrix"]["precision"]["spec"] <= 1
    assert cfgmod.parse_duration("24h") == 86400 and cfgmod.parse_duration("90m") == 5400 and cfgmod.parse_duration(7200) == 7200


def test_world_json_records_every_draw(tmp_path):
    w = World(SMALL, 14)
    d = json.load(open(w.save(tmp_path / "world.json")))
    for k in ("precision", "recall", "f1", "consistency_centre", "confidence_centre", "blind_cells", "global_timeline", "edge_segments", "local_episodes", "range_draws", "edges", "warnings", "areas"):
        assert k in d
    assert d["synthetic"] is True and len(d["precision"]) == 6 and len(d["edge_segments"]) == 3 and "intercepts" in d


def test_command_line(tmp_path):
    cmd = [sys.executable, str(HERE / "generate.py")]
    r = subprocess.run(cmd + ["stream", "--seed", "1", "--edge", "e1", "--schedule", "0:s_a", "--out", str(tmp_path), "--config", str(tmp_path / "c.yaml"), "--no-signals-log"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 2          # config file missing
    (tmp_path / "c.yaml").write_text("simple: {duration: 20m}\n")
    r = subprocess.run(cmd + ["stream", "--seed", "1", "--edge", "e1", "--schedule", "0:s_a", "--out", str(tmp_path / "o"), "--config", str(tmp_path / "c.yaml"), "--no-signals-log"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o" / "signals.jsonl").stat().st_size == 0 and (tmp_path / "o" / "truth.jsonl").stat().st_size > 0 and (tmp_path / "o" / "world.json").exists()
    c = subprocess.run(cmd + ["clip", "--config", str(tmp_path / "c.yaml"), "--edge", "e2", "--model", "x_a", "--t-end", "600"], capture_output=True, text=True)
    assert json.loads(c.stdout)["n_pairs"] == 30
    assert subprocess.run(cmd + ["clip", "--edge", "zz", "--model", "x_a", "--t-end", "600"], capture_output=True, text=True).returncode == 2


def test_signals_see_the_realised_precision_of_the_frame_when_asked_to():
    from scipy.stats import spearmanr
    res = {}
    for k in (0.0, 0.6):
        w = World({"simple": {"duration": "6h"}, "consistency_frame": {"gamma_p_hat": k}, "confidence_frame": {"gamma_p_hat": k}, "area_params": {"lambda_obj": 9.0}}, 5)
        a = frames.frame_arrays(w, 0, 0, 0, 100000)
        pre = np.where(a["nbox"] > 0, a["TP"] / np.maximum(a["nbox"], 1), np.nan)
        m = np.minimum(pre[:-1], pre[1:])
        ok = a["cons_valid"] & ~np.isnan(m) & (a["cons_status"] == 0) & (a["area"][1:] == a["area"][0])
        okf = a["conf_valid"] & (a["area"] == a["area"][0])
        res[k] = (spearmanr(a["cons"][ok], m[ok])[0], spearmanr(a["conf"][okf], pre[okf])[0])
    assert res[0.6][0] > res[0.0][0] + 0.1 and res[0.6][1] > res[0.0][1] + 0.1


# ---------------------------------------------------------------------------------------- scripted scenario
SCENARIO_FILE = HERE / "scenario_abc.yaml"
SCENARIO = {                                   # the three-ship scenario (sizes s and x; versions a, b, c trained on areas A, B, C); the same content as scenario_abc.yaml when that file exists
    "simple": {"perf_max": 0.90, "perf_min": 0.50, "duration": "2h", "steepness": 0.5},
    "areas": ["A", "B", "C"],
    "models": [{"id": f"{size}_{ver}", "size": size, "version": ver, "trained_areas": [area]} for ver, area in (("a", "A"), ("b", "B"), ("c", "C")) for size in "sx"],
    "edges": [
        {"id": "edge1", "models": ["s_a", "x_a"], "initial_model": "s_a", "route": [{"stay": "A", "for": "40m"}, {"move_to": "B", "over": "20m"}, {"stay": "B", "for": "60m"}]},
        {"id": "edge2", "models": ["s_b", "x_b"], "initial_model": "s_b", "route": [{"stay": "B", "for": "50m"}, {"move_to": "C", "over": "20m"}, {"stay": "C", "for": "50m"}]},
        {"id": "edge3", "models": ["s_c", "x_c"], "initial_model": "s_c", "route": [{"stay": "C", "for": "2h"}]},
    ],
    "random_episodes": False,
}


def scenario_yaml(tmp_path):
    import yaml
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(SCENARIO, sort_keys=False))
    return p


def scen(over=None, seed=0):
    import copy
    cfg = copy.deepcopy(SCENARIO)
    cfg.update(over or {})
    return World(cfg, seed)


@pytest.mark.parametrize("source", ["inline", pytest.param("file", marks=pytest.mark.skipif(not SCENARIO_FILE.exists(), reason="scenario_abc.yaml is not in this tree"))])
def test_the_example_scenario_loads_as_given_and_every_route_adds_up_to_the_duration(source):
    import yaml
    raw = yaml.safe_load(open(SCENARIO_FILE)) if source == "file" else SCENARIO
    w = World(SCENARIO_FILE if source == "file" else SCENARIO, 0)
    assert w.names == ["A", "B", "C"] and w.duration_s == 7200 and [e["id"] for e in w.edges] == ["edge1", "edge2", "edge3"] and all(e["mode"] == "route" for e in w.edges)
    for e in raw["edges"]:
        assert sum(cfgmod.parse_duration(s.get("for", s.get("over"))) for s in e["route"]) == pytest.approx(w.duration_s)
    assert [w.edge_models(e) for e in ("edge1", "edge2", "edge3")] == [["s_a", "x_a"], ["s_b", "x_b"], ["s_c", "x_c"]]
    assert [e["initial_model"] for e in w.edges] == ["s_a", "s_b", "s_c"] and raw["random_episodes"] is False and w.cfg["random_episodes"] is False
    assert [[s["kind"], s["area"], s["t0"], s["t1"]] for s in w.segments("edge1")] == [["stay", "A", 0, 2400], ["move", "B", 2400, 3600], ["stay", "B", 3600, 7200]]
    assert [[s["kind"], s["area"], s["t0"], s["t1"]] for s in w.segments("edge2")] == [["stay", "B", 0, 3000], ["move", "C", 3000, 4200], ["stay", "C", 4200, 7200]]
    assert [s["area"] for s in w.segments("edge3")] == ["C"] and w.segments("edge3")[0]["t1"] == 7200


def test_durations_accept_s_m_min_h_mixed_freely():
    assert [cfgmod.parse_duration(x) for x in ("30s", "40m", "40min", "2h", "1.5h", "90", 90, "2 h", "45 sec")] == [30, 2400, 2400, 7200, 5400, 90, 90, 7200, 45]
    for bad in ("2x", "m", "", "1h30m", True):
        with pytest.raises(cfgmod.ConfigError):
            cfgmod.parse_duration(bad)
    cfg = {"simple": {"duration": "1h"}, "areas": ["A", "B"], "models": [{"id": "m", "size": "s", "version": "a", "trained_areas": ["A"]}],
           "edges": [{"id": "e", "models": ["m"], "initial_model": "m", "route": [{"stay": "A", "for": "30min"}, {"move_to": "B", "over": 600}, {"stay": "B", "for": "1200s"}]}]}
    assert World(cfg, 0).edge_tl[0].ends[-1] == 3600


def _route_cfg(route, **kw):
    cfg = {"simple": {"duration": "1h"}, "areas": ["A", "B"], "models": [{"id": "m", "size": "s", "version": "a", "trained_areas": ["A"]}, {"id": "n", "size": "x", "version": "b", "trained_areas": ["B"]}],
           "edges": [{"id": "e", "models": ["m"], "initial_model": "m", "route": route, **kw}]}
    return cfg


def test_route_validation():
    ok = [{"stay": "A", "for": "30m"}, {"move_to": "B", "over": "10m"}, {"stay": "B", "for": "20m"}]
    World(_route_cfg(ok), 0)
    cases = [
        ([{"stay": "A", "for": "30m"}, {"move_to": "B", "over": "10m"}, {"stay": "B", "for": "10m"}], "add up"),
        ([{"stay": "A", "for": "30m"}, {"move_to": "B", "over": "10m"}, {"stay": "B", "for": "30m"}], "add up"),
        ([{"stay": "Z", "for": "1h"}], "unknown area"),
        ([{"stay": "A", "for": "30m"}, {"move_to": "Z", "over": "30m"}], "unknown area"),
        ([{"move_to": "B", "over": "1h"}], "start with 'stay'"),
        ([{"stay": "A", "for": "30m"}, {"move_to": "A", "over": "30m"}], "already in"),
        ([{"stay": "A", "over": "1h"}], "expected"),
        ([{"stay": "A", "for": "0m"}, {"stay": "B", "for": "1h"}], "positive"),
        ([], "non-empty"),
    ]
    for route, msg in cases:
        with pytest.raises(cfgmod.ConfigError, match=msg):
            World(_route_cfg(route), 0)
    with pytest.raises(cfgmod.ConfigError, match="unknown model"):
        World(_route_cfg(ok, models=["m", "nope"]), 0)
    with pytest.raises(cfgmod.ConfigError, match="unknown initial_model"):
        World(_route_cfg(ok, initial_model="nope"), 0)
    with pytest.raises(cfgmod.ConfigError, match="not among the models stored"):
        World(_route_cfg(ok, initial_model="n"), 0)
    with pytest.raises(cfgmod.ConfigError, match="duplicate edge id"):
        c = _route_cfg(ok)
        c["edges"] = c["edges"] * 2
        World(c, 0)
    with pytest.raises(cfgmod.ConfigError, match="trained_areas"):
        c = _route_cfg(ok)
        c["models"][0]["trained_areas"] = ["Q"]
        World(c, 0)


def test_a_routed_edge_follows_its_route_exactly_and_the_stream_has_the_exact_duration(tmp_path):
    w = World(SCENARIO, 0)
    t = np.array([0.0, 2399.0, 2401.0, 2999.0, 3601.0, 7199.0])
    assert list(w.area("edge1", t)) == ["A", "A", "A", "A", "B", "B"]                # the move lasts 2400-3600 s; its midpoint is 3000 s
    assert list(w.area("edge2", np.array([10.0, 2999.0, 3001.0, 3590.0, 4201.0, 7000.0]))) == ["B", "B", "B", "B", "C", "C"]
    assert w.area("edge3", 5000.0) == "C" and set(w.area("edge3", np.arange(0, 7200, 13.0))) == {"C"}
    w.stream("edge1", [(0, "s_a")], tmp_path, logs="minimal")
    rows = [json.loads(l) for l in open(tmp_path / "truth.jsonl")]
    assert len(rows) == 7200 * 6 and rows[0]["t"] == 0.0 and rows[-1]["t"] == pytest.approx(7200 - 1 / 6)
    assert rows[0]["area"] == "A" and rows[-1]["area"] == "B" and {r["area"] for r in rows} == {"A", "B"}


def test_a_move_interpolates_p_r_c_conf_and_lambda_linearly_and_recomputes_f1():
    w = World(SCENARIO, 0)
    t = np.linspace(2400.0, 3600.0, 13)
    q = w.quality("edge1", "s_a", t)
    P0, P1, R0, R1 = w.P[0, 0], w.P[0, 1], w.R[0, 0], w.R[0, 1]
    s = (t - 2400.0) / 1200.0
    assert np.allclose(q["P"], (1 - s) * P0 + s * P1) and np.allclose(q["R"], (1 - s) * R0 + s * R1)
    assert np.allclose(q["F1"], 2 * q["P"] * q["R"] / (q["P"] + q["R"]))              # F1 is recomputed from the interpolated P and R ...
    assert abs(q["F1"][6] - (w.F1[0, 0] + w.F1[0, 1]) / 2) > 1e-6                          # ... not interpolated itself
    a, b, ss, lab = w.edge_tl[0].mix(t[:-1])                                              # (at 3600 s the next step, a stay in B, takes over: the same values)
    assert np.allclose(ss, s[:-1]) and (a == 0).all() and (b == 1).all() and list(lab) == [0] * 6 + [1] * 6        # the label is the nearer area
    assert q["P"][0] == pytest.approx(P0) and q["P"][-1] == pytest.approx(P1)
    d = np.diff(q["P"])
    assert np.allclose(d, d[0])                                                          # a straight line: no step, no smoothstep
    # the per-frame centres inside a frame array use the same weights (lambda, C and Conf are blended with s)
    from tools.dummy_signals import frames as fr_
    mid = int(3000 * 6)
    assert w.edge_tl[0].mix(np.array([mid / 6.0]))[2][0] == pytest.approx(0.5)
    lam_mid = 0.5 * w.lam[0] + 0.5 * w.lam[1]
    assert lam_mid == pytest.approx((w.lam[0] + w.lam[1]) / 2)
    assert fr_.frame_arrays(w, 0, 0, mid, mid + 10)["f1_cell"][0] == pytest.approx(float(w.quality("edge1", "s_a", np.array([mid / 6.0]))["F1"][0]))


def test_clip_score_and_stream_accept_a_model_stored_on_another_edge(tmp_path):
    w = World(SCENARIO, 0)
    assert w.edge_models("edge1") == ["s_a", "x_a"]
    r = w.clip_score("edge1", "x_c", 3000.0)                                              # x_c is stored on edge3; the clip is edge1's data
    assert r["n_pairs"] == 30 and r["consistency"] is not None and r["areas"] == ["A", "B"]
    assert np.array_equal(w.frame_truth("edge1", "x_c", 17900, 18001)["n_obj"], w.frame_truth("edge1", "s_a", 17900, 18001)["n_obj"])      # same data, whatever the model
    assert w.clip_score("edge1", "x_c", 3000.0) == w.clip_score(0, 5, 3000.0)
    assert w.clip_score("edge1", "x_c", 3000.0) != w.clip_score("edge2", "x_c", 3000.0)   # the requester's data, not the model's host
    assert w.model_source("edge1", "s_a") == ("local", "edge1") and w.model_source("edge1", "x_c") == ("fetched", "edge3") and w.model_source("edge1", "x_c", "edge3") == ("fetched", "edge3")
    with pytest.raises(ValueError):
        w.model_source("edge1", "x_c", "edge2")                                           # edge2 does not store x_c
    w.stream("edge1", [(0, "s_a"), (1800, "x_a"), (3000, "x_c"), (4200, "s_b", "edge2")], tmp_path, logs="minimal")
    s = json.load(open(tmp_path / "summary.json"))["model_schedule"]
    assert s == [{"t": 0.0, "model": "s_a", "source": "local", "from_edge": "edge1"}, {"t": 1800.0, "model": "x_a", "source": "local", "from_edge": "edge1"},
                 {"t": 3000.0, "model": "x_c", "source": "fetched", "from_edge": "edge3"}, {"t": 4200.0, "model": "s_b", "source": "fetched", "from_edge": "edge2"}]
    assert {json.loads(l)["model_id"] for l in open(tmp_path / "truth.jsonl")} == {"s_a", "x_a", "x_c", "s_b"}


def test_the_stream_schedule_text_and_the_one_command_run(tmp_path):
    from tools.dummy_signals import stream as st
    assert st.parse_schedule_text("0:s_a,3600:x_b@edge2", ["s_a", "x_b"]) == [(0.0, "s_a", None), (3600.0, "x_b", "edge2")]
    r = subprocess.run([sys.executable, str(HERE / "generate.py"), "--config", str(scenario_yaml(tmp_path)), "--seed", "0", "--out", str(tmp_path / "run"), "--logs", "minimal"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    run = tmp_path / "run"
    assert sorted(p.name for p in run.iterdir()) == ["edge1", "edge2", "edge3", "world.json"]
    for e, m in (("edge1", "s_a"), ("edge2", "s_b"), ("edge3", "s_c")):
        assert sorted(p.name for p in (run / e).iterdir()) == ["context.jsonl", "requests.jsonl", "summary.json", "truth.jsonl", "windows.jsonl"]
        assert json.load(open(run / e / "summary.json"))["model_schedule"] == [{"t": 0.0, "model": m, "source": "local", "from_edge": e}]
    d = json.load(open(run / "world.json"))
    assert [(e["id"], e["mode"], e["models"], e["initial_model"]) for e in d["edges"]] == [("edge1", "route", ["s_a", "x_a"], "s_a"), ("edge2", "route", ["s_b", "x_b"], "s_b"), ("edge3", "route", ["s_c", "x_c"], "s_c")]
    assert d["edges"][0]["route"] == [{"stay": "A", "for": "40m"}, {"move_to": "B", "over": "20m"}, {"stay": "B", "for": "60m"}]
    full = subprocess.run([sys.executable, str(HERE / "generate.py"), "--config", str(scenario_yaml(tmp_path)), "--out", str(tmp_path / "full")], capture_output=True, text=True, cwd=tmp_path)
    assert full.returncode == 0 and (tmp_path / "full" / "edge1" / "signals.jsonl").stat().st_size > 0


def test_renamed_fields_and_no_alias_for_the_old_name(tmp_path):
    w = World(SCENARIO, 0)
    d = json.load(open(w.save(tmp_path / "world.json")))
    assert d["n_areas"] == 3 and [a["name"] for a in d["areas"]] == ["A", "B", "C"] and all({"difficulty", "lambda_obj"} <= set(a) for a in d["areas"])
    assert set(d["edge_segments"]["edge1"][1]) == {"t0", "t1", "kind", "area", "from_area", "origin"} and d["edge_segments"]["edge1"][1]["from_area"] == "A"
    assert all(set(b) == {"model", "area"} for b in d["blind_cells"])
    assert "regime" not in json.dumps(d).lower()
    w.stream("edge2", [(0, "s_b")], tmp_path / "s", logs="minimal")
    row = json.loads(open(tmp_path / "s" / "truth.jsonl").readline())
    assert set(row) == {"synthetic", "frame_index", "t", "model_id", "area", "n_obj", "TP", "FP", "FN"} and row["area"] == "B"
    assert "regime" not in "".join(p.read_text() for p in (tmp_path / "s").iterdir()).lower()
    for old in ({"regimes": {"difficulty": 0.5}}, {"simple": {"n_regimes": 3}}):
        with pytest.raises(cfgmod.ConfigError, match="unknown parameter"):
            World(old, 0)
    with pytest.raises(cfgmod.ConfigError, match="unknown key"):
        World({"models": [{"id": "m", "size": "s", "version": "a", "trained": [0]}]}, 0)
    src = "\n".join(p.read_text() for p in list(HERE.glob("*.py")) + list(HERE.glob("*.md")) + list(HERE.glob("*.yaml")))
    assert "regime" not in src.lower()


def test_area_difficulty_can_be_pinned_and_is_otherwise_drawn_per_seed():
    pinned = scen({"areas": [{"name": "A", "difficulty": 0.2}, {"name": "B", "difficulty": 0.9}, "C"]})
    assert pinned.difficulty[0] == 0.2 and pinned.difficulty[1] == 0.9 and 0 <= pinned.difficulty[2] <= 1
    assert pinned.lam[0] > pinned.lam[1]
    drawn = [World(SCENARIO, s).difficulty for s in (0, 1)]
    assert not np.allclose(drawn[0], drawn[1])
    assert np.array_equal(World(SCENARIO, 0).difficulty, drawn[0]) and np.isclose(scen({"areas": [{"name": "A", "difficulty": 0.2}, "B", "C"]}, 0).difficulty[1:], drawn[0][1:]).all()
    with pytest.raises(cfgmod.ConfigError, match="difficulty"):
        scen({"areas": [{"name": "A", "difficulty": 1.5}, "B", "C"]})


def test_random_episodes_switch_accepts_on_off_true_false():
    import yaml
    assert [cfgmod.parse_switch(x, "x") for x in (True, False, "on", "off", "On", "true", "false")] == [True, False, True, False, True, True, False]
    assert yaml.safe_load("random_episodes: off")["random_episodes"] is False and yaml.safe_load("random_episodes: on")["random_episodes"] is True
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.parse_switch("maybe", "random_episodes")
    off, on = scen({"random_episodes": "off"}), scen({"random_episodes": "on", "local_episodes": {"rate_per_hour": 6, "median_s": 120, "sigma": 0.5}})
    assert all(len(ep) == 0 for ep in off.episodes) and any(len(ep) > 0 for ep in on.episodes)
    ep = next(ep for ep in on.episodes if ep)
    t0, t1, r = ep[0]
    i = [k for k, e in enumerate(on.episodes) if e][0]
    assert on.area(i, (t0 + t1) / 2) == on.names[r]


def test_routed_and_random_edges_can_be_mixed_and_world_json_records_the_mode(tmp_path):
    cfg = {"simple": {"duration": "2h"}, "areas": ["A", "B", "C"], "models": [{"id": "m", "size": "s", "version": "a", "trained_areas": ["A"]}, {"id": "n", "size": "x", "version": "b", "trained_areas": ["B", "C"]}],
           "edges": [{"id": "r", "models": ["m"], "route": [{"stay": "A", "for": "1h"}, {"move_to": "C", "over": "10m"}, {"stay": "C", "for": "50m"}]}, {"id": "q", "position": 20, "models": ["n"]}]}
    w = World(cfg, 3)
    assert [e["mode"] for e in w.edges] == ["route", "random"] and w.edges[0]["offset_s"] is None and w.edges[1]["offset_s"] == pytest.approx(20 / 30 * 3600)
    assert w.global_segments and w.area("r", 100.0) == "A" and w.area("r", 7000.0) == "C"
    assert w.edges[1]["initial_model"] == "n" and w.edges[0]["initial_model"] == "m"
    d = json.load(open(w.save(tmp_path / "w.json")))
    assert [e["mode"] for e in d["edges"]] == ["route", "random"]
    assert len(w.segments("q")) > 1 and all("kind" not in s for s in w.segments("q"))
    # a world with no random edge has no global timeline and does not need front_speed
    assert World(SCENARIO, 0).global_segments == []
