import json
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, asset
from edge_signals.adapt import annotate as A
from edge_signals.adapt import collect as C
from edge_signals.adapt import register as R
from edge_signals.adapt.common import load_cfg, read_stems
from edge_signals.adapt.retrain import check_recipe
from edge_signals.adapt.switch import switch


def stems(clip, n, start=0):
    return [f"{clip}-{i:07d}" for i in range(start, start + n)]


@pytest.fixture
def world(tmp_path):
    """Two adaptation clips, two replay clips, ground-truth label files for all of them."""
    adapt, rep = stems("nir1", 30) + stems("nir2", 20), stems("clr1", 40) + stems("clr2", 40)
    lab = tmp_path / "labels"
    lab.mkdir()
    for s in adapt + rep:
        (lab / f"{s}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (tmp_path / "adapt.txt").write_text("".join(s + "\n" for s in adapt))
    (tmp_path / "replay1.txt").write_text("".join(s + "\n" for s in rep[:40]))
    (tmp_path / "replay2.txt").write_text("".join(s + "\n" for s in rep[40:]))
    cfg = load_cfg("configs/adaptation.yaml", [f"collect.replay.lists=[{tmp_path / 'replay1.txt'}, {tmp_path / 'replay2.txt'}]",
                                               f"annotate.provided.labels_dir={lab}", "annotate.mode=provided"])
    return tmp_path, cfg, adapt, rep


def test_collect_from_a_frame_list_draws_replay_at_the_ratio_without_overlap(world):
    tmp, cfg, adapt, rep = world
    info = C.collect(cfg, tmp / "run", frames_list=tmp / "adapt.txt", ratio=0.5, seed=3)
    assert info["n_adapt"] == 50 and info["n_replay"] == 25 and info["source"]["kind"] == "frame_list"
    got = read_stems(tmp / "run/collect/replay.txt")
    assert len(set(got)) == 25 and not set(got) & set(adapt) and set(got) <= set(rep)
    C.collect(cfg, tmp / "run2", frames_list=tmp / "adapt.txt", ratio=0.5, seed=3)
    assert read_stems(tmp / "run2/collect/replay.txt") == got                      # same seed, same replay
    C.collect(cfg, tmp / "run3", frames_list=tmp / "adapt.txt", ratio=0.5, seed=4)
    assert read_stems(tmp / "run3/collect/replay.txt") != got
    with pytest.raises(SystemExit, match="replay lists hold"):
        C.collect(cfg, tmp / "run4", frames_list=tmp / "adapt.txt", ratio=2.0)      # 100 needed, 80 available


def test_collect_from_a_request_takes_the_frames_since_the_last_reset(world):
    tmp, cfg, adapt, rep = world
    stream = stems("nir1", 30) + stems("nir2", 20)
    (tmp / "stream.txt").write_text("".join(s + "\n" for s in stream))
    (tmp / "requests.jsonl").write_text(json.dumps({"request_id": "step-consistency-2", "t": 4.0, "rule": "step", "stage": "screening"}) + "\n" +
                                        json.dumps({"request_id": "step-consistency-7", "t": 6.0, "rule": "step", "stage": "screening"}) + "\n")
    info = C.collect(cfg, tmp / "r", requests=tmp / "requests.jsonl", request_id="step-consistency-2", stream_list=tmp / "stream.txt", fps=6.0, ratio=0)
    got = read_stems(tmp / "r/collect/adapt.txt")
    assert got == stream[:25] and info["source"]["request_id"] == "step-consistency-2"    # t = index / 6 <= 4.0 -> indices 0..24
    C.collect(cfg, tmp / "r2", requests=tmp / "requests.jsonl", stream_list=tmp / "stream.txt", since_t=4.0, ratio=0)   # default request = the last one
    assert read_stems(tmp / "r2/collect/adapt.txt") == stream[25:37]                     # 4.0 < t <= 6.0
    with pytest.raises(SystemExit, match="not found"):
        C.collect(cfg, tmp / "r3", requests=tmp / "requests.jsonl", request_id="nope", stream_list=tmp / "stream.txt")


def test_annotate_provided_copies_labels_logs_the_delay_and_reports_missing(world):
    tmp, cfg, adapt, rep = world
    cfg["annotate"]["provided"].update(delay_s_fixed=30.0, delay_s_per_frame=0.5)
    C.collect(cfg, tmp / "run", frames_list=tmp / "adapt.txt", ratio=0.5)
    info = A.annotate(cfg, tmp / "run", tmp)
    n = len(list((tmp / "run/annotate/labels").glob("*.txt")))
    assert n == 75 and info["simulated_annotation_delay_s"] == pytest.approx(30 + 0.5 * 50) and info["mode"] == "provided"
    (Path(cfg["annotate"]["provided"]["labels_dir"]) / f"{adapt[3]}.txt").unlink()
    with pytest.raises(SystemExit, match="no provided label"):
        A.annotate(cfg, tmp / "run", tmp)


def test_annotate_pseudo_needs_the_mounted_teacher(world):
    pytest.importorskip("ultralytics")
    tmp, cfg, adapt, rep = world
    C.collect(cfg, tmp / "run", frames_list=tmp / "adapt.txt", ratio=0)
    with pytest.raises(SystemExit, match="teacher weights not found"):
        A.annotate(cfg, tmp / "run", tmp, mode="pseudo", teacher=str(tmp / "missing.pt"))


def test_retrain_recipe_must_set_close_mosaic_explicitly():
    r = dict(load_cfg()["retrain"]["recipe"])
    assert check_recipe(dict(r))["close_mosaic"] == 0
    r.pop("close_mosaic")
    with pytest.raises(SystemExit, match="close_mosaic must be set explicitly"):
        check_recipe(r)
    r = dict(load_cfg()["retrain"]["recipe"], close_mosaic=-1)
    with pytest.raises(SystemExit, match="close_mosaic"):
        check_recipe(r)


def test_registry_entry_and_versions():
    ids = {"params": "p1", "state": "s1"}
    ref = {"feature_drift": {"reference_id": "abc", "backbone_params_id": "p1", "backbone_state_id": "s2"}}
    e = R.build_entry("maritime_s_base.v1", "maritime_s_base.v0", "step-consistency-9", {"epochs": 10}, "sha", "models/w.pt", ids, ref, stream_t=12.0)
    assert e["backbone_params_match"] is True and e["bn_stats_match"] is False and e["reference_ids"]["feature_drift"] == "abc" and e["request_id"] == "step-consistency-9"
    assert R.next_version({"versions": []}, "maritime_s_base.v0") == "maritime_s_base.v1"
    reg = {"versions": [{"version": "maritime_s_base.v1"}, {"version": "maritime_s_base.v2"}, {"version": "other.v7"}]}
    assert R.next_version(reg, "maritime_s_base.v1") == "maritime_s_base.v3"


def test_register_a_real_model_records_hashes_and_bn_stats_match(tmp_path):
    pytest.importorskip("ultralytics")
    cfg = load_cfg()
    ref = sorted((ROOT / "references").glob("feature_drift.*.npz"))[0]
    e = R.register(cfg, tmp_path, asset("maritime_s_base"), parent="maritime_s_base.v0", request_id="req-1", reference_path=ref)
    assert e["version"] == "maritime_s_base.v1" and (tmp_path / "models/maritime_s_base.v1.pt").exists()
    assert e["backbone_params_match"] is True and e["bn_stats_match"] is True            # the shipped model is the reference's model
    reg = json.loads((tmp_path / "registry.json").read_text())
    assert reg["versions"][0]["weights_sha256"] == e["weights_sha256"] and len(e["weights_sha256"]) == 64
    second = R.register(cfg, tmp_path, asset("maritime_s_base"), parent="maritime_s_base.v1", request_id="req-2")
    assert second["version"] == "maritime_s_base.v2" and second["bn_stats_match"] is None


def test_switch_is_a_documented_stub():
    with pytest.raises(NotImplementedError, match="documented stub"):
        switch()


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda(), reason="1-epoch retrain smoke test needs a GPU")
def test_retrain_smoke_one_epoch(tmp_path):
    pytest.importorskip("ultralytics")
    asset("maritime_s_base")
    asset("yolov12n")
    pytest.importorskip("albumentations")
    import make_synthetic_frames as m
    from edge_signals.adapt.retrain import retrain
    files = m.make(tmp_path / "img", clips=2, frames=8, size=(640, 384))
    lab = tmp_path / "labels"
    lab.mkdir()
    for p in files:
        (lab / f"{p.stem}.txt").write_text("0 0.3 0.6 0.11 0.06\n")
    names = [p.stem for p in files]
    (tmp_path / "adapt.txt").write_text("".join(s + "\n" for s in names[:8]))
    (tmp_path / "replay.txt").write_text("".join(s + "\n" for s in names[8:]))
    cfg = load_cfg("configs/adaptation.yaml", [f"collect.replay.lists=[{tmp_path / 'replay.txt'}]", f"annotate.provided.labels_dir={lab}", "annotate.mode=provided",
                                               "retrain.recipe.workers=0", "retrain.recipe.batch=4", "retrain.recipe.epochs=1"])
    run = tmp_path / "run"
    C.collect(cfg, run, frames_list=tmp_path / "adapt.txt", ratio=1.0)
    A.annotate(cfg, run, tmp_path / "img")
    res = retrain(cfg, run, tmp_path / "img")
    assert res["epochs_trained"] == 1 and res["recipe_hidden_albumentations"]["hidden_blur_gray_clahe_active"] is False
    assert "RandomBrightnessContrast" in res["recipe_hidden_albumentations"]["transform"] and res["peak_gpu_MiB"]["peak_alloc_MiB"] > 0
    assert set(res["per_phase_s"]) >= {"dataset_build_s", "train_s", "save_s", "total_s"} and (run / "retrain/weights/last.pt").exists()
    assert not list(tmp_path.glob("*.cache")) and not (ROOT / "runs").exists()          # nothing written outside the run directory


def test_evaluate_excludes_the_videos_of_the_collected_window_from_every_held_out_split(tmp_path, capsys):
    from edge_signals.cli import exclude_collected_videos
    run = tmp_path / "run"
    (run / "collect").mkdir(parents=True)
    (run / "collect/adapt.txt").write_text("".join(s + "\n" for s in stems("nir1", 5) + stems("hz1", 3)))
    (run / "collect/replay.txt").write_text("".join(s + "\n" for s in stems("clr1", 4)))
    lists = {"nir_eval": stems("nir1", 6) + stems("nir9", 4), "haze_eval": stems("hz1", 7), "clear_test": stems("clr7", 5), "clear_calib": stems("clr1", 2) + stems("clr5", 2)}
    files = {}
    for n, st in lists.items():
        (tmp_path / f"{n}.txt").write_text("".join(s + "\n" for s in st))
        files[n] = str(tmp_path / f"{n}.txt")
    kept, rep = exclude_collected_videos(files, run)
    assert set(kept) == {"nir_eval", "clear_test", "clear_calib"}                       # haze_eval: every video excluded -> nothing left to evaluate
    assert read_stems(kept["nir_eval"]) == stems("nir9", 4) and read_stems(kept["clear_test"]) == stems("clr7", 5) and read_stems(kept["clear_calib"]) == stems("clr5", 2)
    assert rep["adapt_videos"] == ["hz1", "nir1"] and rep["replay_videos"] == ["clr1"]
    assert rep["per_split"]["nir_eval"] == {"videos": ["nir1"], "n_frames_before": 10, "n_frames_after": 4}
    assert rep["per_split"]["haze_eval"]["videos"] == ["hz1"] and rep["per_split"]["clear_test"]["videos"] == []
    assert "excluded 1 video(s)" in capsys.readouterr().err
    same, none = exclude_collected_videos(files, run, include_collected=True)          # opt-out: the lists are used as given
    assert same == files and none is None
    same, none = exclude_collected_videos(files, tmp_path / "no_collect_stage")
    assert same == files and none is None


def test_shipped_default_annotate_mode_is_provided_and_pseudo_stays_an_option():
    cfg = load_cfg("configs/adaptation.yaml", [])
    assert cfg["annotate"]["mode"] == "provided" and cfg["collect"]["replay"]["ratio"] == 2.0 and "teacher_weights" in cfg["annotate"]["pseudo"]
    text = Path("configs/adaptation.yaml").read_text()
    assert "FAILURE CONDITION" in text and "teacher is stronger than the student" in text
