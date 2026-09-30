"""Replay order and lenient video resolution: `stream --videos-dir`, `--stream/--split`, `--on-missing skip|error`."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_signals import replay as R  # noqa: E402
from edge_signals.base import ReferenceBundle  # noqa: E402
from edge_signals.cli import main as cli_main  # noqa: E402
from test_video_and_same_video import FakeDetector, write_video  # noqa: E402

ORDER = {"splits": {"s1": ["MVI_0001_VIS", "MVI_0002_VIS"], "s2": ["MVI_0003_NIR"], "s3": ["MVI_0004_VIS_OB"]},
         "default_stream": "st", "streams": {"st": ["s2", "s1"], "solo": ["s3"]}}


def touch(d: Path, *names):
    for n in names:
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        (d / n).write_bytes(b"x")


def items(stream=None, split=None):
    return R.ordered_codes(ORDER, stream, split)


def test_order_is_the_concatenation_of_the_splits_in_the_given_order():
    assert [i["code"] for i in items()] == ["MVI_0003_NIR", "MVI_0001_VIS", "MVI_0002_VIS"]
    assert [i["code"] for i in items(split="s1")] == ["MVI_0001_VIS", "MVI_0002_VIS"] and [i["split"] for i in items("st")][0] == "s2"
    with pytest.raises(SystemExit, match="unknown stream"):
        items("nope")
    with pytest.raises(SystemExit, match="not both"):
        items("st", "s1")


def test_exact_stem_match_case_insensitive_and_any_extension_recursive(tmp_path):
    touch(tmp_path, "a/MVI_0001_VIS.avi", "b/c/mvi_0002_vis.MP4", "MVI_0003_NIR.MOV", "notes.txt", "MVI_0004_VIS_OB.mkv", ".hidden/._MVI_0009.avi")
    r = R.resolve(items(), tmp_path)
    assert [x["code"] for x in r["replayed"]] == ["MVI_0003_NIR", "MVI_0001_VIS", "MVI_0002_VIS"]              # order preserved
    assert all(v["status"] == "exact" for v in r["resolution"].values()) and r["not_replayed"] == []
    assert Path(r["replayed"][2]["file"]).name == "mvi_0002_vis.MP4" and r["n_video_files_found"] == 4


def test_contains_match_when_there_is_no_exact_match(tmp_path):
    touch(tmp_path, "SMD_MVI_0001_VIS_clip.avi", "MVI_0002_VIS.avi", "MVI_0003_NIR.avi")
    r = R.resolve(items(), tmp_path)
    assert r["resolution"]["MVI_0001_VIS"]["status"] == "contains" and r["resolution"]["MVI_0002_VIS"]["status"] == "exact"


def test_exact_wins_over_contains(tmp_path):
    touch(tmp_path, "MVI_0001_VIS.avi", "MVI_0001_VIS_copy.avi")
    assert R.resolve_code("MVI_0001_VIS", R.find_videos(tmp_path))["status"] == "exact"


def test_ambiguity_is_reported_with_candidates_and_treated_as_missing(tmp_path):
    touch(tmp_path, "x/MVI_0001_VIS_a.avi", "y/MVI_0001_VIS_b.avi", "MVI_0002_VIS.avi", "MVI_0003_NIR.avi", "d1/MVI_0002_VIS.mp4")
    r = R.resolve(items(), tmp_path)
    a = r["resolution"]["MVI_0001_VIS"]
    assert a["status"] == "ambiguous" and len(a["candidates"]) == 2
    assert r["resolution"]["MVI_0002_VIS"]["status"] == "ambiguous" and r["resolution"]["MVI_0002_VIS"]["match"] == "exact"    # same stem twice
    assert [x["code"] for x in r["replayed"]] == ["MVI_0003_NIR"] and {m["code"] for m in r["not_replayed"]} == {"MVI_0001_VIS", "MVI_0002_VIS"}


def test_missing_video_is_listed(tmp_path):
    touch(tmp_path, "MVI_0001_VIS.avi", "MVI_0003_NIR.avi")
    r = R.resolve(items(), tmp_path)
    assert r["resolution"]["MVI_0002_VIS"] == {"status": "missing", "split": "s1"} and [x["code"] for x in r["replayed"]] == ["MVI_0003_NIR", "MVI_0001_VIS"]


def test_not_a_directory_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="not a directory"):
        R.resolve(items(), tmp_path / "nope")


# ---- through the command line (a fake detector: no weights needed)
@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    b = ReferenceBundle("healthy_changes", {"previous": np.linspace(0, 0.1, 20), "anchor": np.linspace(0, 0.1, 20)}, {"frame_stride": 5, "fps": 6.0}).save(tmp_path)
    trig = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "trigger.yaml"))
    trig["trigger"]["healthy_changes"] = str(b)
    (tmp_path / "trigger.yaml").write_text(yaml.safe_dump(trig))
    (tmp_path / "order.yaml").write_text(yaml.safe_dump(ORDER))
    import edge_signals.inference as inf

    class Det(FakeDetector):
        hook, hook_index, hook_shape, env = None, None, None, {}
        weights_sha256 = backbone_params_id = backbone_state_id = "x"

        def __init__(self, *a, **k):
            pass
    monkeypatch.setattr(inf, "Detector", Det)
    vids = tmp_path / "vids"
    vids.mkdir()
    return tmp_path, vids


def run(tmp, vids, *extra):
    return cli_main(["stream", "--videos-dir", str(vids), "--order", str(tmp / "order.yaml"), "--trigger", str(tmp / "trigger.yaml"), "--out", str(tmp / "out"), *extra])


def test_skip_is_the_default_the_run_completes_and_resolution_json_records_it(cli_env, capsys):
    tmp, vids = cli_env
    write_video(vids / "MVI_0003_NIR.avi", 40)
    write_video(vids / "MVI_0002_VIS.avi", 25)                                   # MVI_0001_VIS is missing
    run(tmp, vids)
    err = capsys.readouterr().err
    assert "[warning] video MVI_0001_VIS (s1): missing" in err
    res = json.load(open(tmp / "out" / "resolution.json"))
    assert [x["code"] for x in res["replayed"]] == ["MVI_0003_NIR", "MVI_0002_VIS"] and res["resolution"]["MVI_0001_VIS"]["status"] == "missing" and res["stream"] == "st"
    summ = json.load(open(tmp / "out" / "summary.json"))
    assert summ["videos"] == ["MVI_0003_NIR", "MVI_0002_VIS"] and summ["n_frames"] == 8 + 5 and summ["resolution"]["not_replayed"] == ["MVI_0001_VIS"]
    fr = [json.loads(l) for l in open(tmp / "out" / "frames.jsonl")]
    assert [f["video"] for f in fr] == ["MVI_0003_NIR"] * 8 + ["MVI_0002_VIS"] * 5 and fr[8]["frame_index"] == 0


def test_error_mode_stops_with_the_list_and_still_writes_resolution_json(cli_env):
    tmp, vids = cli_env
    write_video(vids / "MVI_0003_NIR.avi", 12)
    with pytest.raises(SystemExit, match="MVI_0001_VIS, MVI_0002_VIS"):
        run(tmp, vids, "--on-missing", "error")
    assert json.load(open(tmp / "out" / "resolution.json"))["not_replayed"][0]["reason"] == "missing"


def test_split_selects_one_split_and_nothing_found_is_an_error(cli_env):
    tmp, vids = cli_env
    write_video(vids / "MVI_0001_VIS.avi", 12)
    run(tmp, vids, "--split", "s1")
    assert json.load(open(tmp / "out" / "summary.json"))["videos"] == ["MVI_0001_VIS"]
    with pytest.raises(SystemExit, match="none of the 1 videos"):
        run(tmp, vids, "--stream", "solo")


def test_videos_dir_is_required():
    with pytest.raises(SystemExit, match="--videos-dir"):
        cli_main(["stream", "--out", "/tmp/never"])


def test_labelled_images_follow_the_replay_order_too(cli_env, capsys, monkeypatch):
    import cv2
    import edge_signals.inference as inf
    from edge_signals.base import FrameRecord
    from conftest import make_boxes
    tmp, _ = cli_env

    class ImgDet(inf.Detector):                                                     # cli_env's fake detector, taking image paths instead of arrays
        def infer(self, images, ids, t0=0.0, fps=6.0, clip_ids=None, device_id="d", model_version="m"):
            return [FrameRecord(t=t0 + i / fps, frame_id=f, boxes=make_boxes(), clip_id=None if clip_ids is None else clip_ids[i]) for i, f in enumerate(ids)]
    monkeypatch.setattr(inf, "Detector", ImgDet)
    img = tmp / "imgs"
    img.mkdir()
    for code, n in (("MVI_0001_VIS", 3), ("MVI_0003_NIR", 4), ("MVI_9999_VIS", 2)):          # alphabetical order would put 0001 first; 9999 is not in the stream
        for k in range(n):
            cv2.imwrite(str(img / f"{code}-{k:07d}.jpg"), np.full((36, 64, 3), 40 * k, np.uint8))
    cli_main(["stream", "--images-dir", str(img), "--stream", "st", "--order", str(tmp / "order.yaml"), "--trigger", str(tmp / "trigger.yaml"), "--out", str(tmp / "out_i")])
    fr = [json.loads(l)["clip_id"] for l in open(tmp / "out_i" / "frames.jsonl")]
    assert fr == ["MVI_0003_NIR"] * 4 + ["MVI_0001_VIS"] * 3                                # replay order; the clip outside the stream is left out; MVI_0002_VIS has no frames
    res = json.load(open(tmp / "out_i" / "resolution.json"))
    assert res["resolution"]["MVI_0002_VIS"]["status"] == "missing" and res["resolution"]["MVI_0003_NIR"]["n_frames"] == 4 and res["input"] == "images"
    assert "MVI_0002_VIS" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="MVI_0002_VIS"):
        cli_main(["stream", "--images-dir", str(img), "--stream", "st", "--on-missing", "error", "--order", str(tmp / "order.yaml"), "--trigger", str(tmp / "trigger.yaml"), "--out", str(tmp / "out_j")])
