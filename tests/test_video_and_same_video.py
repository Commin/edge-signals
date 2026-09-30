"""Video input, the stride guard, same-video collection, the in-stream view and the alignment tool - on tiny synthetic videos."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from conftest import make_boxes  # noqa: E402
from edge_signals import video as V  # noqa: E402
from edge_signals.adapt import collect as C  # noqa: E402
from edge_signals.adapt import streamref as S  # noqa: E402
from edge_signals.adapt.common import load_cfg, read_stems  # noqa: E402
from edge_signals.base import FrameRecord  # noqa: E402
from edge_signals.cli import main as cli_main  # noqa: E402
from edge_signals.config import validate_signals, validate_trigger  # noqa: E402
from edge_signals.engine import SignalEngine  # noqa: E402

W, H = 96, 64


def frame_image(k: int, noise=False):
    """A frame whose content identifies its index k (block position and level), plus optional noise."""
    img = np.full((H, W, 3), 40, np.uint8)
    x = (k * 3) % (W - 20)
    img[10:30, x:x + 20] = 90 + (k * 5) % 150
    img[40:60, 10:30] = (k * 11) % 255
    if noise:
        img = np.clip(img.astype(int) + np.random.RandomState(k).randint(-6, 6, img.shape), 0, 255).astype(np.uint8)
    return img


def write_video(path: Path, n: int, fps=30.0, shift=0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for k in range(n):
        vw.write(frame_image(max(0, k - shift)))
    vw.release()
    return path


class FakeDetector:
    """Stands in for the YOLO detector: one box whose position follows the frame content; consistent between frames."""

    def infer(self, images, ids, t0=0.0, fps=6.0, clip_ids=None, device_id="d", model_version="m"):
        out = []
        for i, (im, fid) in enumerate(zip(images, ids)):
            assert im.shape == (H, W, 3)
            out.append(FrameRecord(t=t0 + i / fps, frame_id=fid, boxes=make_boxes(), clip_id=None if clip_ids is None else clip_ids[i],
                                   device_id=device_id, model_version=model_version))
        return out


def test_iter_frames_keeps_every_fifth_frame_with_its_original_index(tmp_path):
    p = write_video(tmp_path / "MVI_0001_VIS.avi", 53)
    got = list(V.iter_frames(p, 5))
    assert [i for i, _ in got] == list(range(0, 53, 5)) and V.probe(p)["n_frames"] == 53 and abs(V.probe(p)["fps"] - 30.0) < 0.5
    for i, fr in got[:4]:                                   # the decoded frame really is frame i (content check, lossy codec tolerance)
        assert np.abs(fr.astype(int) - frame_image(i).astype(int)).mean() < 8


def test_stream_records_carry_video_name_and_original_index(tmp_path):
    a, b = write_video(tmp_path / "vidA.avi", 31), write_video(tmp_path / "vidB.avi", 16)
    recs = list(V.stream_video_records(FakeDetector(), [a, b], 5, 6.0, chunk=4))
    assert [(r.video, r.frame_index) for r in recs] == [("vidA", i) for i in range(0, 31, 5)] + [("vidB", i) for i in range(0, 16, 5)]
    assert [r.clip_id for r in recs] == [r.video for r in recs] and recs[3].t == pytest.approx(3 / 6.0) and recs[0].frame_id == "vidA@0000000"


def test_stride_is_tied_to_the_calibration():
    assert V.check_stride(5, 6.0, 30.0, 5) == pytest.approx(6.0)
    with pytest.raises(SystemExit, match="differs from the stride the calibration was made at"):
        V.check_stride(3, 6.0, 30.0, 5)
    with pytest.raises(SystemExit, match="does not record the frame stride"):
        V.check_stride(5, 6.0, 30.0, None)
    with pytest.raises(SystemExit, match="gives 5.00 fps"):
        V.check_stride(5, 6.0, 25.0, 5)


def test_cli_refuses_another_stride_before_loading_anything(tmp_path):
    from edge_signals.base import ReferenceBundle
    import yaml
    b = ReferenceBundle("healthy_changes", {"previous": np.linspace(0, 0.1, 20), "anchor": np.linspace(0, 0.1, 20)}, {"frame_stride": 5, "fps": 6.0})
    bp = b.save(tmp_path)
    trig = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "trigger.yaml"))
    trig["trigger"]["healthy_changes"] = str(bp)
    (tmp_path / "trigger.yaml").write_text(yaml.safe_dump(trig))
    p = write_video(tmp_path / "vid.avi", 12)
    with pytest.raises(SystemExit, match="differs from the stride the calibration was made at"):
        cli_main(["stream", "--video", str(p), "--frame-stride", "3", "--trigger", str(tmp_path / "trigger.yaml"), "--out", str(tmp_path / "o")])
    trig["trigger"]["healthy_changes"] = str(ReferenceBundle("healthy_changes", {"previous": np.zeros(3), "anchor": np.zeros(3)}, {}).save(tmp_path / "nostride"))
    (tmp_path / "trigger2.yaml").write_text(yaml.safe_dump(trig))
    with pytest.raises(SystemExit, match="does not record the frame stride"):
        cli_main(["stream", "--video", str(p), "--trigger", str(tmp_path / "trigger2.yaml"), "--out", str(tmp_path / "o")])


def test_requests_and_frame_log_carry_the_original_position(tmp_path):
    from conftest import sig_cfg
    sc = validate_signals(sig_cfg(), fps=6.0)
    tc = validate_trigger({"trigger": {"rules": {"step": {"type": "relative_drop", "enabled": True, "stage": "screening", "comparator": "previous", "delta": 0.2,
                                                          "n_consecutive": 1, "cooldown_windows": 0}}}}, sc)
    frames = []
    for k in range(20):
        b = make_boxes() if k < 10 else make_boxes(shift=(0.0, 0.25 * (k % 2)))
        frames.append(FrameRecord(t=k / 6.0, frame_id=f"v@{5 * k:07d}", boxes=b, clip_id="v", video="v", frame_index=5 * k))
    eng = SignalEngine(sc, tc, fps=6.0, out_dir=tmp_path)
    out = eng.run(frames)
    r = out["triggers"][0]
    assert r.video == "v" and r.frame_index % 5 == 0 and r.frame_index >= 50
    rows = [json.loads(l) for l in open(tmp_path / "frames.jsonl")]
    assert rows[7]["video"] == "v" and rows[7]["frame_index"] == 35


# ---------------------------------------------------------------- mapping, same-video collection
def make_mapping(path: Path, videos: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_index", "image_stem", "split"])
        for v, n in videos.items():
            for k in range(n):
                w.writerow([v, 5 * k, f"{v}-{k:07d}", "x"])
    return path


def make_stream_out(path: Path, order, requests):
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps({"videos": order, "frame_stride": 5}))
    (path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    return path


def req(rid, video, idx, rule="step", stage="screening"):
    return {"request_id": rid, "video": video, "frame_index": idx, "rule": rule, "stage": stage, "t": 0.0}


@pytest.fixture
def world(tmp_path):
    mp = make_mapping(tmp_path / "map.csv", {"vidA": 40, "vidB": 30, "vidC": 50})
    for i, (nm, n) in enumerate([("r1", 60), ("r2", 60)]):
        (tmp_path / f"{nm}.txt").write_text("".join(f"{nm}-{k:07d}\n" for k in range(n)))
    cfg = load_cfg("configs/adaptation.yaml", [f"collect.replay.lists=[{tmp_path / 'r1.txt'}, {tmp_path / 'r2.txt'}]", "collect.min_new_frames=20"])
    return tmp_path, mp, cfg


def test_segments_cover_reset_to_request_across_videos(world):
    tmp, mp, cfg = world
    m = S.load_mapping(mp)
    segs = S.segments(["vidA", "vidB", "vidC"], ("vidA", 100), ("vidC", 60))
    assert segs == [("vidA", 100, None), ("vidB", None, None), ("vidC", None, 60)]
    stems = S.stems_in(m, segs)
    assert stems[0] == "vidA-0000021" and stems.count("vidB-0000000") == 1 and stems[-1] == "vidC-0000012"      # vidA frames 105.., all vidB, vidC up to 60
    assert len(stems) == (40 - 21) + 30 + 13
    assert S.segments(["vidA", "vidB"], None, ("vidA", 20)) == [("vidA", None, 20)]
    with pytest.raises(SystemExit, match="after the request"):
        S.segments(["vidA", "vidB"], ("vidB", 5), ("vidA", 20))


def test_collect_takes_the_labelled_frames_of_the_same_video_before_the_request(world):
    tmp, mp, cfg = world
    so = make_stream_out(tmp / "so", ["vidA", "vidB", "vidC"], [req("step-1", "vidA", 95), req("step-2", "vidC", 60)])
    info = C.collect(cfg, tmp / "run", stream_out=so, mapping=mp, request_id="step-1", ratio=0.5, seed=1)
    adapt = read_stems(tmp / "run/collect/adapt.txt")
    assert info["status"] == "ok" and adapt == [f"vidA-{k:07d}" for k in range(20)] and info["n_replay"] == 10          # frames 0..95 -> 20 labelled frames
    assert info["source"]["request_position"] == {"video": "vidA", "frame_index": 95}
    info2 = C.collect(cfg, tmp / "run2", stream_out=so, mapping=mp, request_id="step-2", since_request_id="step-1", ratio=0)
    a2 = read_stems(tmp / "run2/collect/adapt.txt")
    assert a2[0] == "vidA-0000020" and a2[-1] == "vidC-0000012" and len(a2) == 20 + 30 + 13 and info2["source"]["reset_position"] == {"video": "vidA", "frame_index": 95}


def test_insufficient_data_is_logged_and_no_retraining_starts(world, capsys):
    tmp, mp, cfg = world
    so = make_stream_out(tmp / "so", ["vidA", "vidB"], [req("step-1", "vidA", 40)])              # 9 frames < 20
    info = C.collect(cfg, tmp / "run", stream_out=so, mapping=mp, request_id="step-1")
    assert info["status"] == "insufficient_data" and info["n_adapt"] == 9 and not (tmp / "run/collect/adapt.txt").exists()
    from edge_signals.adapt import annotate as A
    with pytest.raises(SystemExit, match="insufficient data"):
        A.annotate(cfg, tmp / "run", tmp)
    rc = cli_main(["collect", "--out", str(tmp / "run3"), "--stream-out", str(so), "--mapping", str(mp), "--request-id", "step-1", "--min-new-frames", "30",
                   "--set", f"collect.replay.lists=[{tmp / 'r1.txt'}]"])
    assert rc == 3
    assert "insufficient data" in capsys.readouterr().err


def test_in_stream_view_is_the_later_frames_of_the_request_video(world):
    tmp, mp, cfg = world
    m = S.load_mapping(mp)
    later = S.later_frames(m, ("vidB", 60))
    assert later == [f"vidB-{k:07d}" for k in range(13, 30)] and S.later_frames(m, ("vidB", 145)) == []


def test_request_without_a_position_is_refused(world):
    with pytest.raises(SystemExit, match="no video / frame_index"):
        S.position({"request_id": "x"})


# ---------------------------------------------------------------- alignment tool
def build_aligned(tmp_path, n_frames=60, shift=0):
    """A video whose frame 5k is labelled image k (optionally displaced by `shift` frames) + the labelled images and mapping."""
    vid = write_video(tmp_path / "MVI_9999_VIS.avi", n_frames + 5, shift=shift)
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    rows = []
    for k in range(n_frames // 5):
        cv2.imwrite(str(frames / f"MVI_9999_VIS-{k:07d}.jpg"), cv2.resize(frame_image(5 * k), (W * 2, H * 2)))
        rows.append(["MVI_9999_VIS", 5 * k, f"MVI_9999_VIS-{k:07d}", "x"])
    mp = tmp_path / "map.csv"
    with open(mp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_index", "image_stem", "split"])
        w.writerows(rows)
    return vid, frames, mp


def test_alignment_search_finds_stride_5_offset_0_and_flags_a_displaced_video(tmp_path, capsys):
    import sys
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_alignment as CA
    order = tmp_path / "order.yaml"
    order.write_text(yaml.safe_dump({"splits": {"s": ["MVI_9999_VIS"]}, "streams": {"x": ["s"]}, "default_stream": "x"}))
    vid, frames, mp = build_aligned(tmp_path, n_frames=90)
    row = CA.check_video(vid, frames, [f"MVI_9999_VIS-{k:07d}" for k in range(18)])
    assert (row["best_stride"], row["best_offset"]) == (5, 0) and row["margin_db"] > 3 and row["labelled_x_5"] == 90
    args = ["--videos-dir", str(tmp_path), "--frames-dir", str(frames), "--mapping", str(mp), "--order", str(order), "--out", str(tmp_path / "al.json")]
    assert CA.main(args) == 0 and "none" in capsys.readouterr().out and json.load(open(tmp_path / "al.json"))["videos"][0]["code"] == "MVI_9999_VIS"
    t2 = tmp_path / "shifted"
    t2.mkdir()
    vid2, frames2, mp2 = build_aligned(t2, n_frames=90, shift=1)
    rc = CA.main(["--videos-dir", str(t2), "--frames-dir", str(frames2), "--mapping", str(mp2), "--order", str(order)])
    assert rc == 1 and "('MVI_9999_VIS', 5, 1)" in capsys.readouterr().out                    # decoded frame 5k+1 = labelled image k
