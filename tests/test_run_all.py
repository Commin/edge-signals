"""scripts/run_all.sh (option handling only: no docker needed) and scripts/compare_expected.py."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_all.sh"
sys.path.insert(0, str(ROOT / "scripts"))
import compare_expected as CE  # noqa: E402


def sh(*args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT)


def test_help_lists_every_option_and_exits_zero():
    r = sh("--help")
    assert r.returncode == 0
    for opt in ("--work-dir", "--tag", "--videos-dir", "--dns", "--network-host", "--gpus", "--skip-tests", "--skip-retrain", "--no-pull", "--local-assets"):
        assert opt in r.stdout, opt
    assert "set -uo" not in r.stdout


def test_unknown_option_and_conflicting_network_options_are_refused():
    r = sh("--nope")
    assert r.returncode == 2 and "unknown option" in r.stderr
    r = sh("--dns", "1.2.3.4", "--network-host")
    assert r.returncode == 2 and "either --dns or --network-host" in r.stderr
    assert sh("--work-dir").returncode == 2


def test_no_node_specific_values_in_the_script():
    text = SCRIPT.read_text()
    assert not re.search(r"/large/|/home/|172\.16\.|inria|riken|mdxuser", text, re.I)


def test_default_version_matches_the_expected_results_and_the_readme():
    version = re.search(r'^VERSION="(v[0-9.]+)"', SCRIPT.read_text(), re.M).group(1)
    assert yaml.safe_load(open(ROOT / "configs" / "expected_results.yaml"))["release"] == version
    assert f"ghcr.io/commin/edge-signals:{version}" in (ROOT / "README.md").read_text()


def make_run(tmp, mode, requests, n_frames=2222):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "requests.jsonl").write_text("".join(json.dumps({**r, "drop": 0.1, "delta": 0.08, "p_value": 0.02}) + "\n" for r in requests))
    # video runs have a resolution block, image runs have resolution = null (as written by `stream`)
    (tmp / "summary.json").write_text(json.dumps({"n_frames": n_frames, "resolution": {"stream": "drift_calib"} if mode == "video" else None}))
    return tmp


def expected(mode):
    return yaml.safe_load(open(ROOT / "configs" / "expected_results.yaml"))["modes"][mode]["requests"]


def test_compare_expected_matches_the_shipped_expectation(tmp_path, capsys):
    vid = [{"video": x["video"], "frame_index": x["frame_index"]} for x in expected("video")]
    assert CE.main(["--mode", "video", "--out", str(make_run(tmp_path / "v", "video", vid))]) == 0 and "OK" in capsys.readouterr().out
    img = [{"clip_id": c} for c in expected("images")]
    assert CE.main(["--mode", "images", "--out", str(make_run(tmp_path / "i", "images", img))]) == 0


def test_compare_expected_warns_with_the_reason_when_a_window_flips(tmp_path, capsys):
    vid = [{"video": x["video"], "frame_index": x["frame_index"]} for x in expected("video")]
    run = make_run(tmp_path / "v", "video", vid[:-1] + [{"video": "MVI_1617_VIS", "frame_index": 0}])
    assert CE.main(["--mode", "video", "--out", str(run)]) == 1
    out = capsys.readouterr().out
    assert "WARNING" in out and "expected but not raised: MVI_0801_VIS_OB frame 300" in out and "raised but not expected: MVI_1617_VIS frame 0" in out
    assert "close to delta" in out and "drop 0.1000 vs delta 0.08" in out
    assert CE.main(["--mode", "video", "--out", str(make_run(tmp_path / "f", "video", vid, n_frames=100))]) == 1 and "frames processed: 100 instead of 2222" in capsys.readouterr().out


def test_compare_expected_reports_an_unreadable_run(tmp_path, capsys):
    assert CE.main(["--mode", "video", "--out", str(tmp_path / "nothing")]) == 2 and "cannot read" in capsys.readouterr().err


def test_an_unexpected_error_is_exit_2_not_a_difference(tmp_path):
    (tmp_path / "requests.jsonl").write_text("{}\n")                                   # a request without the expected fields
    (tmp_path / "summary.json").write_text(json.dumps({"n_frames": 2222}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_expected.py"), "--mode", "images", "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 2 and "Traceback" in r.stderr


def test_printed_commands_are_real_docker_commands_with_unbuffered_python():
    text = SCRIPT.read_text()
    assert "dk_base" not in text                                                # the printed line must be the docker command, not a shell function name
    assert re.search(r"^DK=\(docker run --rm --user \"\$UIDGID\" --shm-size=2g -e PYTHONUNBUFFERED=1\)$", text, re.M)
    assert "tee -a" in text and "PIPESTATUS" in text                            # live output, kept in the log (tee, not redirect)
