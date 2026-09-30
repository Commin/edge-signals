import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_data as pd  # noqa: E402


def make_cfg(tmp_path, lists):
    cfg = {"source": {"url": None, "sha256": None, "local_dir": None}, "target_dir": str(tmp_path / "out"),
           "input_name_regex": r"^(?P<clip>.+?)_frame(?P<n>\d+)", "reindex": True, "frame_lists": [str(l) for l in lists]}
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_layout_reindexes_numerically_and_checks_lists(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for n in (10, 2, 5):                                   # frame10 must come after frame2 (numeric order)
        (src / f"vidA_frame{n}_jpg.rf.abc.jpg").write_bytes(b"x")
    (src / "vidB_frame0_jpg.rf.def.jpg").write_bytes(b"y")
    lst = tmp_path / "list.txt"
    lst.write_text("vidA-0000000\nvidA-0000002\nvidB-0000000\n")
    pd.main(["--config", str(make_cfg(tmp_path, [lst])), "--source-dir", str(src)])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["vidA-0000000.jpg", "vidA-0000001.jpg", "vidA-0000002.jpg",
                                                                  "vidB-0000000.jpg"]
    lst.write_text("vidA-0000009\n")
    with pytest.raises(SystemExit):
        pd.main(["--config", str(make_cfg(tmp_path, [lst])), "--source-dir", str(src)])


def test_archive_hash_and_traversal_are_checked(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("vidA_frame1.jpg", b"x")
    assert pd.sha256(z) and pd.extract(z, tmp_path / "unpacked") and (tmp_path / "unpacked" / "vidA_frame1.jpg").exists()
    bad = tmp_path / "b.zip"
    with zipfile.ZipFile(bad, "w") as f:
        f.writestr("../evil.jpg", b"x")
    with pytest.raises(SystemExit, match="unsafe"):
        pd.extract(bad, tmp_path / "unpacked_bad")
    with pytest.raises(SystemExit, match="nothing to prepare"):
        pd.main(["--config", str(make_cfg(tmp_path, []))])


def test_synthetic_frames_are_written_with_clip_and_index_names(tmp_path):
    pytest.importorskip("PIL")
    import make_synthetic_frames as m
    files = m.make(tmp_path, clips=2, frames=3)
    assert sorted(p.name for p in files) == ["synth0-0000000.jpg", "synth0-0000001.jpg", "synth0-0000002.jpg",
                                             "synth1-0000000.jpg", "synth1-0000001.jpg", "synth1-0000002.jpg"]
