"""Progress lines must reach a pipe (no TTY) while the process is still running: run fetch-assets and stream as subprocesses, with PYTHONUNBUFFERED removed from the environment."""
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import ASSETS, asset  # noqa: E402
from test_video_and_same_video import write_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def child_env(**extra):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}          # the child must flush by itself
    env.update(PYTHONNOUSERSITE="1", MPLBACKEND="Agg")
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'third_party' / 'consistency'}"
    env.update(extra)
    return env


def read_lines_with_times(proc, stream):
    out = []
    for line in iter(stream.readline, ""):
        out.append((time.monotonic(), line.rstrip("\n"), proc.poll() is None))     # (arrival time, text, was the process still running when it arrived?)
    return out


class SlowServer:
    """Serves `size` bytes at roughly 150 kB/s so that a download lasts a few seconds."""

    def __init__(self, size=500_000):
        data = b"z" * size
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                for i in range(0, len(data), 50_000):
                    self.wfile.write(data[i:i + 50_000])
                    self.wfile.flush()
                    time.sleep(0.3)
        self.data = data
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def test_fetch_assets_progress_reaches_a_pipe_before_the_process_ends(tmp_path):
    import hashlib
    srv = SlowServer()
    try:
        cfg = {"assets": [{"name": "w", "kind": "weights", "url": f"http://127.0.0.1:{srv.port}/w.pt", "sha256": hashlib.sha256(srv.data).hexdigest(), "size_bytes": len(srv.data),
                           "target": "$EDGE_ASSETS/weights/w.pt", "extract": False}]}
        (tmp_path / "assets.yaml").write_text(yaml.safe_dump(cfg))
        proc = subprocess.Popen([sys.executable, "-m", "edge_signals.cli", "fetch-assets", "--assets-config", str(tmp_path / "assets.yaml"), "--assets-dir", str(tmp_path / "a"),
                                 "--data-dir", str(tmp_path / "d")], cwd=ROOT, env=child_env(EDGE_PROGRESS_EVERY="0.5"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        lines = read_lines_with_times(proc, proc.stderr)
        t_end = time.monotonic()
        assert proc.wait() == 0
    finally:
        srv.close()
    assert lines, "no output at all"
    assert "connecting to 127.0.0.1" in lines[0][1] and lines[0][2], "the first progress line must arrive while the process is still running"
    progress = [l for l in lines if " MB" in l[1] and "MB/s" in l[1]]
    assert progress and progress[0][2] and t_end - lines[0][0] > 1.0                      # lines arrive one after the other during the download, not all at the end


def test_stream_progress_lines_reach_a_pipe_before_the_process_ends(tmp_path):
    asset("maritime_s_base")
    vids = tmp_path / "vids"
    vids.mkdir()
    for c in ("MVI_0001_VIS", "MVI_0002_VIS", "MVI_0003_VIS"):
        write_video(vids / f"{c}.avi", 400)                                             # 80 processed frames each
    (tmp_path / "order.yaml").write_text(yaml.safe_dump({"splits": {"s": ["MVI_0001_VIS", "MVI_0002_VIS", "MVI_0003_VIS"]}, "streams": {"x": ["s"]}, "default_stream": "x"}))
    proc = subprocess.Popen([sys.executable, "-m", "edge_signals.cli", "stream", "--videos-dir", str(vids), "--order", str(tmp_path / "order.yaml"), "--out", str(tmp_path / "out")],
                            cwd=ROOT, env=child_env(EDGE_ASSETS=str(ASSETS), YOLO_OFFLINE="True", YOLO_CONFIG_DIR=str(tmp_path / "yolo")), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    lines = read_lines_with_times(proc, proc.stderr)
    t_end = time.monotonic()
    assert proc.wait() == 0, [l[1] for l in lines][-5:]
    prog = [l for l in lines if l[1].startswith("[stream] video")]
    assert [l[1].split(":")[0] for l in prog] == ["[stream] video 1/3 MVI_0001_VIS", "[stream] video 2/3 MVI_0002_VIS", "[stream] video 3/3 MVI_0003_VIS"]
    assert prog[0][2] and t_end - prog[0][0] > 0.2                                       # the first line arrived while videos 2 and 3 were still being processed
    assert "80 frames processed" in prog[0][1] and json.load(open(tmp_path / "out" / "summary.json"))["n_frames"] == 240
