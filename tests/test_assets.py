"""fetch-assets: requests-only downloads with resume, SHA256, Google Drive confirmation, safe extraction, idempotence."""
import hashlib
import http.server
import io
import re
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest
import yaml

from edge_signals import assets as A


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Server:
    """Local HTTP server: /raw (Range-capable), /uc (a Drive-like 'too large to scan' interstitial), /download (needs the confirm token)."""

    def __init__(self, files: dict, drive_mode="form"):
        self.files, self.drive_mode, self.requests = files, drive_mode, []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, data, status=200, ctype="application/octet-stream", extra=None):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Range")))
                path, _, q = self.path.partition("?")
                if path.startswith("/raw/"):
                    data = outer.files[path[5:]]
                    rng = self.headers.get("Range")
                    if rng and (m := re.match(r"bytes=(\d+)-", rng)):
                        start = int(m.group(1))
                        if start >= len(data):
                            return self._send(b"", 416)
                        return self._send(data[start:], 206, extra={"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"})
                    return self._send(data)
                if path == "/uc":                                  # the interstitial
                    fid = re.search(r"id=([^&]+)", q).group(1)
                    if "confirm=abc" in q:                          # the older interstitial's link: now serve the file
                        return self._send(outer.files[fid])
                    if outer.drive_mode == "none":
                        return self._send(b"<html>Sorry, quota exceeded</html>", ctype="text/html; charset=utf-8")
                    if outer.drive_mode == "link":
                        page = f'<html><a href="/uc?export=download&amp;id={fid}&amp;confirm=abc">Download anyway</a></html>'
                    else:
                        page = (f'<html><form id="download-form" action="http://127.0.0.1:{outer.port}/download" method="get">'
                                f'<input type="hidden" name="id" value="{fid}"><input type="hidden" name="export" value="download">'
                                f'<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="u-1"></form></html>')
                    return self._send(page.encode(), ctype="text/html; charset=utf-8")
                if path == "/download" and "confirm=t" in q and "uuid=u-1" in q:
                    return self._send(outer.files[re.search(r"id=([^&]+)", q).group(1)])
                self._send(b"", 404)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    made = []

    def make(files, **kw):
        s = Server(files, **kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


def cfg_for(name, data: bytes, url="", target="$EDGE_ASSETS/weights/w.pt", extract=False, **kw):
    return {"assets": [{"name": name, "kind": "weights", "url": url, "sha256": sha(data), "size_bytes": len(data), "target": target, "extract": extract, **kw}]}


def test_download_verify_and_idempotent(server, tmp_path):
    data = b"weights" * 5000
    s = server({"w.pt": data})
    cfg = cfg_for("w", data, f"http://127.0.0.1:{s.port}/raw/w.pt")
    r = A.fetch_assets(cfg, assets_dir=tmp_path / "a", data_dir=tmp_path / "d")
    assert r["w"]["status"] == "ok" and r["_all_ok"] and (tmp_path / "a/weights/w.pt").read_bytes() == data
    n = len(s.requests)
    r2 = A.fetch_assets(cfg, assets_dir=tmp_path / "a", data_dir=tmp_path / "d")
    assert r2["w"]["status"] == "up_to_date" and len(s.requests) == n                    # nothing was requested again


def test_partial_download_is_resumed_with_a_range_request(server, tmp_path):
    data = bytes(range(256)) * 4000
    s = server({"w.pt": data})
    cfg = cfg_for("w", data, f"http://127.0.0.1:{s.port}/raw/w.pt")
    part = tmp_path / "a/.downloads/w.download.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:300000])
    r = A.fetch_assets(cfg, assets_dir=tmp_path / "a", data_dir=tmp_path / "d")
    assert r["w"]["status"] == "ok" and (tmp_path / "a/weights/w.pt").read_bytes() == data
    assert ("/raw/w.pt", "bytes=300000-") in s.requests


def test_checksum_and_size_mismatch_leave_no_target(server, tmp_path):
    data = b"abc" * 100
    s = server({"w.pt": data})
    bad = cfg_for("w", data, f"http://127.0.0.1:{s.port}/raw/w.pt")
    bad["assets"][0]["sha256"] = "0" * 64
    r = A.fetch_assets(bad, assets_dir=tmp_path / "a", data_dir=tmp_path / "d")
    assert r["w"]["status"] == "failed" and "sha256 mismatch" in r["w"]["reason"] and not r["_all_ok"] and not (tmp_path / "a/weights/w.pt").exists()
    bad2 = cfg_for("w", data, f"http://127.0.0.1:{s.port}/raw/w.pt", size_bytes=7)
    r = A.fetch_assets(bad2, assets_dir=tmp_path / "a", data_dir=tmp_path / "d")
    assert "size" in r["w"]["reason"] and not (tmp_path / "a/weights/w.pt").exists()


def test_missing_url_is_reported_not_silent(tmp_path):
    r = A.fetch_assets(cfg_for("w", b"x", url=""), assets_dir=tmp_path / "a", data_dir=tmp_path / "d")
    assert r["w"]["status"] == "skipped" and "no url" in r["w"]["reason"] and r["_all_ok"] is False


def test_url_override_and_mounted_config(server, tmp_path, monkeypatch, capsys):
    data = b"z" * 1000
    s = server({"w.pt": data})
    cfgfile = tmp_path / "mounted_assets.yaml"
    cfgfile.write_text(yaml.safe_dump(cfg_for("w", data, url="")))
    A.main(["--assets-config", str(cfgfile), "--assets-dir", str(tmp_path / "a"), "--data-dir", str(tmp_path / "d"), "--url", f"w=http://127.0.0.1:{s.port}/raw/w.pt"])
    assert (tmp_path / "a/weights/w.pt").read_bytes() == data                    # no image rebuild: the link came from the command line


@pytest.mark.parametrize("mode", ["form", "link"])
def test_google_drive_large_file_confirmation(server, tmp_path, monkeypatch, mode):
    data = b"big-file" * 1000
    s = server({"FILEID": data}, drive_mode=mode)
    monkeypatch.setattr(A, "DRIVE_BASE", f"http://127.0.0.1:{s.port}")
    monkeypatch.setattr(A, "GDRIVE_HOSTS", ("127.0.0.1", "drive.google.com"))
    for url in ("gdrive:FILEID", f"http://127.0.0.1:{s.port}/file/d/FILEID/view"):
        dst = tmp_path / f"{mode}.bin"
        A.download(url, dst)
        assert dst.read_bytes() == data
        dst.unlink()
    assert A.drive_file_id("https://drive.google.com/file/d/AbC-1_2/view?usp=sharing") == "AbC-1_2"
    assert A.drive_file_id("https://drive.google.com/uc?id=XYZ&export=download") == "XYZ" and A.drive_file_id("https://example.org/a") is None


def test_google_drive_page_without_a_file_is_an_error(server, tmp_path, monkeypatch):
    s = server({"FILEID": b"x"}, drive_mode="none")
    monkeypatch.setattr(A, "DRIVE_BASE", f"http://127.0.0.1:{s.port}")
    with pytest.raises(SystemExit, match="web page instead of the file"):
        A.download("gdrive:FILEID", tmp_path / "x.bin")


def make_tar(path: Path, files: dict):
    with tarfile.open(path, "w") as t:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return path


def test_file_url_archive_is_extracted_under_the_data_volume(tmp_path):
    tar = make_tar(tmp_path / "d.tar", {"frames/a-0000000.jpg": b"1", "labels/a-0000000.txt": b"0 .5 .5 .1 .1\n", "lists/s.txt": b"a-0000000\n"})
    cfg = {"assets": [{"name": "data_s", "kind": "data", "url": f"file://{tar}", "sha256": sha(tar.read_bytes()), "size_bytes": tar.stat().st_size,
                       "target": "$EDGE_DATA", "extract": True}]}
    r = A.fetch_assets(cfg, assets_dir=tmp_path / "a", data_dir=tmp_path / "d")
    assert r["data_s"]["status"] == "ok" and r["data_s"]["n_files"] == 3 and (tmp_path / "d/frames/a-0000000.jpg").exists() and (tmp_path / "d/lists/s.txt").exists()
    assert A.fetch_assets(cfg, assets_dir=tmp_path / "a", data_dir=tmp_path / "d")["data_s"]["status"] == "up_to_date"


def test_unsafe_archive_members_are_refused(tmp_path):
    bad = make_tar(tmp_path / "bad.tar", {"../evil.txt": b"x"})
    with pytest.raises(SystemExit, match="unsafe"):
        A.extract_archive(bad, tmp_path / "out")
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as z:
        z.writestr("/abs.txt", b"x")
    with pytest.raises(SystemExit, match="unsafe"):
        A.extract_archive(tmp_path / "bad.zip", tmp_path / "out2")


def test_a_tar_that_ends_with_a_zip_member_is_still_a_tar(tmp_path):
    ck = tmp_path / "model.pt"
    with zipfile.ZipFile(ck, "w") as z:                      # a torch checkpoint is a zip file
        z.writestr("archive/data.pkl", b"x" * 100)
    tar = tmp_path / "teacher.tar"
    with tarfile.open(tar, "w") as t:
        t.add(ck, arcname="maritime_x_base.pt")
    assert zipfile.is_zipfile(tar)                           # the trap: the tar's tail looks like a zip
    assert A.extract_archive(tar, tmp_path / "out") == 1 and (tmp_path / "out" / "maritime_x_base.pt").is_file()


def test_only_requests_is_used_and_gdown_is_gone():
    root = Path(A.__file__).resolve().parents[1]
    for p in list((root / "edge_signals").rglob("*.py")) + list((root / "scripts").glob("*.py")):
        assert "gdown" not in p.read_text(), p
    assert "import requests" in Path(A.__file__).read_text()


def test_shipped_assets_config_is_well_formed():
    cfg = A.load_assets()
    names = [a["name"] for a in cfg["assets"]]
    assert {"maritime_s_base", "maritime_x_base", "yolov12n"} <= set(names) and len(names) == len(set(names))
    for a in cfg["assets"]:
        assert a["kind"] in ("weights", "reference", "data", "video") and a["target"].startswith(("$EDGE_ASSETS", "$EDGE_DATA"))
        if a["kind"] in ("weights", "data"):
            assert len(a["sha256"]) == 64 and a["size_bytes"] > 0
        assert a["kind"] != "video"                          # videos are not assets: the user points `stream --videos-dir` at their own copy
        if a["name"] != "yolov12n":
            assert a["url"].startswith("gdrive:") and len(a["url"]) > 20     # weights and labelled frames: Google Drive; overridable at run time
    y = next(a for a in cfg["assets"] if a["name"] == "yolov12n")
    assert y["url"].startswith("https://github.com/sunsmarterjie/yolov12/releases/") and y["sha256"] == "37080c2891b94c62998f0bfb552dd70c32f9f2ee36618b9e7b3da49b49e150ac"


# ---------------------------------------------------------------- mocked responses only (no socket is opened)
class FakeResp:
    def __init__(self, status=200, body=b"", ctype="application/octet-stream", headers=None):
        self.status_code, self.body = status, body
        self.headers = {"Content-Type": ctype, **(headers or {})}

    @property
    def text(self):
        return self.body.decode()

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]


class FakeSession:
    def __init__(self, route):
        self.route, self.calls = route, []

    def get(self, url, stream=True, headers=None, timeout=None, allow_redirects=True):
        self.calls.append((url, dict(headers or {})))
        return self.route(url, headers or {})


def test_mocked_drive_confirmation_page_then_file(tmp_path, monkeypatch):
    data = b"payload" * 500
    page = ('<html><form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'
            '<input type="hidden" name="id" value="F1"><input type="hidden" name="export" value="download"><input type="hidden" name="resourcekey" value="0-rk">'
            '<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="u-9"></form></html>')

    def route(url, h):
        if url.startswith("https://drive.usercontent.google.com/download") and "confirm=t" in url and "uuid=u-9" in url and "resourcekey=0-rk" in url:
            return FakeResp(body=data)
        if url.startswith("https://drive.google.com/uc?export=download&id=F1"):
            return FakeResp(body=page.encode(), ctype="text/html; charset=utf-8")
        return FakeResp(404)
    sess = FakeSession(route)
    A.download("https://drive.google.com/file/d/F1/view?usp=sharing&resourcekey=0-rk", tmp_path / "f.bin", session=sess)
    assert (tmp_path / "f.bin").read_bytes() == data and len(sess.calls) == 2
    assert "resourcekey=0-rk" in sess.calls[0][0]                            # the resource key of an older shared file is sent from the start


def test_mocked_resourcekey_is_parsed_from_every_link_form():
    assert A.drive_resourcekey("https://drive.google.com/file/d/F1/view?resourcekey=0-abc") == "0-abc"
    assert A.drive_resourcekey("https://drive.google.com/open?id=F1&resourcekey=0-abc") == "0-abc"
    assert A.drive_resourcekey("gdrive:F1?resourcekey=0-abc") == "0-abc" and A.drive_file_id("gdrive:F1?resourcekey=0-abc") == "F1"
    assert A.drive_resourcekey("https://drive.google.com/file/d/F1/view") is None


def test_mocked_resume_sends_a_range_request_and_appends(tmp_path):
    data = bytes(range(256)) * 100
    seen = []

    def route(url, h):
        seen.append(h.get("Range"))
        start = int(h["Range"][6:-1])
        return FakeResp(206, data[start:], headers={"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"})
    (tmp_path / "f.bin.part").write_bytes(data[:10000])
    A.download("https://example.org/f.bin", tmp_path / "f.bin", session=FakeSession(route))
    assert seen == ["bytes=10000-"] and (tmp_path / "f.bin").read_bytes() == data and not (tmp_path / "f.bin.part").exists()


def test_mocked_server_ignoring_range_restarts_the_download(tmp_path):
    data = b"abcdef" * 1000
    (tmp_path / "f.bin.part").write_bytes(b"garbage")
    A.download("https://example.org/f.bin", tmp_path / "f.bin", session=FakeSession(lambda u, h: FakeResp(200, data)))
    assert (tmp_path / "f.bin").read_bytes() == data


def test_mocked_sha256_mismatch_is_reported_and_leaves_nothing(tmp_path):
    good = b"weights" * 100
    cfg = cfg_for("w", good, "https://example.org/w.pt")
    r = A.fetch_assets(cfg, assets_dir=tmp_path / "a", data_dir=tmp_path / "d", session=FakeSession(lambda u, h: FakeResp(200, b"tampered" * 100)))
    assert r["w"]["status"] == "failed" and "sha256 mismatch" in r["w"]["reason"] and not (tmp_path / "a/weights/w.pt").exists() and not r["_all_ok"]
    assert not list((tmp_path / "a/.downloads").glob("*")) if (tmp_path / "a/.downloads").exists() else True


def test_mocked_http_error_and_drive_quota_page(tmp_path):
    cfg = cfg_for("w", b"x", "https://example.org/w.pt")
    r = A.fetch_assets(cfg, assets_dir=tmp_path / "a", data_dir=tmp_path / "d", session=FakeSession(lambda u, h: FakeResp(403)))
    assert r["w"]["status"] == "failed" and "403" in r["w"]["reason"]
    with pytest.raises(SystemExit, match="web page instead of the file"):
        A.download("gdrive:F1", tmp_path / "q.bin", session=FakeSession(lambda u, h: FakeResp(200, b"<html>Quota exceeded</html>", "text/html")))


# ---------------------------------------------------------------- timeouts, Drive web pages, progress (v0.5.1)
import socket as _socket
import time as _time


class HangingServer:
    """Raw socket server: `mode` = 'silent' (accepts, never answers) or 'stall' (sends the headers and 200,000 body bytes, then goes quiet)."""

    def __init__(self, mode):
        self.mode = mode
        self.sock = _socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.conns = []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self.conns.append(c)
            if self.mode == "stall":
                c.recv(65536)
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 1000000\r\n\r\n" + b"x" * 200000)

    def close(self):
        self.sock.close()
        for c in self.conns:
            c.close()


def test_shipped_timeouts_are_15_s_connect_and_60_s_read():
    assert A.download.__defaults__[2] == (15, 60) and (A.CONNECT_TIMEOUT, A.READ_TIMEOUT) == (15, 60) and A.PROGRESS_EVERY == 10.0


def test_a_server_that_never_answers_times_out_with_the_host_and_the_cause(tmp_path):
    srv = HangingServer("silent")
    try:
        t0 = _time.monotonic()
        with pytest.raises(SystemExit, match=r"127\.0\.0\.1 accepted the connection but sent nothing for 0\.5 s"):
            A.download(f"http://127.0.0.1:{srv.port}/w.pt", tmp_path / "w.bin", timeout=(1, 0.5), label="maritime_s_base")
        assert _time.monotonic() - t0 < 10
    finally:
        srv.close()


def test_a_transfer_that_stalls_keeps_the_partial_file_for_the_next_run(tmp_path):
    srv = HangingServer("stall")
    try:
        with pytest.raises(SystemExit, match=r"stopped sending data for 0\.5 s; the partial file is kept"):
            A.download(f"http://127.0.0.1:{srv.port}/w.pt", tmp_path / "w.bin", timeout=(1, 0.5))
        assert 0 < (tmp_path / "w.bin.part").stat().st_size <= 200000 and not (tmp_path / "w.bin").exists()
    finally:
        srv.close()


def test_connect_failures_name_the_host_and_suggest_network_host_or_dns(tmp_path):
    import requests

    class Boom:
        def __init__(self, exc):
            self.exc = exc

        def get(self, *a, **k):
            raise self.exc
    for exc, text in ((requests.exceptions.ConnectTimeout("t"), "could not connect to drive.google.com within 15 s"),
                      (requests.exceptions.ConnectionError("dns"), "cannot reach drive.google.com")):
        with pytest.raises(SystemExit) as e:
            A.download("https://drive.google.com/file/d/F1/view", tmp_path / "f.bin", session=Boom(exc), label="data_clear_val")
        msg = str(e.value)
        assert text in msg and "data_clear_val" in msg and "--network host" in msg and "--dns <resolver>" in msg and "--network none" in msg


@pytest.mark.parametrize("page,reason", [("<html>Google Drive - Quota exceeded for this file</html>", "quota of this file is exceeded"),
                                         ("<html>You need access. Request access</html>", "not shared with"),
                                         ("<html>Sorry, the file you have requested does not exist.</html>", "does not exist"),
                                         ("<html>something else</html>", "quota, permission or a changed Drive page")])
def test_drive_html_page_stops_with_the_file_id_and_the_reason(tmp_path, page, reason):
    with pytest.raises(SystemExit) as e:
        A.download("gdrive:1AbC_def-9", tmp_path / "f.bin", session=FakeSession(lambda u, h: FakeResp(200, page.encode(), "text/html; charset=utf-8")))
    assert "web page instead of the file" in str(e.value) and "file id 1AbC_def-9" in str(e.value) and reason in str(e.value)


def test_fetch_assets_stops_at_the_first_drive_page_and_keeps_what_it_has(tmp_path):
    good, other = b"first" * 200, b"second" * 200
    cfg = {"assets": [{"name": "a", "kind": "weights", "url": "https://example.org/a.pt", "sha256": sha(good), "size_bytes": len(good), "target": "$EDGE_ASSETS/weights/a.pt", "extract": False},
                      {"name": "b", "kind": "weights", "url": "gdrive:FILE_B", "sha256": sha(other), "size_bytes": len(other), "target": "$EDGE_ASSETS/weights/b.pt", "extract": False},
                      {"name": "c", "kind": "weights", "url": "https://example.org/c.pt", "sha256": sha(good), "size_bytes": len(good), "target": "$EDGE_ASSETS/weights/c.pt", "extract": False}]}
    sess = FakeSession(lambda u, h: FakeResp(200, b"<html>Quota exceeded</html>", "text/html") if "FILE_B" in u else FakeResp(200, good))
    with pytest.raises(SystemExit) as e:
        A.fetch_assets(cfg, assets_dir=tmp_path / "a", data_dir=tmp_path / "d", session=sess)
    assert "b:" in str(e.value) and "file id FILE_B" in str(e.value) and "quota" in str(e.value) and "Stopped" in str(e.value)
    assert (tmp_path / "a/weights/a.pt").read_bytes() == good and not (tmp_path / "a/weights/c.pt").exists()        # a is kept, c was never tried
    assert not any("example.org/c.pt" in u for u, _ in sess.calls)


def test_a_progress_line_is_printed_while_a_file_downloads(tmp_path, monkeypatch, capsys):
    data = b"abcdefgh" * 4096
    monkeypatch.setattr(A, "PROGRESS_EVERY", 0.0)                           # a line after every chunk
    monkeypatch.setattr(A, "NET_CHUNK", 8192)
    A.download("https://example.org/w.pt", tmp_path / "w.bin", session=FakeSession(lambda u, h: FakeResp(200, data)), expected_size=len(data), label="maritime_s_base")
    err = capsys.readouterr().err
    assert "[fetch-assets] maritime_s_base: connecting to example.org" in err
    lines = [l for l in err.splitlines() if "MB" in l and "%" in l]
    assert len(lines) >= 2 and "of 0.0 MB" in lines[0] and "100 %" in lines[-1] and "MB/s" in lines[0]
    assert "maritime_s_base: 0.0 MB received" in err


def test_no_progress_line_is_needed_for_a_short_download_but_the_summary_line_is_always_there(tmp_path, capsys):
    A.download("https://example.org/w.pt", tmp_path / "w.bin", session=FakeSession(lambda u, h: FakeResp(200, b"tiny")), label="x1")
    assert "x1: 0.0 MB received" in capsys.readouterr().err
