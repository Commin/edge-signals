"""fetch-assets: download the runtime assets (weights, data, videos), verify SHA256, extract safely. Idempotent, resumable.

Nothing that is downloaded is part of the repository or of the image. Only `requests` is used (HTTPS, Google Drive including its
large-file confirmation page); `file://` URLs are accepted too (offline copies, tests). After fetch-assets every other command runs offline.

An asset (configs/assets.yaml):  name, kind (weights | reference | data | video), url (empty until published), sha256, size_bytes,
target (a path under /assets or /data; a directory for archives), extract (true for tar / zip archives).
URLs can be overridden with --url NAME=URL or with a mounted config (--assets-config), so a changed link never needs an image rebuild.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import yaml

CHUNK = 1 << 20
ROOT = Path(__file__).resolve().parents[1]
GDRIVE_HOSTS = ("drive.google.com", "docs.google.com", "drive.usercontent.google.com")
DRIVE_BASE = "https://drive.google.com"          # module constants so that tests can point the Drive flow at a local server


class AssetError(SystemExit):
    pass


class DriveError(AssetError):
    """Google Drive sent a web page (quota, permission, wrong id) instead of the file: fetch-assets stops."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- Google Drive
def drive_file_id(url: str) -> Optional[str]:
    """File id of a Google Drive share / download link (or 'gdrive:<id>'), else None."""
    if url.startswith("gdrive:"):
        return url[len("gdrive:"):].split("?")[0]
    u = urlparse(url)
    if u.hostname not in GDRIVE_HOSTS:
        return None
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", u.path)
    if m:
        return m.group(1)
    q = parse_qs(u.query)
    return q["id"][0] if "id" in q else None


def drive_resourcekey(url: str) -> Optional[str]:
    """`resourcekey` of a link to an older shared file (`...?resourcekey=0-abc` or 'gdrive:<id>?resourcekey=0-abc'), else None."""
    q = parse_qs(urlparse(url).query if not url.startswith("gdrive:") else url.partition("?")[2])
    return q["resourcekey"][0] if "resourcekey" in q else None


def drive_confirmation(page: str, file_id: str) -> Optional[str]:
    """URL that continues a Drive 'file too large to scan for viruses' interstitial, from its HTML, or None if the page is something else."""
    m = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', page) or re.search(r'<form[^>]*action="([^"]*download[^"]*)"', page)
    if m:                                                    # current interstitial: a form whose hidden inputs carry confirm / uuid / id
        action = html.unescape(m.group(1))
        params = {n: html.unescape(v) for n, v in re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page)}
        params.setdefault("id", file_id)
        params.setdefault("export", "download")
        return action + ("&" if "?" in action else "?") + "&".join(f"{k}={v}" for k, v in params.items())
    m = re.search(r'href="(/uc\?export=download[^"]*confirm=[^"]+)"', page)
    if m:                                                    # older interstitial: a link with a confirm token
        return DRIVE_BASE + html.unescape(m.group(1))
    return None


# ---------------------------------------------------------------- download
CONNECT_TIMEOUT, READ_TIMEOUT = 15, 60            # seconds: connecting, and waiting for the next bytes
NET_CHUNK = 1 << 16                               # bytes per network read: a stalled connection loses at most this much of the partial file
PROGRESS_EVERY = 10.0                             # a progress line at least every 10 s while a file downloads
NET_HINT = ("no network or DNS in the container? run fetch-assets with --network host or with --dns <resolver> (the other commands run with --network none)")


def _log(msg: str) -> None:
    print(f"[fetch-assets] {msg}", file=sys.stderr, flush=True)


def _host(url: str) -> str:
    return urlparse(url).hostname or url


def _stream_to(resp, part: Path, resume_from: int, expected_size: Optional[int], label: str = "", every: float = None) -> None:
    every = PROGRESS_EVERY if every is None else every
    mode = "ab" if resume_from and resp.status_code == 206 else "wb"
    done = resume_from if mode == "ab" else 0
    total = expected_size or None
    if total is None:
        cl = resp.headers.get("Content-Length")
        total = (int(cl) + done) if cl and cl.isdigit() else None
    t0 = last = time.monotonic()
    with open(part, mode) as f:
        for b in resp.iter_content(NET_CHUNK):
            if b:
                f.write(b)
                done += len(b)
            now = time.monotonic()
            if now - last >= every:
                last = now
                pct = f" ({100 * done / total:.0f} %)" if total else ""
                tot = f" of {total / 1e6:.1f} MB" if total else ""
                _log(f"{label or part.name}: {done / 1e6:.1f} MB{tot}{pct}, {done / 1e6 / max(now - t0, 1e-9):.1f} MB/s")
    _log(f"{label or part.name}: {done / 1e6:.1f} MB received")


def drive_page_reason(page: str) -> str:
    """Why Google Drive sent a web page: the likely cause, from the page text."""
    low = page.lower()
    if "quota" in low or "too many users" in low or "too many" in low:
        return "the download quota of this file is exceeded (try again later, or make a copy of the file in another Drive account)"
    if "request access" in low or "you need access" in low or "sign in" in low or "accounts.google.com" in low or "permission" in low:
        return "the file is not shared with \"anyone with the link\" (permission)"
    if "not found" in low or "404" in low or "can't be found" in low or "does not exist" in low:
        return "the file id does not exist (removed or wrong id)"
    return "quota, permission or a changed Drive page"


def download(url: str, dst: Path, session=None, expected_size: Optional[int] = None, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), label: str = "") -> Path:
    """Download `url` to `dst`. Resumes a partial `<dst>.part` with a Range request. HTTPS, Google Drive and file:// are supported.
    Explicit connect / read timeouts; a progress line at least every PROGRESS_EVERY s; a Drive web page instead of the file stops with the reason and the file id."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(dst) + ".part")
    label = label or dst.name
    if url.startswith("file://"):
        src = Path(unquote(urlparse(url).path))
        if not src.exists():
            raise AssetError(f"file not found: {src}")
        shutil.copyfile(src, part)
        os.replace(part, dst)
        return dst
    import requests
    s = session or requests.Session()
    fid = drive_file_id(url)
    rk = drive_resourcekey(url) if fid else None
    target = f"{DRIVE_BASE}/uc?export=download&id={fid}" + (f"&resourcekey={rk}" if rk else "") if fid else url
    for attempt in range(3):                                # a Drive interstitial needs one extra request
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        host = _host(target)
        _log(f"{label}: connecting to {host}" + (f" (resuming at {have / 1e6:.1f} MB)" if have else ""))
        try:
            r = s.get(target, stream=True, headers=headers, timeout=timeout, allow_redirects=True)
        except requests.exceptions.ConnectTimeout as e:
            raise AssetError(f"{label}: could not connect to {host} within {timeout[0]} s - {NET_HINT} ({type(e).__name__})")
        except requests.exceptions.ReadTimeout as e:
            raise AssetError(f"{label}: {host} accepted the connection but sent nothing for {timeout[1]} s - the server is stalled or the network is filtered; try again ({type(e).__name__})")
        except requests.exceptions.ConnectionError as e:
            raise AssetError(f"{label}: cannot reach {host} - {NET_HINT} ({type(e).__name__})")
        if r.status_code == 416 and have:                   # the partial file is already complete (or wrong): start over once
            part.unlink()
            continue
        if r.status_code >= 400:
            raise AssetError(f"download failed ({r.status_code}) for {url}" + (f" (Google Drive file id {fid})" if fid else ""))
        ctype = r.headers.get("Content-Type", "")
        if fid and "text/html" in ctype:
            page = r.text
            nxt = drive_confirmation(page, fid)
            if not nxt or attempt == 2:
                raise DriveError(f"Google Drive returned a web page instead of the file (file id {fid}): {drive_page_reason(page)}")
            target = nxt
            continue
        if have and r.status_code == 200:                   # the server ignored the Range header: restart
            have = 0
        try:
            _stream_to(r, part, have, expected_size, label)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            if isinstance(e, requests.exceptions.ReadTimeout) or "timed out" in str(e).lower():      # requests reports a stall while iterating as a ConnectionError
                raise AssetError(f"{label}: {host} stopped sending data for {timeout[1]} s; the partial file is kept and the next run resumes it")
            raise AssetError(f"{label}: the connection to {host} dropped ({type(e).__name__}); the partial file is kept and the next run resumes it")
        os.replace(part, dst)
        return dst
    raise AssetError(f"could not download {url}")


# ---------------------------------------------------------------- extraction
def extract_archive(path: Path, target: Path) -> int:
    """Safe extraction (no absolute paths, no '..', no links). tar is checked first: a tar whose last member is a zip (.pt) also looks like a zip."""
    target.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            members = t.getmembers()
            for m in members:
                if Path(m.name).is_absolute() or ".." in Path(m.name).parts or m.issym() or m.islnk():
                    raise AssetError(f"unsafe member in archive: {m.name}")
            t.extractall(target)
            return sum(1 for m in members if m.isfile())
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for m in z.namelist():
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    raise AssetError(f"unsafe path in archive: {m}")
            z.extractall(target)
            return sum(1 for m in z.namelist() if not m.endswith("/"))
    raise AssetError(f"{path.name}: not a zip / tar archive")


# ---------------------------------------------------------------- config and driver
def load_assets(path=None) -> dict:
    return yaml.safe_load(open(path or ROOT / "configs" / "assets.yaml"))


def resolve_target(target: str, assets_dir: Optional[Path], data_dir: Optional[Path]) -> Path:
    """$EDGE_ASSETS / $EDGE_DATA (or the literal /assets, /data) in a target become the given volume directories."""
    t = target
    for tok, base in (("$EDGE_ASSETS", assets_dir), ("${EDGE_ASSETS}", assets_dir), ("$EDGE_DATA", data_dir), ("${EDGE_DATA}", data_dir)):
        if base is not None and t.startswith(tok):
            t = str(base) + t[len(tok):]
    t = os.path.expandvars(t)
    for tok, base in (("/assets", assets_dir), ("/data", data_dir)):
        if base is not None and (t == tok or t.startswith(tok + "/")) and not Path(t).exists():
            return Path(base) / t[len(tok):].lstrip("/")
    return Path(t)


def state_path(assets_dir: Path) -> Path:
    return assets_dir / ".fetched.json"


def fetch_assets(cfg: dict, assets_dir=None, data_dir=None, only: Optional[List[str]] = None, url_overrides: Optional[Dict[str, str]] = None,
                 kinds: Optional[List[str]] = None, session=None, tmp_dir=None) -> dict:
    """Download / verify / extract every (selected) asset. Returns {name: {status, ...}, '_all_ok': bool}."""
    assets_dir = Path(assets_dir or os.environ.get("EDGE_ASSETS") or "/assets")
    data_dir = Path(data_dir or os.environ.get("EDGE_DATA") or "/data")
    over = url_overrides or {}
    tmp = Path(tmp_dir) if tmp_dir else assets_dir / ".downloads"
    st_file = state_path(assets_dir)
    state = json.loads(st_file.read_text()) if st_file.exists() else {}
    res, ok = {}, True
    for a in cfg["assets"]:
        name = a["name"]
        if (only and name not in only) or (kinds and a["kind"] not in kinds):
            continue
        target = resolve_target(a["target"], assets_dir, data_dir)
        url = over.get(name) or a.get("url") or ""
        sha = (a.get("sha256") or "").lower()
        marker = state.get(name)
        present = target.exists() and (target.is_file() or any(target.iterdir()))
        if present and marker and (not sha or marker.get("sha256") == sha):     # idempotent: already fetched and verified
            res[name] = {"status": "up_to_date", "target": str(target)}
            continue
        if not url:
            res[name] = {"status": "skipped", "reason": "no url (set assets.yaml, --url NAME=URL or a mounted --assets-config)"}
            ok = False
            continue
        arch = tmp / f"{name}.download"
        try:
            download(url, arch, session, a.get("size_bytes"), label=name)
        except DriveError as e:                            # quota / permission / wrong id: every further download would fail the same way
            raise SystemExit(f"[fetch-assets] {name}: {e}\nStopped; assets already fetched stay in place and are skipped on the next run.")
        except AssetError as e:
            res[name] = {"status": "failed", "reason": str(e)}
            ok = False
            continue
        got = sha256_file(arch)
        if sha and got != sha:
            arch.unlink()
            res[name] = {"status": "failed", "reason": f"sha256 mismatch (got {got}, expected {sha})"}
            ok = False
            continue
        if a.get("size_bytes") and arch.stat().st_size != int(a["size_bytes"]):
            got_size = arch.stat().st_size
            arch.unlink()
            res[name] = {"status": "failed", "reason": f"size {got_size} != expected {a['size_bytes']}"}
            ok = False
            continue
        if a.get("extract"):
            n = extract_archive(arch, target)
            arch.unlink()
            res[name] = {"status": "ok", "target": str(target), "n_files": n, "sha256": got}
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(arch, target)
            res[name] = {"status": "ok", "target": str(target), "size_bytes": target.stat().st_size, "sha256": got}
        state[name] = {"sha256": got}
        assets_dir.mkdir(parents=True, exist_ok=True)
        st_file.write_text(json.dumps(state, indent=1))
    try:
        if tmp.exists() and not any(tmp.iterdir()):
            tmp.rmdir()
    except OSError:
        pass
    res["_all_ok"] = ok
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="fetch-assets", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-config", help="assets.yaml (default: configs/assets.yaml; mount your own to change links without rebuilding the image)")
    ap.add_argument("--assets-dir", help="weights and references volume (default $EDGE_ASSETS, /assets)")
    ap.add_argument("--data-dir", help="data and videos volume (default $EDGE_DATA, /data)")
    ap.add_argument("--only", nargs="*", help="only these asset names")
    ap.add_argument("--kind", nargs="*", choices=["weights", "reference", "data", "video"], help="only these kinds")
    ap.add_argument("--url", action="append", default=[], metavar="NAME=URL", help="override the URL of an asset (https, Google Drive link or id 'gdrive:<id>', file://)")
    a = ap.parse_args(argv)
    r = fetch_assets(load_assets(a.assets_config), a.assets_dir, a.data_dir, a.only, dict(x.split("=", 1) for x in a.url), a.kind)
    print(json.dumps(r, indent=2))
    if not r["_all_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
