import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "consistency"


def test_vendored_package_is_unchanged():
    lines = [l.split(None, 1) for l in (VENDOR / "FILES.sha256").read_text().splitlines() if l.strip()]
    assert len(lines) > 100
    for digest, rel in lines:
        p = VENDOR / rel.strip()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, rel
    assert re.search(r"commit\s*:\s*[0-9a-f]{40}", (VENDOR / "VENDORED.txt").read_text())


def test_consistency_signal_uses_the_vendored_copy():
    from edge_signals.builtin.consistency import VENDOR as V, _monitor, vendored_commit
    fn = _monitor()
    assert Path(fn.__code__.co_filename).resolve().is_relative_to(V.resolve()) if hasattr(Path, "is_relative_to") else True
    assert len(vendored_commit()) == 40


def test_feature_hook_pools_the_captured_layer_in_the_same_forward_pass():
    torch = pytest.importorskip("torch")
    from edge_signals.inference import FeatureHook
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 6, 3, padding=1))
    hook = FeatureHook(net[0])
    calls = []
    net.register_forward_hook(lambda m, i, o: calls.append(1))
    x = torch.randn(5, 3, 8, 8)
    y = net(x)
    v = hook.pop()
    assert len(calls) == 1 and v.shape == (5, 4) and hook.shape == (4, 8, 8)
    expect = net[0](x).mean(dim=(2, 3)).detach().numpy()
    hook.pop()
    assert abs(v - expect).max() < 1e-6
    hook.remove()
    net(x)
    assert hook.pop().shape[0] == 0


def test_env_guard_accepts_only_the_pinned_fork(monkeypatch, tmp_path):
    import sys
    import types
    from edge_signals import inference
    fake = types.ModuleType("ultralytics")
    fake.__version__ = "8.3.63"
    fake.__file__ = str(tmp_path / "site-packages" / "ultralytics" / "__init__.py")
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.delenv("YOLOV12_DIR", raising=False)
    cfg = {"ultralytics_version": "8.3.63", "fork_commit": "a" * 40, "fork_dir_envvar": "YOLOV12_DIR"}
    monkeypatch.setattr(inference, "installed_fork_commit", lambda: "a" * 40)
    assert inference.env_guard(cfg)["source"] == "pip"
    monkeypatch.setattr(inference, "installed_fork_commit", lambda: "b" * 40)
    with pytest.raises(SystemExit, match="pinned fork commit"):
        inference.env_guard(cfg)
    monkeypatch.setenv("YOLOV12_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(SystemExit, match="not from"):
        inference.env_guard(cfg)
    monkeypatch.setenv("YOLOV12_DIR", str(tmp_path / "site-packages"))
    assert inference.env_guard(cfg)["source"] == "fork_dir"
    fake.__version__ = "8.2.100"
    with pytest.raises(SystemExit, match="pinned"):
        inference.env_guard(cfg)


def test_device_auto_resolves_to_gpu_or_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from edge_signals.inference import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == 0 and resolve_device("cpu") == "cpu" and resolve_device(1) == 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
