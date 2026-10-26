"""Counter-based random numbers: every value is a pure function of (seed, layer, keys...), so results never depend on the order or the range of a query.

A vectorised splitmix64 hash maps the integer keys to 64 bits; uniforms, normals and value noise are derived from it. No generator state exists.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

# layers (what a draw is for)
L_CONFIG, L_TIMELINE, L_EPISODE, L_MATRIX, L_NOBJ, L_LAMBDA, L_Z, L_RECALL, L_PRECISION, L_TP, L_FP, L_CONF, L_CONS, L_WANDER = range(1, 15)

_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_MASK = (1 << 64) - 1


def _mix(x):
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _G
        z = (z ^ (z >> _S30)) * _M1
        z = (z ^ (z >> _S27)) * _M2
        return z ^ (z >> _S31)


def _u64(p):
    if isinstance(p, (int, np.integer)):
        return np.uint64(int(p) & _MASK)
    return np.asarray(p).astype(np.int64).astype(np.uint64)


def key(*parts):
    """64-bit hash of the integer keys (scalars or broadcastable int arrays)."""
    h = np.uint64(0)
    with np.errstate(over="ignore"):
        for i, p in enumerate(parts):
            h = _mix(h ^ _mix(_u64(p) + np.uint64(i + 1) * _G))
    return h


def uniform(*parts):
    """Uniform on the open interval (0, 1)."""
    return ((key(*parts) >> _S11).astype(np.float64) + 0.5) / 9007199254740992.0


def normal(*parts):
    return ndtri(uniform(*parts))


def _norm_const(shape="cosine", n=20001):
    x = np.linspace(0, 1, n, endpoint=False)
    w = (1 - np.cos(np.pi * x)) / 2
    return 1.0 / np.sqrt(np.mean((1 - w) ** 2 + w ** 2))


_VN = _norm_const()


def _knot(df, *parts):
    """Unit-variance knot values: Gaussian, or Student-t with `df` degrees of freedom (heavier tails: occasional large excursions)."""
    if df is None:
        return normal(*parts)
    return student_t_ppf(uniform(*parts), df) / np.sqrt(df / (df - 2.0))


def student_t_ppf(u, df):
    from scipy.stats import t as _t
    return _t.ppf(u, df)


def value_noise(period_frames, frame, *parts, df=None):
    """Smooth random-access noise with unit variance: hashed knots every `period_frames` frames, cosine-interpolated."""
    f = np.asarray(frame, dtype=np.int64)
    P = int(period_frames)
    j, r = np.divmod(f, P)
    x = r / P
    w = (1 - np.cos(np.pi * x)) / 2
    return ((1 - w) * _knot(df, *parts, j) + w * _knot(df, *parts, j + 1)) * _VN


def reflect01(v):
    """Reflect into [0, 1]: a continuous fold, no point mass at the bounds."""
    w = np.mod(v, 2.0)
    return np.where(w > 1.0, 2.0 - w, w)
