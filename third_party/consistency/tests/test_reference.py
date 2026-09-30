"""Numeric equality of the locked monitor against stored reference signals.

The fixture holds 100 transitions (the sample used by the Jetson TX2 correctness gate:
``random.Random(0).sample`` of the 6287 reference rows) with their prediction files and
the expected ``status``, ``N_t``, ``N_t1``, ``K``, ``G_mc`` and ``alarm``.
"""

import csv
import math
import os

from consistency.io import load_predictions
from consistency.monitor import compute_locked_monitor

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _f(text):
    return float(text)


def _abs_err(observed, expected):
    if math.isfinite(observed) and math.isfinite(expected):
        return abs(observed - expected)
    assert math.isnan(observed) == math.isnan(expected), (observed, expected)
    if not math.isnan(observed):
        assert observed == expected  # matching infinities
    return 0.0


def test_reference_signals_exact():
    with open(os.path.join(FIXTURES, "reference_signals.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 100
    store = load_predictions(os.path.join(FIXTURES, "predictions"))

    max_gmc = 0.0
    max_alarm = 0.0
    for ref in rows:
        res = compute_locked_monitor(store.boxes(ref["key"]), store.boxes(ref["next_key"]))
        assert res.status == ref["status"]
        assert (res.N_t, res.N_t1, res.K) == (int(ref["N_t"]), int(ref["N_t1"]), int(ref["K"]))
        max_gmc = max(max_gmc, _abs_err(res.G_mc, _f(ref["G_mc"])))
        max_alarm = max(max_alarm, _abs_err(res.alarm, _f(ref["alarm__Gmc-full-1ch-incl"])))

    assert max_gmc == 0.0
    assert max_alarm == 0.0
