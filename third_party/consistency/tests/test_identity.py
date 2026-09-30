"""Identity anchor and status rules.

With ``G_mc`` replaced by the uncompensated ``G`` (and every unmatched box counted in
full, which is what ``full`` means), the composite ``G * K / max(N_t, N_t1, 1)`` must
equal the uncompensated consistency ``R = G * m`` to machine precision.
"""

import csv
import math
import os

from consistency.decoupled import MatchConfig, Status, compute_frame_pair_signal
from consistency.io import load_predictions
from consistency.monitor import compute_locked_monitor

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _reference_transitions():
    with open(os.path.join(FIXTURES, "reference_signals.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    store = load_predictions(os.path.join(FIXTURES, "predictions"))
    return [(store.boxes(r["key"]), store.boxes(r["next_key"])) for r in rows]


def test_identity_R3_equals_R_on_reference_transitions():
    checked = 0
    worst = 0.0
    for preds_t, preds_t1 in _reference_transitions():
        sig = compute_frame_pair_signal(preds_t, preds_t1, MatchConfig())
        if sig.status is not Status.MATCHED:
            continue
        r3 = sig.G * sig.K / max(sig.N_t, sig.N_t1, 1)
        worst = max(worst, abs(r3 - sig.R))
        checked += 1
    assert checked > 0
    assert worst <= 1e-15


def test_zero_motion_gives_uncompensated_score():
    frame = [[0, 0.30, 0.40, 0.10, 0.10, 0.9], [1, 0.70, 0.60, 0.20, 0.10, 0.8],
             [0, 0.50, 0.20, 0.05, 0.08, 0.7]]
    res = compute_locked_monitor(frame, [list(b) for b in frame])
    sig = compute_frame_pair_signal(frame, [list(b) for b in frame])
    assert (res.dx, res.dy) == (0.0, 0.0)
    # G comes from the vectorised IoU matrix, G_mc from the scalar IoU: equal to ~1e-15
    assert abs(res.G_mc - sig.G) <= 1e-12 and abs(sig.G - 1.0) <= 1e-12
    assert abs(res.consistency - sig.R) <= 1e-12


def test_global_shift_is_compensated():
    frame_t = [[0, 0.30, 0.40, 0.10, 0.10, 0.9], [1, 0.70, 0.60, 0.20, 0.10, 0.8],
               [0, 0.50, 0.20, 0.05, 0.08, 0.7]]
    frame_t1 = [[b[0], b[1] + 0.02, b[2] - 0.01] + b[3:] for b in frame_t]
    res = compute_locked_monitor(frame_t, frame_t1)
    sig = compute_frame_pair_signal(frame_t, frame_t1)
    assert abs(res.dx - 0.02) < 1e-12 and abs(res.dy + 0.01) < 1e-12
    assert res.G < 1.0 and res.G == sig.G
    assert res.G_mc > 0.999999


def test_status_rules():
    box = [[0, 0.5, 0.5, 0.1, 0.1, 0.9]]
    empty = compute_locked_monitor([], [])
    assert empty.status == "EMPTY_PAIR"
    assert math.isnan(empty.alarm) and math.isnan(empty.consistency)

    one_sided = compute_locked_monitor(box, [])
    assert one_sided.status == "ONE_SIDED"
    assert one_sided.alarm == 1.0 and one_sided.consistency == 0.0

    # different class: both frames non-empty, no admissible pair
    no_match = compute_locked_monitor(box, [[1, 0.5, 0.5, 0.1, 0.1, 0.9]])
    assert no_match.status == "NO_MATCH"
    assert no_match.alarm == 1.0 and no_match.consistency == 0.0

    # IoU below the 0.05 floor: rejected even for the same class
    far = compute_locked_monitor(box, [[0, 0.9, 0.9, 0.1, 0.1, 0.9]])
    assert far.status == "NO_MATCH"
