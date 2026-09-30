"""Compare the requests of a stream run with configs/expected_results.yaml.

    ./run.sh compare-expected --mode images|video --out RUN_DIR [--expected configs/expected_results.yaml]

RUN_DIR is the --out directory of `stream` (requests.jsonl, summary.json). Exit code 0: same requests; 1: differences (a WARNING with an explanation - windows whose drop is close to
the threshold can flip between input modes and, rarely, between GPUs); 2: the run or the expectation cannot be read.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

WHY = ("Why this can happen: the trigger fires when a window's consistency drops by more than delta against the previous window. A window whose drop is close to delta can fall on either side "
       "when the input differs slightly (decoded video vs the JPEG frames the threshold was calibrated on: median window statistic difference about 0.005, 90th percentile 0.039) or when the GPU/driver "
       "changes the last digits of the detections. Compare the drops listed above with delta before suspecting the installation; a different NUMBER of frames or a crash would be a real problem.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["images", "video"], required=True)
    ap.add_argument("--out", required=True, help="output directory of the stream run")
    ap.add_argument("--expected", default=str(ROOT / "configs" / "expected_results.yaml"))
    a = ap.parse_args(argv)
    try:
        exp = yaml.safe_load(open(a.expected))
        m = exp["modes"][a.mode]
        reqs = [json.loads(l) for l in open(Path(a.out) / "requests.jsonl") if l.strip()]
        summ = json.load(open(Path(a.out) / "summary.json"))
    except (OSError, KeyError, ValueError) as e:
        print(f"[compare-expected] cannot read the run or the expectation: {e}", file=sys.stderr)
        return 2
    if a.mode == "video":
        got = [(r["video"], r["frame_index"]) for r in reqs]
        want = [(x["video"], x["frame_index"]) for x in m["requests"]]
        fmt = lambda k: f"{k[0]} frame {k[1]}"
    else:
        got = [r["clip_id"] for r in reqs]
        want = list(m["requests"])
        fmt = str
    print(f"[compare-expected] mode {a.mode}, stream {(summ.get('resolution') or {}).get('stream') or exp['stream']}: {len(got)} request(s), {summ['n_frames']} frames "
          f"(expected {len(want)} request(s), {exp['n_frames']} frames; release {exp['release']})")
    for k, r in zip(range(len(reqs)), reqs):
        print(f"  request {k + 1}: {fmt(got[k])}  drop {r['drop']:.4f} vs delta {r['delta']}  p={r['p_value']:.4f}")
    missing = [w for w in want if w not in got]
    extra = [g for g in got if g not in want]
    problems = []
    if summ["n_frames"] != exp["n_frames"]:
        problems.append(f"frames processed: {summ['n_frames']} instead of {exp['n_frames']} (missing videos or frames? see resolution.json)")
    if missing:
        problems.append("expected but not raised: " + ", ".join(fmt(w) for w in missing))
    if extra:
        problems.append("raised but not expected: " + ", ".join(fmt(g) for g in extra))
    if not problems and got != want:
        problems.append("same requests but in a different order")
    if not problems:
        print("[compare-expected] OK: the requests are exactly the expected ones")
        return 0
    print("[compare-expected] WARNING: the run differs from the expected results:")
    for p in problems:
        print("  - " + p)
    print("  " + WHY)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                   # an unexpected error is not "the results differ": exit 2, never 1
        import traceback
        traceback.print_exc()
        sys.exit(2)
