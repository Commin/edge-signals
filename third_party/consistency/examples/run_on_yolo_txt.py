"""Score every consecutive frame pair in a folder of YOLO ``.txt`` prediction files.

    python examples/run_on_yolo_txt.py --pred-dir path/to/predictions [--out scores.csv]

Files are ordered numerically by frame index; a pair is formed only between consecutive
frames of the same sequence prefix (see ``consistency/io.py`` for the accepted names).
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from consistency.io import load_predictions  # noqa: E402
from consistency.monitor import compute_locked_monitor  # noqa: E402

FIELDS = ["key", "next_key", "status", "N_t", "N_t1", "K", "G", "G_mc", "dx", "dy",
          "consistency", "alarm"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, help="folder of YOLO .txt predictions")
    ap.add_argument("--out", default="-", help="CSV output path (default: stdout)")
    args = ap.parse_args(argv)

    store = load_predictions(args.pred_dir)
    out = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    try:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for tr in store.transitions():
            res = compute_locked_monitor(store.boxes(tr.key), store.boxes(tr.next_key)).to_dict()
            row = {"key": tr.key, "next_key": tr.next_key}
            row.update({k: res[k] for k in FIELDS[2:]})
            writer.writerow(row)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
