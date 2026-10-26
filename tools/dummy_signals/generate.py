#!/usr/bin/env python3
"""Synthetic multi-edge streams in the real edge_signals log format. Synthetic data: see README.md.

  generate.py        --config C [--seed N] --out DIR [--logs minimal]    (no subcommand) world.json plus, for every edge, the stream of its initial model in DIR/<edge>/
  generate.py world  [--config C] [--seed N] --out DIR                    write world.json (all parameters and draws = ground truth)
  generate.py stream [--config C] [--seed N] --edge E --schedule 0:s_a,3600:x_b@edge2 --out DIR [--delta D] [--no-signals-log]
                                                                          signals / windows / requests / context / frames.jsonl, truth.jsonl, summary.json (+ world.json);
                                                                          --logs minimal: windows, requests, context and truth only
  generate.py clip   [--config C] [--seed N] --edge E --model M --t-end T [--n-pairs 30]
                                                                          clip score of model M on the data of edge E (the requester) for the 30 pairs ending at T

The model schedule is an INPUT: this tool selects nothing. Nothing is written inside the repository; --out is yours.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.dummy_signals import config as cfgmod  # noqa: E402
from tools.dummy_signals import stream as st  # noqa: E402
from tools.dummy_signals.world import World  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("world", "stream", "clip", "run", "-h", "--help"):
        argv = ["run"] + argv                                                    # the default command
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "world", "stream", "clip"):
        p = sub.add_parser(name)
        p.add_argument("--config", help="YAML overrides of tools/dummy_signals/default_config.yaml")
        p.add_argument("--seed", type=int, default=0)
        if name != "clip":
            p.add_argument("--out", required=True)
        if name in ("stream", "clip"):
            p.add_argument("--edge", required=True)
        if name == "stream":
            p.add_argument("--schedule", required=True, help="'0:s_a,3600:x_b@edge2': model id from time t (seconds); @edge = the edge it is fetched from (default: the first edge that stores it)")
        if name in ("run", "stream"):
            p.add_argument("--delta", type=float, help="trigger threshold (default: configs/trigger.yaml)")
            p.add_argument("--no-signals-log", action="store_true", help="leave signals.jsonl empty")
            p.add_argument("--logs", choices=["full", "minimal"], default="full", help="minimal: windows, requests, context and truth only (no per-pair signals.jsonl, no frames.jsonl)")
        if name == "clip":
            p.add_argument("--model", required=True)
            p.add_argument("--t-end", type=float, required=True)
            p.add_argument("--n-pairs", type=int, default=30)
    a = ap.parse_args(argv)
    try:
        w = World(a.config, a.seed)
        if a.cmd == "world":
            print(w.save(Path(a.out) / "world.json"))
        elif a.cmd == "run":
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            w.save(out / "world.json")
            for e in w.edges:
                w.stream(e["id"], [(0, e["initial_model"])], out / e["id"], not a.no_signals_log, a.delta, a.logs)
                s = json.load(open(out / e["id"] / "summary.json"))
                print(f"{e['id']}: model {e['initial_model']}, {s['n_frames']} frames, {s['n_windows_by_signal']} windows, {s['n_requests']} requests -> {out / e['id']}")
            print(f"world.json -> {out / 'world.json'}")
        elif a.cmd == "stream":
            Path(a.out).mkdir(parents=True, exist_ok=True)
            w.save(Path(a.out) / "world.json")
            sched = st.parse_schedule_text(a.schedule, [m["id"] for m in w.cfg["models"]])
            w.stream(a.edge, sched, a.out, not a.no_signals_log, a.delta, a.logs)
            print(json.dumps(json.load(open(Path(a.out) / "summary.json")), indent=2))
        else:
            print(json.dumps(w.clip_score(a.edge, a.model, a.t_end, a.n_pairs)))
    except (cfgmod.ConfigError, ValueError, KeyError, OSError) as e:
        print(f"generate.py: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
