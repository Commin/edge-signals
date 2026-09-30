"""Command line: stream (run the signal stage on a folder / list of frames) and fit-reference (feature_drift reference)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import registry
from .base import ReferenceBundle
from .config import load_configs, load_yaml
from .engine import SignalEngine
from .streams import order_frames, read_list, stream_records

ROOT = Path(__file__).resolve().parents[1]


def rp(p) -> Path:
    """Config path: $ENV variables are expanded (EDGE_ASSETS, EDGE_DATA); relative paths are taken from the package root."""
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return p if p.is_absolute() else ROOT / p


def load_references(signals_cfg: dict) -> dict:
    refs = {}
    for name, s in signals_cfg.items():
        ref = (s.get("params") or {}).get("reference")
        if s["measure"] and ref:
            refs[name] = ReferenceBundle.load(rp(ref))
    return refs


def needs_features(signals_cfg: dict) -> bool:
    reg = registry.available()
    return any(s["measure"] and "features" in reg[n].requires for n, s in signals_cfg.items())


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def collect(a) -> list:
    if a.list:
        return order_frames(read_list(a.list, a.images_root))
    d = Path(a.images_dir)
    return order_frames(p for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("."))


def order_images_by_replay(paths, a, out):
    """Labelled-image mode with --stream / --split: keep the frames of the clips named by the replay order and put the clips in that order
    (frames in numeric order inside a clip). Writes resolution.json (code -> number of frames, or missing)."""
    from . import replay
    from .streams import clip_and_index
    order = replay.load_order(rp(a.order))
    items = replay.ordered_codes(order, a.stream, a.split)
    by = {}
    for p in paths:
        by.setdefault(clip_and_index(p.stem)[0], []).append(p)
    ordered, res = [], {}
    for it in items:
        if it["code"] in by:
            ordered += order_frames(by[it["code"]])
            res[it["code"]] = {"status": "found", "n_frames": len(by[it["code"]]), "split": it["split"]}
        else:
            res[it["code"]] = {"status": "missing", "split": it["split"]}
    missing = [c for c, v in res.items() if v["status"] == "missing"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolution.json").write_text(json.dumps({"input": "images", "stream": f"split:{a.split}" if a.split else (a.stream or order["default_stream"]),
                                                     "on_missing": a.on_missing, "resolution": res, "n_frames": len(ordered), "not_replayed": missing}, indent=2))
    for c in missing:
        print(f"[{'error' if a.on_missing == 'error' else 'warning'}] clip {c} ({res[c]['split']}): no labelled frames found", file=sys.stderr)
    if missing and a.on_missing == "error":
        sys.exit(f"{len(missing)} clip(s) without frames: {', '.join(missing)} (see {out / 'resolution.json'})")
    return ordered


def progress_lines(records, eng, total=None, out=None):
    """Pass the records through and print one line per video (or clip) when it is finished: name, frames processed, requests so far.
    A video is finished when the next one starts (the engine has then processed all its frames) or when the stream ends."""
    state = {"key": None, "n": 0, "k": 0}
    out = out or sys.stderr

    def line():
        if state["key"] is not None:
            of = f"/{total}" if total else ""
            print(f"[stream] video {state['k']}{of} {state['key']}: {state['n']} frames processed, {eng.n_requests} request(s) so far (stream total {eng.n_frames} frames)", file=out, flush=True)
    for r in records:
        key = r.video or r.clip_id
        if key != state["key"]:
            line()
            state.update(key=key, n=0, k=state["k"] + 1)
        state["n"] += 1
        yield r
    line()                                            # the stream is exhausted: the engine has processed the last frame of the last video


def cmd_stream(a):
    from .inference import Detector
    inf = load_yaml(rp(a.inference))
    sc, tc = load_configs(rp(a.signals), rp(a.trigger), inf['stream']['fps'])
    refs = load_references(sc)
    out = rp(a.out) if not Path(a.out).is_absolute() else Path(a.out)
    healthy = ReferenceBundle.load(rp(tc["healthy_changes"])) if tc.get("healthy_changes") else None
    st = inf["stream"]
    videos, stride, resolution = None, None, None
    if a.videos_dir or a.video or a.video_list:       # the stride is checked BEFORE anything heavy is loaded
        from . import replay
        from . import video as V
        if a.videos_dir:                              # the replay order decides which videos, in which order; the folder is the user's own copy
            order = replay.load_order(rp(a.order))
            resolution = replay.resolve(replay.ordered_codes(order, a.stream, a.split), a.videos_dir, a.on_missing)
            resolution["order_file"] = str(rp(a.order))
            resolution["stream"] = f"split:{a.split}" if a.split else (a.stream or order["default_stream"])
            out.mkdir(parents=True, exist_ok=True)
            replay.write_resolution(resolution, out)
            for m in resolution["not_replayed"]:
                extra = f" (candidates: {', '.join(m['candidates'])})" if m.get("candidates") else ""
                print(f"[{'error' if a.on_missing == 'error' else 'warning'}] video {m['code']} ({m['split']}): {m['reason']}{extra}", file=sys.stderr)
            if resolution["not_replayed"] and a.on_missing == "error":
                sys.exit(f"{len(resolution['not_replayed'])} video(s) missing or ambiguous: {', '.join(m['code'] for m in resolution['not_replayed'])} (see {out / 'resolution.json'})")
            videos = [Path(x["file"]) for x in resolution["replayed"]]
        else:
            videos = [Path(x) for x in (a.video or [])] + (V.read_video_list(a.video_list) if a.video_list else [])
        if not videos:
            sys.exit("no input videos" + (f": none of the {resolution['n_requested']} videos of the order was found in {a.videos_dir}" if resolution else ""))
        stride = a.frame_stride or (st.get("video") or {}).get("frame_stride", 5)
        calibrated = healthy.meta.get("frame_stride") if healthy is not None else tc.get("calibrated_stride")
        for p in videos:
            V.check_stride(stride, st["fps"], V.probe(p)["fps"], calibrated)
    else:
        if not (a.list or a.images_dir):
            sys.exit("stream needs --videos-dir <folder with the original videos> (see README), or --images-dir / --list for a folder of frames")
        paths = collect(a)
        if paths and (a.stream or a.split):
            paths = order_images_by_replay(paths, a, out)
        if not paths:
            sys.exit("no input images")
    det = Detector(rp(inf["model"]["weights"]), inf["predict"], inf["env"], want_features=needs_features(sc))
    for n, r in refs.items():
        if r.meta.get("backbone_params_id") != det.backbone_params_id:
            print(f"[warning] reference of '{n}' was fitted on a different backbone: its values will be INVALID "
                  f"(backbone_mismatch)", file=sys.stderr)
    eng = SignalEngine(sc, tc, refs, out_dir=out, fps=st["fps"], healthy=healthy)
    if videos:
        records = V.stream_video_records(det, videos, stride, st["fps"], min(inf["predict"]["chunk"], (st.get("video") or {}).get("chunk", 16)),
                                         st["device_id"], inf["model"]["version"])
    else:
        records = stream_records(det, paths, st["fps"], inf["predict"]["chunk"], st["device_id"], inf["model"]["version"])
    n_units = len(videos) if videos else len({(p.stem.rsplit("-", 1)[0]) for p in paths})
    prog = progress_lines(records, eng, n_units)
    t0 = time.perf_counter()
    res = eng.run(prog)
    summary = {"input": "video" if videos else "images", "videos": [Path(p).stem for p in videos] if videos else None,
               "resolution": None if resolution is None else {"stream": resolution["stream"], "n_requested": resolution["n_requested"], "n_replayed": len(resolution["replayed"]), "not_replayed": [m["code"] for m in resolution["not_replayed"]]}, "frame_stride": stride,
               "n_frames": eng.n_frames, "n_pairs": eng.n_pairs, "n_windows": len(res["windows"]),
               "n_windows_by_signal": {n: sum(1 for w in res["windows"] if w.signal == n) for n in sorted({w.signal for w in res["windows"]})},
               "n_windows_note": "n_windows counts window records over ALL signals (one per signal per window, as in windows.jsonl); n_windows_by_signal gives the count per signal (consistency windows = the trigger's windows)",
               "n_requests": len(res["triggers"]), "requests_by_rule": {r: sum(1 for x in res["triggers"] if x.rule == r) for r in {x.rule for x in res["triggers"]}}, "n_scene_cuts": len(res["context"]),
               "dropped_partial_windows": eng.dropped_partial_windows, "wall_s": round(time.perf_counter() - t0, 2),
               "signal_stage_s_per_signal": {k: round(v, 4) for k, v in eng.timing.items()},
               "feature_hook": {"layer_index": det.hook_index, "output_shape": det.hook_shape} if det.hook else None,
               "model": {"version": inf["model"]["version"], "weights_sha256": det.weights_sha256, "backbone_params_id": det.backbone_params_id, "backbone_state_id": det.backbone_state_id},
               "references": {n: r.id for n, r in refs.items()}, "healthy_changes": None if healthy is None else healthy.id, "python": platform.python_version(), **det.env,
               "config_sha256": {Path(x).name: sha(rp(x)) for x in (a.signals, a.trigger, a.inference)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_fit_reference(a):
    from .builtin.feature_drift import FeatureDrift
    from .inference import Detector
    inf = load_yaml(rp(a.inference))
    det = Detector(rp(inf["model"]["weights"]), inf["predict"], inf["env"], want_features=True)
    paths = collect(a)
    st = inf["stream"]
    recs = list(stream_records(det, paths, st["fps"], inf["predict"]["chunk"], st["device_id"], inf["model"]["version"]))
    params = dict(load_yaml(rp(a.signals))["signals"]["feature_drift"].get("params", {}))
    params.update(hook_layer=det.hook_index, hook_output_shape=list(det.hook_shape))
    bundle = FeatureDrift.fit_reference(recs, params)
    path = bundle.save(rp(a.out))
    print(json.dumps({"reference_id": bundle.id, "path": str(path), **bundle.meta}, indent=2))


# ---------------------------------------------------------------- adaptation stages
def _adapt(a):
    from .adapt.common import load_cfg
    return load_cfg(a.adapt_config, a.set)


def cmd_collect(a):
    from .adapt.collect import collect
    info = collect(_adapt(a), a.out, a.frames_list, a.requests, a.request_id, a.stream_list, a.since_t, a.fps, a.replay_ratio, a.replay_seed,
                   stream_out=a.stream_out, mapping=a.mapping, since_request_id=a.since_request_id, min_new_frames=a.min_new_frames)
    print(json.dumps(info, indent=2))
    if info.get("status") == "insufficient_data":       # logged; no retraining starts (exit code 3 lets a script branch)
        print(f"[collect] insufficient data: {info['n_adapt']} new labelled frames < minimum {info['min_new_frames']}; no retraining", file=sys.stderr)
        return 3


def cmd_annotate(a):
    from .adapt.annotate import annotate
    print(json.dumps(annotate(_adapt(a), a.out, a.images_root, a.mode, a.labels_dir, a.replay_labels_dir, a.teacher, a.check_labels_dir), indent=2))


def cmd_retrain(a):
    from .adapt.retrain import retrain
    inf = load_yaml(rp(a.inference))
    from .inference import env_guard
    env_guard(inf["env"])
    r = retrain(_adapt(a), a.out, a.images_root, a.base, a.seed, a.epochs)
    print(json.dumps({k: r[k] for k in ("weights", "weights_sha256", "per_phase_s", "peak_gpu_MiB", "epochs_trained")}, indent=2))


def exclude_collected_videos(evals: dict, run, include_collected: bool = False):
    """Held-out means held out: every video that is in the request's collected window (adaptation frames) or in its replay frames is removed from
    every held-out list. Reads <run>/collect/{adapt,replay}.txt; without them (no collect stage in this run) nothing is excluded.
    Returns (filtered lists {name: path}, report) - report = None when nothing applies."""
    from .adapt.common import read_stems
    cdir = Path(run) / "collect"
    if include_collected or not (cdir / "adapt.txt").exists():
        return evals, None
    vid = lambda st: st.rsplit("-", 1)[0]
    adapt = sorted({vid(x) for x in read_stems(cdir / "adapt.txt")})
    replay = sorted({vid(x) for x in read_stems(cdir / "replay.txt")}) if (cdir / "replay.txt").exists() else []
    bad = set(adapt) | set(replay)
    out, per = {}, {}
    (Path(run) / "evaluate").mkdir(parents=True, exist_ok=True)
    for n, lst in evals.items():
        stems = read_stems(lst)
        keep = [x for x in stems if vid(x) not in bad]
        gone = sorted({vid(x) for x in stems if vid(x) in bad})
        per[n] = {"videos": gone, "n_frames_before": len(stems), "n_frames_after": len(keep)}
        if keep:
            f = Path(run) / "evaluate" / f"{n}.held_out.txt"
            f.write_text("".join(x + "\n" for x in keep))
            out[n] = str(f)
    rep = {"rule": "videos of the collected window (adaptation frames) and of the replay frames are excluded from every held-out split", "adapt_videos": adapt, "replay_videos": replay, "per_split": per}
    for n, e in per.items():
        if e["videos"]:
            print(f"[evaluate] held-out split {n}: excluded {len(e['videos'])} video(s) of the collected window ({e['n_frames_before']} -> {e['n_frames_after']} frames): {', '.join(e['videos'])}", file=sys.stderr)
    return out, rep


def cmd_evaluate(a):
    from .adapt.evaluate import evaluate
    from .inference import env_guard
    env_guard(load_yaml(rp(a.inference))["env"])
    evals = dict(x.split("=", 1) for x in a.eval)
    if not evals and not a.in_stream:
        sys.exit("evaluate needs --eval NAME=LIST and/or --in-stream STREAM_OUT")
    weights = a.weights or str(Path(a.out) / "retrain" / "weights" / "last.pt")
    evals, excluded = exclude_collected_videos(evals, a.out, a.include_collected)
    views = {n: "heldout" for n in evals}
    if a.in_stream:                                    # view (a): the later frames of the request's video, after the request (expected optimistic)
        from .adapt.streamref import find_request, later_frames, load_mapping, load_stream, position
        _, reqs = load_stream(a.in_stream)
        pos = position(find_request(reqs, a.request_id))
        stems = later_frames(load_mapping(rp(a.mapping)), pos)
        lst = Path(a.out) / "evaluate" / f"{a.tag}_in_stream_after_request.txt"
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("".join(s + "\n" for s in stems))
        evals = {"in_stream_after_request": str(lst), **evals}
        views["in_stream_after_request"] = "in_stream (same video(s), after the request; expected optimistic)"
    res = evaluate(_adapt(a), a.out, weights, evals, a.images_root, a.labels_dir, a.tag, a.fps)
    for n, v in views.items():
        res["splits"][n]["view"] = v
    if excluded is not None:
        res["excluded_collected_videos"] = excluded
        for n, e in excluded["per_split"].items():
            if n in res["splits"]:
                res["splits"][n]["excluded_videos"], res["splits"][n]["n_frames_before_exclusion"] = e["videos"], e["n_frames_before"]
            else:
                res["splits"][n] = {"n_frames": 0, "skipped": "every video of this split is in the collected window (nothing held out)", "excluded_videos": e["videos"],
                                    "n_frames_before_exclusion": e["n_frames_before"], "view": "heldout"}
    from .adapt.common import jdump
    jdump(res, Path(a.out) / "evaluate" / f"{a.tag}.json")
    print(json.dumps({n: ({"skipped": s["skipped"]} if "skipped" in s else
                          {"pooled_mAP50": s["LABEL_REQUIRED"]["pooled_mAP50"], "proxy": s["LABEL_FREE"]["proxy_consistency_most_recent_window"]["consistency_window_median"]})
                      for n, s in res["splits"].items()}, indent=2))


def cmd_register(a):
    from .adapt.register import register
    from .inference import env_guard
    env_guard(load_yaml(rp(a.inference))["env"])
    weights = a.weights or str(Path(a.out) / "retrain" / "weights" / "last.pt")
    recipe = a.recipe_json or str(Path(a.out) / "retrain" / "retrain.json")
    ref = a.reference or (load_yaml(rp("configs/signals.yaml"))["signals"].get("feature_drift", {}).get("params") or {}).get("reference")
    print(json.dumps(register(_adapt(a), a.out, weights, a.parent, a.request_id, recipe if Path(recipe).exists() else None, ref, a.registry, a.stream_t,
                              a.family), indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fetch-assets":
        from . import assets
        return assets.main(argv[1:])
    ap = argparse.ArgumentParser(prog="edge_signals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("collect", cmd_collect), ("annotate", cmd_annotate), ("retrain", cmd_retrain), ("evaluate", cmd_evaluate), ("register", cmd_register)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="run directory: every stage writes only here")
        p.add_argument("--adapt-config", default="configs/adaptation.yaml")
        p.add_argument("--set", action="append", default=[], help="override a config value: section.key=value (repeatable)")
        p.add_argument("--inference", default="configs/inference.yaml")
        p.add_argument("--images-root")
        p.set_defaults(fn=fn)
        if name == "collect":
            p.add_argument("--frames-list", help="adaptation frames (one per line); alternative to --requests")
            p.add_argument("--requests", help="requests.jsonl of a stream run")
            p.add_argument("--request-id")
            p.add_argument("--stream-list", help="the frame list the stream ran on (defines stream time)")
            p.add_argument("--since-t", type=float, help="stream time of the last reset (default: the last registered version's, else the stream start)")
            p.add_argument("--fps", type=float, default=6.0)
            p.add_argument("--replay-ratio", type=float)
            p.add_argument("--replay-seed", type=int)
            p.add_argument("--stream-out", help="output directory of a `stream --video` run: same-video collection (needs --mapping)")
            p.add_argument("--mapping", default="data/mapping/frame_map.csv", help="(video, original frame index) <-> labelled image stem")
            p.add_argument("--since-request-id", help="the last reset: a request of the same stream run (default: the stream start)")
            p.add_argument("--min-new-frames", type=int, help="fewer new labelled frames than this: log 'insufficient data', no retraining")
        if name == "annotate":
            p.add_argument("--mode", choices=["provided", "pseudo"])
            p.add_argument("--labels-dir")
            p.add_argument("--replay-labels-dir")
            p.add_argument("--teacher", help="teacher weights (pseudo mode)")
            p.add_argument("--check-labels-dir", help="ground truth to score pseudo-labels against (LABEL-REQUIRED, reported separately)")
        if name == "retrain":
            p.add_argument("--base")
            p.add_argument("--seed", type=int)
            p.add_argument("--epochs", type=int)
        if name == "evaluate":
            p.add_argument("--weights")
            p.add_argument("--eval", action="append", default=[], metavar="NAME=LIST", help="labelled frame list to evaluate on (repeatable); optional with --in-stream")
            p.add_argument("--labels-dir", required=True)
            p.add_argument("--tag", required=True)
            p.add_argument("--fps", type=float, default=6.0)
            p.add_argument("--in-stream", metavar="STREAM_OUT", help="also evaluate the later frames of the request's video after the request (needs --request-id and --mapping)")
            p.add_argument("--request-id")
            p.add_argument("--mapping", default="data/mapping/frame_map.csv")
            p.add_argument("--include-collected", action="store_true", help="do NOT exclude the videos of the collected window from the held-out splits (default: they are excluded and reported)")
        if name == "register":
            p.add_argument("--weights")
            p.add_argument("--parent")
            p.add_argument("--request-id")
            p.add_argument("--recipe-json")
            p.add_argument("--reference", help="feature_drift reference bundle (default: from configs/signals.yaml)")
            p.add_argument("--registry")
            p.add_argument("--stream-t", type=float)
            p.add_argument("--family")
    for name in ("stream", "fit-reference"):
        p = sub.add_parser(name)
        p.add_argument("--list", help="text file: one frame per line (path or stem)")
        p.add_argument("--images-dir", help="folder of frames (used when --list is not given)")
        p.add_argument("--images-root", help="root for stems / relative paths in --list")
        p.add_argument("--out", default="outputs" if name == "stream" else "references")
        p.add_argument("--signals", default="configs/signals.yaml")
        p.add_argument("--trigger", default="configs/trigger.yaml")
        p.add_argument("--inference", default="configs/inference.yaml")
        if name == "stream":
            p.add_argument("--videos-dir", help="folder with YOUR copy of the original SMD videos (searched recursively; .avi .mp4 .mov .mkv); which videos and in which order: the replay order file")
            p.add_argument("--stream", help="named stream of the order file (default: its default_stream); with --videos-dir, or with --images-dir/--list to replay the labelled frames in the same order")
            p.add_argument("--split", help="replay one split of the order file instead of a stream")
            p.add_argument("--order", default="configs/replay_order.yaml", help="replay order file")
            p.add_argument("--on-missing", choices=["skip", "error"], default="skip", help="a video that is missing or ambiguous in --videos-dir: skip it with a warning (default) or stop")
            p.add_argument("--video", nargs="+", help="explicit original video file(s), processed in this order (alternative to --videos-dir)")
            p.add_argument("--video-list", help="text file: one video path per line")
            p.add_argument("--frame-stride", type=int, help="keep every N-th frame (default: configs/inference.yaml stream.video.frame_stride = 5, i.e. 6 fps from 30 fps)")
    a = ap.parse_args(argv)
    if hasattr(a, "fn"):
        return a.fn(a)
    {"stream": cmd_stream, "fit-reference": cmd_fit_reference}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
