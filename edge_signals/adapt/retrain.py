"""retrain: fine-tune from the active model on <run>/annotate (adaptation frames + replay frames).

The recipe lives in YAML (configs/adaptation.yaml: retrain.recipe). close_mosaic MUST be set explicitly. The fork's hidden Albumentations
(Blur, MedianBlur, ToGray, CLAHE) are replaced by the configured list and verified off from the dataset's actual transform pipeline; the
result is written to the log and to retrain.json, and the stage fails if a hidden transform is active. ultralytics fixes the DataLoader
generator seed to a constant, so `seed` would not change data order or augmentation; it is re-implemented here with seed offset (seed 0
= ultralytics' default behaviour). Logs wall-clock per phase and peak GPU memory. Writes only under <run>/retrain/.
"""
from __future__ import annotations

import csv
import os
import shutil
import time
from pathlib import Path

import yaml

from .common import jdump, jload, rp, sha256_file, stage_dir

REQUIRED = ("freeze", "epochs", "mosaic", "close_mosaic", "imgsz", "batch", "optimizer", "seed", "workers", "deterministic", "device", "augment", "albumentations")
HIDDEN = ("Blur", "MedianBlur", "ToGray", "CLAHE")


def check_recipe(r: dict) -> dict:
    miss = [k for k in REQUIRED if k not in r]
    if miss:
        raise SystemExit(f"retrain.recipe is missing {miss}" + ("  (close_mosaic must be set explicitly)" if "close_mosaic" in miss else ""))
    if not isinstance(r["close_mosaic"], int) or r["close_mosaic"] < 0:
        raise SystemExit("retrain.recipe.close_mosaic must be an integer >= 0 (0 = mosaic stays on for the whole retrain)")
    if not isinstance(r["freeze"], int) or r["freeze"] < 0:
        raise SystemExit("retrain.recipe.freeze must be an integer >= 0 (number of leading layers to freeze; 9 = the backbone)")
    return r


def patch_albumentations(spec):
    """Replace the fork's hidden Albumentations list with the configured one (no change to the fork's files)."""
    import ultralytics.data.augment as aug

    def init(self, p=1.0):
        self.p, self.transform, self.contains_spatial = p, None, False
        if spec:
            import albumentations as A
            self.transform = A.Compose([getattr(A, d["name"])(**{k: v for k, v in d.items() if k != "name"}) for d in spec])
    aug.Albumentations.__init__ = init


def patch_loader_seed(seed):
    """Generator seed = ultralytics' constant + RANK + seed (seed 0 = default behaviour)."""
    import torch
    from torch.utils.data import distributed
    import ultralytics.data.build as B
    import ultralytics.models.yolo.detect.train as DT
    from ultralytics.data.utils import PIN_MEMORY
    from ultralytics.utils import RANK

    def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
        batch = min(batch, len(dataset))
        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), workers)
        sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK + int(seed))
        return B.InfiniteDataLoader(dataset=dataset, batch_size=batch, shuffle=shuffle and sampler is None, num_workers=nw, sampler=sampler,
                                    pin_memory=PIN_MEMORY, collate_fn=getattr(dataset, "collate_fn", None), worker_init_fn=B.seed_worker,
                                    generator=generator)
    DT.build_dataloader = build_dataloader


def build_dataset(run: Path, images_root, names, label_dir: Path) -> Path:
    """<run>/retrain/data: image symlinks + label copies of adaptation and replay frames, train.txt, data.yaml (train = val: no selection)."""
    d = stage_dir(run, "retrain") / "data"
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    stems = [l.strip() for l in open(run / "collect" / "adapt.txt") if l.strip()] + [l.strip() for l in open(run / "collect" / "replay.txt") if l.strip()]
    for s in stems:
        src = Path(images_root) / f"{s}.jpg"
        if not src.exists():
            raise SystemExit(f"image not found: {src}")
        (d / "images" / f"{s}.jpg").symlink_to(src.resolve())
        shutil.copy2(label_dir / f"{s}.txt", d / "labels" / f"{s}.txt")
    lt = d / "train.txt"
    lt.write_text("".join(f"{d / 'images' / (s + '.jpg')}\n" for s in stems))
    (d / "data.yaml").write_text(yaml.safe_dump({"path": str(d), "train": str(lt), "val": str(lt), "nc": len(names), "names": list(names)}, sort_keys=False))
    return d / "data.yaml"


def retrain(cfg: dict, run, images_root, base=None, seed=None, epochs=None) -> dict:
    import torch
    from ultralytics import YOLO
    run = Path(run)
    rcfg = cfg["retrain"]
    recipe = check_recipe(dict(rcfg["recipe"]))
    from ..inference import resolve_device
    recipe["device"] = resolve_device(recipe["device"])
    if seed is not None:
        recipe["seed"] = int(seed)
    if epochs is not None:
        recipe["epochs"] = int(epochs)
    ann = jload(run / "annotate" / "annotate.json")
    if ann is None:
        raise SystemExit("run annotate first (missing annotate/annotate.json)")
    out = stage_dir(run, "retrain")
    phases = {}
    t0 = time.perf_counter()
    base_w = rp(base or rcfg["base"])
    names = list(YOLO(str(base_w)).names.values())
    data_yaml = build_dataset(run, images_root, names, run / "annotate" / "labels")
    phases["dataset_build_s"] = round(time.perf_counter() - t0, 2)
    spec = recipe["albumentations"]
    patch_albumentations(spec)
    patch_loader_seed(recipe["seed"])
    info = {}

    def albu_effective(trainer):
        a = [x for x in trainer.train_loader.dataset.transforms.transforms if type(x).__name__ == "Albumentations"]
        tr = str(a[0].transform) if a and a[0].transform is not None else None
        active = bool(tr and any(n in tr for n in HIDDEN))
        info.update(found=bool(a), transform=tr, hidden_blur_gray_clahe_active=active, spec=spec)
        print(f"[retrain] albumentations effective: {tr!r}; hidden Blur/MedianBlur/ToGray/CLAHE active: {active}", flush=True)
        if active:
            raise SystemExit("hidden albumentations are active - refusing to train")

    work = out / "work"                       # the trainer's AMP self-check looks for yolov12n.pt in the working directory
    work.mkdir(exist_ok=True)
    amp = rp(rcfg["amp_check_weights"])
    if amp.exists() and not (work / "yolov12n.pt").exists():
        (work / "yolov12n.pt").symlink_to(amp.resolve())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    cwd = os.getcwd()
    os.chdir(work)
    t1 = time.perf_counter()
    try:
        m = YOLO(str(base_w))
        m.add_callback("on_train_start", albu_effective)
        m.train(data=str(data_yaml), epochs=recipe["epochs"], patience=recipe.get("patience", 1000), imgsz=recipe["imgsz"], batch=recipe["batch"],
                freeze=recipe["freeze"], optimizer=recipe["optimizer"], seed=recipe["seed"], deterministic=recipe["deterministic"],
                workers=recipe["workers"], device=recipe["device"], mosaic=recipe["mosaic"], close_mosaic=recipe["close_mosaic"], val=False,
                plots=False, cache=False, save_period=-1, project=str(out), name="train", exist_ok=True, verbose=False, **recipe["augment"])
    finally:
        os.chdir(cwd)
    phases["train_s"] = round(time.perf_counter() - t1, 2)
    peak = {"peak_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)} \
        if torch.cuda.is_available() else {"peak_alloc_MiB": None, "peak_reserved_MiB": None}
    for ev in (out / "train").glob("events.out.tfevents*"):
        ev.unlink()
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(out / "train" / "results.csv"))]
    prev, spe = None, []
    for r in rows:
        tt = float(r["time"])
        spe.append(tt if prev is None or tt < prev else tt - prev)
        prev = tt
    t2 = time.perf_counter()
    wdir = out / "weights"
    wdir.mkdir(exist_ok=True)
    shutil.copy2(out / "train" / "weights" / "last.pt", wdir / "last.pt")
    phases["save_s"] = round(time.perf_counter() - t2, 2)
    res = {"recipe": recipe, "recipe_hidden_albumentations": info, "base_weights": str(base_w), "base_sha256": sha256_file(base_w),
           "weights": str(wdir / "last.pt"), "weights_sha256": sha256_file(wdir / "last.pt"), "epochs_trained": len(spe),
           "s_per_epoch": [round(x, 2) for x in spe], "per_phase_s": {**phases, "total_s": round(time.perf_counter() - t0, 2)},
           "train_s": phases["train_s"], "seed": recipe["seed"], "peak_gpu_MiB": peak, "n_train_frames": ann["n_adapt"] + ann["n_replay"],
           "n_adapt": ann["n_adapt"], "n_replay": ann["n_replay"], "annotate_mode": ann["mode"]}
    jdump(res, out / "retrain.json")
    print(f"[retrain] done: {res['per_phase_s']}, peak {peak}", flush=True)
    return res
