"""Aşama 2 — Data cleaning: eli bul, kırp, boyutlandır, train / val / test olarak böl.

Girdi:  dataset/<sınıf>/*      (ham fotoğraflar, dokunulmaz)
Çıktı:  data/crops/<sınıf>/          224x224 kırpılmış eller
        data/no_hand/<sınıf>/        elin bulunamadığı fotoğrafların önizlemesi (karar senin)
        data/split/<train|val|test>/<sınıf>/
        data/manifest.csv            her fotoğraf için kaynak, skor, kutu ve split
        data/preview_grid.jpg        her sınıftan kırpılmış örnekler

Bölme: art arda çekilen fotoğraflar birbirine çok benzediği için tek tek değil,
5'li ardışık gruplar halinde bölünür. Böylece neredeyse aynı iki kare biri
train'e biri test'e düşmez ve test skoru şişmez.

Çalıştırma:  .venv/bin/python tools/prepare_data.py
"""

import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rps_crop  # noqa: E402

pillow_heif.register_heif_opener()

RAW = ROOT / "dataset"
OUT = ROOT / "data"
CLASSES = ("rock", "paper", "scissors")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
GROUP = 5
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

_landmarker = None


def _init():
    global _landmarker
    _landmarker = rps_crop.make_landmarker()


def source_of(path: Path) -> str:
    return "laptop" if path.name.startswith("laptop_") else "phone"


def process(item):
    label, path = item
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    found = rps_crop.find_hand(_landmarker, img)
    stem = f"{path.stem}"
    if found is None:
        prev = img.copy()
        prev.thumbnail((800, 800))
        prev.save(OUT / "no_hand" / label / f"{stem}.jpg", quality=85)
        return dict(file=path.name, label=label, source=source_of(path), hand=0)
    box, score, handed = found
    rps_crop.crop_hand(img, box).save(OUT / "crops" / label / f"{stem}.jpg", quality=95)
    return dict(
        file=path.name, label=label, source=source_of(path), hand=1,
        score=round(score, 3), handedness=handed,
        box="%d,%d,%d,%d" % tuple(map(round, box)),
        box_frac=round(max(box[2] - box[0], box[3] - box[1]) / max(img.size), 3),
    )


def assign_splits(rows):
    """Her sınıf + kaynak içinde 5'li ardışık grupları %70/%15/%15 dağıt."""
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["label"], r["source"])].append(r)
    for rs in buckets.values():
        rs.sort(key=lambda r: r["file"])
        groups = [rs[i:i + GROUP] for i in range(0, len(rs), GROUP)]
        rng.shuffle(groups)
        n = len(groups)
        n_val = max(1, round(n * SPLITS["val"])) if n >= 3 else 0
        n_test = max(1, round(n * SPLITS["test"])) if n >= 3 else 0
        for i, g in enumerate(groups):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for r in g:
                r["split"] = split


def preview_grid(rows, per_class=8, t=150):
    grid = Image.new("RGB", (per_class * t, len(CLASSES) * (t + 22)), "white")
    d = ImageDraw.Draw(grid)
    rng = random.Random(SEED)
    for ci, c in enumerate(CLASSES):
        cand = [r for r in rows if r["label"] == c and r["hand"]]
        pick = rng.sample(cand, min(per_class, len(cand)))
        y = ci * (t + 22)
        d.text((4, y + 4), c, fill="black")
        for i, r in enumerate(pick):
            im = Image.open(OUT / "crops" / c / f"{Path(r['file']).stem}.jpg").resize((t, t))
            grid.paste(im, (i * t, y + 22))
    grid.save(OUT / "preview_grid.jpg", quality=90)


def main():
    for sub in ("crops", "no_hand", "split"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    for c in CLASSES:
        (OUT / "crops" / c).mkdir(parents=True)
        (OUT / "no_hand" / c).mkdir(parents=True)

    items = [(c, p) for c in CLASSES for p in sorted((RAW / c).iterdir()) if p.suffix.lower() in EXTS]
    print(f"{len(items)} fotoğraf işleniyor…")
    # Tek süreç: MediaPipe çok süreçte ara sıra çöküp havuzu kilitliyordu (~0.5 sn / fotoğraf)
    _init()
    rows = []
    for i, item in enumerate(items, 1):
        rows.append(process(item))
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)

    hands = [r for r in rows if r["hand"]]
    assign_splits(hands)
    for r in hands:
        dst = OUT / "split" / r["split"] / r["label"]
        dst.mkdir(parents=True, exist_ok=True)
        name = f"{Path(r['file']).stem}.jpg"
        shutil.copy2(OUT / "crops" / r["label"] / name, dst / name)

    keys = ["file", "label", "source", "hand", "split", "score", "handedness", "box", "box_frac"]
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in keys} for r in rows)
    preview_grid(rows)

    print("\nEl bulunamayan:", dict(Counter(r["label"] for r in rows if not r["hand"])) or "yok")
    cnt = Counter((r["split"], r["label"]) for r in hands)
    print(f"\n| split | {' | '.join(CLASSES)} | toplam |\n|---|---|---|---|---|")
    for s in SPLITS:
        vals = [cnt[(s, c)] for c in CLASSES]
        print(f"| {s} | {' | '.join(map(str, vals))} | {sum(vals)} |")
    tot = [sum(cnt[(s, c)] for s in SPLITS) for c in CLASSES]
    print(f"| toplam | {' | '.join(map(str, tot))} | {sum(tot)} |")


if __name__ == "__main__":
    main()
