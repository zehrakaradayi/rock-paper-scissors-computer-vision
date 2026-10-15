"""Ödev (5 hata) — zor durum fotoğraflarını kırp ve böl. Mevcut data/split/ bölünmesine DOKUNMAZ.

Girdi:  dataset_zor/<durum>/<sınıf>/*      (bugünkü odada çekilen yeni fotoğraflar)
Çıktı:  data/zor/train/<durum>/<sınıf>/   eğitime eklenebilecek kırpılmış eller
        data/zor/test/<durum>/<sınıf>/    "zor test" seti: eğitimde ASLA kullanılmaz
        data/zor/manifest.csv
        data/zor/sheet_<durum>.jpg        kırpmaların önizlemesi

Kurallar:
- Sadece MediaPipe'ın eli bulduğu fotoğraflar kullanılır (bulunamayan el canlı sayfada da sınıflandırılmaz).
- Sınıflar dengeli: her (durum, split) için her sınıftan aynı sayıda fotoğraf alınır (çekim sırasıyla ilk K tanesi,
  K = en az fotoğrafı olan sınıfın sayısı).
- Bölme prepare_data.py'deki gibi 5'li ardışık gruplarla: art arda çekilmiş neredeyse aynı iki kare
  biri train'e biri test'e düşmesin. 1. ve 3. grup test (10), diğerleri train.
- "asagi": ilk çekilen 10'ar fotoğraf test (düzeltme 1 bunlarla ölçüldü), sonradan çekilenler train (düzeltme 1b).

Çalıştırma:  .venv/bin/python tools/prepare_zor.py
"""

import csv
import shutil
import sys
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rps_crop  # noqa: E402

pillow_heif.register_heif_opener()

RAW = ROOT / "dataset_zor"
OUT = ROOT / "data" / "zor"
CLASSES = ("rock", "paper", "scissors")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
GROUP = 5
MODES = {"asagi": "first10", "yuz": "groups", "cizgili": "groups"}


def assign(found: list[Path], all_files: list[Path], mode: str) -> dict[str, list[Path]]:
    """Eli bulunan dosyaları (çekim sırasıyla) train / test'e ayırır."""
    if mode == "first10":
        first = set(all_files[:10])  # ilk çekim seansı
        return {"test": [f for f in found if f in first], "train": [f for f in found if f not in first]}
    out = {"train": [], "test": []}
    for i, f in enumerate(found):
        out["test" if (i // GROUP) in (1, 3) else "train"].append(f)
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    landmarker = rps_crop.make_landmarker()
    rows, crops = [], {}
    for cond, mode in MODES.items():
        files = {c: sorted(p for p in (RAW / cond / c).glob("*") if p.suffix.lower() in EXTS) for c in CLASSES}
        if not all(files.values()):
            print(f"{cond}: fotoğraf eksik — atlandı"); continue
        found = {c: [] for c in CLASSES}
        for c in CLASSES:
            for f in files[c]:
                img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                hit = rps_crop.find_hand(landmarker, img)
                if hit is None:
                    rows.append(dict(file=f.name, cond=cond, label=c, split="", hand=0)); continue
                crops[f] = rps_crop.crop_hand(img, hit[0])
                found[c].append(f)
        parts = {c: assign(found[c], files[c], mode) for c in CLASSES}
        thumbs = []
        for split in ("train", "test"):
            k = min(len(parts[c][split]) for c in CLASSES)  # denge: en az olan sınıfa göre
            for r, c in enumerate(CLASSES):
                for f in parts[c][split]:
                    used = f in parts[c][split][:k]
                    rows.append(dict(file=f.name, cond=cond, label=c, split=split if used else "fazla", hand=1))
                    if used:
                        dst = OUT / split / cond / c / (f.stem + ".jpg")
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        crops[f].save(dst, quality=95)
                        thumbs.append((r, crops[f], split))
            print(f"{cond:8s} {split:5s} her sınıftan {k}  (eli bulunan: " +
                  ", ".join(f"{c} {len(parts[c][split])}" for c in CLASSES) + ")")
        # önizleme: satır = sınıf, kırmızı çerçeve = test
        T, per_row = 96, max(sum(1 for t in thumbs if t[0] == r) for r in range(3))
        sheet = Image.new("RGB", (T * per_row, T * 3), "white")
        d, col = ImageDraw.Draw(sheet), [0, 0, 0]
        for r, im, split in thumbs:
            x, y = col[r] * T, r * T; col[r] += 1
            sheet.paste(im.resize((T - 4, T - 4)), (x + 2, y + 2))
            if split == "test":
                d.rectangle([x, y, x + T - 1, y + T - 1], outline="red", width=2)
        sheet.save(OUT / f"sheet_{cond}.jpg", quality=90)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "cond", "label", "split", "hand"])
        w.writeheader(); w.writerows(rows)
    print("el bulunamadı:", sum(1 for x in rows if not x["hand"]), "fotoğraf (manifest.csv'de hand=0)")


if __name__ == "__main__":
    main()
