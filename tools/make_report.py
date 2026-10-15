"""Ödev (5 hata) — teslim PDF'i: odev_5_hata.pdf

Girdi:  mistakes/*/              deneme sayfasında kaydedilen hatalar (ekran görüntüsü, kırpma, Grad-CAM, info.json)
        models/model_b*.pt       Bölüm 2 ve Bölüm 4'te eğitilen Model B'ler (seed 42)
        models/odev_sonuclar_3seed.json   Bölüm 4.8: her adımın 3 seed sonucu
Metinler (neden / düzeltme / sonuç) aşağıdaki MISTAKES listesinde.

Çalıştırma:  .venv/bin/python tools/make_report.py
"""

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUT = ROOT / "odev_5_hata.pdf"
TR = {"rock": "Taş", "paper": "Kağıt", "scissors": "Makas"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

STEPS = [  # (kısa ad, odev_sonuclar_3seed.json anahtarı, seed 42 ağırlıkları)
    ("Önce", "Önce: Model B", "model_b"),
    ("Düz. 1", "Düzeltme 1: döndürme ±180°", "model_b_fix1"),
    ("Düz. 1b", "Düzeltme 1b: + parmaklar aşağı fotoğrafları", "model_b_fix1b"),
    ("Düz. 2", "Düzeltme 2: + yüz/saç fotoğrafları", "model_b_fix2"),
    ("Düz. 3", "Düzeltme 3: + çizgili duvar fotoğrafları", "model_b_fix3"),
]
FINAL = "Düz. 3"


def load(name):
    m = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=3)
    m.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu"))
    return m.eval()


def gradcam(model, crop: Image.Image):
    """training.ipynb'deki gradcam() ile aynı (son aşama). (bindirilmiş görüntü, tahmin, güven) döner."""
    classes = json.loads((MODELS / "model_b.json").read_text())["classes"]
    arr = np.asarray(crop.resize((224, 224)), dtype=np.float32) / 255.0
    x = torch.from_numpy(((arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)).requires_grad_(True)
    acts, grads = {}, {}
    layer = model.stages[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    out = model(x)
    cls = int(out.argmax(1))
    model.zero_grad(); out[0, cls].backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().numpy()
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    img = arr * 0.55 + heat * 0.45
    return np.clip(img, 0, 1), classes[cls], float(out.softmax(1)[0, cls].detach())


def wrap(s, width=108):
    return "\n".join(textwrap.fill(p, width) for p in s.split("\n"))


def mean_cell(runs, step_key, col):
    v = runs[step_key][col]
    return f"{np.mean(v):.3f} ({min(v):.2f}–{max(v):.2f})"


def main():
    runs = json.loads((MODELS / "odev_sonuclar_3seed.json").read_text())
    cols = list(next(iter(runs.values())).keys())
    col_of = {c.split(":")[-1].split("(")[0].strip(): c for c in cols}  # "asagi" → "Zor: asagi (27)"
    items = [d for d in sorted((ROOT / "mistakes").glob("*"))
             if (d / "info.json").exists() and json.loads((d / "info.json").read_text())["wrong"]]
    assert len(items) == len(TEXT) == 5, (len(items), len(TEXT))
    models = {short: load(name) for short, _, name in STEPS}

    with PdfPages(OUT) as pdf:
        # --- 1. sayfa: özet ---
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "Ödev: 5 Hata — Model B (ConvNeXt-Tiny)", fontsize=18, weight="bold")
        fig.text(0.07, 0.925, "Rock Paper Scissors Vision · 15 Ekim 2026", fontsize=10, color="#555")
        fig.text(0.07, 0.905, wrap(INTRO), fontsize=8.6, va="top", linespacing=1.45)
        ax = fig.add_axes([0.07, 0.44, 0.86, 0.26]); ax.axis("off")
        short_cols = ["Eski test (70)"] + [c for c in cols if c.startswith("Zor")]
        table = [[short] + [mean_cell(runs, key, c) for c in short_cols] for short, key, _ in STEPS]
        t = ax.table(cellText=table, colLabels=["Adım"] + [c.replace("Zor: ", "Zor\n") for c in short_cols],
                     loc="upper center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(7.4); t.scale(1, 2.3)
        for (r, c), cell in t.get_celld().items():
            if r == 0:
                cell.set_facecolor("#eef2f7"); cell.set_text_props(weight="bold")
        fig.text(0.07, 0.705, "Macro F1 — 3 eğitimin (seed 42, 1, 2) ortalaması, parantez içinde en düşük–en yüksek",
                 fontsize=9, weight="bold")
        fig.text(0.07, 0.43, wrap(SUMMARY), fontsize=8.6, va="top", linespacing=1.45)
        pdf.savefig(fig); plt.close(fig)

        # --- her hata için bir sayfa ---
        for k, (d, txt) in enumerate(zip(items, TEXT), start=1):
            info = json.loads((d / "info.json").read_text())
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.07, 0.955, f"Hata {k} / 5", fontsize=16, weight="bold")
            fig.text(0.07, 0.93, f"Gösterilen: {TR[info['showed']]}   ·   Model B dedi: {TR[info['said']]} "
                                 f"(%{info['conf'] * 100:.0f})", fontsize=11.5, color="#b91c1c", weight="bold")
            ax = fig.add_axes([0.07, 0.625, 0.86, 0.29])
            ax.imshow(Image.open(d / "screenshot.jpg")); ax.axis("off")
            fig.text(0.5, 0.615, "Deneme sayfasının ekran görüntüsü (ayna görüntü, kaydedildiği an)", fontsize=7.5,
                     color="#555", ha="center")
            crop = Image.open(d / "crop.jpg").convert("RGB")
            panels = [("Modelin gördüğü (kırpma)", np.asarray(crop.resize((224, 224))) / 255.0, None)]
            for short in ("Önce", txt["after"], FINAL) if txt["after"] != FINAL else ("Önce", FINAL):
                img, pred, conf = gradcam(models[short], crop)
                panels.append((f"Grad-CAM · {short}\n{TR[info['showed']]} → {TR[pred]} %{conf * 100:.0f}", img,
                               pred == info["showed"]))
            w = 0.86 / len(panels)
            for j, (title, img, ok) in enumerate(panels):
                a = fig.add_axes([0.07 + j * w, 0.425, w - 0.01, 0.155]); a.imshow(img); a.axis("off")
                a.set_title(title, fontsize=7.5, color="#555" if ok is None else ("#15803d" if ok else "#b91c1c"))
            body = (f"Neden?\n{txt['why']}\n\nDüzeltme\n{txt['fix']}\n\nSonuç\n{txt['result']}")
            fig.text(0.07, 0.39, wrap(body), fontsize=8.2, va="top", linespacing=1.35)
            # önce / sonra tablosu (3 seed ortalaması)
            ax = fig.add_axes([0.07, 0.02, 0.86, 0.125]); ax.axis("off")
            rel = col_of[txt["col"]]
            rows = []
            for short, key, _ in STEPS:
                if short not in txt["rows"]:
                    continue
                _, pred, conf = gradcam(models[short], crop)
                rows.append([short, f"{TR[pred]} %{conf * 100:.0f}", mean_cell(runs, key, rel),
                             mean_cell(runs, key, col_of["hepsi"]), mean_cell(runs, key, "Eski test (70)")])
            t = ax.table(cellText=rows, colLabels=["Adım", "Bu karede (seed 42)", rel, "Zor: hepsi (87)", "Eski test (70)"],
                         loc="upper center", cellLoc="center")
            t.auto_set_font_size(False); t.set_fontsize(7.4); t.scale(1, 1.6)
            for (r, c), cell in t.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#eef2f7"); cell.set_text_props(weight="bold")
            fig.text(0.07, 0.15, "Macro F1 önce / sonra (3 seed ortalaması, en düşük–en yüksek)", fontsize=8.5, weight="bold")
            pdf.savefig(fig); plt.close(fig)
    print(f"{OUT.name} yazıldı ({OUT.stat().st_size / 1e6:.1f} MB)")


INTRO = """\
Model B'yi deneme sayfasında kendi elimle denedim ve yanıldığı 5 anı kaydettim. Beşi de modelin eğitim fotoğraflarını \
çektiğim odadan (10 Ekim) farklı bir odada: çizgili duvar, sıcak ışık, desenli kol.
Nasıl ölçtüm? Eski test seti (70 fotoğraf) Model B'de zaten ~1.000 olduğu için iyileşmeyi gösteremiyor. Bu yüzden bugünkü \
odada her zor durum için yeni fotoğraflar çektim. Bir kısmını SADECE teste ayırdım ("zor test", 87 fotoğraf; eğitime hiç \
girmedi), kalanını eğitime ekledim. Sınıflar her zaman dengeli (her sınıfa aynı sayıda fotoğraf), her adımda tek bir \
değişiklik yaptım ve Model B'yi aynı tarifle (Bölüm 2.2) baştan eğittim. Eski test, bir düzeltmenin başka bir şeyi bozup \
bozmadığını gösteriyor.
Şans payı: tek bir eğitimin sonucu, sadece seed değişince ±0.05 oynayabiliyor (1 fotoğraf ≈ 0.03). Bu yüzden her adımı 3 \
farklı seed ile eğittim; tablodaki sayılar 3 eğitimin ortalaması.
Düzeltmeler (her biri bir öncekinin üstüne): 1) augmentation'da döndürme ±20° → ±180°; 1b) her sınıfa 11 "parmaklar aşağı" \
fotoğrafı; 2) her sınıfa 10 "el yüzün / saçın önünde" fotoğrafı; 3) her sınıfa 11 "çizgili duvar önünde" fotoğrafı."""

SUMMARY = """\
Sonuç: zor testte macro F1 0.710 → 0.863 (+0.15); eski test 0.995 → 0.990, yani hiçbir düzeltme eski testi bozmadı.
En net etki Düzeltme 2'de: "yüz" zor testi 0.728 → 0.897 ve aralıklar neredeyse hiç çakışmıyor. Düzeltme 1, 1b ve 3'ün \
kendi hedef sütunlarındaki artışlar gerçek ama seed'ler arası aralıklar çakışıyor; tek eğitimle bakılsaydı Düzeltme 3 \
yanlışlıkla "kötüleştirdi" görünecekti (seed 42: 0.872 → 0.797).
5 hatanın 3'ü düzeldi (3, 4, 5). Hata 1 (aşağı bakan makas) ve hata 2 (yandan görünen kağıt) son modelde hâlâ yanlış; \
nedenleri ve bir sonraki adım kendi sayfalarında."""

TEXT = [
    dict(col="asagi", after="Düz. 1b", rows=["Önce", "Düz. 1", "Düz. 1b", "Düz. 3"],
         why="Parmaklar aşağıyı gösteriyor ve kutunun yarısı desenli kol. Eğitim fotoğraflarının hepsinde parmaklar yukarıdaydı "
             "ve kol düz/koyu renkliydi. Grad-CAM parmaklara değil bileğe ve desenli kola bakıyor: model bu kareyi tanıdığı bir şeye "
             "benzetemiyor ve yumruk (Taş) diyor.",
         fix="Düzeltme 1: augmentation'da döndürme ±20° → ±180° (yeni fotoğraf yok). Bu, zor testteki aşağı makasları düzeltmedi "
             "(seed 42: 9 makasın 8'i hâlâ Taş), çünkü döndürülmüş bir eğitim fotoğrafında el hâlâ avuç içi tarafından ve düz kolla "
             "görünüyor. Düzeltme 1b: her sınıfa 11 gerçek \"parmaklar aşağı\" fotoğrafı.",
         result="\"Aşağı\" zor testi 0.553 → 0.668 → 0.713 yükseldi, ama bu karenin kendisi düzelmedi: Taş yerine artık Kağıt diyor. "
                "Eğitime eklenen aşağı fotoğraflarının çoğunda kol çıplaktı, bu karede ise desenli kol kutunun yarısını kaplıyor. "
                "Sonraki adım: aynı desenli kıyafetle, kol kutuya girecek şekilde aşağı fotoğrafları çekmek."),
    dict(col="cizgili", after=FINAL, rows=["Önce", "Düz. 2", "Düz. 3"],
         why="El kameraya yan dönük (sadece kenarı görünüyor), küçük ve uzakta; arkada dikey çizgili duvar paneli var. Grad-CAM "
             "ele değil duvardaki çizgilere dağılmış: model dikey çizgileri parmak sanıp Makas diyor.",
         fix="Düzeltme 3: her sınıfa 11 \"çizgili duvar önünde\" fotoğrafı.",
         result="\"Çizgili\" zor testi 0.878 → 0.922 yükseldi; ama bu kare son modelde de Makas. Çektiğim çizgili duvar fotoğraflarında "
                "el çoğunlukla avuç içi kameraya dönük ve yakındı; bu karedeki gibi yan dönük, uzak bir kağıt neredeyse hiç yoktu. Yani "
                "düzeltme genel olarak çizgili duvarda işe yaradı, ama bu karedeki asıl zor durumu (yan dönük el) veriye koymadım. "
                "Sonraki adım: her sınıftan yan dönük ve uzak el fotoğrafları."),
    dict(col="yuz", after="Düz. 2", rows=["Önce", "Düz. 1b", "Düz. 2", "Düz. 3"],
         why="El yüzün ve saçın önünde; saç telleri parmaklarla karışıyor ve parmakların bir kısmı saçın içinde kayboluyor. Grad-CAM "
             "parmaklara değil eklemlere ve kola bakıyor; iki parmağı göremeyince Taş diyor. Eğitim fotoğraflarında yüz ya da saç "
             "neredeyse hiç yoktu.",
         fix="Düzeltme 2: her sınıfa (Taş, Kağıt, Makas) 10 \"el yüzün / saçın önünde\" fotoğrafı. Sadece Makas'a eklemek sınıfları "
             "dengesizleştirir ve modele \"kutuda yüz/saç = Makas\" kısayolunu öğretebilirdi.",
         result="Düzeldi: bu kare Düzeltme 2'den sonra Makas. \"Yüz\" zor testi 0.728 → 0.897 (en net iyileşme; seed aralıkları "
                "neredeyse çakışmıyor). Eski test 0.976 → 0.985, bozulmadı."),
    dict(col="yuz", after="Düz. 2", rows=["Önce", "Düz. 1", "Düz. 2", "Düz. 3"],
         why="El saçın önünde, makasın iki parmağı birbirine çok yakın ve arkadan turuncu bir lamba ışığı var. Grad-CAM parmak "
             "uçlarında ama bitişik iki parmak açık bir eli andırıyor: Kağıt %48, Makas %40, yani model kararsız.",
         fix="Düzeltme 2 (hata 3 ile aynı): her sınıfa 10 \"el yüzün / saçın önünde\" fotoğrafı.",
         result="Düzeldi; aslında Düzeltme 1'den (döndürme ±180°) sonra Makas'a döndü ve Düzeltme 2 ile güveni arttı (%55 → %66). "
                "Düzeltme 1 tek başına \"yüz\" zor testini düşürmüştü (0.694 → 0.631); asıl kalıcı iyileşmeyi Düzeltme 2 getirdi "
                "(0.728 → 0.897)."),
    dict(col="asagi", after="Düz. 1", rows=["Önce", "Düz. 1", "Düz. 1b", "Düz. 3"],
         why="Açık el (Kağıt) ama parmaklar aşağıyı gösteriyor; kutunun üst kısmı desenli kol. Grad-CAM parmaklara değil kola ve "
             "kutunun üst kenarına bakıyor. Eğitimde aşağı bakan el yoktu.",
         fix="Düzeltme 1: augmentation'da döndürme ±20° → ±180°. Sonra Düzeltme 1b: her sınıfa 11 \"parmaklar aşağı\" fotoğrafı.",
         result="Düzeldi: Düzeltme 1'den sonra Kağıt ve Grad-CAM koldan elin ortasına kaydı; 1b'den sonra güven %45 → %75. "
                "\"Aşağı\" zor testi 0.553 → 0.713, eski test 0.995 → 0.976 (aralık içinde, bozulmadı)."),
]

if __name__ == "__main__":
    main()
