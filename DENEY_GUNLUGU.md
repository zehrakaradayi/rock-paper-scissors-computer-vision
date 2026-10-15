# Deney Günlüğü — Rock Paper Scissors Vision

## Aşama 1 — Dataset Toplama

- Araç: `tools/capture_server.py` (laptop kamerasıyla tek tıkla çekim + telefon fotoğrafı yükleme)
- Klasör yapısı: `dataset/rock`, `dataset/paper`, `dataset/scissors` (git'e eklenmez)
- Fotoğraf sayıları (toplam 452):

  | Sınıf | Toplam | Laptop | Telefon |
  |---|---|---|---|
  | rock | 152 | 106 | 46 |
  | paper | 150 | 11 | 139 |
  | scissors | 150 | 7 | 143 |

- Not: 143 makas fotoğrafı yanlışlıkla `rock/` klasörüne yüklenmişti; görsel kontrolle `scissors/` klasörüne taşındı.
- Not: paper ve scissors neredeyse tamamen telefonla çekildi; demo laptop kamerasıyla yapılacağı için bu bir risk.

## Aşama 2 — Data Cleaning

- Script: `tools/prepare_data.py`, kırpma kuralı: `rps_crop.py` (deneme sayfası da aynı kuralı kullanacak)
- El bulma: MediaPipe Hand Landmarker (`models/hand_landmarker.task`), 21 noktanın kutusu → uzun kenar × 1.4 kare → dışarı taşan kısım siyah → 224×224
- El bulunamayan: 17 fotoğraf (rock 8, paper 2, scissors 7) → `data/no_hand/`, eğitime girmedi.
  Çoğu çok uzak, bulanık ya da parmakları kadraj dışında kalan çok yakın çekimler.
- Bölme: %70 / %15 / %15, sınıf + kaynak (laptop/telefon) içinde **5'li ardışık gruplar** halinde
  (art arda çekilmiş neredeyse aynı kareler train ve test'e bölünmesin diye), seed = 42

  | split | rock | paper | scissors | toplam |
  |---|---|---|---|---|
  | train | 94 | 101 | 103 | 298 |
  | val | 25 | 22 | 20 | 67 |
  | test | 25 | 25 | 20 | 70 |
  | toplam | 144 | 148 | 143 | 435 |

- Görsel kontrol: 435 kırpmanın hepsine bakıldı; el ortada ve tam, yanlış sınıfta görsel yok.
- Teknik not: sunucu ARM (aarch64) ve MediaPipe 1.x'in istediği libEGL/libGLESv2 yüklü değil →
  `tools/setup_gl_libs.sh` Debian paketlerinden `vendor/gl/` içine indirir.

## Aşama 3 — Model Seçimi

**Başlangıç modeli: Model A (sıfırdan CNN).** Model B, Model A kamerada denendikten sonra eğitilecek.

### Model A — Kendi CNN'im (sıfırdan, onaylandı)

| Katman | Çıkış |
|---|---|
| Giriş | 224×224×3 |
| Blok 1: Conv3×3(32) → BatchNorm → ReLU → MaxPool | 112×112×32 |
| Blok 2: Conv3×3(64) → BatchNorm → ReLU → MaxPool | 56×56×64 |
| Blok 3: Conv3×3(128) → BatchNorm → ReLU → MaxPool | 28×28×128 |
| Blok 4: Conv3×3(128) → BatchNorm → ReLU → MaxPool | 14×14×128 |
| Blok 5: Conv3×3(256) → BatchNorm → ReLU → MaxPool | 7×7×256 |
| Flatten → Dropout 0.5 → Linear(12544→2000) → ReLU → Dropout 0.5 → Linear(2000→3) | 3 |

- Parametre: **25.6M** (conv blokları 0.54M, tam bağlı katmanlar 25.1M)
- Eğitim: AdamW, **lr 3e-4**, cosine, weight decay 0.05, label smoothing 0.1, 60 epoch; en iyi val macro F1'deki ağırlıklar saklanır
- lr kararı (sadece validation'a bakılarak): 12 epoch'luk ön denemede lr 1e-3 → en iyi val F1 0.764 (ilk epoch loss 10.7, dalgalı);
  lr 3e-4 → en iyi val F1 0.910. Model A için 3e-4 seçildi.
- Risk: parametrelerin %98'i tam bağlı katmanda, 298 train fotoğrafıyla ezberleme (overfitting) riski yüksek

### Model B — ConvNeXt-Tiny (pretrained)

- `timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=3)`, **27.8M** parametre
- CPU fine-tuning planı (48 vCPU, GPU yok, 224 px):
  1. Gövde donuk, sadece `head` eğitilir: 5 epoch, lr 1e-3
  2. Son aşama (`stages[3]`, 15.5M) + `head` çözülür: 10 epoch, lr 1e-4
- Ölçülen hız: 32'lik batch'te tüm ağ eğitilirken 5.1 sn (Model A: 0.6 sn)

### Adil karşılaştırma
Aynı train / val / test bölünmesi, aynı augmentation, aynı normalizasyon (ImageNet mean/std), aynı 224 px giriş.

### Canlı eğitim paneli
`live.py` + `live_cnn.html`, [exposure-academy-animations](https://github.com/Darkosxl/exposure-academy-animations) `animations/` klasöründen değiştirilmeden kopyalandı (commit `2cd4b5a`).

### Tahminim
**Model B az farkla kazanır.** İkisi de iyi olur ama pretrained ConvNeXt-Tiny biraz önde olur.

## Aşama 4 — Training ve Metrikler

### Model A — Sıfırdan CNN
- 60 epoch, lr 3e-4, epoch başına ~4 sn (CPU, 48 thread); toplam ~5 dk
- En iyi validation macro F1: **1.000** (epoch 31, bu ağırlıklar kullanıldı)
- Son epoch: train F1 0.990, val F1 0.970 → train ile val arası fark küçük, belirgin overfitting yok
- **Test macro F1: 0.987** (70 fotoğrafta 1 hata: bir taş kağıt sanıldı)

  | Sınıf | F1 |
  |---|---|
  | Taş | 0.980 |
  | Kağıt | 0.980 |
  | Makas | 1.000 |

- ONNX: `model_a.onnx` (103 MB, sadece sunucuda), `model_a.int8.onnx` (27 MB, repoda); int8 test macro F1 0.987
- Not: test fotoğrafları train ile aynı çekim seanslarından (aynı oda, ışık, kişi). Skor çok yüksek ama gerçek
  hayattaki başarıyı asıl canlı deneme gösterecek.

### Model B — ConvNeXt-Tiny (pretrained)
- Adım 1 (gövde donuk, sadece head, 3.843 parametre): 5 epoch, lr 1e-3 → val F1 0.880
- Adım 2 (stages[3] + head, 15.5M parametre): 10 epoch, lr 1e-4 → val F1 ilk epoch'ta 0.984'e sıçradı
- En iyi validation macro F1: **1.000** (epoch 10); toplam ~2.5 dk
- Son epoch: train F1 1.000, val F1 1.000
- **Test macro F1: 1.000** (70 fotoğrafın hepsi doğru)
- ONNX: `model_b.onnx` (111 MB, sadece sunucuda), `model_b.int8.onnx` (34 MB, repoda); int8 test macro F1 0.985

### Karşılaştırma

| Model | Parametre | Epoch | En iyi val macro F1 | Test macro F1 | F1 Taş | F1 Kağıt | F1 Makas | int8 test macro F1 |
|---|---|---|---|---|---|---|---|---|
| Model A — Sıfırdan CNN | 25.6M | 60 | 1.000 | **0.987** | 0.980 | 0.980 | 1.000 | 0.987 |
| Model B — ConvNeXt-Tiny (pretrained) | 27.8M | 15 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | 0.985 |

- **Rakamlara göre önde: Model B** (test macro F1 1.000 vs 0.987). Fark 70 test fotoğrafında sadece 1 hata;
  bu kadar küçük bir test setinde iki model rakamsal olarak neredeyse berabere.
- **Tahminim ("B az farkla kazanır") doğru çıktı.**
- Pretraining'in asıl farkı hızda: B, 15 epoch / ~2.5 dk'da val F1 1.000'e ulaştı; A 60 epoch / ~5 dk sürdü.
- **Grad-CAM:** A makas ve taşta büyük ölçüde parmakların etrafındaki **arka plana** (duvar) bakıyor; B elin
  kendisine (parmaklar, yumruğun ortası) daha çok odaklanıyor. A'nın farklı arka planlarda zorlanması bekleniyor.
- int8 sürümlerde B 1 test fotoğrafını daha kaçırdı (0.985); A'da fark yok.

## Aşama 5 — Dene ve Karar Ver

- Deneme sayfası: `play.py` + `play.html` (localhost:8001). Model sunucunun CPU'sunda çalışır; el MediaPipe ile bulunur
  ve eğitimdekiyle aynı kuralla (`rps_crop.py`) kırpılır. İki model yan yana: kutu, tahmin, % confidence, olasılık çubukları.
- **Seçilen model: Model A (sıfırdan CNN).**
- Neden: canlı denemede kamerada **Model A daha emin ve daha tutarlıydı**; Model B daha çok karıştırdı / yüzdeleri daha çok oynadı.
- Metrikler ve canlı deneme aynı şeyi söyledi mi? **Hayır.** Testte B önde (1.000 vs 0.987) ama fark 70 fotoğrafta tek bir
  hataydı; o kadar küçük bir farkı canlı deneme tersine çevirdi. Test seti train ile aynı seanslardan geldiği için gerçek
  kamerayı tam temsil etmiyor; karar için canlı deneme daha belirleyici oldu.
- Akılda tutulacak risk: Grad-CAM'de A'nın makas ve taşta arka plana baktığı görülmüştü; eğitimdekinden çok farklı bir
  arka planda zorlanabilir.

## Aşama 6 — Productize ve Deploy

- **Kaybeden kaldırıldı:** Model B'nin dosyaları (`model_b.onnx`, `model_b.int8.onnx`, `model_b.pt`, `model_b.json`) silindi.
  `training.ipynb`'deki Model B bölümü ve karşılaştırma, eğitimin kaydı olarak çıktılarıyla birlikte duruyor.
- **Web uygulaması:** `web/` (Next.js 15). Model tarayıcıda çalışır (ONNX Runtime Web, `model_a.int8.onnx`, 27 MB);
  el MediaPipe Hand Landmarker ile tarayıcıda bulunur ve eğitimdekiyle aynı kuralla kırpılır (kare × 1.4, dışı siyah, 224 px).
  Sunucu yok, API anahtarı yok; kamera görüntüsü cihazdan çıkmaz.
- Arayüz: başlık + açıklama, büyük kamera görüntüsü, bounding box + tahmin + % confidence, olasılık çubukları,
  yükleniyor / hata / "Elini kameraya göster" durumları, model adı + macro F1 bilgi kartı, mobil uyumlu (ön kamera).
- Headless Chromium + sahte kamera ile test (her sınıftan bir test fotoğrafı):

  | Fotoğraf | Tarayıcı (int8) | Python sunucu (fp32) |
  |---|---|---|
  | Taş | Taş %82 | Taş %87 |
  | Kağıt | Kağıt %89 | Kağıt %89 |
  | Makas | Makas %97 | Makas %95 |

  Ayrıca: el yokken "Elini kameraya göster", kamera izni reddedilince "Kameraya erişilemedi. Lütfen izin ver.",
  390 px telefon ekranında yatay taşma yok.
- **GitHub:** https://github.com/zehrakaradayi/rock-paper-scissors-computer-vision (main)
- **Vercel (canlı):** https://web-zehra7.vercel.app — Root Directory `web`, environment variable yok.
  Vercel'in varsayılan "Vercel Authentication" koruması kapatıldı (açıkken link sadece Vercel hesabıyla açılıyordu).
- Canlı sitede headless Chromium + sahte kamera testi (390 px telefon ekranı, Model A int8):

  | Fotoğraf | Canlı site |
  |---|---|
  | Taş | Taş %96 |
  | Kağıt | Kağıt %93 |
  | Makas | Makas %97 |
