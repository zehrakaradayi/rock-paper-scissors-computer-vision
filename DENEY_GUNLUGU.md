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

_(yapılacak)_

## Aşama 5 — Dene ve Karar Ver

_(yapılacak)_

## Aşama 6 — Productize ve Deploy

_(yapılacak)_
