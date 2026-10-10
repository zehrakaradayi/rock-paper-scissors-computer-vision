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

_(yapılacak)_

## Aşama 4 — Training ve Metrikler

_(yapılacak)_

## Aşama 5 — Dene ve Karar Ver

_(yapılacak)_

## Aşama 6 — Productize ve Deploy

_(yapılacak)_
