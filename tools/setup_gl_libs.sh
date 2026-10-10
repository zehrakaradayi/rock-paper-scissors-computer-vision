#!/usr/bin/env bash
# MediaPipe 1.x, sunucuda yüklü olmayan libEGL.so.1 / libGLESv2.so.2 kütüphanelerini ister.
# sudo olmadan Debian'ın resmi paketlerinden indirip vendor/gl/ klasörüne açar.
# rps_crop.py bu klasörü otomatik yükler.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
for p in libegl1 libgles2 libglvnd0; do
  curl -sSfL -o "$TMP/$p.deb" "https://deb.debian.org/debian/pool/main/libg/libglvnd/${p}_1.7.0-3+b1_$(dpkg --print-architecture).deb"
  dpkg-deb -x "$TMP/$p.deb" "$TMP/x"
done
mkdir -p vendor/gl
cp -a "$TMP"/x/usr/lib/*-linux-gnu/*.so* vendor/gl/
rm -rf "$TMP"
ls vendor/gl
