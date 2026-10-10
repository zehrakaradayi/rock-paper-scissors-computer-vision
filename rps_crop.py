"""El bulma + kırpma — eğitimde ve deneme sayfasında AYNI kural kullanılır.

Kural: MediaPipe Hand Landmarker'ın 21 noktasının kutusunu al, uzun kenarı
MARGIN katına çıkararak kare yap, kare görüntü dışına taşarsa siyahla doldur,
sonra SIZE x SIZE'a küçült.
"""

import ctypes
from pathlib import Path

# Sunucuda olmayan EGL/GLES kütüphaneleri (tools/setup_gl_libs.sh indirir)
_GL = Path(__file__).resolve().parent / "vendor" / "gl"
for _name in ("libGLdispatch.so.0", "libEGL.so.1", "libGLESv2.so.2"):
    if (_GL / _name).exists():
        ctypes.CDLL(str(_GL / _name), mode=ctypes.RTLD_GLOBAL)

import mediapipe as mp  # noqa: E402
import numpy as np
from PIL import Image  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
MARGIN = 1.4  # kutuyu eller kesilmesin diye büyütme oranı
SIZE = 224  # modelin giriş boyutu
DETECT_MAX_SIDE = 1280  # el arama bu boyuta küçültülmüş görüntüde yapılır


def make_landmarker(min_conf: float = 0.3) -> vision.HandLandmarker:
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
        )
    )


def find_hand(landmarker, img: Image.Image):
    """Elin kutusunu (x0, y0, x1, y1) orijinal piksel koordinatında ve skorunu döndürür; el yoksa None."""
    small = img.copy()
    small.thumbnail((DETECT_MAX_SIDE, DETECT_MAX_SIDE))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(small)))
    if not res.hand_landmarks:
        return None
    W, H = img.size
    xs = [p.x * W for p in res.hand_landmarks[0]]
    ys = [p.y * H for p in res.hand_landmarks[0]]
    hand = res.handedness[0][0]
    return (min(xs), min(ys), max(xs), max(ys)), hand.score, hand.category_name


def square_box(box, margin: float = MARGIN):
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * margin
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)


def crop_hand(img: Image.Image, box, size: int = SIZE) -> Image.Image:
    """Kare kutuyu kırp (dışarı taşan kısım siyah) ve size x size'a getir."""
    # PIL.crop görüntü dışını siyah (0) ile doldurur
    return img.crop(square_box(box)).resize((size, size), Image.BICUBIC)
