"""Aşama 5 — Deneme sayfası: eğitilen model(ler)i kendi elinle dene.

Tarayıcı kameradan kareyi gönderir → sunucu MediaPipe ile eli bulur, eğitimdekiyle AYNI kuralla kırpar
(rps_crop.py) → models/ içindeki her ONNX modeli çalıştırır → kutu + olasılıklar döner.
Model sunucunun CPU'sunda çalışır. Yeni bir model eğitilince sayfayı yenilemen yeter.

Çalıştırma:  .venv/bin/python play.py   →  http://localhost:8001
"""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

import rps_crop

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PAGE = ROOT / "play.html"
PORT = 8001
ORDER = ["rock", "paper", "scissors"]  # sayfada Taş, Kağıt, Makas sırası

_landmarker = rps_crop.make_landmarker(min_conf=0.5)
_lm_lock = threading.Lock()
_models: dict[str, dict] = {}  # isim → {sess, meta, mtime}
_models_lock = threading.Lock()


def load_models() -> list[dict]:
    """models/<isim>.json + models/<isim>.onnx çiftlerini yükler; değişen/yeni dosyaları yeniden okur."""
    with _models_lock:
        for name in list(_models):  # dosyası silinen model sayfadan da kalkar
            if not (MODELS / f"{name}.onnx").exists():
                del _models[name]
        for meta_path in sorted(MODELS.glob("*.json")):
            onnx = meta_path.with_suffix(".onnx")
            if not onnx.exists():
                continue
            name, mtime = meta_path.stem, onnx.stat().st_mtime
            if name in _models and _models[name]["mtime"] == mtime:
                continue
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 8
            _models[name] = dict(
                sess=ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"]),
                meta=json.loads(meta_path.read_text()), mtime=mtime,
            )
            print(f"model yüklendi: {onnx.name}", flush=True)
        return [dict(name=n, title=m["meta"]["title"], macro_f1=m["meta"]["test"]["macro_f1"])
                for n, m in sorted(_models.items())]


def predict(jpeg: bytes) -> dict:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(jpeg))).convert("RGB")
    with _lm_lock:
        found = rps_crop.find_hand(_landmarker, img)
    if found is None:
        return {"hand": False}
    box = found[0]
    crop = rps_crop.crop_hand(img, box)
    W, H = img.size
    x0, y0, x1, y1 = rps_crop.square_box(box)
    out = {"hand": True, "box": [x0 / W, y0 / H, x1 / W, y1 / H], "models": {}}

    buf = io.BytesIO()
    crop.resize((112, 112)).save(buf, "JPEG", quality=80)
    out["crop"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    arr = np.asarray(crop, dtype=np.float32) / 255.0
    with _models_lock:
        models = list(_models.items())
    for name, m in models:
        meta = m["meta"]
        x = ((arr - meta["mean"]) / meta["std"]).transpose(2, 0, 1)[None].astype(np.float32)
        logits = m["sess"].run(None, {"input": x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        probs = {c: float(p[meta["classes"].index(c)]) for c in ORDER}
        best = max(probs, key=probs.get)
        out["models"][name] = {"probs": probs, "label": best, "conf": probs[best]}
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/models":
            self._send(200, json.dumps(load_models()).encode())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            return self._send(404, b"not found", "text/plain")
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            self._send(200, json.dumps(predict(data)).encode())
        except Exception as e:  # bozuk kare vb.: sayfa hata durumunu göstersin
            self._send(500, json.dumps({"error": str(e)}).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("modeller:", [m["name"] for m in load_models()] or "henüz yok")
    print(f"Deneme sayfası: http://localhost:{PORT}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
