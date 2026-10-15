"""Aşama 5 — Deneme sayfası: eğitilen model(ler)i kendi elinle dene.

Tarayıcı kameradan kareyi gönderir → sunucu MediaPipe ile eli bulur, eğitimdekiyle AYNI kuralla kırpar
(rps_crop.py) → models/ içindeki her ONNX modeli çalıştırır → kutu + olasılıklar döner.
Model sunucunun CPU'sunda çalışır. Yeni bir model eğitilince sayfayı yenilemen yeter.

Ödev (5 hata): "Hatayı kaydet" → mistakes/<zaman>_<gösterilen>/ içine ekran görüntüsü, kırpılmış el,
Model B'nin Grad-CAM'i (models/model_b.pt gerekir) ve tahminler (info.json) yazılır.

Çalıştırma:  .venv/bin/python play.py   →  http://localhost:8001
"""

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps

import rps_crop

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PAGE = ROOT / "play.html"
MISTAKES = ROOT / "mistakes"
CAM_MODEL = "model_b"  # Grad-CAM bu modelin PyTorch ağırlıklarıyla hesaplanır
PORT = 8001
ORDER = ["rock", "paper", "scissors"]  # sayfada Taş, Kağıt, Makas sırası

_landmarker = rps_crop.make_landmarker(min_conf=0.5)
_lm_lock = threading.Lock()
_models: dict[str, dict] = {}  # isim → {sess, meta, mtime}
_models_lock = threading.Lock()
_cam: dict = {}  # mtime, model, layer
_cam_lock = threading.Lock()
NAMES = {"rock": "Taş", "paper": "Kağıt", "scissors": "Makas"}
COLORS = {"rock": (194, 65, 12), "paper": (29, 78, 216), "scissors": (21, 128, 61)}


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


def predict(jpeg: bytes, keep: bool = False) -> dict:
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
    if keep:  # kaydetme için ham görüntü ve kırpma da lazım
        out["_img"], out["_crop"] = img, crop
    return out


def font(size: int):
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), size)


def cam_model():
    """models/model_b.pt değiştiyse yeniden yükler; yoksa None."""
    pt = MODELS / f"{CAM_MODEL}.pt"
    if not pt.exists():
        return None
    if _cam.get("mtime") != pt.stat().st_mtime:
        import timm
        import torch
        meta = json.loads((MODELS / f"{CAM_MODEL}.json").read_text())
        model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False,
                                  num_classes=len(meta["classes"]))
        model.load_state_dict(torch.load(pt, map_location="cpu"))
        model.eval()
        _cam.update(mtime=pt.stat().st_mtime, model=model, layer=model.stages[-1], meta=meta)
    return _cam


def gradcam(crop: Image.Image) -> Image.Image | None:
    """training.ipynb'deki gradcam() ile aynı: son aşamanın aktivasyonları × tahmin edilen sınıfın gradyanı."""
    import matplotlib
    import torch
    import torch.nn.functional as F
    with _cam_lock:
        c = cam_model()
        if c is None:
            return None
        meta, model, layer = c["meta"], c["model"], c["layer"]
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        x = torch.from_numpy(((arr - meta["mean"]) / meta["std"]).transpose(2, 0, 1)[None].astype(np.float32))
        acts, grads = {}, {}
        h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
        x.requires_grad_(True)
        out = model(x)
        model.zero_grad(); out[0, int(out.argmax(1))].backward()
        h1.remove(); h2.remove()
        w = grads["g"].mean((2, 3), keepdim=True)
        cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().numpy()
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return Image.blend(crop, Image.fromarray(heat), 0.45)


def screenshot(img: Image.Image, box, r: dict) -> Image.Image:
    """Sayfada görünen hali: ayna görüntü + kutu + 'Tahmin %güven'."""
    shot = ImageOps.mirror(img)
    W, H = shot.size
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = (1 - x1) * W, y0 * H, (1 - x0) * W, y1 * H
    col, lw = COLORS[r["label"]], max(3, W // 200)
    d = ImageDraw.Draw(shot)
    d.rectangle([bx0, by0, bx1, by1], outline=col, width=lw)
    f = font(max(18, W // 26))
    text = f"{NAMES[r['label']]} %{round(r['conf'] * 100)}"
    tw, th = d.textlength(text, font=f) + 16, max(28, W // 18)
    ty = by0 if by0 - th < 0 else by0 - th
    d.rectangle([bx0, ty, bx0 + tw, ty + th], fill=col)
    d.text((bx0 + 8, ty + th / 2), text, font=f, fill="white", anchor="lm")
    return shot


def save_mistake(jpeg: bytes, showed: str) -> dict:
    out = predict(jpeg, keep=True)
    if not out["hand"]:
        return {"ok": False, "error": "El bulunamadı, tekrar dene."}
    if CAM_MODEL not in out["models"]:
        return {"ok": False, "error": f"{CAM_MODEL} yüklü değil."}
    img, crop = out.pop("_img"), out.pop("_crop")
    r = out["models"][CAM_MODEL]
    folder = MISTAKES / f"{time.strftime('%Y%m%d_%H%M%S')}_{showed}"
    folder.mkdir(parents=True, exist_ok=True)
    img.save(folder / "frame.jpg", quality=92)  # ham kare (aynasız, kutusuz)
    shot = screenshot(img, out["box"], r)
    shot.save(folder / "screenshot.jpg", quality=90)
    crop.save(folder / "crop.jpg", quality=95)
    cam = gradcam(crop)
    if cam is not None:
        cam.save(folder / "gradcam.jpg", quality=95)
    info = dict(showed=showed, said=r["label"], conf=r["conf"], probs=r["probs"], wrong=r["label"] != showed,
                model=CAM_MODEL, box=out["box"], all_models=out["models"])
    (folder / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return {"ok": True, "id": folder.name, **{k: info[k] for k in ("showed", "said", "conf", "wrong")}}


def list_mistakes() -> list[dict]:
    items = []
    for d in sorted(MISTAKES.glob("*/info.json")):
        info = json.loads(d.read_text())
        items.append(dict(id=d.parent.name, showed=info["showed"], said=info["said"], conf=info["conf"],
                          wrong=info["wrong"], has_cam=(d.parent / "gradcam.jpg").exists()))
    return items


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
        elif path == "/mistakes":
            self._send(200, json.dumps(list_mistakes()).encode())
        elif path.startswith("/mistakes/") and path.endswith(".jpg"):
            f = (MISTAKES / path[len("/mistakes/"):]).resolve()
            if f.is_relative_to(MISTAKES) and f.exists():
                self._send(200, f.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path not in ("/predict", "/save"):
            return self._send(404, b"not found", "text/plain")
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            if url.path == "/save":
                showed = parse_qs(url.query).get("showed", [""])[0]
                if showed not in ORDER:
                    return self._send(400, b'{"error": "showed"}')
                return self._send(200, json.dumps(save_mistake(data, showed)).encode())
            self._send(200, json.dumps(predict(data)).encode())
        except Exception as e:  # bozuk kare vb.: sayfa hata durumunu göstersin
            self._send(500, json.dumps({"error": str(e)}).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("modeller:", [m["name"] for m in load_models()] or "henüz yok")
    print(f"Deneme sayfası: http://localhost:{PORT}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
