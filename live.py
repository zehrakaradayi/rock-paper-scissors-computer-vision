"""Eğitim sürerken canlı güncellenen paneller (önceden çekilmiş video değil, gerçek eğitimden çizilir).

Eğitim başlayınca tarayıcıda bir sayfa açılır (live_cnn.html). Eğitilen batch'in fotoğrafları kayar, biri büyür.
Bir damga (hep aynı kanal) fotoğrafın üstünden geçer; bir yer damgaya ne kadar benziyorsa 3D şehirdeki kulesi
o kadar yükselir. Değerler modelin O ANKİ ağırlıklarından birkaç saniyede bir alınır, model öğrendikçe değişir.

CNNLive      Model A, 5 blok: Conv → BatchNorm (su çizgisi ortalamada) → ReLU (suyun altı silinir) → MaxPool
             (her 2x2'nin en uzunu kalır). Blok 1 yavaş, 2-5 hızlı; sonunda Taş / Kağıt / Makas.
FineTuneLive Model B (ConvNeXt), 4 aşama: Patchify (fotoğraf karelere bölünür, şehir bir anda çıkar), sonra her
             aşamadan bir blok: Conv 7x7 → LayerNorm → GELU (suyun altı ezilir, tamamen silinmez) → Residual
             (eski şehir altta geri gelir, yenisi üstüne eklenir), aşamalar arasında Downsample. Donuk (öğrenmeyen)
             aşamalar gri, öğrenen aşamalar mavi çizilir; 2. adımda son aşama griden maviye döner.

İkisi de notebook'taki train(..., view=...) tarafından her batch'ten sonra çağrılır (x, y: o anki batch).
Sayfayı eğitim sürecinin içindeki küçük bir sunucu verir. VS Code ile uzak sunucuya bağlıysan sayfa senin
bilgisayarında açılır; açılmazsa hücre çıktısındaki linke tıkla.
"""
import base64
import io
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

TR = {"paper": "Kağıt", "rock": "Taş", "scissors": "Makas"}
LIVE_HTML = Path(__file__).with_name("live_cnn.html")
_STATE, _LOCK, _SERVER = {"id": 0, "epoch": 0, "epochs": 0}, threading.Lock(), None


def _serve():
    """Panel sayfası için küçük yerel sunucu (eğitim sürecinin içinde, arka planda): / sayfa, /state değerler."""
    global _SERVER
    if _SERVER:
        return _SERVER.server_address[1]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                since = parse_qs(urlparse(self.path).query).get("since", [""])[0]
                with _LOCK:  # değerler değişmediyse sadece epoch sayacı gider
                    body = json.dumps(_STATE if since != str(_STATE["id"]) else
                                      {k: _STATE[k] for k in ("id", "epoch", "epochs")}).encode()
                kind = "application/json"
            else:
                body, kind = LIVE_HTML.read_bytes(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # her istekte notebook'a satır basmasın
            pass

    for port in range(8002, 8020):  # 8001 deneme sayfası (play.py)
        try:
            _SERVER = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        raise OSError("8002-8019 arası boş port yok")
    threading.Thread(target=_SERVER.serve_forever, daemon=True).start()
    return port


def _jpeg(rgb, size):
    """(H, W, 3) 0..1 görüntü → sayfada gösterilecek küçük JPEG (data URL)."""
    im = Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).resize((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _grid(a, g):
    """(H, W) aktivasyon → g x g kule yüksekliği (bölge ortalaması)."""
    return F.adaptive_avg_pool2d(a[None, None].float(), g)[0, 0].tolist()


def _unit(w):
    return (w / w.abs().max().clamp_min(1e-8)).tolist()


class _Browser:
    every = 3.0  # sn; sayfa bir turu ~16 sn'de oynatır ve her tur başında en yeni değerleri alır

    def __init__(self, kind, title, ds, to_rgb, classes, epochs, window):
        self.to_rgb, self.classes, self.epochs = to_rgb, classes, epochs
        names = [c for c in ("rock", "paper", "scissors") if c in classes]  # Taş, Kağıt, Makas sırası
        self.order = [classes.index(c) for c in names] if len(names) == len(classes) else list(range(len(classes)))
        samples = [ds[ds.targets.index(k)] for k in range(len(classes))]  # train() hiç batch vermezse
        self.batch = torch.stack([s[0] for s in samples]), torch.tensor([s[1] for s in samples])
        self.channels, self.turn, self.last = None, 0, 0.0
        with _LOCK:
            _STATE.clear()
            _STATE.update(id=0, epoch=1, epochs=epochs, kind=kind, title=title)
        url = f"http://localhost:{_serve()}/"
        print(f"Canlı panel: {url}  (tarayıcıda açılmadıysa bu linke tıkla)")
        if window:
            webbrowser.open(url)  # VS Code uzak bağlantıda $BROWSER sayesinde senin bilgisayarında açılır

    def __call__(self, model, history, batch, n_batches, force=False, x=None, y=None):
        with _LOCK:
            _STATE["epoch"] = min(len(history) + 1, self.epochs)
        if x is not None:
            self.batch = x, y  # epoch sonundaki çağrı batch vermez: son batch kullanılır
        if time.time() - self.last < self.every:
            return
        x, y = (t.to(next(model.parameters()).device) for t in self.batch)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            snap = self._snapshot(model, x, y)
        model.train(was_training)
        with _LOCK:
            _STATE.update(snap, id=_STATE["id"] + 1)
        self.last = time.time()

    def _snapshot(self, model, x, y):
        """Batch'ten sıradaki sınıfın bir fotoğrafı + modelin o anki değerleri (_values)."""
        want = self.order[self.turn % len(self.order)]
        self.turn += 1
        hits = (y == want).nonzero()
        pick = int(hits[0]) if len(hits) else 0
        img = x[pick:pick + 1]
        return dict(self._values(model, img), thumbs=[_jpeg(self.to_rgb(t.cpu()), 48) for t in x],
                    photo=_jpeg(self.to_rgb(img[0].cpu()), 224), pick=pick, label=int(y[pick]),
                    names=[TR.get(c, c) for c in self.classes], order=self.order)

    def _hooked(self, model, img, mods):
        """mods: {isim: modül}. Modeli çalıştırır; her modülün (giriş, çıkış)'ını ve olasılıkları döner."""
        acts = {}
        hooks = [m.register_forward_hook(lambda _m, i, o, k=k: acts.__setitem__(k, (i[0][0], o[0])))
                 for k, m in mods.items()]
        probs = model(img).softmax(1)[0].tolist()
        for h in hooks:
            h.remove()
        return acts, probs

    def close(self):
        """Eğitim bitti: sayfa son halini oynatmaya devam eder."""


class CNNLive(_Browser):
    GRIDS = [8, 4, 2, 1, 1]  # her bloğun çıktısı bu kadar kuleye ortalanır: şehir blok blok küçülür

    def __init__(self, model, ds, to_rgb, classes, epochs, window="auto"):
        self.blocks = [m for m in model if isinstance(m, nn.Sequential)]
        super().__init__("A", "Model A öğreniyor", ds, to_rgb, classes, epochs, window)

    def _values(self, model, img):
        acts, probs = self._hooked(model, img, {(b, s): m for b, blk in enumerate(self.blocks)
                                                for s, m in enumerate(blk[:2])})
        if self.channels is None:  # takip edilen damga: ilk anda en çok tepki veren kanal, eğitim boyunca aynı
            self.channels = [int(acts[b, 1][1].flatten(1).std(1).argmax()) for b in range(len(self.blocks))]
        blocks = [{"conv": _grid(acts[b, 0][1][c], g), "bn": _grid(acts[b, 1][1][c], g)}
                  for b, (c, g) in enumerate(zip(self.channels, self.GRIDS))]
        w = self.blocks[0][0].weight[self.channels[0]].mean(0)  # damganın 3x3 ağırlıkları (renk kanallarının ortalaması)
        return dict(blocks=blocks, stamp=_unit(w), probs=probs)


class FineTuneLive(_Browser):
    GRIDS = [16, 8, 4, 2]  # aşama başına kule ızgarası (gerçek: 56, 28, 14, 7)

    def __init__(self, model, ds, to_rgb, classes, epochs, window="auto"):
        self.stages, self.head = list(model.stages), model.head
        super().__init__("B", "Model B öğreniyor", ds, to_rgb, classes, epochs, window)

    def _values(self, model, img):
        mods = {}
        for s, st in enumerate(self.stages):  # her aşamanın ilk bloğu o aşamayı temsil eder
            blk = st.blocks[0]
            mods.update({(s, "blk"): blk, (s, "conv"): blk.conv_dw, (s, "ln"): blk.norm})
        acts, probs = self._hooked(model, img, mods)
        ch = lambda a, n: a if a.shape[0] == n else a.permute(2, 0, 1)  # LayerNorm çıktısı (H, W, C) gelir
        if self.channels is None:
            # ponytail: el genelde kırpımın ortasında; ortası ortalamadan en çok yüksek kanal = ele en çok tepki veren
            def hand(a):
                h, w = a.shape[1:]
                mid = a[:, h // 4:h - h // 4, w // 4:w - w // 4].mean((1, 2))
                return int((mid - a.mean((1, 2))).argmax())
            self.channels = [hand(acts[s, "blk"][1]) for s in range(len(self.stages))]
        stages = []
        for s, (c, g) in enumerate(zip(self.channels, self.GRIDS)):
            inp, out = acts[s, "blk"]
            stages.append({"inp": _grid(inp[c], g), "conv": _grid(acts[s, "conv"][1][c], g),
                           "ln": _grid(ch(acts[s, "ln"][1], inp.shape[0])[c], g), "out": _grid(out[c], g)})
        learns = lambda m: any(p.requires_grad for p in m.parameters())
        return dict(stages=stages, stamp=_unit(self.stages[0].blocks[0].conv_dw.weight[self.channels[0], 0]),
                    frozen=[not learns(st) for st in self.stages], head_frozen=not learns(self.head),
                    counts=[len(st.blocks) for st in self.stages], probs=probs)
