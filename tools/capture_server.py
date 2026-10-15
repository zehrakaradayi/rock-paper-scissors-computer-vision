"""Aşama 1 — Dataset toplama aracı.

Laptop kamerasıyla tek tıkla fotoğraf çekip dataset/<sınıf>/ klasörüne kaydeder.
Telefonla çekilen fotoğrafları da aynı sayfadan yükleyebilirsin.

Çalıştırma:  python3 tools/capture_server.py   →  http://localhost:8765
             python3 tools/capture_server.py 8766 dataset_zor/yuz   (başka port + başka kayıt klasörü)
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "dataset")
PAGE = Path(__file__).resolve().parent / "capture.html"
CLASSES = ("rock", "paper", "scissors")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

last_saved: list[Path] = []


def counts() -> dict:
    return {
        c: sum(1 for p in (DATASET / c).iterdir() if p.suffix.lower() in EXTS)
        for c in CLASSES
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/counts":
            self._json(counts())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)

        if url.path == "/undo":
            if last_saved:
                p = last_saved.pop()
                p.unlink(missing_ok=True)
            return self._json(counts())

        if url.path != "/save":
            return self._send(404, b"not found", "text/plain")

        label = q.get("label", [""])[0]
        source = q.get("source", ["laptop"])[0]
        ext = q.get("ext", [".jpg"])[0].lower()
        if label not in CLASSES or source not in ("laptop", "phone") or ext not in EXTS:
            return self._json({"error": "geçersiz istek"}, 400)

        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not data:
            return self._json({"error": "boş dosya"}, 400)

        name = f"{source}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 10**6:06d}{ext}"
        out = DATASET / label / name
        out.write_bytes(data)
        last_saved.append(out)
        self._json(counts())

    def log_message(self, fmt, *args):
        print(self.address_string(), fmt % args, flush=True)


if __name__ == "__main__":
    for c in CLASSES:
        (DATASET / c).mkdir(parents=True, exist_ok=True)
    print(f"Fotoğraf çekme sayfası: http://localhost:{PORT}")
    print(f"Kayıt klasörü: {DATASET}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
