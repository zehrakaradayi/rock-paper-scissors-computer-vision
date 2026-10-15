"use client";

import { useEffect, useRef, useState } from "react";
import { BASE, handBox, loadClassifier, loadHandLandmarker, NAMES, ORDER, type Box, type ModelMeta, type Prediction } from "@/lib/rps";

const COLORS = { rock: "#c2410c", paper: "#1d4ed8", scissors: "#15803d" } as const;

type Status =
  | { kind: "loading"; text: string }
  | { kind: "error"; text: string }
  | { kind: "running" };

export default function RpsCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<Status>({ kind: "loading", text: "Model yükleniyor…" });
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);

  useEffect(() => {
    let stopped = false, raf = 0;
    let stream: MediaStream | null = null;

    (async () => {
      let m: ModelMeta, landmarker: Awaited<ReturnType<typeof loadHandLandmarker>>;
      let classify: Awaited<ReturnType<typeof loadClassifier>>;
      try {
        m = await (await fetch(BASE + "/models/model_a.json")).json();
        setMeta(m);
        [landmarker, classify] = await Promise.all([
          loadHandLandmarker(),
          loadClassifier(m, (p) => setStatus({ kind: "loading", text: `Model yükleniyor… %${Math.round(p * 100)}` })),
        ]);
      } catch (e) {
        console.error(e);
        return setStatus({ kind: "error", text: "Model yüklenemedi. Sayfayı yenilemeyi dene." });
      }
      if (stopped) return;

      setStatus({ kind: "loading", text: "Kamera açılıyor…" });
      const video = videoRef.current!;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();
      } catch {
        return setStatus({ kind: "error", text: "Kameraya erişilemedi. Lütfen izin ver." });
      }
      if (stopped) return;
      setStatus({ kind: "running" });

      let busy = false, box: Box | null = null, last: Prediction | null = null, lastTs = 0;
      const canvas = canvasRef.current!;

      const frame = () => {
        if (stopped) return;
        const vw = video.videoWidth, vh = video.videoHeight;
        if (vw && video.readyState >= 2) {
          const ts = Math.max(performance.now(), lastTs + 1);
          lastTs = ts;
          const hands = landmarker.detectForVideo(video, ts).landmarks;
          box = hands.length ? handBox(hands[0], vw, vh, m.crop_margin) : null;
          if (!box) {
            last = null;
            setPred(null);
          } else if (!busy) {
            busy = true;
            const b = box;
            classify(video, b)
              .then((p) => {
                if (box) { last = p; setPred(p); }
              })
              .catch(console.error)
              .finally(() => (busy = false));
          }
          draw(canvas, vw, vh, box, last);
        }
        raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);
    })();

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const showHint = status.kind === "running" && !pred;

  return (
    <>
      <div className="stage">
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} />
        {status.kind !== "running" && (
          <div className={`overlay ${status.kind === "error" ? "error" : ""}`}>
            {status.kind === "loading" && <span className="spinner" aria-hidden />}
            <p>{status.text}</p>
          </div>
        )}
        {showHint && <div className="hint">Elini kameraya göster</div>}
      </div>

      <div className="bars" aria-live="polite">
        {ORDER.map((c) => {
          const p = pred ? pred.probs[c] : 0;
          return (
            <div key={c} className={`bar ${pred?.label === c ? "top" : ""}`}>
              <span>{NAMES[c]}</span>
              <div className="track"><div className="fill" style={{ width: `${p * 100}%`, background: COLORS[c] }} /></div>
              <span className="pct">{pred ? `%${Math.round(p * 100)}` : "–"}</span>
            </div>
          );
        })}
      </div>

      <aside className="info">
        <div>
          <span className="k">Model</span>
          <span>{meta ? meta.title : "…"}</span>
        </div>
        <div>
          <span className="k">Macro F1 (test)</span>
          <span>{meta ? meta.test_macro_f1.toFixed(3) : "…"}</span>
        </div>
        <p>Model tamamen tarayıcında çalışır; kamera görüntün hiçbir yere gönderilmez.</p>
      </aside>
    </>
  );
}

// Kamera görüntüsü CSS ile ayna çevrili; kutuyu da aynı şekilde çevirerek çiziyoruz, yazıyı düz bırakıyoruz.
function draw(canvas: HTMLCanvasElement, vw: number, vh: number, box: Box | null, pred: Prediction | null) {
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw;
    canvas.height = vh;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, vw, vh);
  if (!box || !pred) return;
  const color = COLORS[pred.label];
  const x = vw - (box.x + box.size), y = box.y, s = box.size;
  const lw = Math.max(3, vw / 220);
  ctx.lineWidth = lw;
  ctx.strokeStyle = color;
  ctx.strokeRect(x, y, s, s);

  const text = `${NAMES[pred.label]} %${Math.round(pred.conf * 100)}`;
  const fs = Math.max(20, vw / 24);
  ctx.font = `600 ${fs}px system-ui, -apple-system, sans-serif`;
  const tw = ctx.measureText(text).width + fs * 0.7, th = fs * 1.5;
  const tx = Math.min(Math.max(0, x - lw / 2), vw - tw);
  const ty = y - th >= 0 ? y - th : Math.max(0, y);
  ctx.fillStyle = color;
  ctx.fillRect(tx, ty, tw, th);
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, tx + fs * 0.35, ty + th / 2);
}
