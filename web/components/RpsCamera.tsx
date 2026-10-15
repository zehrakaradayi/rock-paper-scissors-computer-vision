"use client";

import { useEffect, useRef, useState } from "react";
import {
  BASE, handBox, loadClassifier, loadHandLandmarker, NAMES, ORDER,
  type Box, type Classifier, type ModelMeta, type Prediction,
} from "@/lib/rps";

const COLORS = { rock: "#c2410c", paper: "#1d4ed8", scissors: "#15803d" } as const;

// ?model=b gibi bir bağlantıyla doğrudan o model açılır
const MODELS = [
  { id: "model_a", key: "a", label: "Model A" },
  { id: "model_b", key: "b", label: "Model B" },
  { id: "model_b_fix", key: "b-fix", label: "B düzeltilmiş" },
] as const;
type ModelId = (typeof MODELS)[number]["id"];

type Status =
  | { kind: "loading"; text: string }
  | { kind: "error"; text: string }
  | { kind: "running" };

type Active = { id: ModelId; meta: ModelMeta; clf: Classifier };

export default function RpsCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const activeRef = useRef<Active | null>(null);
  const [modelId, setModelId] = useState<ModelId | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "loading", text: "Model yükleniyor…" });
  const [modelLoad, setModelLoad] = useState<{ title: string; pct: number } | null>(null);
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [hasModel, setHasModel] = useState(false);

  useEffect(() => {
    const key = new URLSearchParams(location.search).get("model");
    setModelId(MODELS.find((m) => m.key === key)?.id ?? "model_a");
  }, []);

  // Seçilen modeli yükle; hazır olunca eskisinin yerine geç
  useEffect(() => {
    if (!modelId) return;
    let cancelled = false;
    const prev = activeRef.current;
    activeRef.current = null;
    setPred(null);
    prev?.clf.release().catch(console.error);

    (async () => {
      try {
        const m: ModelMeta = await (await fetch(`${BASE}/models/${modelId}.json`)).json();
        if (cancelled) return;
        setMeta(m);
        setModelLoad({ title: m.title, pct: 0 });
        const clf = await loadClassifier(m, (p) => !cancelled && setModelLoad({ title: m.title, pct: p }));
        if (cancelled) return void clf.release();
        activeRef.current = { id: modelId, meta: m, clf };
        setModelLoad(null);
        setHasModel(true);
      } catch (e) {
        console.error(e);
        if (!cancelled) setStatus({ kind: "error", text: "Model yüklenemedi. Sayfayı yenilemeyi dene." });
      }
    })();
    return () => { cancelled = true; };
  }, [modelId]);

  // El bulucu + kamera + döngü: bir kez açılır, model değişince yeniden başlamaz
  useEffect(() => {
    let stopped = false, raf = 0;
    let stream: MediaStream | null = null;

    (async () => {
      let landmarker: Awaited<ReturnType<typeof loadHandLandmarker>>;
      try {
        landmarker = await loadHandLandmarker();
      } catch (e) {
        console.error(e);
        return setStatus({ kind: "error", text: "El bulucu yüklenemedi. Sayfayı yenilemeyi dene." });
      }
      if (stopped) return;

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
      setStatus((s) => (s.kind === "error" ? s : { kind: "running" }));

      let busy = false, box: Box | null = null, last: Prediction | null = null, lastTs = 0;
      const canvas = canvasRef.current!;

      const frame = () => {
        if (stopped) return;
        const vw = video.videoWidth, vh = video.videoHeight;
        const active = activeRef.current;
        if (vw && video.readyState >= 2) {
          const ts = Math.max(performance.now(), lastTs + 1);
          lastTs = ts;
          const hands = active ? landmarker.detectForVideo(video, ts).landmarks : [];
          box = hands.length ? handBox(hands[0], vw, vh, active!.meta.crop_margin) : null;
          if (!box) {
            last = null;
            setPred(null);
          } else if (!busy) {
            busy = true;
            active!.clf
              .classify(video, box)
              .then((p) => {
                if (box && activeRef.current === active) { last = p; setPred(p); }
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

  const pick = (id: ModelId) => {
    if (id === modelId) return;
    setModelId(id);
    const url = new URL(location.href);
    url.searchParams.set("model", MODELS.find((m) => m.id === id)!.key);
    history.replaceState(null, "", url);
  };

  const loadText = modelLoad && `${modelLoad.title} yükleniyor… %${Math.round(modelLoad.pct * 100)}`;
  // ilk açılışta tam ekran, model değişirken sadece küçük bir şerit
  const overlay =
    status.kind === "error" ? status.text
    : status.kind === "loading" || !hasModel ? (loadText ?? (status.kind === "loading" ? status.text : "Model yükleniyor…"))
    : null;
  const showHint = status.kind === "running" && !modelLoad && !pred;

  return (
    <>
      <div className="models" role="radiogroup" aria-label="Model seç">
        {MODELS.map((m) => (
          <button
            key={m.id}
            role="radio"
            aria-checked={modelId === m.id}
            className={modelId === m.id ? "on" : ""}
            onClick={() => pick(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="stage">
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} />
        {overlay && (
          <div className={`overlay ${status.kind === "error" ? "error" : ""}`}>
            {status.kind !== "error" && <span className="spinner" aria-hidden />}
            <p>{overlay}</p>
          </div>
        )}
        {!overlay && loadText && <div className="loading-pill">{loadText}</div>}
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
        {meta?.zor_macro_f1 != null && (
          <div>
            <span className="k">Zor test</span>
            <span>{meta.zor_macro_f1.toFixed(3)}</span>
          </div>
        )}
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
