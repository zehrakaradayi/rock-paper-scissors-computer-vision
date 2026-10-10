// El bulma + kırpma + sınıflandırma. Hepsi tarayıcıda çalışır.
// Kırpma kuralı eğitimdekiyle (rps_crop.py) AYNI: 21 el noktasının kutusu → uzun kenar × 1.4 kare →
// görüntü dışına taşan kısım siyah → 224×224 → ImageNet normalizasyonu.

export const ORDER = ["rock", "paper", "scissors"] as const;
export type Label = (typeof ORDER)[number];
export const NAMES: Record<Label, string> = { rock: "Taş", paper: "Kağıt", scissors: "Makas" };

export type ModelMeta = {
  title: string;
  classes: string[];
  input_size: number;
  mean: number[];
  std: number[];
  crop_margin: number;
  test_macro_f1: number;
  file: string;
};
export type Box = { x: number; y: number; size: number }; // video pikseli, kare
export type Prediction = { probs: Record<Label, number>; label: Label; conf: number };

type Landmarker = {
  detectForVideo(video: HTMLVideoElement, ts: number): { landmarks: { x: number; y: number }[][] };
};

// public/ altındaki ES modüllerini paketleyiciye dokundurmadan yükle
const load = (url: string) => import(/* webpackIgnore: true */ url);

export async function loadHandLandmarker(): Promise<Landmarker> {
  const { FilesetResolver, HandLandmarker } = await load("/mediapipe/vision_bundle.mjs");
  const files = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: "/models/hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: 1,
  });
  try {
    return await HandLandmarker.createFromOptions(files, options("GPU"));
  } catch {
    return await HandLandmarker.createFromOptions(files, options("CPU"));
  }
}

export async function loadClassifier(meta: ModelMeta, onProgress: (pct: number) => void) {
  const ort = await load("/ort/ort.wasm.bundle.min.mjs");
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  const res = await fetch(meta.file);
  if (!res.ok || !res.body) throw new Error(`model indirilemedi (${res.status})`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress(got / total);
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    bytes.set(c, off);
    off += c.length;
  }
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });

  const S = meta.input_size;
  const crop = document.createElement("canvas");
  crop.width = crop.height = S;
  const ctx = crop.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = "high";

  return async function classify(video: HTMLVideoElement, box: Box): Promise<Prediction> {
    // kare kutuyu kırp; görüntü dışında kalan kısım siyah kalır (PIL.crop ile aynı)
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, S, S);
    const vw = video.videoWidth, vh = video.videoHeight, k = S / box.size;
    const sx = Math.max(0, box.x), sy = Math.max(0, box.y);
    const ex = Math.min(vw, box.x + box.size), ey = Math.min(vh, box.y + box.size);
    if (ex > sx && ey > sy) {
      ctx.drawImage(video, sx, sy, ex - sx, ey - sy, (sx - box.x) * k, (sy - box.y) * k, (ex - sx) * k, (ey - sy) * k);
    }
    const px = ctx.getImageData(0, 0, S, S).data;
    const x = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) {
      for (let c = 0; c < 3; c++) x[c * S * S + i] = (px[i * 4 + c] / 255 - meta.mean[c]) / meta.std[c];
    }
    const out = await session.run({ input: new ort.Tensor("float32", x, [1, 3, S, S]) });
    const logits = out.logits.data as Float32Array;
    const max = Math.max(...logits);
    const exp = Array.from(logits, (v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probs = Object.fromEntries(ORDER.map((c) => [c, exp[meta.classes.indexOf(c)] / sum])) as Record<Label, number>;
    const label = ORDER.reduce((a, b) => (probs[b] > probs[a] ? b : a));
    return { probs, label, conf: probs[label] };
  };
}

export function handBox(landmarks: { x: number; y: number }[], vw: number, vh: number, margin: number): Box {
  const xs = landmarks.map((p) => p.x * vw), ys = landmarks.map((p) => p.y * vh);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const size = Math.max(x1 - x0, y1 - y0) * margin;
  return { x: (x0 + x1) / 2 - size / 2, y: (y0 + y1) / 2 - size / 2, size };
}
