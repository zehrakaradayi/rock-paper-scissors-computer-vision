// Tarayıcıda çalışan kütüphanelerin dosyalarını (wasm) public/ altına kopyalar.
// Böylece her şey kendi sitemizden yüklenir, dışarıdaki bir CDN'e bağımlı kalmayız.
import { cpSync, mkdirSync } from "node:fs";

const copy = (from, to, files) => {
  mkdirSync(to, { recursive: true });
  for (const f of files) cpSync(`${from}/${f}`, `${to}/${f}`);
};

copy("node_modules/onnxruntime-web/dist", "public/ort", [
  "ort.wasm.bundle.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
]);
copy("node_modules/@mediapipe/tasks-vision", "public/mediapipe", ["vision_bundle.mjs"]);
copy("node_modules/@mediapipe/tasks-vision/wasm", "public/mediapipe/wasm", [
  "vision_wasm_internal.js",
  "vision_wasm_internal.wasm",
  "vision_wasm_nosimd_internal.js",
  "vision_wasm_nosimd_internal.wasm",
]);
console.log("wasm dosyaları public/ altına kopyalandı");
