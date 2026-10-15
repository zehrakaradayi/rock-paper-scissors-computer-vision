/** @type {import('next').NextConfig} */
const nextConfig = {
  // Alt yolda çalışırken (ör. NEXT_PUBLIC_BASE_PATH=/lab/proxy/3000) _next/ dosyaları da o önekle istenir
  assetPrefix: process.env.NEXT_PUBLIC_BASE_PATH || undefined,
  // Cross-origin isolation: ONNX Runtime Web'in birden fazla thread kullanabilmesi için gerekli.
  // Tüm dosyalar kendi sitemizden geldiği için bir şeyi bozmaz.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

export default nextConfig;
