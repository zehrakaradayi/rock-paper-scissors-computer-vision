/** @type {import('next').NextConfig} */
const nextConfig = {
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
