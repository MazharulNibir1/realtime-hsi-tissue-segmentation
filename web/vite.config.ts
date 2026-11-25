import { defineConfig } from "vite";

// Cross-origin isolation (COOP + COEP) enables SharedArrayBuffer, which multithreaded
// ONNX Runtime Web WASM needs. Production sets the same headers in vercel.json.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022", chunkSizeWarningLimit: 1200 },
});
