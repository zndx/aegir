import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Silently swallow proxy errors during startup — the Aegir gateway may not
// be running yet when the Vite dev server starts (or running in a different
// terminal). Matches ~/local/src/zndx/atelier/ui/vite.config.ts.
const silenceProxyError = (err: Error, _req: unknown, _res: unknown) => {
  if ((err as NodeJS.ErrnoException).code === "ECONNREFUSED") return;
  console.error("[vite] proxy error:", err.message);
};

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: true,
    proxy: {
      "/api": {
        // Aegir gateway port — different from Atelier's 8090 to avoid
        // collision when both projects run on the same dev box.
        target: "http://localhost:8091",
        changeOrigin: true,
        configure: (proxy) => { proxy.on("error", silenceProxyError); },
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
