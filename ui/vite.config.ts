import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Silently swallow proxy errors during startup — the Aegir gateway may not
// be running yet when the Vite dev server starts (or running in a different
// terminal). Matches ~/local/src/zndx/atelier/ui/vite.config.ts.
const silenceProxyError = (err: Error, _req: unknown, _res: unknown) => {
  if ((err as NodeJS.ErrnoException).code === "ECONNREFUSED") return;
  console.error("[vite] proxy error:", err.message);
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // @xyflow/react is reachable only through a lazy import (ProvenanceGraph), so Vite's startup dep-scan never
  // pre-bundles it — opening Provenance then triggers an on-demand re-optimize and the in-flight dynamic import
  // 504s ("Outdated Optimize Dep"). Force it into the startup optimization so the lazy chunk loads cleanly.
  optimizeDeps: {
    include: ["@xyflow/react"],
  },
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
      // Live HoloViews panel servers (topology B). In prod this is the gateway's reverse proxy
      // (nginx/traefik); in dev Vite stands in. ws:true carries the Bokeh-server session socket so
      // the chord stays same-origin (air-gapped) and embeds cleanly via server_document.
      "/viz": {
        target: "http://localhost:5006",
        changeOrigin: true,
        ws: true,
        configure: (proxy) => { proxy.on("error", silenceProxyError); },
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
