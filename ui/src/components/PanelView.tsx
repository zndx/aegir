import { useEffect, useRef, useState } from "react";
import { Alert } from "antd";

import { useColorMode } from "../theme/colorMode";

interface PanelViewProps {
  /** Bokeh app name, served by the bokeh server at /viz/<app> behind the gateway proxy. */
  app: string;
  /** Query args forwarded to the app's session (e.g. {lens: "lens/terms"}). */
  params?: Record<string, string>;
  height?: number | string;
}

// BokehJS bundles served by the bokeh server (same-origin via the /viz proxy → air-gapped, and the
// GraphRenderer-correct build — unlike npm @bokeh/bokehjs). Loaded once, globally, before the autoload.
const BOKEH_BUNDLES = ["bokeh", "bokeh-gl", "bokeh-widgets", "bokeh-tables", "bokeh-mathjax"]
  .map((n) => `/viz/static/js/${n}.min.js`);

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[data-bk-src="${CSS.escape(src)}"]`)) { resolve(); return; }
    const s = document.createElement("script");
    s.src = src;
    s.async = false;                 // preserve order: core before gl/widgets/tables
    s.dataset.bkSrc = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

async function ensureBokeh(): Promise<void> {
  for (const url of BOKEH_BUNDLES) await loadScript(url);
}

/**
 * Embeds a LIVE HoloViews view (served by a bokeh server behind the gateway proxy) into a div.
 *
 * We preload the bokeh server's own BokehJS, then inject the `bokeh.embed.server_document` bootstrap
 * from `/api/viz/<app>/embed`. With `window.Bokeh` already defined, the autoload skips its async
 * self-loader (whose `.onload` chain breaks under dynamic injection → "reading 'safely' of undefined")
 * and renders immediately. Everything is same-origin (air-gapped); no npm `@bokeh/bokehjs`, no iframe.
 * StrictMode-safe: dedupe per `app|params` via a data-attr on the persistent host, no mid-load teardown.
 */
export default function PanelView({ app, params, height = 540 }: PanelViewProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  // the live doc theme follows the UI mode (aegir.viz.theme) — mode is part of the session key,
  // so toggling re-embeds the panel with a fresh, matching-theme session
  const mode = useColorMode();
  const qs = new URLSearchParams({ ...(params ?? {}), mode }).toString();

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const key = `${app}|${qs}`;
    if (host.dataset.bkKey === key) return;   // StrictMode remount / unchanged → keep the live view
    host.dataset.bkKey = key;
    setError(null);
    host.innerHTML = "";
    (async () => {
      try {
        await ensureBokeh();
        const res = await fetch(`/api/viz/${app}/embed${qs ? `?${qs}` : ""}`);
        if (!res.ok) throw new Error(`embed bootstrap ${res.status}`);
        const html = await res.text();
        const parsed = document.createElement("div");
        parsed.innerHTML = html;
        const target = document.createElement("div");
        host.appendChild(target);
        parsed.querySelectorAll("script").forEach((old) => {
          const s = document.createElement("script");
          for (const a of Array.from(old.attributes)) s.setAttribute(a.name, a.value);
          s.textContent = old.textContent;
          target.appendChild(s);
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [app, qs]);

  if (error) {
    return (
      <Alert type="warning" showIcon style={{ margin: "4px 0 12px" }}
        message={`Live view '${app}' failed to load`} description={error} />
    );
  }
  return <div ref={hostRef} style={{ minHeight: height, display: "grid", placeItems: "center" }} />;
}
