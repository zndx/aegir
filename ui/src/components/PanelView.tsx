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
  // GESTURE GATE (RH): plain scroll must never be trapped by the embed — the page scrolls; the
  // plot's wheel/pan tools engage only while Ctrl/⌘ is held (the maps-embed idiom). Mechanism:
  // capture-phase interception on the WRAPPER runs before the bokeh canvas's own listeners —
  // stopPropagation starves the canvas of the event WITHOUT preventDefault, so native page scroll
  // proceeds. Hover/click are untouched (tooltips keep working); one-finger touch scrolls the page,
  // multi-touch reaches the plot. A transient hint chip teaches the modifier.
  const [hint, setHint] = useState(false);
  const hintTimer = useRef<number | undefined>(undefined);
  const gateWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) return;                 // modifier held → the plot gets the gesture
    e.stopPropagation();
    setHint(true);
    window.clearTimeout(hintTimer.current);
    hintTimer.current = window.setTimeout(() => setHint(false), 1400);
  };
  const gateTouch = (e: React.TouchEvent) => {
    if (e.touches.length === 1) e.stopPropagation();    // one finger = page scroll; two = plot gesture
  };
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
  return (
    <div style={{ position: "relative", overflow: "hidden", maxWidth: "100%" }}
         onWheelCapture={gateWheel} onTouchMoveCapture={gateTouch}>
      {/* a plain block with definite width — a scale_width bokeh root SIZES FROM its container, and a
          centered grid cell sizes from content (circular → 0×0, the invisible-embed failure mode) */}
      <div ref={hostRef} style={{ minHeight: height, width: "100%" }} />
      <div style={{
        position: "absolute", top: 10, left: "50%", transform: "translateX(-50%)",
        padding: "3px 10px", borderRadius: 12, fontSize: 11, whiteSpace: "nowrap",
        background: "var(--color-kumo-elevated)", border: "1px solid var(--color-kumo-hairline)",
        color: "var(--text-color-kumo-subtle)", pointerEvents: "none",
        opacity: hint ? 1 : 0, transition: "opacity 0.25s ease",
      }}>
        hold Ctrl to zoom / pan the chart
      </div>
    </div>
  );
}
