import { useEffect, useId, useRef, useState } from "react";
import { Alert, Spin } from "antd";

// BokehJS is bundled via @bokeh/bokehjs (no CDN). KNOWN LIMITATION: the vite-bundled build
// mishandles GraphRenderer (Chord/Graph/Sankey) — curves/scatter embed fine, graph elements throw.
// scripts/experiment_panel_lineup.py renders those live via Panel/Bokeh-server (python-bokeh's JS,
// also fully air-gapped) — the convergence path. Types are bundled with the package; the import has
// side effects that register Bokeh on the global scope, which is what Bokeh.embed.embed_item
// expects from the JSON produced by bokeh.embed.json_item() on the server.
import * as Bokeh from "@bokeh/bokehjs";

type BokehJsonItem = unknown;

interface BokehPlotProps {
  /** URL returning a static Bokeh JSON document. */
  src: string;
  /** Human-readable label, shown on error. */
  title?: string;
  /** Optional fixed height; Bokeh docs are also self-sized. */
  height?: number | string;
}

/**
 * Fetches a pre-rendered Bokeh JSON document and mounts it into a div via
 * ``Bokeh.embed.embed_item``. Uses a unique DOM id per instance so multiple
 * plots can coexist on one page.
 *
 * Current implementation: plots are **static** (no DynamicMap, no Python callback) — this suits
 * simple curves/scatter. The convergence target for interactive/graph viz is live HoloViews via
 * Panel (Bokeh-server), not this static-embed path. If the server's backing JSON changes, we refetch
 * on mount by providing a new ``src`` URL with a cache-buster query string (the caller controls this).
 */
export default function BokehPlot({ src, title, height = 360 }: BokehPlotProps) {
  const divId = useId().replace(/:/g, "_");
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setLoading(true);

    (async () => {
      try {
        const res = await fetch(src, { headers: { "Cache-Control": "no-cache" } });
        if (!res.ok) {
          throw new Error(`${res.status} ${res.statusText}`);
        }
        const doc = (await res.json()) as BokehJsonItem;
        if (cancelled) return;
        const host = hostRef.current;
        if (!host) return;
        // Clear prior content (re-renders on src change).
        host.innerHTML = `<div id="${divId}"></div>`;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (Bokeh as any).embed.embed_item(doc, divId);
        setLoading(false);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [src, divId]);

  if (error) {
    return (
      <Alert
        type="error"
        showIcon
        message={title ? `Plot '${title}' failed to load` : "Plot failed to load"}
        description={error}
      />
    );
  }

  return (
    <div style={{ position: "relative", minHeight: height }}>
      {loading && (
        <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>
          <Spin />
        </div>
      )}
      <div ref={hostRef} style={{ minHeight: height }} />
    </div>
  );
}
