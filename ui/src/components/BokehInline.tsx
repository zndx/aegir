import { useEffect, useId, useRef, useState } from "react";
import { Alert } from "antd";

// Sibling of BokehPlot for docs that are already in hand (no fetch). The lineup inlines a
// pre-rendered Bokeh json_item onto a lens note's `chord` frontmatter, so we embed the parsed
// object directly. Same bundled-BokehJS, same embed_item mount; we just skip the network round-trip.
import * as Bokeh from "@bokeh/bokehjs";

interface BokehInlineProps {
  /** A parsed Bokeh json_item document (target_id / doc / root_id), as produced by bokeh.embed.json_item. */
  doc: unknown;
  /** Human-readable label, shown on error. */
  title?: string;
  height?: number | string;
}

/**
 * Mounts an in-hand Bokeh JSON document into a div via ``Bokeh.embed.embed_item``. Unique DOM id per
 * instance so multiple chords can coexist. Re-embeds when ``doc`` changes (lens switch swaps the chord).
 */
export default function BokehInline({ doc, title, height = 480 }: BokehInlineProps) {
  const divId = useId().replace(/:/g, "_");
  const hostRef = useRef<HTMLDivElement | null>(null);
  const runRef = useRef(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    const host = hostRef.current;
    if (!host || !doc) return;
    // A *unique* target per effect run: useId is stable, so under StrictMode's mount→unmount→mount
    // (and on lens switches) embed_item would otherwise append a second plot into the same div.
    const targetId = `${divId}-${++runRef.current}`;
    host.innerHTML = `<div id="${targetId}"></div>`;
    (async () => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (Bokeh as any).embed.embed_item(doc, targetId);
        if (cancelled) host.innerHTML = "";   // a remount superseded us — drop this view
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; host.innerHTML = ""; };  // remove on unmount / re-run
  }, [doc, divId]);

  if (error) {
    return (
      <Alert type="warning" showIcon style={{ margin: "4px 0 12px" }}
        message={title ? `Chord '${title}' failed to render` : "Chord failed to render"}
        description={error} />
    );
  }

  return <div ref={hostRef} style={{ minHeight: height, display: "grid", placeItems: "center" }} />;
}
