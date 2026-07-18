import { Flow } from "@cloudflare/kumo";

// The relational-table panel's ERD header, on Kumo Flow (org norm: cldr-design-template promotes
// Kumo as the common visual library; RH 2026-07-18). A STATIC orientation glyph — no zoom, no click
// (RH: "they only help users cognize and orient to their position within our rendered corpora") —
// showing the focal table's first-order neighborhood as three stages:
//   referencing tables (fan-in) → focal table (column preview) → FK targets + views (fan-out).
// The payload is precomputed by the KB projector into the note's `erd` frontmatter. Views render
// `disabled` (Kumo dims their connectors) — derived surfaces, visually subordinate to base tables.

export interface ErdNode { id: string; kind: "table" | "junction" | "view"; focal?: boolean; cols?: string[]; known?: boolean; }
export interface ErdEdge { source: string; target: string; label?: string; }
export interface ErdData {
  focal: string;
  nodes: ErdNode[];
  edges: ErdEdge[];
  more?: { referenced_by?: number; views?: number };
}

/** A neighbor card: table name, kind, and the FK column that ties it to the focal table. */
function NeighborCard({ n, fkCol }: { n: ErdNode; fkCol?: string }) {
  const junction = n.kind === "junction";
  const view = n.kind === "view";
  return (
    <div
      className={
        "rounded-md px-2.5 py-1.5 text-left " +
        (view
          ? "border border-dashed border-kumo-line bg-kumo-base"
          : junction
            ? "border border-kumo-line bg-kumo-elevated"
            : "border border-kumo-line bg-kumo-elevated")
      }
      style={{ maxWidth: "11rem" }}
    >
      <div className={"font-mono text-[11px] break-all " + (view ? "text-kumo-subtle" : "text-kumo-default")}>
        {n.id}
      </div>
      <div className="text-[8.5px] uppercase tracking-wide text-kumo-inactive">
        {junction ? "junction" : n.kind}
        {fkCol ? <span className="ml-1 font-mono normal-case text-kumo-subtle">· {fkCol}</span> : null}
      </div>
    </div>
  );
}

/** The focal table card: name + a short column:type preview (the panel's full detail sits below). */
function FocalCard({ erd }: { erd: ErdData }) {
  const focal = erd.nodes.find((n) => n.focal);
  return (
    <div className="rounded-md border-2 border-kumo-brand bg-kumo-recessed px-3 py-2 text-left" style={{ maxWidth: "13rem" }}>
      <div className="font-mono text-[12px] font-semibold text-kumo-strong break-all">{erd.focal}</div>
      <div className="text-[8.5px] uppercase tracking-wide text-kumo-inactive mb-1">
        {focal?.kind === "junction" ? "junction table" : "table"}
      </div>
      {focal?.cols?.length ? (
        <div className="border-t border-kumo-hairline pt-1 font-mono text-[9px] leading-relaxed text-kumo-subtle">
          {focal.cols.map((c) => <div key={c}>{c}</div>)}
        </div>
      ) : null}
    </div>
  );
}

export default function RelationalErd({ erd }: { erd: ErdData; onLink?: (id: string) => void }) {
  const focal = erd.focal;
  const fkColOf = (id: string) =>
    erd.edges.find((e) => (e.source === id && e.target === focal) || (e.source === focal && e.target === id))?.label;
  const incoming = erd.nodes.filter(
    (n) => !n.focal && n.kind !== "view" && erd.edges.some((e) => e.source === n.id && e.target === focal),
  );
  const outgoing = erd.nodes.filter(
    (n) => !n.focal && n.kind !== "view" && !incoming.includes(n)
      && erd.edges.some((e) => e.source === focal && e.target === n.id),
  );
  const views = erd.nodes.filter((n) => n.kind === "view");
  const more = erd.more ?? {};
  const moreBits = [
    more.referenced_by ? `+${more.referenced_by} more referencing` : null,
    more.views ? `+${more.views} more views` : null,
  ].filter(Boolean).join(" · ");

  return (
    <div className="mb-2 -mx-1">
      <div className="rounded-md border border-kumo-hairline bg-kumo-base p-2 overflow-x-auto">
        <Flow orientation="horizontal" align="center">
          {incoming.length > 0 && (
            <Flow.Parallel align="end">
              {incoming.map((n) => (
                <Flow.Node key={n.id} render={<NeighborCard n={n} fkCol={fkColOf(n.id)} />} />
              ))}
            </Flow.Parallel>
          )}
          <Flow.Node render={<FocalCard erd={erd} />} />
          {(outgoing.length > 0 || views.length > 0) && (
            <Flow.Parallel>
              {outgoing.map((n) => (
                <Flow.Node key={n.id} render={<NeighborCard n={n} fkCol={fkColOf(n.id)} />} />
              ))}
              {views.map((n) => (
                <Flow.Node key={n.id} disabled render={<NeighborCard n={n} />} />
              ))}
            </Flow.Parallel>
          )}
        </Flow>
      </div>
      <div className="py-0.5 text-center text-[11px] text-kumo-inactive">
        first-order ERD · referencing → <span className="text-kumo-subtle">this table</span> → FK targets · views dimmed
        {moreBits ? ` · ${moreBits}` : ""}
      </div>
    </div>
  );
}
