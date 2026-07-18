import { CloseOutlined } from "@ant-design/icons";
import { Tag, Typography } from "antd";
import { lazy, Suspense, useEffect } from "react";
import type { ReactNode } from "react";

import PanelView from "./PanelView";

// Lazy so @xyflow/react is code-split off the main lineup path — the heavy graph lib loads only when a
// Provenance panel opens, so it can never break the rest of the lineup (and the dev server pulls it on demand).
const ProvenanceGraph = lazy(() => import("./ProvenanceGraph"));
const RelationalErd = lazy(() => import("./RelationalErd"));

const { Text } = Typography;

export interface KBNote {
  id: string;
  // The gateway's /api/kb/note response carries the root-less id as `name` (not `id`); prefer it.
  name?: string;
  title: string;
  kind: string;
  data_product: string;
  root?: string;
  body: string;
  links?: string[];
  // Training notes name the bokeh-server app their panel mounts (sweeps_app / reward_app / …).
  viz_app?: string;
  // Provenance notes carry the AGE node id their ego-graph centers on (null = seed anchor).
  ego_focal?: number | null;
  // Lens notes may opt out of the live chord embed (trunk lenses — the chords are release-era).
  chord?: boolean;
  // Relational-table notes carry their precomputed first-order ERD (rendered by RelationalErd).
  erd?: import("./RelationalErd").ErdData;
}

// Flag accent per Data Product — the FedWiki "flag" reinterpreted as a lens color.
const ACCENT: Record<string, string> = {
  ontology: "#6366f1",
  relational: "#13a884",
  content: "#e08e0b",
  lens: "#8a5cf6",
};

// Inline tokens: [[id|label]] (lineup edge) · **bold** · `code`
function renderInline(text: string, onLink: (id: string) => void, kp: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\[\[([^\]|]+)(?:\|([^\]]*))?\]\]|\*\*([^*]+)\*\*|`([^`]+)`/g;
  let last = 0;
  let i = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) {
      const id = m[1];
      const label = m[2] ?? id.split("/").pop() ?? id;
      out.push(
        <a
          key={`${kp}-${i}`}
          onClick={(e) => { e.preventDefault(); onLink(id); }}
          style={{ color: "var(--text-color-kumo-link)", cursor: "pointer", fontWeight: 500 }}
          title={id}
        >
          {label}
        </a>,
      );
    } else if (m[3] !== undefined) {
      out.push(<strong key={`${kp}-${i}`}>{m[3]}</strong>);
    } else if (m[4] !== undefined) {
      out.push(
        <code key={`${kp}-${i}`} style={{ background: "var(--color-kumo-recessed)", padding: "1px 4px", borderRadius: 3, fontSize: 12 }}>
          {m[4]}
        </code>,
      );
    }
    last = re.lastIndex;
    i++;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// Split a markdown table row on `|`, but NOT inside a [[id|label]] wikilink (whose label
// separator is also `|`). Without this, pivot cells with labeled edges get shredded.
function cells(line: string): string[] {
  const inner = line.replace(/^\||\|$/g, "");
  const out: string[] = [];
  let buf = "";
  let depth = 0;
  for (let i = 0; i < inner.length; i++) {
    if (inner[i] === "[" && inner[i + 1] === "[") { depth++; buf += "[["; i++; continue; }
    if (inner[i] === "]" && inner[i + 1] === "]") { depth = Math.max(0, depth - 1); buf += "]]"; i++; continue; }
    if (inner[i] === "|" && depth === 0) { out.push(buf.trim()); buf = ""; continue; }
    buf += inner[i];
  }
  out.push(buf.trim());
  return out;
}

function renderBody(body: string, onLink: (id: string) => void): ReactNode[] {
  const lines = body.split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const ln = lines[i];
    if (ln.trim().startsWith("|")) {
      const rows: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) { rows.push(lines[i]); i++; }
      const header = cells(rows[0]);
      const data = rows.slice(1).filter((r) => !/^\s*\|[\s:|-]+\|\s*$/.test(r)).map(cells);
      blocks.push(
        <table key={`t-${i}`} style={{ borderCollapse: "collapse", fontSize: 12, margin: "6px 0", width: "100%" }}>
          <thead>
            <tr>{header.map((h, j) => (
              <th key={j} style={{ textAlign: "left", borderBottom: "1px solid var(--color-kumo-line)", padding: "2px 6px", color: "var(--text-color-kumo-subtle)" }}>{h}</th>
            ))}</tr>
          </thead>
          <tbody>
            {data.map((r, ri) => (
              <tr key={ri}>{r.map((c, cj) => (
                <td key={cj} style={{ borderBottom: "1px solid var(--color-kumo-hairline)", padding: "2px 6px" }}>{renderInline(c, onLink, `td-${i}-${ri}-${cj}`)}</td>
              ))}</tr>
            ))}
          </tbody>
        </table>,
      );
      continue;
    }
    if (ln.trim().startsWith("- ")) {
      const items: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("- ")) { items.push(lines[i].trim().slice(2)); i++; }
      blocks.push(
        <ul key={`u-${i}`} style={{ margin: "4px 0", paddingLeft: 18 }}>
          {items.map((it, j) => <li key={j} style={{ margin: "2px 0" }}>{renderInline(it, onLink, `li-${i}-${j}`)}</li>)}
        </ul>,
      );
      continue;
    }
    if (ln.trim().startsWith("```")) {
      const lang = ln.trim().slice(3).trim().toLowerCase();
      const buf: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) { buf.push(lines[i]); i++; }
      if (i < lines.length) i++; // consume closing fence
      const raw = buf.join("\n");
      let obj: any = null;
      if (lang === "json" || raw.trim().startsWith("{") || raw.trim().startsWith("[")) {
        try { obj = JSON.parse(raw); } catch { obj = null; }
      }
      if (obj && Array.isArray(obj.tables)) {
        blocks.push(
          <div key={`j-${i}`} style={{ margin: "8px 0" }}>
            {obj.tables.map((t: any, ti: number) => (
              <div key={ti} style={{ margin: "8px 0" }}>
                <div style={{ fontSize: 11, color: "var(--text-color-kumo-subtle)", fontFamily: "monospace", marginBottom: 2 }}>
                  {String(t.name ?? `table ${ti + 1}`)} · {(t.rows?.length ?? 0)} rows
                </div>
                <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
                  {Array.isArray(t.columns) && t.columns.length > 0 && (
                    <thead><tr>{t.columns.map((c: any, cj: number) => (
                      <th key={cj} style={{ textAlign: "left", borderBottom: "1px solid var(--color-kumo-line)", padding: "2px 6px", color: "var(--text-color-kumo-subtle)" }}>{String(c)}</th>
                    ))}</tr></thead>
                  )}
                  <tbody>
                    {(t.rows || []).slice(0, 100).map((r: any, ri: number) => (
                      <tr key={ri}>{(Array.isArray(r) ? r : [r]).map((c: any, cj: number) => (
                        <td key={cj} style={{ borderBottom: "1px solid var(--color-kumo-hairline)", padding: "2px 6px" }}>{String(c)}</td>
                      ))}</tr>
                    ))}
                  </tbody>
                </table>
                {(t.rows?.length ?? 0) > 100 && (
                  <div style={{ fontSize: 11, color: "var(--text-color-kumo-inactive)" }}>…{t.rows.length - 100} more rows</div>
                )}
              </div>
            ))}
          </div>,
        );
        continue;
      }
      blocks.push(
        <pre key={`c-${i}`} style={{ background: "var(--color-kumo-recessed)", padding: 8, borderRadius: 4, fontSize: 11.5, overflowX: "auto", margin: "6px 0" }}>
          {obj ? JSON.stringify(obj, null, 2) : raw}
        </pre>,
      );
      continue;
    }
    if (ln.trim() === "") { i++; continue; }
    blocks.push(<p key={`p-${i}`} style={{ margin: "6px 0", lineHeight: 1.5 }}>{renderInline(ln, onLink, `p-${i}`)}</p>);
    i++;
  }
  return blocks;
}

interface Props {
  note: KBNote | null;
  loading?: boolean;
  onLink: (id: string) => void;
  onClose: () => void;
}

function LineupPanel({ note, loading, onLink, onClose }: Props) {
  const accent = ACCENT[note?.data_product ?? "lens"] ?? "#999";
  // Training panels carry wide viz (the sweeps PCP) — give them room so all axes show without scroll.
  const basis = note?.kind === "training" ? "52rem" : "33rem";

  // A viz panel's nodes can be tapped to open a note (e.g. the Provenance DAG → its data-product lens).
  // The bokeh app dispatches `aegir:open-note` on the shared window (no iframe); we route the matching
  // app's event through this panel's own `onLink` — the same trail navigation as a lens link (opens the
  // target as a narrow panel just to the right of this one).
  const vizApp = note?.viz_app;
  useEffect(() => {
    if (!vizApp) return;
    const handler = (e: Event) => {
      const d = (e as CustomEvent).detail;
      if (d?.app === vizApp && d?.id) onLink(d.id);
    };
    window.addEventListener("aegir:open-note", handler);
    return () => window.removeEventListener("aegir:open-note", handler);
  }, [vizApp, onLink]);
  return (
    <div
      style={{
        flex: `0 0 ${basis}`, maxWidth: basis, height: "100%", overflowY: "auto",
        background: "var(--color-kumo-base)", border: "1px solid var(--color-kumo-hairline)", borderTop: `3px solid ${accent}`,
        borderRadius: 4, boxShadow: "0 1px 4px rgba(0,0,0,0.35)", marginRight: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, padding: "10px 16px 6px", borderBottom: "1px solid var(--color-kumo-hairline)" }}>
        <Text strong style={{ fontSize: 16 }}>{note?.title ?? (loading ? "…" : "not found")}</Text>
        {note && <Tag color={accent} style={{ marginInlineStart: "auto" }}>{note.kind}</Tag>}
        <CloseOutlined onClick={onClose} style={{ cursor: "pointer", color: "var(--text-color-kumo-inactive)", fontSize: 12 }} />
      </div>
      <div style={{ padding: "8px 16px 16px", fontSize: 13.5, color: "var(--text-color-kumo-default)" }}>
        {/* live embeds render for current/scratch only — archive kastens are frozen */}
        {note?.kind === "lens" && note.chord !== false && note.root !== "archive" && (
          <div style={{ margin: "0 -4px 10px", borderBottom: "1px solid var(--color-kumo-hairline)", paddingBottom: 6 }}>
            <PanelView app="lineup_app" params={{ lens: note.name ?? note.id, root: note.root ?? "current" }} />
          </div>
        )}
        {note?.kind === "training" && note.viz_app && note.root !== "archive" && (
          <div style={{ margin: "0 -4px 10px", borderBottom: "1px solid var(--color-kumo-hairline)", paddingBottom: 6 }}>
            <PanelView app={note.viz_app} height={460} />
          </div>
        )}
        {note?.kind === "provenance" && (
          <Suspense fallback={<Text type="secondary" style={{ fontSize: 12 }}>loading graph…</Text>}>
            <ProvenanceGraph focal={note.ego_focal ?? null} onLink={onLink} />
          </Suspense>
        )}
        {note?.kind === "relational-table" && note.erd && (
          <Suspense fallback={<Text type="secondary" style={{ fontSize: 12 }}>loading ERD…</Text>}>
            <RelationalErd erd={note.erd} onLink={onLink} />
          </Suspense>
        )}
        {note ? renderBody(note.body, onLink)
          : <Text type="secondary">{loading ? "loading…" : "This note is not in the current projection."}</Text>}
      </div>
    </div>
  );
}

export default LineupPanel;
