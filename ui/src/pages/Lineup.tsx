import { Segmented, Typography } from "antd";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import LineupPanel from "../components/LineupPanel";
import type { KBNote } from "../components/LineupPanel";

const { Text } = Typography;

interface IndexEntry {
  id: string; title: string; kind: string; data_product: string; root: string; relpath: string; links: string[];
}
interface KBIndex { counts: { total: number; by_data_product: Record<string, number> }; notes: IndexEntry[]; }

const LENSES = [
  { key: "terms", label: "Terms", seed: "lens/terms", hint: "ontology vocabulary" },
  { key: "schema", label: "Schema", seed: "lens/schema", hint: "relational projection" },
  { key: "content", label: "Content", seed: "lens/content", hint: "corpus + topics" },
];
const ROOTS = ["current", "scratch", "archive"];

function tab(active: boolean): React.CSSProperties {
  return {
    display: "block", padding: "5px 10px", borderRadius: 4, cursor: "pointer", fontSize: 13,
    color: active ? "#fff" : "#333", background: active ? "#4f7cff" : "transparent",
    textDecoration: "none",
  };
}

function Lineup() {
  const [params] = useSearchParams();
  const lensKey = params.get("lens") || "terms";
  const seed = LENSES.find((l) => l.key === lensKey)?.seed || "lens/terms";

  const [root, setRoot] = useState("current");
  const [index, setIndex] = useState<KBIndex | null>(null);
  const [trail, setTrail] = useState<string[]>([seed]);
  const [cache, setCache] = useState<Record<string, KBNote | null>>({});
  const requested = useRef<Set<string>>(new Set());

  useEffect(() => {
    fetch("/api/kb/index").then((r) => (r.ok ? r.json() : null)).then(setIndex).catch(() => setIndex(null));
  }, []);

  const fetchNote = useCallback((id: string) => {
    if (requested.current.has(id)) return;
    requested.current.add(id);
    fetch(`/api/kb/note/${id}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((n: KBNote | null) => setCache((c) => ({ ...c, [id]: n })))
      .catch(() => setCache((c) => ({ ...c, [id]: null })));
  }, []);

  useEffect(() => { trail.forEach((id) => { if (!(id in cache)) fetchNote(id); }); }, [trail, cache, fetchNote]);
  useEffect(() => { setTrail([seed]); }, [seed]);

  const openFrom = (fromIdx: number) => (targetId: string) =>
    setTrail((t) => [...t.slice(0, fromIdx + 1), targetId]);

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      <div style={{ flex: "0 0 210px", borderRight: "1px solid #e5e5e5", padding: "14px 12px", overflowY: "auto", background: "#fafafb" }}>
        <Text strong style={{ fontSize: 11, color: "#999", letterSpacing: 0.5 }}>LENS</Text>
        <div style={{ display: "flex", flexDirection: "column", gap: 2, margin: "8px 0 18px" }}>
          {LENSES.map((l) => (
            <a key={l.key} onClick={() => setTrail([l.seed])} style={tab(trail[0] === l.seed)} title={l.hint}>
              {l.label}
              <span style={{ float: "right", fontSize: 11, opacity: 0.6 }}>{l.hint}</span>
            </a>
          ))}
        </div>
        <Text strong style={{ fontSize: 11, color: "#999", letterSpacing: 0.5 }}>MATURITY</Text>
        <Segmented size="small" block options={ROOTS} value={root} onChange={(v) => setRoot(v as string)} style={{ marginTop: 8 }} />
        <Text type="secondary" style={{ fontSize: 11, display: "block", marginTop: 8 }}>
          {root === "current" ? "the live projection (what we know)" : `${root} populates via the lifecycle`}
        </Text>
        {index && (
          <div style={{ marginTop: 18, fontSize: 11, color: "#aaa" }}>
            {index.counts.total} notes ·{" "}
            {Object.entries(index.counts.by_data_product).map(([k, v]) => `${v} ${k}`).join(" · ")}
          </div>
        )}
      </div>

      <div style={{ flex: 1, display: "flex", overflowX: "auto", padding: 14, background: "#ececef" }}>
        {trail.map((id, i) => (
          <LineupPanel
            key={`${id}-${i}`}
            note={cache[id] ?? null}
            loading={!(id in cache)}
            onLink={openFrom(i)}
            onClose={() => setTrail((t) => (t.length > 1 ? t.slice(0, Math.max(1, i)) : t))}
          />
        ))}
      </div>
    </div>
  );
}

export default Lineup;
