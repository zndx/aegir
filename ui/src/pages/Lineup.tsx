import { Select, Typography } from "antd";
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
  { key: "terms", label: "Terms", seed: "lens/terms", hint: "the lexicon" },
  { key: "schema", label: "Schema", seed: "lens/schema", hint: "relational projection" },
  { key: "content", label: "Content", seed: "lens/content", hint: "corpus + topics" },
];
// Training/experiment entries (live HoloViews panels). Extensible as requirements materialize —
// see docs/current/src/roadmap/leaderboard_observatory.md.
const TRAINING = [
  { key: "sweeps", label: "Sweeps", seed: "training/sweeps", hint: "parallel coords" },
  { key: "reward", label: "Reward", seed: "training/reward", hint: "GRPO dynamics" },
  { key: "provenance", label: "Provenance", seed: "training/provenance", hint: "lineage DAG" },
  { key: "metrics", label: "Metrics", seed: "training/metrics", hint: "gates & measures" },
];
const ROOTS = ["archive", "current", "scratch"];

function tab(active: boolean): React.CSSProperties {
  return {
    display: "block", padding: "5px 10px", borderRadius: 4, cursor: "pointer", fontSize: 13,
    color: active ? "#fff" : "#333", background: active ? "#4f7cff" : "transparent",
    textDecoration: "none",
  };
}

// Persist the panel trail per-seed in sessionStorage so it survives a re-render/remount (e.g. on window
// blur→focus / alt-tab). Without this, `trail` re-initializes to [seed] and the user loses their place.
const trailKey = (s: string) => `lineup:trail:${s}`;
function loadTrail(seed: string): string[] {
  try {
    const v = sessionStorage.getItem(trailKey(seed));
    if (v) { const a = JSON.parse(v); if (Array.isArray(a) && a.length) return a as string[]; }
  } catch { /* sessionStorage unavailable */ }
  return [seed];
}
function saveTrail(seed: string, trail: string[]): void {
  try { sessionStorage.setItem(trailKey(seed), JSON.stringify(trail)); } catch { /* ignore */ }
}

function Lineup() {
  const [params] = useSearchParams();
  const lensKey = params.get("lens") || "terms";
  // ?open=<note id> deep-links a specific note (e.g. training/sweeps from the Landing card); else the lens.
  const seed = params.get("open") || LENSES.find((l) => l.key === lensKey)?.seed || "lens/terms";

  const [root, setRoot] = useState("current");
  const [index, setIndex] = useState<KBIndex | null>(null);
  const [trail, setTrail] = useState<string[]>(() => loadTrail(seed));
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
  // Restore (not reset) the trail when the seed changes or the component remounts — survives alt-tab.
  useEffect(() => { setTrail(loadTrail(seed)); }, [seed]);
  useEffect(() => { saveTrail(seed, trail); }, [trail, seed]);

  const openFrom = (fromIdx: number) => (targetId: string) =>
    setTrail((t) => [...t.slice(0, fromIdx + 1), targetId]);

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      <div style={{ flex: "0 0 210px", borderRight: "1px solid #e5e5e5", padding: "14px 12px", overflowY: "auto", background: "#fafafb" }}>
        <Text strong style={{ fontSize: 11, color: "#999", letterSpacing: 0.5 }}>SECTION</Text>
        <Select
          value={root}
          size="small"
          style={{ width: "100%", margin: "6px 0 18px" }}
          onChange={(v) => { setRoot(v); setTrail(v === "current" ? [seed] : []); }}
          options={ROOTS.map((r) => ({ value: r, label: r.charAt(0).toUpperCase() + r.slice(1) }))}
        />
        {root === "current" ? (
          <>
            <Text strong style={{ fontSize: 11, color: "#999", letterSpacing: 0.5 }}>LENS</Text>
            <div style={{ display: "flex", flexDirection: "column", gap: 2, margin: "8px 0 8px" }}>
              {LENSES.map((l) => (
                <a key={l.key} onClick={() => setTrail([l.seed])} style={tab(trail[0] === l.seed)} title={l.hint}>
                  {l.label}
                  <span style={{ float: "right", fontSize: 11, opacity: 0.6 }}>{l.hint}</span>
                </a>
              ))}
            </div>
            <Text type="secondary" style={{ fontSize: 11, display: "block", marginTop: 4 }}>
              collections × lens — the live projection (what we know)
            </Text>
            <Text strong style={{ fontSize: 11, color: "#999", letterSpacing: 0.5, display: "block", marginTop: 16 }}>TRAINING</Text>
            <div style={{ display: "flex", flexDirection: "column", gap: 2, margin: "8px 0 8px" }}>
              {TRAINING.map((t) => (
                <a key={t.key} onClick={() => setTrail([t.seed])} style={tab(trail[0] === t.seed)} title={t.hint}>
                  {t.label}
                  <span style={{ float: "right", fontSize: 11, opacity: 0.6 }}>{t.hint}</span>
                </a>
              ))}
            </div>
          </>
        ) : (
          <>
            <Text strong style={{ fontSize: 11, color: "#999", letterSpacing: 0.5 }}>{root.toUpperCase()}</Text>
            <div style={{ display: "flex", flexDirection: "column", gap: 2, margin: "8px 0 8px" }}>
              {(() => {
                // Archive: show entry points (snapshot registry + authored notes), NOT the
                // thousands of namespaced notes inside a frozen snapshot (reachable by drilling in).
                const entries = (index?.notes || []).filter(
                  (n) => n.root === root &&
                    (root !== "archive" || n.kind === "archive-snapshot" || n.kind.endsWith("-note")));
                return entries.length ? entries.map((n) => (
                  <a key={n.id} onClick={() => setTrail([n.id])} style={tab(trail[0] === n.id)} title={n.id}>{n.title}</a>
                )) : <Text type="secondary" style={{ fontSize: 12 }}>empty — populates via the lifecycle</Text>;
              })()}
            </div>
          </>
        )}
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
