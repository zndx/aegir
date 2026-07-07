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
// Roots are refs (RH 2026-07-06, git-style): current = the latest sdg-corpora release kasten ·
// scratch = TRUNK (the live projection — incremental work lands here) · archive = past releases
// + snapshots + tombstones. The SAME sidebar layout serves every root; entries appear where the
// root's kasten has them, and note fetches resolve ids within the active root.
const ROOTS = ["archive", "current", "scratch"];

function tab(active: boolean): React.CSSProperties {
  return {
    display: "block", padding: "5px 10px", borderRadius: 4, cursor: "pointer", fontSize: 13,
    color: active ? "#fff" : "#333", background: active ? "#4f7cff" : "transparent",
    textDecoration: "none",
  };
}

// Persist the panel trail per layer+seed in sessionStorage so it survives a re-render/remount
// (e.g. on window blur→focus / alt-tab) and per-layer trails don't bleed into each other.
// The layer is the root, plus the selected archived kasten when browsing archive.
const trailKey = (layer: string, seed: string) => `lineup:trail:${layer}:${seed}`;
function loadTrail(layer: string, seed: string): string[] | null {
  try {
    const v = sessionStorage.getItem(trailKey(layer, seed));
    if (v) { const a = JSON.parse(v); if (Array.isArray(a) && a.length) return a as string[]; }
  } catch { /* sessionStorage unavailable */ }
  return null;
}
function saveTrail(layer: string, seed: string, trail: string[]): void {
  try { if (trail.length) sessionStorage.setItem(trailKey(layer, seed), JSON.stringify(trail)); } catch { /* ignore */ }
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <>
      <Text strong style={{ fontSize: 11, color: "#999", letterSpacing: 0.5, display: "block", marginTop: 16 }}>
        {title}
      </Text>
      <div style={{ display: "flex", flexDirection: "column", gap: 2, margin: "8px 0 8px" }}>{children}</div>
    </>
  );
}

function Lineup() {
  const [params] = useSearchParams();
  const lensKey = params.get("lens") || "terms";
  // ?open=<note id> deep-links a specific note (e.g. training/sweeps from the Landing card); else the lens.
  const seed = params.get("open") || LENSES.find((l) => l.key === lensKey)?.seed || "lens/terms";

  const [root, setRoot] = useState("current");
  const [index, setIndex] = useState<KBIndex | null>(null);
  const [trail, setTrail] = useState<string[]>([]);
  const [cache, setCache] = useState<Record<string, KBNote | null>>({});
  const requested = useRef<Set<string>>(new Set());
  const ck = useCallback((id: string) => `${root}:${id}`, [root]);

  // Archived kastens — the namespaced, self-contained freezes of past current/ projections
  // (kb-snapshot; release freezes land the same way). Each was "the lineup when it was
  // current", so it carries the same lens/corpus/training surfaces under its key prefix.
  // The archive side-nav browses the NEWEST kasten only (older freezes are retained on disk
  // and stay reachable by id, just not via the side-nav).
  const kastens = Array.from(new Set((index?.notes || [])
    .filter((n) => n.root === "archive" && n.id.endsWith("/lens/terms"))
    .map((n) => n.id.slice(0, -"/lens/terms".length)))).sort().reverse();
  const K = root === "archive" ? kastens[0] ?? null : null;
  const prefix = K ? `${K}/` : "";
  const layer = K ? `${root}:${K}` : root;

  useEffect(() => {
    fetch("/api/kb/index").then((r) => (r.ok ? r.json() : null)).then(setIndex).catch(() => setIndex(null));
  }, []);

  const fetchNote = useCallback((id: string) => {
    const k = `${root}:${id}`;
    if (requested.current.has(k)) return;
    requested.current.add(k);
    fetch(`/api/kb/note/${id}?root=${root}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((n: KBNote | null) => setCache((c) => ({ ...c, [k]: n })))
      .catch(() => setCache((c) => ({ ...c, [k]: null })));
  }, [root]);

  useEffect(() => { trail.forEach((id) => { if (!(ck(id) in cache)) fetchNote(id); }); }, [trail, cache, fetchNote, ck]);
  // Restore (not reset) the trail per layer+seed; else start at the (kasten-prefixed) seed if
  // the layer has it, else the layer's first natural trailhead (lens → release → snapshot).
  useEffect(() => {
    const saved = loadTrail(layer, seed);
    if (saved) { setTrail(saved); return; }
    const inRoot = (id: string) => (index?.notes || []).some((n) => n.root === root && n.id === id);
    if (inRoot(prefix + seed)) { setTrail([prefix + seed]); return; }
    if (inRoot(prefix + "lens/terms")) { setTrail([prefix + "lens/terms"]); return; }
    const first = (index?.notes || []).find((n) => n.root === root &&
      (n.kind === "lens" || n.kind === "release-note" || n.kind === "corpus"));
    setTrail(first ? [first.id] : []);
  }, [seed, root, index, layer, prefix]);
  useEffect(() => { saveTrail(layer, seed, trail); }, [trail, seed, layer]);

  const openFrom = (fromIdx: number) => (targetId: string) =>
    setTrail((t) => [...t.slice(0, fromIdx + 1), targetId]);

  // The SAME group structure for every root (LENS / TRAINING); entries resolve through the
  // layer's prefix (empty for current/scratch; the newest kasten key for archive). Corpus
  // surfaces are reached through the content/schema lenses, not a sidebar group.
  const rootNotes = (index?.notes || []).filter((n) => n.root === root);
  const lenses = LENSES.filter((l) => rootNotes.some((n) => n.id === prefix + l.seed));
  const training = TRAINING.filter((t) => rootNotes.some((n) => n.id === prefix + t.seed));

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      <div style={{ flex: "0 0 210px", borderRight: "1px solid #e5e5e5", padding: "14px 12px", overflowY: "auto", background: "#fafafb" }}>
        <Text strong style={{ fontSize: 11, color: "#999", letterSpacing: 0.5 }}>SECTION</Text>
        <Select
          value={root}
          size="small"
          style={{ width: "100%", margin: "6px 0 2px" }}
          onChange={setRoot}
          options={ROOTS.map((r) => ({ value: r, label: r.charAt(0).toUpperCase() + r.slice(1) }))}
        />
        {lenses.length > 0 && (
          <Group title="LENS">
            {lenses.map((l) => (
              <a key={l.key} onClick={() => setTrail([prefix + l.seed])} style={tab(trail[0] === prefix + l.seed)} title={l.hint}>
                {l.label}
                <span style={{ float: "right", fontSize: 11, opacity: 0.6 }}>{l.hint}</span>
              </a>
            ))}
          </Group>
        )}
        {training.length > 0 && (
          <Group title="TRAINING">
            {training.map((t) => (
              <a key={t.key} onClick={() => setTrail([prefix + t.seed])} style={tab(trail[0] === prefix + t.seed)} title={t.hint}>
                {t.label}
                <span style={{ float: "right", fontSize: 11, opacity: 0.6 }}>{t.hint}</span>
              </a>
            ))}
          </Group>
        )}
        {!lenses.length && !training.length && (
          <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 16 }}>
            empty — populates via the release lifecycle
          </Text>
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
            key={`${root}-${id}-${i}`}
            note={cache[ck(id)] ?? null}
            loading={!(ck(id) in cache)}
            onLink={openFrom(i)}
            onClose={() => setTrail((t) => (t.length > 1 ? t.slice(0, Math.max(1, i)) : t))}
          />
        ))}
      </div>
    </div>
  );
}

export default Lineup;
