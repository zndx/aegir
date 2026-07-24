import { CaretDoubleLeft, CaretDoubleRight } from "@phosphor-icons/react";
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
  { key: "terms", label: "Lexicon", seed: "lens/terms", hint: "aperture · terms · topics" },
  { key: "schema", label: "Schema", seed: "lens/schema", hint: "relational projection" },
  { key: "content", label: "Content", seed: "lens/content", hint: "corpus + topics" },
];
// Training/experiment entries (live HoloViews panels). Extensible as requirements materialize —
// see docs/current/src/roadmap/leaderboard_observatory.md.
const TRAINING = [
  { key: "sweeps", label: "Sweeps", seed: "training/sweeps", hint: "parallel coords" },
  { key: "reward", label: "Reward", seed: "training/reward", hint: "GRPO dynamics" },
  { key: "metrics", label: "Metrics", seed: "training/metrics", hint: "gates & measures" },
];
// PROVENANCE — sibling of LENS/TRAINING (RH 2026-07-21): Lineage = the internal DAG (Atlas/OL
// ego-graph, formerly the 'Provenance' training entry); Sources = the EPISTEMIC lineage —
// every external source directly referenced, in scope via SDG.
const PROVENANCE = [
  { key: "lineage", label: "Lineage", seed: "training/provenance", hint: "internal DAG" },
  { key: "reasoning", label: "Reasoning", seed: "provenance/reasoning", hint: "retained thinking traces" },
  { key: "sources", label: "Sources", seed: "provenance/sources", hint: "external inputs" },
];
// Roots are refs (RH 2026-07-06, git-style): current = the latest sdg-corpora release kasten ·
// scratch = TRUNK (the live projection — incremental work lands here) · archive = past releases
// + snapshots + tombstones. The SAME sidebar layout serves every root; entries appear where the
// root's kasten has them, and note fetches resolve ids within the active root.
const ROOTS = ["archive", "current", "scratch"];

function tab(active: boolean): React.CSSProperties {
  return {
    display: "block", padding: "5px 10px", borderRadius: 4, cursor: "pointer", fontSize: 13,
    color: active ? "#fff" : "var(--text-color-kumo-default)", background: active ? "var(--color-kumo-brand)" : "transparent",
    textDecoration: "none",
  };
}

// Persist the panel trail per layer+seed in sessionStorage so it survives a re-render/remount
// (e.g. on window blur→focus / alt-tab) and per-layer trails don't bleed into each other.
// The layer is the root, plus the selected archived kasten when browsing archive.
// One ACTIVE trail per layer (v2 — RH 2026-07-21): the URL tracks the FOCUSED panel, so a
// seed-keyed store broke on remount (alt-tab): the remount re-seeded from the last panel and
// restored nothing. The layer key restores the whole path regardless of focus.
const trailKey = (layer: string) => `lineup:trail2:${layer}`;
function loadTrail(layer: string): string[] | null {
  try {
    const v = sessionStorage.getItem(trailKey(layer));
    if (v) { const a = JSON.parse(v); if (Array.isArray(a) && a.length) return a as string[]; }
  } catch { /* sessionStorage unavailable */ }
  return null;
}
function saveTrail(layer: string, trail: string[]): void {
  try { if (trail.length) sessionStorage.setItem(trailKey(layer), JSON.stringify(trail)); } catch { /* ignore */ }
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <>
      <Text strong style={{ fontSize: 11, color: "var(--text-color-kumo-inactive)", letterSpacing: 0.5, display: "block", marginTop: 16 }}>
        {title}
      </Text>
      <div style={{ display: "flex", flexDirection: "column", gap: 2, margin: "8px 0 8px" }}>{children}</div>
    </>
  );
}

function Lineup() {
  const [params, setParams] = useSearchParams();
  const lensKey = params.get("lens") || "terms";
  // ?open=<note id> deep-links a specific note; captured ONCE at mount — the URL-sync effect
  // below WRITES ?open as you browse, and a reactive seed would feed back into the trail-restore
  // effect and reset the trail to a single panel on every click (the direct-linking regression).
  const [initialOpen] = useState(() => params.get("open"));
  const seed = initialOpen || LENSES.find((l) => l.key === lensKey)?.seed || "lens/terms";

  const [root, setRoot] = useState(ROOTS.includes(params.get("root") || "") ? params.get("root")! : "current");
  const [index, setIndex] = useState<KBIndex | null | undefined>(undefined);
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

  // A single failed index fetch must not latch an empty nav (e.g. a rebuild mid-write):
  // retry with backoff until it lands.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = (attempt: number) => {
      fetch("/api/kb/index")
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((j) => { if (!cancelled) setIndex(j); })
        .catch(() => {
          if (cancelled) return;
          if (attempt < 5) timer = setTimeout(() => load(attempt + 1), 1500 * (attempt + 1));
          else setIndex(null);              // exhausted — surface 'unreachable', never fake 'empty'
        });
    };
    load(0);
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
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
  const restoredFor = useRef<string | null>(null);
  useEffect(() => {
    if (restoredFor.current === layer) return;     // once per layer — later seed churn never clobbers
    restoredFor.current = layer;
    const saved = loadTrail(layer);
    if (saved && (!initialOpen || saved.includes(initialOpen))) {
      prevLen.current = saved.length;
      setTrail(saved);                             // the FULL path survives remounts (alt-tab)
      return;
    }
    const inRoot = (id: string) => (index?.notes || []).some((n) => n.root === root && n.id === id);
    if (initialOpen) { setTrail([initialOpen]); return; }   // genuinely new deep link
    if (inRoot(prefix + seed)) { setTrail([prefix + seed]); return; }
    if (inRoot(prefix + "lens/terms")) { setTrail([prefix + "lens/terms"]); return; }
    const first = (index?.notes || []).find((n) => n.root === root &&
      (n.kind === "lens" || n.kind === "release-note" || n.kind === "corpus"));
    setTrail(first ? [first.id] : []);
  }, [seed, root, index, layer, prefix, initialOpen]);
  useEffect(() => { saveTrail(layer, trail); }, [trail, layer]);
  // shareable URLs with FEDWIKI HISTORY SEMANTICS (RH 2026-07-21): extending the lineup
  // PUSHES history (back walks the path leftward, as Ward's lineup does); truncation/replacement
  // REPLACES. lastNav distinguishes our own writes from inbound navigation.
  const lastNav = useRef<string | null>(null);
  const prevLen = useRef(0);
  useEffect(() => {
    if (!trail.length) return;
    const focused = trail[trail.length - 1];
    const extend = trail.length > prevLen.current;
    prevLen.current = trail.length;
    lastNav.current = focused;
    setParams((prev) => {
      const q = new URLSearchParams(prev);
      q.set("open", focused);
      q.set("root", root);
      return q.toString() === prev.toString() ? prev : q;
    }, { replace: !extend });
  }, [trail, root, setParams]);
  // inbound navigation (back/forward buttons, a pasted/edited URL in-place): reconcile the trail.
  // back to an id already in the trail = truncate right of it (the lineup walks back); a novel id
  // extends the current lineup (forward / hand-edited deep link).
  useEffect(() => {
    const pOpen = params.get("open");
    if (!pOpen || pOpen === lastNav.current) return;
    lastNav.current = pOpen;
    setTrail((t) => {
      const i = t.indexOf(pOpen);
      if (i >= 0) { prevLen.current = i + 1; return t.slice(0, i + 1); }
      prevLen.current = t.length + 1;
      return [...t, pOpen];
    });
  }, [params]);

  const openFrom = (fromIdx: number) => (targetId: string) =>
    setTrail((t) => [...t.slice(0, fromIdx + 1), targetId]);

  // the path-aware HermiT step: the verification result joins the lineup rightmost
  const [verifying, setVerifying] = useState(false);
  const verifyPath = () => {
    if (verifying) return;
    setVerifying(true);
    fetch("/api/kb/verify-path", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trail }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j: { note_id: string }) => { if (j.note_id) openFrom(trail.length - 1)(j.note_id); })
      .catch(() => { /* verdictless — the panel simply doesn't open */ })
      .finally(() => setVerifying(false));
  };
  const verifyBar = trail.length > 1 ? (
    <div style={{ position: "absolute", top: 6, right: 18, zIndex: 3, fontSize: 12 }}>
      <a onClick={verifyPath} style={{ color: "var(--text-color-kumo-link)", cursor: "pointer" }}>
        {verifying ? "verifying path…" : "⚖ verify path"}
      </a>
    </div>
  ) : null;

  // The SAME group structure for every root (LENS / TRAINING); entries resolve through the
  // layer's prefix (empty for current/scratch; the newest kasten key for archive). Corpus
  // surfaces are reached through the content/schema lenses, not a sidebar group.
  const rootNotes = (index?.notes || []).filter((n) => n.root === root);
  const lenses = LENSES.filter((l) => rootNotes.some((n) => n.id === prefix + l.seed));
  const training = TRAINING.filter((t) => rootNotes.some((n) => n.id === prefix + t.seed));
  const provenance = PROVENANCE.filter((t) => rootNotes.some((n) => n.id === prefix + t.seed));

  // UXR 2026-07-19: collapsible left-nav — the close control lives at the BOTTOM of the rail.
  const [navClosed, setNavClosed] = useState<boolean>(() => {
    try { return sessionStorage.getItem("lineup:nav-closed") === "1"; } catch { return false; }
  });
  const toggleNav = () => setNavClosed((c) => {
    try { sessionStorage.setItem("lineup:nav-closed", c ? "0" : "1"); } catch { /* ignore */ }
    return !c;
  });
  const navToggle = (closed: boolean) => (
    <a
      onClick={toggleNav}
      title={closed ? "open navigation" : "close navigation"}
      style={{ marginTop: "auto", paddingTop: 10, display: "flex", alignItems: "center", gap: 6,
               justifyContent: closed ? "center" : "flex-start", cursor: "pointer", fontSize: 12,
               color: "var(--text-color-kumo-inactive)" }}
    >
      {closed ? <CaretDoubleRight size={14} /> : <><CaretDoubleLeft size={14} /> close</>}
    </a>
  );

  if (navClosed) {
    return (
      <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
        <div style={{ flex: "0 0 30px", borderRight: "1px solid var(--color-kumo-hairline)", padding: "14px 4px", display: "flex", flexDirection: "column", background: "var(--color-kumo-base)" }}>
          {navToggle(true)}
        </div>
        <div style={{ flex: 1, position: "relative", display: "flex", overflowX: "auto", padding: 14, background: "var(--color-kumo-canvas)" }}>
          {verifyBar}
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

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      <div style={{ flex: "0 0 210px", borderRight: "1px solid var(--color-kumo-hairline)", padding: "14px 12px", overflowY: "auto", background: "var(--color-kumo-base)", display: "flex", flexDirection: "column" }}>
        <Text strong style={{ fontSize: 11, color: "var(--text-color-kumo-inactive)", letterSpacing: 0.5 }}>SECTION</Text>
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
        {provenance.length > 0 && (
          <Group title="PROVENANCE">
            {provenance.map((t) => (
              <a key={t.key} onClick={() => setTrail([prefix + t.seed])} style={tab(trail[0] === prefix + t.seed)} title={t.hint}>
                {t.label}
                <span style={{ float: "right", fontSize: 11, opacity: 0.6 }}>{t.hint}</span>
              </a>
            ))}
          </Group>
        )}
        {index === undefined && (
          <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 16 }}>
            loading index…
          </Text>
        )}
        {index === null && (
          <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 16 }}>
            index unreachable — gateway down or zero-trust session expired.{" "}
            <a onClick={() => window.location.reload()}>reload</a>
          </Text>
        )}
        {index && !lenses.length && !training.length && (
          <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 16 }}>
            empty — populates via the release lifecycle
          </Text>
        )}
        {index && (
          <div style={{ marginTop: 18, fontSize: 11, color: "var(--text-color-kumo-inactive)" }}>
            {index.counts.total} notes ·{" "}
            {Object.entries(index.counts.by_data_product).map(([k, v]) => `${v} ${k}`).join(" · ")}
          </div>
        )}
        {navToggle(false)}
      </div>

      <div style={{ flex: 1, position: "relative", display: "flex", overflowX: "auto", padding: 14, background: "var(--color-kumo-canvas)" }}>
        {verifyBar}
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
