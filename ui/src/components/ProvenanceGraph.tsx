import {
  Background, Controls, Handle, Position, ReactFlow, ReactFlowProvider,
} from "@xyflow/react";
import type { Edge, Node, NodeProps } from "@xyflow/react";
import { Component, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import "@xyflow/react/dist/style.css";

// The lineup Provenance panel: a first-order-neighbor EGO-GRAPH of one lineage node, read from
// /api/provenance/ego. Clicking a neighbor opens ITS ego-graph in a new panel (panel-trail) — the lineage is
// walked node-by-node, not shown as one static DAG. Built to match Atelier's working ReactFlow pattern:
// ReactFlowProvider + a memoized custom node type with Handles (NOT the default node with a JSX data.label).

interface Neighbor { vid: number; label: string; name: string; edge: string; out: boolean; }
interface Ego {
  focal: { vid: number; label: string; name: string } | null;
  neighbors: Neighbor[];
  truncated?: boolean; cap?: number; error?: string;
}

const LABEL_COLOR: Record<string, string> = {
  Family: "#8a5cf6", Topic: "#e08e0b", Template: "#4f7cff", Chapter: "#e08e0b",
  Column: "#13a884", Dataset: "#13a884", Run: "#d4663a", Job: "#d4663a",
};

interface ArtifactData { name: string; label: string; color: string; focal: boolean; }

function ArtifactNode({ data }: NodeProps) {
  const d = data as unknown as ArtifactData;
  return (
    <div style={{
      background: d.focal ? "var(--color-kumo-recessed)" : "var(--color-kumo-elevated)",
      color: d.focal ? "var(--text-color-kumo-strong)" : "var(--text-color-kumo-default)",
      border: `2px solid ${d.focal ? "var(--color-kumo-brand)" : d.color}`, borderRadius: 8, padding: "5px 8px",
      width: 150, textAlign: "center", fontSize: 11, cursor: d.focal ? "default" : "pointer",
    }}>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <div style={{ fontWeight: 600 }}>{d.name}</div>
      <div style={{ fontSize: 9, opacity: 0.6 }}>{d.label}</div>
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
    </div>
  );
}

// Module-level so the object identity is stable — ReactFlow errors on a fresh nodeTypes object each render.
const nodeTypes = { artifact: ArtifactNode };

function Graph({ ego, onLink }: { ego: Ego; onLink: (id: string) => void }) {
  const fid = String(ego.focal!.vid);
  const { nodes, edges } = useMemo<{ nodes: Node[]; edges: Edge[] }>(() => {
    const focal = ego.focal!;
    // dedupe neighbor NODES by vid (a node reached by >1 edge appears once); drop self-references
    const byVid = new Map<string, Neighbor>();
    for (const n of ego.neighbors) {
      const k = String(n.vid);
      if (k !== fid && !byVid.has(k)) byVid.set(k, n);
    }
    const uniq = [...byVid.values()];
    const place = (arr: Neighbor[], side: number) =>
      arr.map((n, i) => ({
        n, x: side * 340,
        y: ((arr.length === 1 ? 0.5 : i / (arr.length - 1)) - 0.5) * Math.max(140, arr.length * 76),
      }));
    const placed = [...place(uniq.filter((n) => n.out), 1), ...place(uniq.filter((n) => !n.out), -1)];
    const ns: Node[] = [
      { id: fid, type: "artifact", position: { x: 0, y: 0 },
        data: { name: focal.name, label: focal.label, color: "#4f7cff", focal: true } },
      ...placed.map(({ n, x, y }): Node => ({
        id: String(n.vid), type: "artifact", position: { x, y },
        data: { name: n.name, label: n.label, color: LABEL_COLOR[n.label] || "#bbb", focal: false } })),
    ];
    // one edge per relationship, unique id (incl. type), only between nodes that exist
    const seen = new Set<string>();
    const es: Edge[] = [];
    for (const n of ego.neighbors) {
      const nid = String(n.vid);
      if (nid === fid || !byVid.has(nid)) continue;
      const [source, target] = n.out ? [fid, nid] : [nid, fid];
      const id = `${source}->${target}:${n.edge}`;
      if (seen.has(id)) continue;
      seen.add(id);
      es.push({ id, source, target, label: n.edge, style: { stroke: "#3d3d4d" },
                labelStyle: { fontSize: 9, fill: "#9ca3af" }, labelBgStyle: { fill: "#1a1a25", fillOpacity: 0.9 } });
    }
    return { nodes: ns, edges: es };
  }, [ego, fid]);

  return (
    <ReactFlow
      nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.2 }}
      nodesDraggable={false} nodesConnectable={false} elementsSelectable
      onNodeClick={(_, node) => { if (node.id !== fid) onLink(`provenance/${node.id}`); }}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#2e2e3a" gap={16} />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

function Inner({ focal, onLink }: { focal: number | null; onLink: (id: string) => void }) {
  const [ego, setEgo] = useState<Ego | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let live = true;
    setLoading(true);
    const q = focal == null ? "" : `?focal=${focal}`;
    fetch(`/api/provenance/ego${q}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: Ego | null) => { if (live) { setEgo(d); setLoading(false); } })
      .catch(() => { if (live) { setEgo(null); setLoading(false); } });
    return () => { live = false; };
  }, [focal]);

  if (loading) return <div style={{ padding: 16, color: "var(--text-color-kumo-subtle)", fontSize: 12 }}>loading lineage…</div>;
  if (!ego || ego.error || !ego.focal)
    return (
      <div style={{ padding: 16, color: "var(--text-color-kumo-subtle)", fontSize: 12 }}>
        provenance graph unavailable{ego?.error ? ` (${ego.error})` : ""} — is aegir_hx up?
      </div>
    );

  return (
    <div style={{ margin: "0 -4px 8px" }}>
      <div style={{ height: 440, width: "100%", border: "1px solid var(--color-kumo-hairline)", borderRadius: 6, background: "var(--color-kumo-base)" }}>
        <ReactFlowProvider>
          <Graph ego={ego} onLink={onLink} />
        </ReactFlowProvider>
      </div>
      <div style={{ fontSize: 11, color: "var(--text-color-kumo-subtle)", textAlign: "center", padding: "3px 0" }}>
        <b>{ego.focal.name}</b> ({ego.focal.label}) · {ego.neighbors.length} neighbors
        {ego.truncated ? ` (capped at ${ego.cap}; more exist)` : ""} · click a node to walk the lineage →
      </div>
    </div>
  );
}

// A render error in the graph must never take down the whole lineup panel — contain it here.
class Boundary extends Component<{ children: ReactNode }, { err: string | null }> {
  state = { err: null as string | null };
  static getDerivedStateFromError(e: unknown) { return { err: e instanceof Error ? e.message : String(e) }; }
  render() {
    if (this.state.err)
      return <div style={{ padding: 16, color: "var(--text-color-kumo-danger)", fontSize: 12 }}>
        provenance graph failed to render: {this.state.err} (see the browser console)</div>;
    return this.props.children;
  }
}

export default function ProvenanceGraph(props: { focal: number | null; onLink: (id: string) => void }) {
  return <Boundary><Inner {...props} /></Boundary>;
}
