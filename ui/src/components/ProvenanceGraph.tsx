import { ReactFlow, Background, Controls } from "@xyflow/react";
import type { Node, Edge } from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";

// The lineup Provenance panel: a first-order-neighbor EGO-GRAPH of one lineage node, read from
// /api/provenance/ego. Clicking a neighbor opens ITS ego-graph in a new panel (panel-trail) — the lineage
// is walked node-by-node, in true lineup fashion, rather than shown as one static DAG.

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

function nodeLabel(name: string, label: string) {
  return (
    <div style={{ lineHeight: 1.2 }}>
      <div style={{ fontWeight: 600, fontSize: 11 }}>{name}</div>
      <div style={{ fontSize: 9, opacity: 0.55 }}>{label}</div>
    </div>
  );
}

function ProvenanceGraph({ focal, onLink }: { focal: number | null; onLink: (id: string) => void }) {
  const [ego, setEgo] = useState<Ego | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const q = focal == null ? "" : `?focal=${focal}`;
    fetch(`/api/provenance/ego${q}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: Ego | null) => { setEgo(d); setLoading(false); })
      .catch(() => { setEgo(null); setLoading(false); });
  }, [focal]);

  const { nodes, edges } = useMemo<{ nodes: Node[]; edges: Edge[] }>(() => {
    if (!ego?.focal) return { nodes: [], edges: [] };
    const fid = String(ego.focal.vid);
    // outgoing (this node → derived) to the right, incoming (sources → this node) to the left
    const place = (arr: Neighbor[], side: number) =>
      arr.map((n, i) => ({
        n, x: side * 340,
        y: ((arr.length === 1 ? 0.5 : i / (arr.length - 1)) - 0.5) * Math.max(140, arr.length * 76),
      }));
    const placed = [
      ...place(ego.neighbors.filter((n) => n.out), 1),
      ...place(ego.neighbors.filter((n) => !n.out), -1),
    ];
    const ns: Node[] = [
      {
        id: fid, position: { x: 0, y: 0 }, data: { label: nodeLabel(ego.focal.name, ego.focal.label) },
        style: { background: "#222", color: "#fff", border: "2px solid #4f7cff", borderRadius: 8,
                 padding: 6, width: 150, textAlign: "center" },
      },
      ...placed.map(({ n, x, y }): Node => ({
        id: String(n.vid), position: { x, y }, data: { label: nodeLabel(n.name, n.label) },
        style: { background: "#fff", color: "#333", border: `2px solid ${LABEL_COLOR[n.label] || "#bbb"}`,
                 borderRadius: 8, padding: 6, width: 150, textAlign: "center", cursor: "pointer" },
      })),
    ];
    const es: Edge[] = ego.neighbors.map((n): Edge => {
      const nid = String(n.vid);
      const [source, target] = n.out ? [fid, nid] : [nid, fid];
      return { id: `${source}->${target}`, source, target, label: n.edge,
               style: { stroke: "#ccc" }, labelStyle: { fontSize: 9, fill: "#999" },
               labelBgStyle: { fill: "#fff", fillOpacity: 0.8 } };
    });
    return { nodes: ns, edges: es };
  }, [ego]);

  if (loading) return <div style={{ padding: 16, color: "#888", fontSize: 12 }}>loading lineage…</div>;
  if (!ego || ego.error || !ego.focal)
    return (
      <div style={{ padding: 16, color: "#888", fontSize: 12 }}>
        provenance graph unavailable{ego?.error ? ` (${ego.error})` : ""} — is aegir_hx up?
      </div>
    );

  return (
    <div style={{ margin: "0 -4px 8px" }}>
      <div style={{ height: 440, border: "1px solid #f0f0f0", borderRadius: 6, background: "#fcfcfd" }}>
        <ReactFlow
          nodes={nodes} edges={edges} fitView fitViewOptions={{ padding: 0.2 }}
          nodesDraggable={false} nodesConnectable={false} elementsSelectable
          onNodeClick={(_, node) => { if (ego.focal && node.id !== String(ego.focal.vid)) onLink(`provenance/${node.id}`); }}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#eee" gap={16} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <div style={{ fontSize: 11, color: "#888", textAlign: "center", padding: "3px 0" }}>
        <b>{ego.focal.name}</b> ({ego.focal.label}) · {ego.neighbors.length} neighbors
        {ego.truncated ? ` (capped at ${ego.cap}; more exist)` : ""} · click a node to walk the lineage →
      </div>
    </div>
  );
}

export default ProvenanceGraph;
