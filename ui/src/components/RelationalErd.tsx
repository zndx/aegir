import {
  Background, Handle, Position, ReactFlow, ReactFlowProvider,
} from "@xyflow/react";
import type { Edge, Node, NodeProps } from "@xyflow/react";
import { Component, useMemo } from "react";
import type { ReactNode } from "react";
import "@xyflow/react/dist/style.css";

// The relational-table panel's ERD header (RH 2026-07-18 UX pass): the focal table's FIRST-ORDER
// relational neighborhood — FK targets right, referencing tables left, views below, junctions tinted —
// precomputed by the KB projector into the note's `erd` frontmatter (no client recomputation). Tapping
// a table opens its note panel (the same walk grammar as the provenance ego-graph); views are terminal.
// Built on @xyflow/react like ProvenanceGraph — deliberately NOT a second graph lib (Kumo has none).

export interface ErdNode { id: string; kind: "table" | "junction" | "view"; focal?: boolean; cols?: string[]; known?: boolean; }
export interface ErdEdge { source: string; target: string; label?: string; }
export interface ErdData {
  focal: string;
  nodes: ErdNode[];
  edges: ErdEdge[];
  more?: { referenced_by?: number; views?: number };
}

const KIND_STYLE: Record<string, { border: string; bg: string; dash?: boolean }> = {
  table: { border: "#13a884", bg: "#fff" },
  junction: { border: "#8a5cf6", bg: "#faf8ff" },
  view: { border: "#b9bec7", bg: "#fafbfc", dash: true },
};

interface TableData { label: string; kind: string; focal: boolean; cols?: string[]; clickable: boolean; }

function TableNode({ data }: NodeProps) {
  const d = data as unknown as TableData;
  const s = KIND_STYLE[d.kind] ?? KIND_STYLE.table;
  return (
    <div style={{
      background: d.focal ? "#1f2430" : s.bg, color: d.focal ? "#fff" : "#333",
      border: `${s.dash ? "1.5px dashed" : "2px solid"} ${d.focal ? "#13a884" : s.border}`,
      borderRadius: 8, padding: "6px 9px", width: d.focal ? 216 : 168,
      fontSize: 11, cursor: d.clickable ? "pointer" : "default",
    }}>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <div style={{ fontWeight: 600, fontFamily: "monospace", fontSize: d.focal ? 12 : 11 }}>{d.label}</div>
      <div style={{ fontSize: 8.5, opacity: 0.55, marginBottom: d.focal && d.cols?.length ? 4 : 0 }}>{d.kind}</div>
      {d.focal && d.cols?.length ? (
        <div style={{
          borderTop: "1px solid rgba(255,255,255,0.25)", paddingTop: 3, fontFamily: "monospace",
          fontSize: 9, lineHeight: 1.5, textAlign: "left", opacity: 0.9,
        }}>
          {d.cols.map((c) => <div key={c}>{c}</div>)}
        </div>
      ) : null}
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
    </div>
  );
}

// Module-level: ReactFlow requires a stable nodeTypes object identity across renders.
const nodeTypes = { erdTable: TableNode };

function Graph({ erd, onLink }: { erd: ErdData; onLink: (id: string) => void }) {
  const { nodes, edges } = useMemo<{ nodes: Node[]; edges: Edge[] }>(() => {
    const focal = erd.focal;
    const incoming = erd.nodes.filter((n) => !n.focal && n.kind !== "view"
      && erd.edges.some((e) => e.source === n.id && e.target === focal));
    const outgoing = erd.nodes.filter((n) => !n.focal && n.kind !== "view" && !incoming.includes(n)
      && erd.edges.some((e) => e.source === focal && e.target === n.id));
    const viewNodes = erd.nodes.filter((n) => n.kind === "view");
    const col = (arr: ErdNode[], x: number, yPad = 0): Node[] =>
      arr.map((n, i) => ({
        id: n.id, type: "erdTable",
        position: { x, y: yPad + ((arr.length === 1 ? 0.5 : i / (arr.length - 1)) - 0.5) * Math.max(120, arr.length * 64) },
        data: {
          label: n.id, kind: n.kind, focal: false,
          clickable: n.kind !== "view" && n.known !== false,
        },
      }));
    const focalNode = erd.nodes.find((n) => n.focal);
    const ns: Node[] = [
      { id: focal, type: "erdTable", position: { x: 0, y: 0 },
        data: { label: focal, kind: focalNode?.kind ?? "table", focal: true, cols: focalNode?.cols, clickable: false } },
      ...col(incoming, -300), ...col(outgoing, 330),
      ...viewNodes.map((n, i): Node => ({
        id: n.id, type: "erdTable",
        position: { x: (i - (viewNodes.length - 1) / 2) * 200, y: 190 + (i % 2) * 46 },
        data: { label: n.id, kind: "view", focal: false, clickable: false },
      })),
    ];
    const seen = new Set(ns.map((n) => n.id));
    const es: Edge[] = erd.edges
      .filter((e) => seen.has(e.source) && seen.has(e.target))
      .map((e) => ({
        id: `${e.source}->${e.target}:${e.label ?? ""}`, source: e.source, target: e.target,
        label: e.label, animated: e.label === "view",
        style: { stroke: e.label === "view" ? "#d5d9e0" : "#c2c8d0" },
        labelStyle: { fontSize: 9, fill: "#8a94a3", fontFamily: "monospace" },
        labelBgStyle: { fill: "#fff", fillOpacity: 0.85 },
      }));
    return { nodes: ns, edges: es };
  }, [erd]);

  return (
    <ReactFlow
      nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.15 }}
      nodesDraggable={false} nodesConnectable={false} elementsSelectable
      onNodeClick={(_, node) => {
        const d = node.data as unknown as TableData;
        if (d.clickable) onLink(`relational/table/${node.id}`);
      }}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#eef0f3" gap={16} />
    </ReactFlow>
  );
}

class Boundary extends Component<{ children: ReactNode }, { err: string | null }> {
  state = { err: null as string | null };
  static getDerivedStateFromError(e: unknown) { return { err: e instanceof Error ? e.message : String(e) }; }
  render() {
    if (this.state.err)
      return <div style={{ padding: 12, color: "#c0392b", fontSize: 12 }}>ERD failed to render: {this.state.err}</div>;
    return this.props.children;
  }
}

export default function RelationalErd({ erd, onLink }: { erd: ErdData; onLink: (id: string) => void }) {
  const more = erd.more ?? {};
  const moreBits = [
    more.referenced_by ? `+${more.referenced_by} more referencing tables` : null,
    more.views ? `+${more.views} more views` : null,
  ].filter(Boolean).join(" · ");
  return (
    <Boundary>
      <div style={{ margin: "0 -4px 8px" }}>
        <div style={{ height: 320, width: "100%", border: "1px solid #f0f0f0", borderRadius: 6, background: "#fcfcfd" }}>
          <ReactFlowProvider>
            <Graph erd={erd} onLink={onLink} />
          </ReactFlowProvider>
        </div>
        <div style={{ fontSize: 11, color: "#888", textAlign: "center", padding: "3px 0" }}>
          first-order ERD · ← referencing · FK targets → · views below{moreBits ? ` · ${moreBits}` : ""} · tap a table to walk
        </div>
      </div>
    </Boundary>
  );
}
