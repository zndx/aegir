import { useEffect, useState } from "react";
import { Alert, Badge, Button, Drawer, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import BokehPlot from "../components/BokehPlot";

const { Title, Text, Paragraph } = Typography;

interface Row {
  run_id: string;
  task: string;
  model_size: string;
  num_params: number | null;
  num_epochs: number | null;
  best_val_macro_f1: number | null;
  best_val_micro_f1: number | null;
  last_train_loss: number | null;
  last_val_loss: number | null;
  git_short_sha: string | null;
  utc_start: string | null;
  utc_end: string | null;
  smoke_test: boolean | null;
}

interface LeaderboardResponse {
  rows: Row[];
  count: number;
}

interface DetailPayload {
  metadata: Record<string, unknown>;
  metrics: unknown;
  plots: string[];
}

function fmtF1(v: number | null) {
  return v == null ? "—" : v.toFixed(4);
}

function fmtParams(n: number | null) {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function Leaderboards() {
  const [rows, setRows] = useState<Row[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [active, setActive] = useState<Row | null>(null);
  const [detail, setDetail] = useState<DetailPayload | null>(null);

  useEffect(() => {
    fetch("/api/leaderboard")
      .then(r => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json() as Promise<LeaderboardResponse>;
      })
      .then(d => setRows(d.rows))
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!active) { setDetail(null); return; }
    fetch(`/api/runs/${encodeURIComponent(active.run_id)}`)
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`${r.status}`)))
      .then(setDetail)
      .catch(e => setError(e instanceof Error ? e.message : String(e)));
  }, [active]);

  const columns: ColumnsType<Row> = [
    {
      title: "Run",
      dataIndex: "run_id",
      key: "run_id",
      render: (run_id: string, row) => (
        <Space direction="vertical" size={0}>
          <Text strong>{run_id}</Text>
          {row.smoke_test && <Tag color="orange">smoke</Tag>}
        </Space>
      ),
      fixed: "left",
    },
    { title: "Task", dataIndex: "task", key: "task" },
    { title: "Model", dataIndex: "model_size", key: "model_size", align: "center" },
    {
      title: "Params", dataIndex: "num_params", key: "num_params", align: "right",
      render: fmtParams, sorter: (a, b) => (a.num_params ?? 0) - (b.num_params ?? 0),
    },
    {
      title: "Epochs", dataIndex: "num_epochs", key: "num_epochs", align: "right",
      render: (v: number | null) => v ?? "—",
    },
    {
      title: "Micro F1", dataIndex: "best_val_micro_f1", key: "micro", align: "right",
      render: fmtF1,
      sorter: (a, b) => (a.best_val_micro_f1 ?? 0) - (b.best_val_micro_f1 ?? 0),
    },
    {
      title: "Macro F1", dataIndex: "best_val_macro_f1", key: "macro", align: "right",
      render: fmtF1,
      sorter: (a, b) => (a.best_val_macro_f1 ?? 0) - (b.best_val_macro_f1 ?? 0),
      defaultSortOrder: "descend",
    },
    { title: "Git", dataIndex: "git_short_sha", key: "sha", render: v => v ?? "—" },
    { title: "Started", dataIndex: "utc_start", key: "utc_start" },
  ];

  return (
    <>
      <Title level={2} style={{ marginBottom: 4 }}>Leaderboards</Title>
      <Paragraph>
        One row per training run. Sidecars are read from{" "}
        <Text code>outputs/runs/</Text> on the gateway host. Click a row to see
        loss, F1, and per-stage chunking diagnostics plotted offline via Bokeh.
      </Paragraph>

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message={
          <>
            External baselines pending{" "}
            <Badge color="purple" text="M2" />
          </>
        }
        description={
          <>
            Aegir runs appear here today. Comparison against Nemotron 3 Nano,
            OpenAI OSS 20b and REVEAL baselines arrives in M2 once their
            inference harness lands.
          </>
        }
      />

      {error && <Alert type="error" message={error} style={{ marginBottom: 12 }} />}

      <Table<Row>
        rowKey="run_id"
        loading={loading}
        dataSource={rows}
        columns={columns}
        pagination={{ pageSize: 25, showSizeChanger: true }}
        scroll={{ x: 1100 }}
        onRow={(row) => ({ onClick: () => setActive(row), style: { cursor: "pointer" } })}
      />

      <Drawer
        title={active ? `Run: ${active.run_id}` : "Run details"}
        placement="right"
        width={780}
        open={active !== null}
        onClose={() => setActive(null)}
        extra={active && (
          <Space>
            <Text type="secondary">
              macro F1 {fmtF1(active.best_val_macro_f1)} · micro F1 {fmtF1(active.best_val_micro_f1)}
            </Text>
          </Space>
        )}
      >
        {active && detail ? (
          <Space direction="vertical" size="middle" style={{ width: "100%" }}>
            {detail.plots.includes("loss") && (
              <BokehPlot title="Loss" src={`/api/runs/${encodeURIComponent(active.run_id)}/plot/loss`} />
            )}
            {detail.plots.includes("f1") && (
              <BokehPlot title="F1" src={`/api/runs/${encodeURIComponent(active.run_id)}/plot/f1`} />
            )}
            {detail.plots.filter(p => p.startsWith("boundary_stage")).map(name => (
              <BokehPlot key={name} title={name} src={`/api/runs/${encodeURIComponent(active.run_id)}/plot/${encodeURIComponent(name)}`} />
            ))}
            {detail.plots.length === 0 && (
              <Alert
                type="warning"
                message="No plots written for this run"
                description="If the run crashed before finalize(), plot files are skipped. Check metadata.json for the utc_end timestamp."
              />
            )}
            <Button type="link" href={`/api/runs/${encodeURIComponent(active.run_id)}`} target="_blank">
              View raw JSON
            </Button>
          </Space>
        ) : <Text>Loading run...</Text>}
      </Drawer>
    </>
  );
}
