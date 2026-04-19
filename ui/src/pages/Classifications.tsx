import { useEffect, useState } from "react";
import { Alert, Card, Col, Descriptions, Row, Statistic, Table, Tag, Tree, Typography } from "antd";
import type { DataNode } from "antd/es/tree";

const { Title, Paragraph, Text } = Typography;

interface LatestPayload {
  run_id: string | null;
  task?: string;
  model_size?: string;
  micro_f1?: number | null;
  macro_f1?: number | null;
  num_classes?: number | null;
  per_class_f1?: number[] | null;
  message?: string;
}

interface CatalogChild {
  name: string;
  kind: string;
  num_labels: number;
  parent: string;
  attributes: { name: string; type: string }[];
}

interface CatalogRoot {
  name: string;
  kind: string;
  description: string;
  children: CatalogChild[];
}

interface CatalogPayload {
  classifications: CatalogRoot[];
  source: string;
  editable: boolean;
}

// Apache Atlas vernacular:
//   - Classification: a named typed tag, optionally inheriting from a parent.
//   - Entity: a thing that can be tagged (table, column, ...).
// We expose Aegir's task registry as an Atlas-compatible catalog so corpus
// metadata round-trips cleanly with any existing Atlas deployment.
function toTreeData(catalog: CatalogPayload | null): DataNode[] {
  if (!catalog) return [];
  return catalog.classifications.map((root) => ({
    key: root.name,
    title: (
      <span>
        <strong>{root.name}</strong>{" "}
        <Tag color="purple">{root.kind}</Tag>
      </span>
    ),
    children: root.children.map((c) => ({
      key: `${root.name}/${c.name}`,
      title: (
        <span>
          {c.name} <Tag color="blue">{c.num_labels} labels</Tag>
        </span>
      ),
    })),
  }));
}

export default function Classifications() {
  const [latest, setLatest] = useState<LatestPayload | null>(null);
  const [catalog, setCatalog] = useState<CatalogPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/classifications/latest")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setLatest)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    fetch("/api/classifications/catalog")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setCatalog)
      .catch(() => setCatalog(null));
  }, []);

  const taskTableData =
    catalog?.classifications[0]?.children.map((c) => ({
      key: c.name,
      name: c.name,
      kind: c.kind,
      num_labels: c.num_labels,
      attributes: c.attributes.map((a) => `${a.name}:${a.type}`).join(", "),
    })) ?? [];

  return (
    <>
      <Title level={2} style={{ marginBottom: 4 }}>
        Classifications
      </Title>
      <Paragraph>
        Apache Atlas-style classification catalog. Browse the task vocabularies
        Aegir trains against — each registered task is a Classification whose
        children are its labels. The shape is Atlas-compatible so corpus-level
        metadata can round-trip with any existing Atlas deployment without
        reshape.
      </Paragraph>

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message={
          <>
            Corpus sync + human-in-the-loop review{" "}
            <Tag color="purple">M2</Tag>
          </>
        }
        description="M1 ships the read-only catalog wired to the task registry, plus the latest-run aggregate metrics. M2 brings Atlas-endpoint sync, editable classifications with BFO-grounded parents, and per-entity tagging feedback into the training corpus."
      />

      {error && (
        <Alert type="error" message={error} style={{ marginBottom: 12 }} />
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Card title="Classification catalog" size="small">
            {catalog === null ? (
              <Text>Loading catalog...</Text>
            ) : (
              <Tree
                defaultExpandAll
                treeData={toTreeData(catalog)}
                selectable={false}
              />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <Card title="Tasks" size="small">
            <Table
              size="small"
              pagination={false}
              dataSource={taskTableData}
              columns={[
                { title: "Task", dataIndex: "name" },
                { title: "Kind", dataIndex: "kind" },
                { title: "Labels", dataIndex: "num_labels", align: "right" },
                {
                  title: "Attributes",
                  dataIndex: "attributes",
                  ellipsis: true,
                },
              ]}
            />
          </Card>
        </Col>
      </Row>

      <Title level={4} style={{ marginTop: 24 }}>
        Latest run
      </Title>
      {latest === null ? (
        <Card>
          <Text>Loading latest run...</Text>
        </Card>
      ) : latest.run_id === null ? (
        <Card>
          <Alert
            type="warning"
            message="No runs yet"
            description={
              latest.message ??
              "Train a model to populate this panel. Try: just run-train --task gt-signals-dbpedia --model-size tiny --epochs 3"
            }
          />
        </Card>
      ) : (
        <Card
          title={
            <>
              <Text code>{latest.run_id}</Text>
            </>
          }
          extra={
            <Text type="secondary">
              {latest.task} · {latest.model_size}
            </Text>
          }
        >
          <Row gutter={16}>
            <Col span={8}>
              <Statistic
                title="Micro F1"
                value={latest.micro_f1 ?? 0}
                precision={4}
                valueStyle={{ color: "#4f7cff" }}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Macro F1"
                value={latest.macro_f1 ?? 0}
                precision={4}
                valueStyle={{ color: "#4f7cff" }}
              />
            </Col>
            <Col span={8}>
              <Statistic title="Classes" value={latest.num_classes ?? 0} />
            </Col>
          </Row>

          <Descriptions size="small" column={1} style={{ marginTop: 16 }}>
            <Descriptions.Item label="Per-class F1">
              {latest.per_class_f1 == null ? (
                <Text type="secondary">
                  Not persisted in M1 sidecar. See M2 metrics schema.
                </Text>
              ) : (
                <Text>{latest.per_class_f1.length} values available</Text>
              )}
            </Descriptions.Item>
          </Descriptions>
        </Card>
      )}
    </>
  );
}
