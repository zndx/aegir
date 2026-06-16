import {
  BarChartOutlined,
  BookOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  TagsOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { Card, Col, Row, Statistic, Tag, Typography } from "antd";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

const { Paragraph, Text, Title } = Typography;

interface StatsPayload {
  service: { ok: boolean; version: string; runs_dir_exists: boolean };
  runs: { count: number; latest: { run_id?: string; task?: string } | null };
  tasks: { count: number };
  terms: { count: number; note: string };
}

function statusLabel(stats: StatsPayload | null): string {
  if (!stats?.service) return "Unknown";
  if (!stats.service.ok) return "Disconnected";
  if (!stats.service.runs_dir_exists) return "Degraded";
  return "Connected";
}

function statusColor(stats: StatsPayload | null): string {
  if (!stats?.service?.ok) return "#ff4d4f";
  if (!stats.service.runs_dir_exists) return "#faad14";
  return "#52c41a";
}

function statusPrefix(stats: StatsPayload | null) {
  if (!stats?.service?.ok) return <CloseCircleOutlined />;
  if (!stats.service.runs_dir_exists) return <WarningOutlined />;
  return <CheckCircleOutlined />;
}

function Landing() {
  const [stats, setStats] = useState<StatsPayload | null>(null);

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  return (
    <>
      <Typography>
        <Title level={2} style={{ marginBottom: 8 }}>
          Aegir
        </Title>
        <Paragraph>
          A foundational, domain-adapted system for relational database
          metadata understanding. Aegir combines H-Net-style dynamic chunking
          with RWKV-7 time mixing to provide Column Type Annotation (CTA),
          Column Property Annotation (CPA), and cross-table Data Element
          Discovery (DED) over wide tables.
        </Paragraph>
        <Paragraph>
          This UI is an air-gap-ready alternative to heavy experiment
          trackers (W&amp;B, MLflow). Run artifacts land on disk as JSON +
          static Bokeh plots; the gateway reads them directly. No tracking
          daemon, no external telemetry.
        </Paragraph>
      </Typography>

      {/* Stats cards — mirrors the Atelier 4-card top row so the two UIs
          converge cleanly when they merge next week. */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Service"
              value={statusLabel(stats)}
              prefix={statusPrefix(stats)}
              valueStyle={{ color: statusColor(stats) }}
            />
            {stats?.service?.version && (
              <Tag color="blue" style={{ marginTop: 8 }}>
                v{stats.service.version}
              </Tag>
            )}
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Link to="/leaderboards">
            <Card hoverable>
              <Statistic
                title="Runs"
                value={stats?.runs?.count ?? "—"}
                prefix={<BarChartOutlined />}
              />
              {stats?.runs?.latest?.task && (
                <Text type="secondary" style={{ fontSize: 11 }}>
                  latest: {stats.runs.latest.task}
                </Text>
              )}
            </Card>
          </Link>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Tasks"
              value={stats?.tasks?.count ?? "—"}
              prefix={<ExperimentOutlined />}
            />
            <Text type="secondary" style={{ fontSize: 11 }}>
              registered benchmarks
            </Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Link to="/lineup?lens=terms">
            <Card hoverable>
              <Statistic
                title="Terms"
                value={stats?.terms?.count ?? "—"}
                prefix={<BookOutlined />}
              />
              <Text type="secondary" style={{ fontSize: 11 }}>
                explore the ontology vocabulary →
              </Text>
            </Card>
          </Link>
        </Col>
      </Row>

      {/* Lenses — entry points into the projected KB (the lineup). The cards are
          logical front doors; panels + links are the flexible substrate beneath.
          Lineup navigation primitive: Ward Cunningham's federated wiki. */}
      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} md={8}>
          <Link to="/lineup?lens=terms" style={{ textDecoration: "none" }}>
            <Card hoverable title="Terms" extra={<BookOutlined />}>
              <Paragraph>
                Browse the ontology vocabulary by family — each term opens to its
                verbalization, axiom, BFO/CCO anchor, and relational projection.
              </Paragraph>
              <Text type="secondary">lens · ontology</Text>
            </Card>
          </Link>
        </Col>
        <Col xs={24} md={8}>
          <Link to="/lineup?lens=schema" style={{ textDecoration: "none" }}>
            <Card hoverable title="Schema" extra={<DatabaseOutlined />}>
              <Paragraph>
                Walk the relational projection — each table realizes an ontology term
                (the ontology↔DDL pivot); columns carry typed slots.
              </Paragraph>
              <Text type="secondary">lens · relational</Text>
            </Card>
          </Link>
        </Col>
        <Col xs={24} md={8}>
          <Link to="/lineup?lens=content" style={{ textDecoration: "none" }}>
            <Card hoverable title="Content" extra={<TagsOutlined />}>
              <Paragraph>
                Read the textbook-quality corpus and the FinePDFs topics it covers,
                linked back to the terms each chapter realizes.
              </Paragraph>
              <Text type="secondary">lens · content</Text>
            </Card>
          </Link>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} md={8}>
          <Link to="/leaderboards" style={{ textDecoration: "none" }}>
            <Card hoverable title="Leaderboards" extra={<BarChartOutlined />}>
              <Paragraph>
                One row per training run: task, model size, F1, boundary
                diagnostics, git SHA. Row-click opens loss/F1/chunking plots.
              </Paragraph>
              <Text type="secondary">M1 · live</Text>
            </Card>
          </Link>
        </Col>
        <Col xs={24} md={8}>
          <Link to="/classifications" style={{ textDecoration: "none" }}>
            <Card hoverable title="Classifications" extra={<TagsOutlined />}>
              <Paragraph>
                Apache Atlas-style classification catalog. Browse the task
                vocabularies Aegir trains against; M2 wires in human-in-the-
                loop review and corpus-level Atlas sync.
              </Paragraph>
              <Text type="secondary">M1 · read-only catalog</Text>
            </Card>
          </Link>
        </Col>
        <Col xs={24} md={8}>
          <Link to="/ontologies" style={{ textDecoration: "none" }}>
            <Card hoverable title="Ontologies" extra={<DatabaseOutlined />}>
              <Paragraph>
                Upload a bespoke vocabulary (CSV/XLSX) and get BERTSubs-style
                subsumption suggestions against the shared ICE/BFO training
                ontology. M1 ships the upload + review shell with a stub
                predictor; M2 lands the real embedding-based scorer.
              </Paragraph>
              <Text type="secondary">M1 · upload + stub predictor</Text>
            </Card>
          </Link>
        </Col>
      </Row>
    </>
  );
}

export default Landing;
