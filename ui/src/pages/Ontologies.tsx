import { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Input,
  Row,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
  message,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import type { UploadFile } from "antd/es/upload/interface";
import { InboxOutlined, UploadOutlined } from "@ant-design/icons";

const { Title, Paragraph, Text } = Typography;

interface DbpediaType {
  label: string;
  count: number;
}

interface DbpediaPayload {
  source: string;
  exists: boolean;
  total_columns?: number;
  num_types?: number;
  types?: DbpediaType[];
}

interface OntologyNode {
  id: string;
  parent: string;
  kind: string;
  description?: string;
}

interface VocabularyPayload {
  name: string;
  upper: string;
  nodes: OntologyNode[];
  note?: string;
}

interface SubsumptionCandidate {
  iri: string;
  label: string;
  confidence: number;
  path: string[];
}

interface SubsumptionSuggestion {
  term: string;
  candidates: SubsumptionCandidate[];
  reasoning: string;
}

interface SubsumptionResponse {
  suggestions: SubsumptionSuggestion[];
  count: number;
  predictor: string;
}

// Parse a CSV/TSV/XLSX-export paste into a list of {name, description}
// triples. For M1 this keeps the upload flow synchronous and client-side
// only; M2 will accept multipart uploads with the real file directly.
function parseVocabText(raw: string): { name: string; description?: string }[] {
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length === 0) return [];
  const delim = lines[0].includes("\t") ? "\t" : ",";
  const [header, ...rest] = lines;
  const headers = header.split(delim).map((h) => h.trim().toLowerCase());
  const nameIdx = headers.findIndex(
    (h) => h === "name" || h === "term" || h === "label",
  );
  const descIdx = headers.findIndex(
    (h) => h === "description" || h === "definition" || h === "desc",
  );
  if (nameIdx < 0) {
    // No header row detected — treat each line as a bare term.
    return [header, ...rest].map((line) => ({ name: line }));
  }
  return rest.map((line) => {
    const cells = line.split(delim);
    return {
      name: (cells[nameIdx] ?? "").trim(),
      description: descIdx >= 0 ? (cells[descIdx] ?? "").trim() : undefined,
    };
  });
}

export default function Ontologies() {
  const [dbpedia, setDbpedia] = useState<DbpediaPayload | null>(null);
  const [vocab, setVocab] = useState<VocabularyPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const [rawText, setRawText] = useState("");
  const [suggestions, setSuggestions] =
    useState<SubsumptionResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetch("/api/ontology/dbpedia-types")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setDbpedia)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    fetch("/api/ontology/vocabulary")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setVocab)
      .catch(() => setVocab(null));
  }, []);

  const submitSubsumption = async () => {
    const parsed = parseVocabText(rawText);
    if (parsed.length === 0) {
      message.warning("No terms found to map");
      return;
    }
    setSubmitting(true);
    try {
      const r = await fetch("/api/ontology/subsume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ terms: parsed }),
      });
      if (!r.ok) throw new Error(`${r.status}`);
      const data: SubsumptionResponse = await r.json();
      setSuggestions(data);
      message.success(
        `Mapped ${data.count} term${data.count === 1 ? "" : "s"}`,
      );
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const onFileSelect = (file: UploadFile) => {
    // antd Upload passes a wrapped object; we only want the text contents.
    const raw = (file as unknown as File) ?? null;
    if (!raw || typeof (raw as File).text !== "function") return false;
    (raw as File)
      .text()
      .then((t) => setRawText(t))
      .catch(() => message.error("Failed to read file"));
    return false; // prevent auto-upload; we handle the text client-side
  };

  const dbpediaFiltered = (dbpedia?.types ?? []).filter(
    (t) => !query || t.label.toLowerCase().includes(query.toLowerCase()),
  );

  const dbpediaCols: ColumnsType<DbpediaType> = [
    {
      title: "DBpedia type",
      dataIndex: "label",
      render: (v: string) => <Text code>{v}</Text>,
      sorter: (a, b) => a.label.localeCompare(b.label),
      defaultSortOrder: "ascend",
    },
    {
      title: "Columns",
      dataIndex: "count",
      align: "right",
      sorter: (a, b) => a.count - b.count,
    },
  ];

  const suggestionCols: ColumnsType<SubsumptionSuggestion> = [
    { title: "Your term", dataIndex: "term", key: "term" },
    {
      title: "Top match",
      key: "top",
      render: (_, r) => {
        const c = r.candidates[0];
        if (!c) return "—";
        return (
          <Space direction="vertical" size={0}>
            <Text code>{c.iri}</Text>
            <Text type="secondary" style={{ fontSize: 11 }}>
              {c.label}
            </Text>
          </Space>
        );
      },
    },
    {
      title: "Confidence",
      key: "conf",
      align: "right",
      render: (_, r) =>
        r.candidates[0] ? (
          <Tag
            color={
              r.candidates[0].confidence > 0.5
                ? "green"
                : r.candidates[0].confidence > 0.2
                  ? "orange"
                  : "default"
            }
          >
            {(r.candidates[0].confidence * 100).toFixed(1)}%
          </Tag>
        ) : (
          "—"
        ),
    },
    {
      title: "Path",
      key: "path",
      render: (_, r) =>
        r.candidates[0] ? (
          <Text type="secondary" style={{ fontSize: 11 }}>
            {r.candidates[0].path.join(" → ")}
          </Text>
        ) : null,
    },
  ];

  return (
    <>
      <Title level={2} style={{ marginBottom: 4 }}>
        Ontologies
      </Title>
      <Paragraph>
        Map a bespoke vocabulary into our shared ICE/BFO-grounded training
        ontology, or browse the label vocabularies the model already sees.
        Upload a CSV/TSV with a <Text code>name</Text> column (optional{" "}
        <Text code>description</Text>), or paste terms directly, and get
        subsumption suggestions back. M1 ships the upload + review shell with
        a stub predictor; M2 replaces the body with a{" "}
        <Text italic>
          BERTSubs-style contextual-embedding subsumption predictor
        </Text>{" "}
        (Chen et al.).
      </Paragraph>

      {error && (
        <Alert type="error" message={error} style={{ marginBottom: 12 }} />
      )}

      <Tabs
        defaultActiveKey="upload"
        items={[
          {
            key: "upload",
            label: "Map your vocabulary",
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24} lg={12}>
                  <Card
                    title={
                      <>
                        Upload / paste terms <Tag color="purple">M1 stub</Tag>
                      </>
                    }
                    size="small"
                  >
                    <Upload.Dragger
                      accept=".csv,.tsv,.txt"
                      beforeUpload={onFileSelect}
                      showUploadList={false}
                    >
                      <p className="ant-upload-drag-icon">
                        <InboxOutlined />
                      </p>
                      <p className="ant-upload-text">
                        Drop a CSV/TSV here, or paste below
                      </p>
                      <p className="ant-upload-hint" style={{ fontSize: 11 }}>
                        Required column: <code>name</code>. Optional:{" "}
                        <code>description</code>.
                      </p>
                    </Upload.Dragger>
                    <Input.TextArea
                      rows={6}
                      style={{ marginTop: 12, fontFamily: "monospace" }}
                      placeholder={
                        "name,description\nCustomer,A person or org with a commercial relationship\nInvoice,Itemized request for payment"
                      }
                      value={rawText}
                      onChange={(e) => setRawText(e.target.value)}
                    />
                    <Button
                      type="primary"
                      icon={<UploadOutlined />}
                      loading={submitting}
                      onClick={submitSubsumption}
                      style={{ marginTop: 12 }}
                      disabled={!rawText.trim()}
                    >
                      Predict subsumption
                    </Button>
                  </Card>
                </Col>
                <Col xs={24} lg={12}>
                  <Card title="Suggestions" size="small">
                    {suggestions === null ? (
                      <Text type="secondary">
                        Upload or paste a vocabulary and click{" "}
                        <em>Predict subsumption</em>. Each term gets a
                        ranked list of ontology parents with a confidence
                        score and full BFO path.
                      </Text>
                    ) : (
                      <>
                        <Alert
                          type={
                            suggestions.predictor === "stub-m1"
                              ? "warning"
                              : "info"
                          }
                          showIcon
                          style={{ marginBottom: 12 }}
                          message={
                            <>
                              Predictor: <Text code>{suggestions.predictor}</Text>
                            </>
                          }
                          description={
                            suggestions.predictor === "stub-m1"
                              ? "M1 placeholder — every term maps to ICE:DataElement at low confidence. The shape stabilizes here; M2 plugs in the BERTSubs-style scorer."
                              : null
                          }
                        />
                        <Table<SubsumptionSuggestion>
                          rowKey="term"
                          size="small"
                          pagination={false}
                          dataSource={suggestions.suggestions}
                          columns={suggestionCols}
                        />
                      </>
                    )}
                  </Card>
                </Col>
              </Row>
            ),
          },
          {
            key: "ice-bfo",
            label: "ICE/BFO skeleton",
            children: (
              <Card size="small">
                <Paragraph>
                  The upper-level alignment that bespoke vocabularies map
                  into. M1 ships a skeleton; M2 replaces it with the
                  ontology-extraction pipeline output described in{" "}
                  <Text code>docs/src/pretraining.md</Text>.
                </Paragraph>
                {vocab ? (
                  <Table
                    size="small"
                    pagination={false}
                    rowKey="id"
                    dataSource={vocab.nodes}
                    columns={[
                      {
                        title: "IRI",
                        dataIndex: "id",
                        render: (v: string) => <Text code>{v}</Text>,
                      },
                      { title: "Parent", dataIndex: "parent" },
                      {
                        title: "Kind",
                        dataIndex: "kind",
                        render: (k: string) => (
                          <Tag
                            color={
                              k === "upper"
                                ? "blue"
                                : k === "mid"
                                  ? "geekblue"
                                  : "default"
                            }
                          >
                            {k}
                          </Tag>
                        ),
                      },
                      {
                        title: "Description",
                        dataIndex: "description",
                        render: (d?: string) =>
                          d ? <Text>{d}</Text> : null,
                      },
                    ]}
                  />
                ) : (
                  <Text>Loading...</Text>
                )}
              </Card>
            ),
          },
          {
            key: "dbpedia",
            label: "DBpedia (gt-signals)",
            children: (
              <>
                {dbpedia === null ? (
                  <Card>
                    <Text>Loading ontology...</Text>
                  </Card>
                ) : !dbpedia.exists ? (
                  <Alert
                    type="warning"
                    message="GitTables ground-truth JSON not found"
                    description={
                      <>
                        Expected at <Text code>{dbpedia.source}</Text>. Set{" "}
                        <Text code>AEGIR_GITTABLES_DIR</Text> to override.
                      </>
                    }
                  />
                ) : (
                  <>
                    <Card style={{ marginBottom: 16 }}>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr 2fr",
                          gap: 24,
                        }}
                      >
                        <Statistic
                          title="Distinct types"
                          value={dbpedia.num_types ?? 0}
                        />
                        <Statistic
                          title="Labeled columns"
                          value={dbpedia.total_columns ?? 0}
                        />
                        <div>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            Source
                          </Text>
                          <div>
                            <Text code>{dbpedia.source}</Text>
                          </div>
                        </div>
                      </div>
                    </Card>
                    <Input.Search
                      placeholder="Filter types (e.g. genus, year, author)"
                      allowClear
                      onChange={(e) => setQuery(e.target.value)}
                      style={{ maxWidth: 320, marginBottom: 16 }}
                    />
                    <Table<DbpediaType>
                      rowKey="label"
                      dataSource={dbpediaFiltered}
                      columns={dbpediaCols}
                      pagination={{ pageSize: 20, showSizeChanger: true }}
                    />
                  </>
                )}
              </>
            ),
          },
        ]}
      />
    </>
  );
}
