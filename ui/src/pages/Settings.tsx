import { useRef, useState } from "react";
import { Alert, Typography } from "antd";

import { useBrand } from "../theme/brand";

const { Paragraph, Text, Title } = Typography;

export default function Settings() {
  const { settings, loading, error, selectBrand, uploadCustom } = useBrand();
  const [flash, setFlash] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  if (loading && !settings) {
    return <Text type="secondary">Loading settings…</Text>;
  }

  async function onSelect(id: string) {
    setBusy(true);
    setFlash("");
    try {
      await selectBrand(id);
      setFlash("Applied to this deployment.");
    } catch (e) {
      setFlash(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(file: File) {
    setBusy(true);
    setFlash("");
    try {
      await uploadCustom(file);
      setFlash("Custom pack uploaded and applied.");
    } catch (e) {
      setFlash(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  const brands = settings?.brands ?? [];
  const selected = settings?.brand_id;

  return (
    <div className="settings-page">
      <Title level={2} style={{ marginBottom: 8 }}>Settings</Title>
      <Paragraph type="secondary">
        Deployment-wide configuration for this Ægir process. Changes apply to{" "}
        <strong>all clients</strong> and persist under{" "}
        <Text code>{settings?.config_path ?? "build/config/aegir-ui.json"}</Text>.
      </Paragraph>

      {error && <Alert type="warning" showIcon message={error} style={{ marginBottom: 16 }} />}

      <section className="settings-card" id="branding">
        <Title level={4}>Branding</Title>
        <Paragraph type="secondary">
          Logo pack only — theme stays Keiretsu; product name stays <strong>Ægir</strong>.
          Listing is <strong>Weathership</strong>, then <strong>Cloudera</strong>, then
          an empty <Text code>Custom</Text> slot for a <Text code>.tgz</Text> pack
          (flat <Text code>logo.svg</Text> + <Text code>favicon.svg</Text>, or a
          Weathership kit). Click a pack to apply it for the whole deployment.
        </Paragraph>

        <input
          ref={fileRef}
          type="file"
          accept=".tgz,.tar.gz,application/gzip,application/x-gzip"
          className="brand-file-input"
          aria-label="Upload custom brand pack"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void onUpload(file);
          }}
        />

        <div className="brand-grid" role="radiogroup" aria-label="Logo brand pack">
          {brands.map((b) => {
            if (!b.has_logo && b.id === "custom") {
              return (
                <button
                  key={b.id}
                  type="button"
                  className="brand-card brand-card-empty"
                  disabled={busy}
                  onClick={() => fileRef.current?.click()}
                >
                  <span className="brand-card-preview" aria-hidden="true">
                    <svg className="brand-plus" viewBox="0 0 32 32" width="32" height="32" focusable="false">
                      <path fill="currentColor" d="M15 6h2v20h-2z" />
                      <path fill="currentColor" d="M6 15h20v2H6z" />
                    </svg>
                  </span>
                  <span className="brand-card-meta">
                    <span className="brand-card-name">Custom</span>
                    <span className="brand-card-id">upload .tgz</span>
                  </span>
                </button>
              );
            }
            return (
              <label key={b.id} className={`brand-card${selected === b.id ? " selected" : ""}`}>
                <input
                  type="radio"
                  name="brand_id"
                  value={b.id}
                  checked={selected === b.id}
                  disabled={busy}
                  onChange={() => void onSelect(b.id)}
                />
                <span className="brand-card-preview">
                  {b.logo_href && <img src={b.logo_href} alt={b.logo_alt} height={28} />}
                </span>
                <span className="brand-card-meta">
                  <span className="brand-card-name">{b.display_name}</span>
                  <span className="brand-card-id">{b.id}</span>
                </span>
              </label>
            );
          })}
        </div>
        {flash && <p className={flash.startsWith("Applied") || flash.startsWith("Custom") ? "saved-hint" : "settings-error"}>{flash}</p>}
      </section>
    </div>
  );
}
