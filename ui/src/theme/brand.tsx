import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { useColorMode } from "./colorMode";

export interface BrandOption {
  id: string;
  display_name: string;
  logo_href: string;
  logo_href_light?: string;
  favicon_href?: string;
  logo_alt: string;
  has_logo: boolean;
  source?: string;
}

export interface SettingsPayload {
  brand_id: string;
  brand: BrandOption;
  brands: BrandOption[];
  config_path: string;
}

interface BrandContextValue {
  settings: SettingsPayload | null;
  loading: boolean;
  error: string;
  refresh: () => Promise<void>;
  selectBrand: (brandId: string) => Promise<void>;
  uploadCustom: (file: File) => Promise<void>;
}

const BrandContext = createContext<BrandContextValue | null>(null);

function applyFavicon(href: string): void {
  if (!href) return;
  let link = document.querySelector<HTMLLinkElement>('link[rel="icon"][data-aegir-brand="1"]');
  if (!link) {
    link = document.createElement("link");
    link.rel = "icon";
    link.type = "image/svg+xml";
    link.dataset.aegirBrand = "1";
    document.head.appendChild(link);
  }
  link.href = href;
}

export function BrandProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<SettingsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const r = await fetch("/api/settings");
    if (!r.ok) throw new Error(`settings ${r.status}`);
    const data = (await r.json()) as SettingsPayload;
    setSettings(data);
    applyFavicon(data.brand?.favicon_href || data.brand?.logo_href || "");
  }, []);

  useEffect(() => {
    refresh()
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [refresh]);

  const selectBrand = useCallback(async (brandId: string) => {
    setError("");
    const r = await fetch("/api/settings/brand", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ brand_id: brandId }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `brand update ${r.status}`);
    setSettings(data as SettingsPayload);
    applyFavicon((data as SettingsPayload).brand?.favicon_href || "");
  }, []);

  const uploadCustom = useCallback(async (file: File) => {
    setError("");
    const body = new FormData();
    body.append("pack", file);
    const r = await fetch("/api/settings/brand/upload", { method: "POST", body });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `upload ${r.status}`);
    setSettings(data as SettingsPayload);
    applyFavicon((data as SettingsPayload).brand?.favicon_href || "");
  }, []);

  const value = useMemo(
    () => ({ settings, loading, error, refresh, selectBrand, uploadCustom }),
    [settings, loading, error, refresh, selectBrand, uploadCustom],
  );
  return <BrandContext.Provider value={value}>{children}</BrandContext.Provider>;
}

export function useBrand(): BrandContextValue {
  const ctx = useContext(BrandContext);
  if (!ctx) throw new Error("useBrand requires BrandProvider");
  return ctx;
}

export function useBrandLogo(): { href: string; alt: string } | null {
  const { settings } = useBrand();
  const mode = useColorMode();
  const brand = settings?.brand;
  if (!brand?.has_logo) return null;
  const href = mode === "light" && brand.logo_href_light ? brand.logo_href_light : brand.logo_href;
  return { href, alt: brand.logo_alt || brand.display_name };
}
