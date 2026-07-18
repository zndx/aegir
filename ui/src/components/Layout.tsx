import { Moon, Sun } from "@phosphor-icons/react";
import { Layout as AntLayout, Typography } from "antd";
import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";

import { getColorMode, toggleColorMode, useColorMode } from "../theme/colorMode";

const { Header, Content, Footer } = AntLayout;
const { Text } = Typography;

const NAV_ITEMS = [
  { path: "/lineup",         label: "Lineup" },
  { path: "/leaderboards",   label: "Leaderboards" },
  { path: "/classifications", label: "Classifications" },
  { path: "/ontologies",     label: "Ontologies" },
];

interface LayoutProps {
  children: ReactNode;
  fullHeight?: boolean;
}

// Phosphor per the org norm; re-renders with the live mode so the glyph shows the DESTINATION mode.
function ModeGlyph() {
  const mode = useColorMode();
  return mode === "dark" ? <Sun size={15} weight="bold" /> : <Moon size={15} weight="bold" />;
}

function Layout({ children, fullHeight }: LayoutProps) {
  const { pathname } = useLocation();

  return (
    <AntLayout
      style={fullHeight
        ? { height: "100vh", overflow: "hidden" }
        : { minHeight: "100vh" }
      }
    >
      <Header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          background: "var(--color-kumo-base)",
          borderBottom: "1px solid var(--color-kumo-hairline)",
          padding: "0 clamp(12px, 2vw, 24px)",
        }}
      >
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Text
            strong
            style={{ color: "var(--text-color-kumo-strong)", fontSize: 20, letterSpacing: 0.5 }}
          >
            Ægir
          </Text>
          <Text style={{ color: "var(--text-color-kumo-subtle)", fontSize: 12 }}>
            v0.2.0 · relational metadata tagging
          </Text>
        </Link>
        <nav style={{ display: "flex", gap: 4, marginLeft: 16 }}>
          {NAV_ITEMS.map(({ path, label }) => {
            const active = pathname === path || pathname.startsWith(path + "/");
            return (
              <Link
                key={path}
                to={path}
                style={{
                  color: active ? "var(--text-color-kumo-strong)" : "var(--text-color-kumo-subtle)",
                  padding: "4px 12px",
                  borderRadius: 4,
                  fontSize: 14,
                  textDecoration: "none",
                  background: active ? "var(--color-kumo-fill)" : "transparent",
                  transition: "all 0.2s",
                }}
              >
                {label}
              </Link>
            );
          })}
        </nav>
        <button
          aria-label="Toggle color mode"
          title="Toggle color mode"
          onClick={() => toggleColorMode(getColorMode())}
          style={{
            marginLeft: "auto", display: "flex", alignItems: "center", justifyContent: "center",
            width: 30, height: 30, borderRadius: 6, cursor: "pointer",
            background: "transparent", border: "1px solid var(--color-kumo-hairline)",
            color: "var(--text-color-kumo-subtle)",
          }}
        >
          <ModeGlyph />
        </button>
      </Header>
      <Content
        style={fullHeight
          ? { padding: "0 8px", display: "flex", flexDirection: "column", overflow: "hidden" }
          : { padding: "24px clamp(16px, 3vw, 48px)" }
        }
      >
        {children}
      </Content>
      {!fullHeight && (
        <Footer style={{ textAlign: "center" }}>
          <Text type="secondary">
            Aegir &mdash; air-gap-friendly hierarchical sequence modeling
          </Text>
        </Footer>
      )}
    </AntLayout>
  );
}

export default Layout;
