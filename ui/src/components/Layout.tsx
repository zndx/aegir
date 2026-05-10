import { Layout as AntLayout, Typography } from "antd";
import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";

const { Header, Content, Footer } = AntLayout;
const { Text } = Typography;

const NAV_ITEMS = [
  { path: "/leaderboards",   label: "Leaderboards" },
  { path: "/classifications", label: "Classifications" },
  { path: "/ontologies",     label: "Ontologies" },
];

interface LayoutProps {
  children: ReactNode;
  fullHeight?: boolean;
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
          background: "#1f1f2a",
          padding: "0 clamp(12px, 2vw, 24px)",
        }}
      >
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Text
            strong
            style={{ color: "#fff", fontSize: 20, letterSpacing: 0.5 }}
          >
            Ægir
          </Text>
          <Text style={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}>
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
                  color: active ? "#fff" : "rgba(255,255,255,0.65)",
                  padding: "4px 12px",
                  borderRadius: 4,
                  fontSize: 14,
                  textDecoration: "none",
                  background: active ? "rgba(255,255,255,0.1)" : "transparent",
                  transition: "all 0.2s",
                }}
              >
                {label}
              </Link>
            );
          })}
        </nav>
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
