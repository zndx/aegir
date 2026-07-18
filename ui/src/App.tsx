import { ConfigProvider, theme } from "antd";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Suspense, lazy } from "react";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";
import { useColorMode } from "./theme/colorMode";

const Leaderboards = lazy(() => import("./pages/Leaderboards"));
const Classifications = lazy(() => import("./pages/Classifications"));
const Ontologies = lazy(() => import("./pages/Ontologies"));
const Lineup = lazy(() => import("./pages/Lineup"));

function App() {
  // org norm (cldr-design-template): dark default, toggleable; the antd BRIDGE follows the kumo
  // data-mode, with neutrals aligned to theme-cloudera.css tokens per mode until Kumo migration completes.
  const mode = useColorMode();
  const dark = mode === "dark";
  return (
    <ConfigProvider
      theme={{
        algorithm: dark ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: dark
          ? { colorPrimary: "#6366f1", borderRadius: 6, colorBgBase: "#12121a", colorTextBase: "#e5e7eb" }
          : { colorPrimary: "#6366f1", borderRadius: 6, colorBgBase: "#ffffff", colorTextBase: "#1f2937" },
      }}
    >
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout><Landing /></Layout>} />
          <Route
            path="/leaderboards"
            element={
              <Layout>
                <Suspense fallback={null}>
                  <Leaderboards />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/classifications"
            element={
              <Layout>
                <Suspense fallback={null}>
                  <Classifications />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/ontologies"
            element={
              <Layout>
                <Suspense fallback={null}>
                  <Ontologies />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/lineup"
            element={
              <Layout fullHeight>
                <Suspense fallback={null}>
                  <Lineup />
                </Suspense>
              </Layout>
            }
          />
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
  );
}

export default App;
