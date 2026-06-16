import { ConfigProvider, theme } from "antd";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Suspense, lazy } from "react";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";

const Leaderboards = lazy(() => import("./pages/Leaderboards"));
const Classifications = lazy(() => import("./pages/Classifications"));
const Ontologies = lazy(() => import("./pages/Ontologies"));
const Lineup = lazy(() => import("./pages/Lineup"));

function App() {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: { colorPrimary: "#4f7cff", borderRadius: 6 },
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
