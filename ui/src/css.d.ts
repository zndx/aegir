// Side-effect CSS imports (e.g. `@xyflow/react/dist/style.css` base styles) — declared as an ambient module
// so tsc accepts the import; vite handles the actual bundling. The first such import in this UI (antd is
// CSS-in-JS), so the declaration lives here rather than a vite-env.d.ts.
declare module "*.css";
