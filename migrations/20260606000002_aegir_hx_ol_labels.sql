-- migrate:up
-- ============================================================================
-- OpenLineage-standard labels for aegir_hx (Marquez-compatible lineage layer)
-- ============================================================================
-- The extended OL variant rides the same aegir_hx graph as the Atlas entities:
--   (Dataset)-[:INPUT_TO]->(Run)-[:OUTPUTS]->(Dataset),  (Job)-[:EXECUTES]->(Run)
-- OL dataset names may be our entity qualifiedNames (chapter:…, topic:…), so a
-- RunEvent's lineage links to the existing semantic nodes — the "extension".
-- Idempotent (guarded), like 20260606000001.

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

DO $aegir_hx_ol$
DECLARE
  lbl text;
  vlabels text[] := ARRAY['Dataset','Job'];
  elabels text[] := ARRAY['INPUT_TO','OUTPUTS','EXECUTES'];
BEGIN
  FOREACH lbl IN ARRAY vlabels LOOP
    IF NOT EXISTS (
      SELECT 1 FROM ag_catalog.ag_label l JOIN ag_catalog.ag_graph g ON l.graph = g.graphid
      WHERE g.name = 'aegir_hx' AND l.name = lbl
    ) THEN
      PERFORM ag_catalog.create_vlabel('aegir_hx', lbl);
    END IF;
  END LOOP;

  FOREACH lbl IN ARRAY elabels LOOP
    IF NOT EXISTS (
      SELECT 1 FROM ag_catalog.ag_label l JOIN ag_catalog.ag_graph g ON l.graph = g.graphid
      WHERE g.name = 'aegir_hx' AND l.name = lbl
    ) THEN
      PERFORM ag_catalog.create_elabel('aegir_hx', lbl);
    END IF;
  END LOOP;
END $aegir_hx_ol$;

SET search_path = "$user", public;

-- migrate:down
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
-- AGE has no drop_label; labels are removed with the graph (see 20260606000001 down).
SET search_path = "$user", public;
