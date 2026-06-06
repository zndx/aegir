-- migrate:up
-- ============================================================================
-- Aegir semantic provenance graph on Apache AGE  (aegir_hx)
-- ============================================================================
-- The graph is the SOURCE OF TRUTH. There is deliberately NO lineage_events
-- JSONB fallback: a fallback erodes the graph-centric architecture (code
-- silently degrades to it). AGE is REQUIRED — provisioned in devenv.nix — and
-- the bootstrap fails LOUDLY (no fallback) if it is absent.
--
-- Idempotent: every create_graph / create_vlabel / create_elabel is guarded by
-- an existence check, so the migration is safe to re-run and survives partial or
-- out-of-band application (AGE's create_* have no IF NOT EXISTS of their own).
--
-- Scope: a separate `aegir_hx` graph (future federation into the Weathership
-- `wx_hx` portfolio graph). Labels are Atlas-entity-aligned so the governance SDK
-- projects the same nodes as Atlas v2 entities + classifications, while
-- OpenLineage runs/datasets ride the same graph.

CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

DO $aegir_hx$
DECLARE
  lbl text;
  -- vertex labels — the closed loop's semantic entities
  vlabels text[] := ARRAY[
    'FinePDFsDoc','Topic','Template','Slot','Family','Chapter','Unit','Table',
    'Column','EvidenceSpan','Claim','Skill','Model','Run'];
  -- edge labels — relationships, incl. the RE_GROUNDS_TO loop closure
  elabels text[] := ARRAY[
    'IN_CLUSTER','RELEVANT_TO','HAS_SLOT','IN_FAMILY','SEEDED_BY','USES_EVIDENCE',
    'SELECTED','PRODUCED','GENERATED','COMPOSED_OF','HAS_COLUMN','INSTANTIATES',
    'FK','ASSERTS','GROUNDED_TO','FROM_DOC','RE_GROUNDS_TO'];
BEGIN
  IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'aegir_hx') THEN
    PERFORM ag_catalog.create_graph('aegir_hx');
  END IF;

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
END $aegir_hx$;

SET search_path = "$user", public;

-- migrate:down
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
SELECT drop_graph('aegir_hx', true);
SET search_path = "$user", public;
