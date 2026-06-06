-- migrate:up
-- ============================================================================
-- Aegir semantic provenance graph on Apache AGE  (aegir_hx)
-- ============================================================================
-- The graph is the SOURCE OF TRUTH. There is deliberately NO lineage_events
-- JSONB fallback table: a fallback erodes the graph-centric architecture (code
-- silently degrades to it). AGE is REQUIRED — provisioned in devenv.nix — and
-- this migration fails LOUDLY if it is absent. Same fail-fast posture as the
-- ColBERT MaxSim bridge (no silent degradation to a weaker substrate).
--
-- Scope: a separate `aegir_hx` graph (future federation into the Weathership
-- `wx_hx` portfolio graph). Entity/edge labels are Atlas-entity-aligned so the
-- governance SDK (aegir.governance) can project the same nodes as Atlas v2
-- entities + classifications, while OpenLineage runs/datasets ride the same graph.

CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT create_graph('aegir_hx');

-- Vertex labels — the closed loop's semantic entities ------------------------
SELECT create_vlabel('aegir_hx', 'FinePDFsDoc');   -- source document (the fixed point)
SELECT create_vlabel('aegir_hx', 'Topic');         -- frozen BERTopic cluster + centroid
SELECT create_vlabel('aegir_hx', 'Template');      -- OWL/BFO catalog template
SELECT create_vlabel('aegir_hx', 'Slot');          -- typed template slot
SELECT create_vlabel('aegir_hx', 'Family');        -- template family (F1..F7)
SELECT create_vlabel('aegir_hx', 'Chapter');       -- generated chapter
SELECT create_vlabel('aegir_hx', 'Unit');          -- corpus unit (prose/example/links/diagram)
SELECT create_vlabel('aegir_hx', 'Table');         -- generated relational table (S2)
SELECT create_vlabel('aegir_hx', 'Column');        -- table column (the CTA/CPA target)
SELECT create_vlabel('aegir_hx', 'EvidenceSpan');  -- grounding span from a FinePDFsDoc
SELECT create_vlabel('aegir_hx', 'Claim');         -- asserted claim
SELECT create_vlabel('aegir_hx', 'Skill');         -- S1..S7
SELECT create_vlabel('aegir_hx', 'Model');         -- GLM / Grok (generation)
SELECT create_vlabel('aegir_hx', 'Run');           -- calibration episode (OpenLineage run)

-- Edge labels — relationships, incl. the RE_GROUNDS_TO loop closure ----------
SELECT create_elabel('aegir_hx', 'IN_CLUSTER');     -- FinePDFsDoc -> Topic
SELECT create_elabel('aegir_hx', 'RELEVANT_TO');    -- Topic -> Template  {cos}
SELECT create_elabel('aegir_hx', 'HAS_SLOT');       -- Template -> Slot
SELECT create_elabel('aegir_hx', 'IN_FAMILY');      -- Template -> Family
SELECT create_elabel('aegir_hx', 'SEEDED_BY');      -- Run -> Topic
SELECT create_elabel('aegir_hx', 'USES_EVIDENCE');  -- Run -> EvidenceSpan
SELECT create_elabel('aegir_hx', 'SELECTED');       -- Run -> Template
SELECT create_elabel('aegir_hx', 'PRODUCED');       -- Skill -> Unit  {repairs}
SELECT create_elabel('aegir_hx', 'GENERATED');      -- Model -> Unit
SELECT create_elabel('aegir_hx', 'COMPOSED_OF');    -- Chapter -> Unit
SELECT create_elabel('aegir_hx', 'HAS_COLUMN');     -- Table -> Column
SELECT create_elabel('aegir_hx', 'INSTANTIATES');   -- Column -> Slot  {type_ok}
SELECT create_elabel('aegir_hx', 'FK');             -- Table -> Table  {via_slot}
SELECT create_elabel('aegir_hx', 'ASSERTS');        -- Unit -> Claim
SELECT create_elabel('aegir_hx', 'GROUNDED_TO');    -- Claim -> EvidenceSpan
SELECT create_elabel('aegir_hx', 'FROM_DOC');       -- EvidenceSpan -> FinePDFsDoc
SELECT create_elabel('aegir_hx', 'RE_GROUNDS_TO');  -- Chapter -> Topic  {cosine, rank, hit_at_1}

SET search_path = "$user", public;

-- migrate:down
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
SELECT drop_graph('aegir_hx', true);
SET search_path = "$user", public;
