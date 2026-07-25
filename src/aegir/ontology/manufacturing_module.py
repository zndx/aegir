"""manufacturing_module — the Manufacturing sector's OWL entity cluster (#44 increment (b)).

The MFG_LOGISTICS territory as BFO/CCO-grounded classes in OUR namespace (bare
names, RH-ratified 2026-07-25): the logical foundations we are uniquely
positioned to define, with SysMLv2 correspondences cited to CENSUS-CONFIRMED
metaclasses only (build/foreign/sysmlv2/census_30.json — the language
represents, the ontology grounds).

Grounding parents are the realized ontology's LIVE opaque IRIs (external-index
verified; coined-alias ban honored):
  bfo:0000040  material entity      · cco:ont00000192  Facility
  cco:ont00000958  Information Content Entity
  cco:ont00000853  Descriptive Information Content Entity
  cco:ont00001359  Act of Manufacturing

ON THE CRITICAL PATH (RH 2026-07-25: no per-module feature flags — one unified
path via ``just metaflow``): appended unconditionally in
build_realized_ontology (the RELATIONAL_CONCEPTS/SPEC_MAPPINGS precedent); the
flow's own gates prove it on-path. verify_manufacturing_module.py remains the
FAST DEV-LOOP PROBE (isolated HermiT + kvasir + the parity/underspecification
diff), not a staging gate.

COMPLETENESS DOCTRINE (RH 2026-07-25): the REQUIREMENT to produce a complete,
functional relational schema DRIVES ontology completeness — a class the
lowering folds or drops is UNDERSPECIFIED, and underspecification is a
remediation signal co-equal with logical inconsistency (never an accepted
lowering behavior — silent folds are the Sweep-B sin at the lowering boundary).
kvasir complains NATIVELY (Plan.underspecified + stderr, kvasir 95da21b); the
dev-loop probe consumes that signal, never re-derives it. First measured case:
Warehouse folded silently for lacking a distinguishing feature → remediated
with sdg:facilityCode (what a WMS actually tracks), not appeasement.

FIDELITY-PROBE TENSIONS RESOLVED HERE (Phase 1c quality signal):
  * ordered collections → ``sdg:lineNumber`` (xsd:integer) on PurchaseOrderLine
    — the sequence_number discipline, recorded not improvised.
  * derived-layer posture: stock aggregates/valuations are VIEWS over these
    persisted classes, never persisted classes themselves (the 79%-derived
    finding: persist the core, derive the rest).
"""

MANUFACTURING_OMN = """
ObjectProperty: sdg:withinFacility
    Annotations: rdfs:label "within facility"

ObjectProperty: sdg:partOfPurchaseOrder
    Annotations: rdfs:label "part of purchase order"

ObjectProperty: sdg:documentsReceiptAgainst
    Annotations: rdfs:label "documents receipt against"

ObjectProperty: sdg:authorizesManufacturingAct
    Annotations: rdfs:label "authorizes manufacturing act"

ObjectProperty: sdg:stockOfItem
    Annotations: rdfs:label "stock of item"

ObjectProperty: sdg:stockAtLocation
    Annotations: rdfs:label "stock at location"

DataProperty: sdg:lineNumber
    Annotations: rdfs:label "line number",
        rdfs:comment "Ordered-collection discipline (fidelity-probe tension resolved): explicit sequence within the owning order — lowers to the sequence_number column."

DataProperty: sdg:quantityOrdered
    Annotations: rdfs:label "quantity ordered"

DataProperty: sdg:quantityOnHand
    Annotations: rdfs:label "quantity on hand"

DataProperty: sdg:facilityCode
    Annotations: rdfs:label "facility code",
        rdfs:comment "The operational identifier a WMS/ERP tracks for a facility — the underspecification-signal remediation that earned Warehouse its table."

Class: sdg:InventoryItem
    Annotations: rdfs:label "Inventory Item",
        rdfs:comment "A material entity under inventory management: identified, stocked, counted, reserved, and issued. Inventory-hood is a borne role; the class grounds the artifact side. Representable atop SysMLv2 as a domain-specific ItemUsage (census-confirmed metaclass)."
    SubClassOf: bfo:0000040

Class: sdg:Warehouse
    Annotations: rdfs:label "Warehouse",
        rdfs:comment "A facility organizing storage locations, put-away, and cycle counting, identified operationally by its facility code."
    SubClassOf: cco:ont00000192,
        sdg:facilityCode some xsd:string

Class: sdg:StorageLocation
    Annotations: rdfs:label "Storage Location",
        rdfs:comment "An addressable location where inventory items are held — bin, rack, shelf, or floor position — within a facility."
    SubClassOf: cco:ont00000192,
        sdg:withinFacility some cco:ont00000192

Class: sdg:PurchaseOrder
    Annotations: rdfs:label "Purchase Order",
        rdfs:comment "A directive information content entity ordering supply from a supplier: the procure-to-pay workflow's opening act. Representable atop SysMLv2 as a domain-specific ItemDefinition specialization with attribute usages (census-confirmed metaclasses)."
    SubClassOf: cco:ont00000958

Class: sdg:PurchaseOrderLine
    Annotations: rdfs:label "Purchase Order Line",
        rdfs:comment "One ordered line of a purchase order: an item, a quantity, a promised date. Carries an explicit line number (ordered-collection discipline)."
    SubClassOf: cco:ont00000958,
        sdg:partOfPurchaseOrder exactly 1 sdg:PurchaseOrder,
        sdg:lineNumber some xsd:integer,
        sdg:quantityOrdered some xsd:decimal

Class: sdg:GoodsReceipt
    Annotations: rdfs:label "Goods Receipt",
        rdfs:comment "A descriptive information content entity documenting that ordered material arrived: receiving's record against a purchase order."
    SubClassOf: cco:ont00000853,
        sdg:documentsReceiptAgainst some sdg:PurchaseOrder

Class: sdg:WorkOrder
    Annotations: rdfs:label "Work Order",
        rdfs:comment "A directive information content entity authorizing production: releases routings to work cells and lines. Grounds directly against the CCO Act of Manufacturing. Representable atop SysMLv2 as directing domain-specific ActionUsages (census-confirmed metaclass)."
    SubClassOf: cco:ont00000958,
        sdg:authorizesManufacturingAct some cco:ont00001359

Class: sdg:StockLevel
    Annotations: rdfs:label "Stock Level",
        rdfs:comment "A descriptive information content entity stating the quantity of one inventory item at one storage location at a time — the persisted fact; aggregates and valuations are DERIVED views (the fidelity probe's persist-the-core discipline). Representable atop SysMLv2 as an AttributeUsage (census-confirmed metaclass)."
    SubClassOf: cco:ont00000853,
        sdg:stockOfItem exactly 1 sdg:InventoryItem,
        sdg:stockAtLocation exactly 1 sdg:StorageLocation,
        sdg:quantityOnHand some xsd:decimal
"""
