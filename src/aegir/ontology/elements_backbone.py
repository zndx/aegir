"""elements_backbone — the SysML identity backbone as OWL (#44 increment: yardstick §2).

The thin elements + relations layer of RH's DDL yardstick, authored as OUR
classes (bare names) grounded on the ICE spine: a model-element record
DESIGNATES model content — it is information about a modeled system, not the
system. The fidelity probe validated this shape from the external's own
statistics (79% of SysMLv2's metamodel features are derived views; the
persisted core is elements + owned relationships — exactly this backbone).

IGNITES THE DORMANT ENUMS: sdg:metaclass and sdg:relationshipKind are the
lowersToProperty targets of the census-verified SysMLv2Metaclasses /
SysMLv2RelationshipKinds Collections — this module is their first consumer, so
kvasir's scheme→lookup path seeds sysml metaclass + relationship-kind reference
tables (20 + 10 census-cased values) with FKs from the backbone columns.

ON THE CRITICAL PATH (no flags). Definition/Usage and ownership are modeled as
OPTIONAL self-references (`only` — universal, no existence commitment: roots
have no owner, definitions have no definition); source/target on relationships
are `exactly 1` (a relationship without ends is nothing). The composite-FK
metaclass-discrimination pattern (UNIQUE(id, metaclass) + subtype FK pairs) is
kvasir-side emission work — a later (c) leg; the backbone tables land now.

Relation to spec_mappings' sdg:Sysml* entities: those are the ANNOTATION-layer
interop mappings (in-scope-external evidence); these are the OPERATIONAL
classes whose rows the semantic layer stores. Distinct jobs, deliberately
distinct names.
"""

ELEMENTS_BACKBONE_OMN = """
ObjectProperty: sdg:elementOwner
    Annotations: rdfs:label "element owner",
        rdfs:comment "Ownership/containment: the namespace-side element that owns this one. OPTIONAL (roots have no owner) — the persisted core's single containment axis, per the fidelity probe."

ObjectProperty: sdg:elementDefinition
    Annotations: rdfs:label "element definition",
        rdfs:comment "The Definition/Usage pattern's usage-side link: a usage resolves its definition. OPTIONAL (definitions have none)."

ObjectProperty: sdg:relationshipSource
    Annotations: rdfs:label "relationship source"

ObjectProperty: sdg:relationshipTarget
    Annotations: rdfs:label "relationship target"

DataProperty: sdg:qualifiedName
    Annotations: rdfs:label "qualified name"

DataProperty: sdg:isDefinition
    Annotations: rdfs:label "is definition"

ObjectProperty: sdg:partOfModelProject
    Annotations: rdfs:label "part of model project"

ObjectProperty: sdg:previousCommit
    Annotations: rdfs:label "previous commit",
        rdfs:comment "OPTIONAL self-reference: the interchange chain's parent commit. HEAD-STATE STORE (ruled 2026-07-25): history is the procession + OL/Atlas lineage — this is the SysML-API interchange shape, never a second git."

ObjectProperty: sdg:inCommit
    Annotations: rdfs:label "in commit"

DataProperty: sdg:commitMessage
    Annotations: rdfs:label "commit message"

Class: sdg:ModelProject
    Annotations: rdfs:label "Model Project",
        rdfs:comment "A record naming one modeling initiative — the initiative layer's root (yardstick §1). An information content entity about the modeling effort, not the modeled system."
    SubClassOf: cco:ont00000958,
        sdg:qualifiedName some xsd:string

Class: sdg:ModelCommit
    Annotations: rdfs:label "Model Commit",
        rdfs:comment "A record of one interchange commit within a model project (yardstick §1) — head-state semantics: the chain exists for interchange, the procession owns history."
    SubClassOf: cco:ont00000958,
        sdg:partOfModelProject exactly 1 sdg:ModelProject,
        sdg:commitMessage some xsd:string,
        sdg:previousCommit only sdg:ModelCommit

Class: sdg:ModelElement
    Annotations: rdfs:label "Model Element",
        rdfs:comment "A record designating one element of a modeled system — the identity backbone's unit, typed by the census-verified SysMLv2 metaclass vocabulary. An information content entity ABOUT the model, not a part of the plant. Representable atop SysMLv2 as Element (census-confirmed root metaclass, abstract)."
    SubClassOf: cco:ont00000958,
        sdg:metaclass some xsd:string,
        sdg:isDefinition some xsd:boolean,
        sdg:elementOwner only sdg:ModelElement,
        sdg:elementDefinition only sdg:ModelElement,
        sdg:partOfModelProject exactly 1 sdg:ModelProject,
        sdg:inCommit only sdg:ModelCommit

Class: sdg:ModelRelationship
    Annotations: rdfs:label "Model Relationship",
        rdfs:comment "A record designating one reified relationship between model elements, kinded by the census-verified relationship vocabulary (the three SDG relational flattenings included, marked as ours). Source and target are mandatory — a relationship without ends designates nothing. Representable atop SysMLv2 as Relationship (census-confirmed, abstract)."
    SubClassOf: cco:ont00000958,
        sdg:relationshipKind some xsd:string,
        sdg:relationshipSource exactly 1 sdg:ModelElement,
        sdg:relationshipTarget exactly 1 sdg:ModelElement
"""
