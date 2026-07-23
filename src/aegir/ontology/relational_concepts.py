"""relational_concepts — the ontology's closure over its own relational projection.

RH ruling (2026-07-23): ALL relational entities require ontology associations — someone
provided with only the database must be able to tag each entity with its ontological
source. EAV and association tables included. FinePDFs supplies domain entropy; these
RELATIONAL-DATA concepts are ours to author deliberately, grounded like everything else
(⊑ cco:ont00000958, Information Content Entity — the verified ICE anchor; no guessed
IRIs per the coined-alias rule), HermiT-certified, shipped in every release so every
sdg-corpora consumer benefits.

The classes name the schema PATTERNS the generators emit (recorded by construction in
naming_map kinds, construct FK structure, and key_plan audit intent) and the COLUMN
ROLES (surrogate identifiers, FK references, audit timestamps, domain attributes).
"""

RELATIONAL_CONCEPTS_OMN = """
Class: sdg:RelationalSchemaEntity
    Annotations: rdfs:label "Relational Schema Entity", iao:0000115 "An information content entity that is an element of a relational schema realized from the ontology — the closure class: every generated relational entity is an instance of exactly one of its subclasses."
    SubClassOf: cco:ont00000958

Class: sdg:EntityTable
    Annotations: rdfs:label "Entity Table", iao:0000115 "A relational table realizing a single ontology class: one row per instance, columns realizing the class's attributes and relations."
    SubClassOf: sdg:RelationalSchemaEntity

Class: sdg:AssociationTable
    Annotations: rdfs:label "Association Table", iao:0000115 "A relational table realizing a many-to-many relation between ontology classes: its non-plumbing columns are foreign-key references to the participant classes' tables."
    SubClassOf: sdg:RelationalSchemaEntity

Class: sdg:LookupTable
    Annotations: rdfs:label "Lookup Table", iao:0000115 "A relational table realizing an enumerated kind: one row per admissible value of a closed value set."
    SubClassOf: sdg:RelationalSchemaEntity

Class: sdg:EntityAttributeValueTable
    Annotations: rdfs:label "Entity-Attribute-Value Table", iao:0000115 "A relational table realizing the EAV pattern: rows are (entity, attribute, value) triples whose attribute column ranges over ontology properties rather than realizing a single class's fixed columns."
    SubClassOf: sdg:RelationalSchemaEntity

Class: sdg:RelationalView
    Annotations: rdfs:label "Relational View", iao:0000115 "A derived relational entity composed over tables, intended for embedding generated prose: each view column is an entity knowable completely because it remains grounded in the ontology."
    SubClassOf: sdg:RelationalSchemaEntity

Class: sdg:RelationalColumn
    Annotations: rdfs:label "Relational Column", iao:0000115 "An information content entity that is a column of a relational schema element."
    SubClassOf: cco:ont00000958

Class: sdg:SurrogateIdentifierColumn
    Annotations: rdfs:label "Surrogate Identifier Column", iao:0000115 "A column bearing a generated identifier designating the row's instance; carries no domain semantics beyond designation."
    SubClassOf: sdg:RelationalColumn

Class: sdg:ForeignKeyColumn
    Annotations: rdfs:label "Foreign Key Column", iao:0000115 "A column realizing a relation to another schema element's identified instances — the relational realization of an ontology object property."
    SubClassOf: sdg:RelationalColumn

Class: sdg:AuditTimestampColumn
    Annotations: rdfs:label "Audit Timestamp Column", iao:0000115 "A column recording row lifecycle instants (creation, update) — provenance plumbing, planned by the generator's audit intent; asserts nothing about the domain."
    SubClassOf: sdg:RelationalColumn

Class: sdg:AttributeColumn
    Annotations: rdfs:label "Attribute Column", iao:0000115 "A column realizing an ontology data property of the table's realized class."
    SubClassOf: sdg:RelationalColumn

ObjectProperty: sdg:realizedFromClass
    Annotations: rdfs:label "realized from class", iao:0000115 "Relates a relational schema entity to the ontology class whose realization it is — the association that lets a database-only consumer tag each entity with its ontological source."
    Domain: sdg:RelationalSchemaEntity
"""
