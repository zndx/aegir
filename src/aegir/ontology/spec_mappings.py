"""spec_mappings — PRODML + SysMLv2 enter the SDG namespace as MAPPED ontology entities.

RH ruling (2026-07-23): we are CERTAIN both specifications enter SDG as mapped
entities so we can inter-operate with any conforming data source — HDF5 files
structured to both specs are already in the downstream analytical queue. Shipped as a
release candidate under the versioned-release doctrine (external review → refined
releases; the impending CAS push refines category placements and grows coverage).

Design rules honored:
- Coined-alias ban: sdg: coinage only, grounded ⊑ verified opaque BFO/CCO numerics
  (bfo:0000015 process · bfo:0000040 material entity · bfo:0000004 independent
  continuant · cco:ont00000958 information content entity — all declared in the
  realize scaffold). No invented bfo:/cco: names.
- Category honesty: wells/fibers/samples are material entities; acquisitions/analyses
  are processes; reports/datasets/model elements are ICEs. SysMLv2 model elements are
  ICEs describing systems — never the systems themselves.
- Collision-proof `Prodml*`/`Sysml*` locals (the entity-derived union holds ~2,700
  classes; bare `Well`/`Requirement` would be identity-merge CAS work, not an RC).
- Every class carries `rdfs:seeAlso` to its source-spec object — the mapping record
  a database/HDF5-side consumer joins on.

The SKOS side (PRODML/SYSML ConceptSchemes + narrower concepts) lives in
domain_concepts.ttl; the triad checker joins the layers.
"""

SPEC_MAPPINGS_OMN = """
Class: sdg:ProdmlWell
    Annotations: rdfs:label "PRODML Well", iao:0000115 "A well as identified in Energistics energyml common: the engineered material entity whose production PRODML objects report on.", rdfs:seeAlso <http://www.energistics.org/energyml/data/commonv2#Well>
    SubClassOf: bfo:0000040

Class: sdg:ProdmlWellbore
    Annotations: rdfs:label "PRODML Wellbore", iao:0000115 "A wellbore of a well per energyml common — the drilled independent continuant along which sensing installations and tests are located.", rdfs:seeAlso <http://www.energistics.org/energyml/data/commonv2#Wellbore>
    SubClassOf: bfo:0000004

Class: sdg:ProdmlFiberOpticalPath
    Annotations: rdfs:label "PRODML Fiber Optical Path", iao:0000115 "The installed optical fiber material entity along a wellbore that distributed sensing acquisitions interrogate.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#FiberOpticalPath>
    SubClassOf: bfo:0000040

Class: sdg:ProdmlFluidSample
    Annotations: rdfs:label "PRODML Fluid Sample", iao:0000115 "A reservoir fluid sample material entity with chain-of-custody, the input to fluid analyses.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#FluidSample>
    SubClassOf: bfo:0000040

Class: sdg:ProdmlFluidAnalysis
    Annotations: rdfs:label "PRODML Fluid Analysis", iao:0000115 "A laboratory analysis process over a fluid sample — PVT studies and compositional assays in the PRODML fluid analysis family.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#FluidAnalysis>
    SubClassOf: bfo:0000015

Class: sdg:ProdmlDasAcquisition
    Annotations: rdfs:label "PRODML DAS Acquisition", iao:0000115 "A distributed acoustic sensing acquisition process interrogating a fiber optical path; its raw/processed arrays land in HDF5 per the PRODML DAS transfer.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#DasAcquisition>
    SubClassOf: bfo:0000015

Class: sdg:ProdmlDtsSurvey
    Annotations: rdfs:label "PRODML DTS Survey", iao:0000115 "A distributed temperature sensing survey process along an installed system, yielding measurement traces.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#DtsInstalledSystem>
    SubClassOf: bfo:0000015

Class: sdg:ProdmlReport
    Annotations: rdfs:label "PRODML Report", iao:0000115 "An information content entity of the PRODML reporting families — the parent of product volume, fluid analysis, production operation reports and sensing datasets.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2>
    SubClassOf: cco:ont00000958

Class: sdg:ProdmlProductVolumeReport
    Annotations: rdfs:label "PRODML Product Volume Report", iao:0000115 "Reported production volumes and dispositions per well, facility, and period — PRODML's product volume object.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#ProductVolume>
    SubClassOf: sdg:ProdmlReport

Class: sdg:ProdmlFluidAnalysisReport
    Annotations: rdfs:label "PRODML Fluid Analysis Report", iao:0000115 "The reported results of a fluid analysis — the ICE output of the analysis process.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#FluidAnalysisReport>
    SubClassOf: sdg:ProdmlReport

Class: sdg:ProdmlProductionOperationReport
    Annotations: rdfs:label "PRODML Production Operation Report", iao:0000115 "Daily production operations reporting: activities, shutdowns, safety events, operational parameters across producing assets.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#ProductionOperation>
    SubClassOf: sdg:ProdmlReport

Class: sdg:ProdmlDasRawDataSet
    Annotations: rdfs:label "PRODML DAS Raw Data Set", iao:0000115 "The raw/FBE/spectra array dataset of a DAS acquisition — the HDF5-resident payload a conforming file's groups and datasets map to.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#DasRaw>
    SubClassOf: sdg:ProdmlReport

Class: sdg:ProdmlDtsMeasurementTrace
    Annotations: rdfs:label "PRODML DTS Measurement Trace", iao:0000115 "A temperature-versus-depth measurement trace produced by a DTS survey.", rdfs:seeAlso <http://www.energistics.org/energyml/data/prodmlv2#DtsMeasurement>
    SubClassOf: sdg:ProdmlReport

ObjectProperty: sdg:prodmlConcernsWell
    Annotations: rdfs:label "prodml concerns well", iao:0000115 "Relates a PRODML report to the well its content reports on — the join a database-only or HDF5-only consumer uses to tag payloads back to the asset."
    Domain: sdg:ProdmlReport
    Range: sdg:ProdmlWell

Class: sdg:SysmlModelElement
    Annotations: rdfs:label "SysML Model Element", iao:0000115 "An information content entity of a SysMLv2/KerML model — model elements describe systems and are never the described systems themselves.", rdfs:seeAlso <https://www.omg.org/spec/SysML/>
    SubClassOf: cco:ont00000958

Class: sdg:SysmlPackage
    Annotations: rdfs:label "SysML Package", iao:0000115 "A namespace-owning container model element grouping definitions and usages.", rdfs:seeAlso <https://www.omg.org/spec/SysML/Package>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlRequirement
    Annotations: rdfs:label "SysML Requirement", iao:0000115 "A requirement definition or usage: a constraint-bearing model element stating what a system shall satisfy, with traceability structure.", rdfs:seeAlso <https://www.omg.org/spec/SysML/RequirementDefinition>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlPartDefinition
    Annotations: rdfs:label "SysML Part Definition", iao:0000115 "A definition of a class of systems or parts — the structural type SysMLv2 part usages instantiate.", rdfs:seeAlso <https://www.omg.org/spec/SysML/PartDefinition>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlPartUsage
    Annotations: rdfs:label "SysML Part Usage", iao:0000115 "A usage of a part within a structural decomposition — the compositional occurrence of a part definition.", rdfs:seeAlso <https://www.omg.org/spec/SysML/PartUsage>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlPortDefinition
    Annotations: rdfs:label "SysML Port Definition", iao:0000115 "A definition of an interaction point through which parts connect and exchange.", rdfs:seeAlso <https://www.omg.org/spec/SysML/PortDefinition>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlConnectionUsage
    Annotations: rdfs:label "SysML Connection Usage", iao:0000115 "A connection between usages — the modeled link binding ports/parts in a structural decomposition.", rdfs:seeAlso <https://www.omg.org/spec/SysML/ConnectionUsage>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlActionDefinition
    Annotations: rdfs:label "SysML Action Definition", iao:0000115 "A behavioral definition of actions a system performs, composable into sequences and control structures.", rdfs:seeAlso <https://www.omg.org/spec/SysML/ActionDefinition>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlStateDefinition
    Annotations: rdfs:label "SysML State Definition", iao:0000115 "A behavioral definition of system states and transitions — SysMLv2 state-based occurrence modeling.", rdfs:seeAlso <https://www.omg.org/spec/SysML/StateDefinition>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlConstraintDefinition
    Annotations: rdfs:label "SysML Constraint Definition", iao:0000115 "A parametric constraint definition — an analyzable relation over attribute usages.", rdfs:seeAlso <https://www.omg.org/spec/SysML/ConstraintDefinition>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlAttributeUsage
    Annotations: rdfs:label "SysML Attribute Usage", iao:0000115 "An attribute usage carrying a typed value slot on a definition or usage.", rdfs:seeAlso <https://www.omg.org/spec/SysML/AttributeUsage>
    SubClassOf: sdg:SysmlModelElement

Class: sdg:SysmlVerificationCase
    Annotations: rdfs:label "SysML Verification Case", iao:0000115 "A verification case binding requirements to analysis, inspection, and test evidence — SysMLv2 verification practice.", rdfs:seeAlso <https://www.omg.org/spec/SysML/VerificationCaseDefinition>
    SubClassOf: sdg:SysmlModelElement

ObjectProperty: sdg:sysmlVerifies
    Annotations: rdfs:label "sysml verifies", iao:0000115 "Relates a verification case to the requirement it verifies — the load-bearing MBSE traceability relation."
    Domain: sdg:SysmlVerificationCase
    Range: sdg:SysmlRequirement
"""
