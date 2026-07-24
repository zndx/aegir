Source: https://re-magazine.ireb.org/articles/discovering-system-requirements-through-sysml
Anchor: MFG_REQUIREMENT

An application of the IREB Handbook of Requirements Modeling

Written by Gildas Premel-Cabic
15 September 2021 · 9 minutes read

For Business Analysts, Requirement Engineers, and System Analysts, the greatest innovation brought by the Systems Modeling Language (SysML) is the introduction of three things: the requirement as a modeling element type, the requirement relationships, and the requirement diagram. Requirement specification with SysML has become a topic of articles and books, as well as the IREB Requirements Modeling Handbook, which considers how to precisely model requirements with SysML diagrams. Yet the manual does not show these SysML requirement items in action, as they do not model the actual content of the requirement — this still remains textually specified.

This article illustrates the application to a study case of the IREB manual, including the use of the SysML requirement diagram.

1 Requirement, Requirement Model & Requirement Diagram

The SysML requirement type element serves as a basis to specify text-based requirements. Its true value relies on the ability to link it to many SysML modeling element types: requirement, but also block, actor, use case, activity, interaction, state machine, and more. The requirements and the requirement relationships form the requirement model, which is fully depicted with the SysML requirement diagram.

With SysML, System Analysts can relate the individual text-based system requirements to the elements that model the system and thus show where the requirements originate from. This traceability capacity is rarely shown in articles and books dealing with requirement modeling.

2 Case Study

The case study is inspired by the Network Pump®. The Pump enables software applications operating on a lower security network to pass information to software applications on a higher security level network automatically, in a secure and reliable manner. These applications are denoted "Low App" and "High App" in the Pump's SysML model.

In this paper, the SysML model does not attempt to depict exactly the Pump. It rather illustrates how SysML can be used in the analysis phase of a system which is seen as a black box. The outcome of the phase is a comprehensive and consistent set of requirements that the system must fulfill.

3 SysML Models Initiated During the Requirement Elicitation Phase

3.1 Requirement Models

The business and user needs, business rules, system performances, and constraints found out during the elicitation phase are modeled as SysML requirements, each defined by an identifier and a textual specification.

The SysML requirement diagram is the place where requirements can be related. SysML offers seven requirement relationships. The semantics of these relations is not controlled: a requirement can be related to another requirement by any type of requirement relationship. However, links between requirements enforce the consistency of the requirement set.

The "deriveReq" relationship is used to relate a requirement to its source; the "refine" relationship can be used instead. The general-purpose "trace" relationship is used to state that a business rule applies to a requirement.

3.2 Context Models

The stakeholders of the Pump can be depicted in a SysML block definition diagram (BDD). According to ISO 29148, low and high applications — which are external systems connected to the Pump — are not stakeholders, even if they are sources of requirements (interface requirements).

The operational environment of the Pump is depicted by a block definition diagram together with an internal block diagram (IBD), providing both a structure model and an interface model.
