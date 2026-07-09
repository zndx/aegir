# End-to-End Example

> **Deferred framing.** This walkthrough traces a passage through
> the earlier exploratory SysMLv2 / ORM pipeline for
> ontology-grounded synthetic data that has been superseded. The
> active synthetic-data pipeline is `just metaflow`
> (`src/aegir/flows/sdg_corpora_flow.py`), documented in the
> [semantic-engine authoritative
> reference](../ontology/production_state.md). This page is
> preserved in the repository for archival continuity.

This walkthrough traces a single educational text passage through the entire pretraining pipeline -- from raw text to a validated training example. Every intermediate representation is shown concretely, making the abstract pipeline tangible.

## Hero Diagram

```d2
direction: down

step1: Step 1: Input Text {
  style.fill: "#fce4ec"
  style.stroke: "#c62828"
  text: "Educational passage about\nhospital patient management"
}

step2: Step 2: Ontology Extraction {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"
  text: "BFO-grounded entities:\nPatient, Encounter, Diagnosis,\nMedication, Provider"
}

step3: Step 3: SysMLv2 Model {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"
  text: "Block definitions, ports,\nstate machines, constraints"
}

step4: Step 4: Data Objects {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"
  text: "Python dataclasses with\ntyped fields and relationships"
}

step5: Step 5: Relational Schema {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"
  text: "5 tables, 23 columns,\n8 foreign keys, 4 constraints"
}

step6: Step 6: Synthetic Data {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"
  text: "200 patients, 1,400 encounters,\n3,200 diagnoses, populated"
}

step7: Step 7: Serialized Input {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"
  text: "Byte-serialized CSV tables\n(what Aegir actually sees)"
}

step8: Step 8: Training Target {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"
  text: "Data element predictions:\nPatientDemographics, ClinicalEncounter,\nDiagnosisRecord, MedicationOrder"
}

step9: Step 9: Validation {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"
  text: "Predicted elements map back\nto source ontology entities ✓"
}

step1 -> step2: "LLM extraction"
step2 -> step3: "openCAESAR mapping"
step3 -> step4: "code generation"
step4 -> step5: "ORM projection"
step5 -> step6: "procedural population"
step6 -> step7: "byte serialization"
step7 -> step8: "Aegir prediction"
step8 -> step9: "compare against\nsource ontology"
```

---

## Step 1: Input Text

A passage from a PDF about hospital emergency department workflows:

> Emergency departments manage patient flow through a structured triage process. When a patient arrives, a triage nurse assesses their condition and assigns an acuity level using the Emergency Severity Index (ESI), ranging from 1 (resuscitation) to 5 (non-urgent). Each patient encounter records the presenting complaint, vital signs at triage, the assigned provider, and any diagnostic tests ordered.
>
> Diagnoses are coded using ICD-10-CM, with a primary diagnosis and optional secondary diagnoses recorded per encounter. Medications prescribed during the encounter are tracked with the drug name, dosage, route of administration, and the prescribing provider. The encounter concludes with a disposition decision: admission, discharge, transfer, or observation.

This is a typical educational passage: clear, structured, and rich in implicit ontological content.

## Step 2: Ontology Extraction

The LLM receives the passage with a structured extraction prompt and produces a BFO-grounded ontology fragment:

**Classes** (with BFO alignment):

| Class | BFO Parent | Properties |
|-------|-----------|------------|
| `Patient` | `BFO:Object` | patient_id, date_of_birth, gender, address |
| `Encounter` | `BFO:Process` | encounter_id, encounter_date, presenting_complaint, disposition |
| `Provider` | `BFO:Role` | provider_id, name, specialty, license_number |
| `Diagnosis` | `BFO:GDC` | diagnosis_id, icd10_code, description, is_primary |
| `Medication` | `BFO:GDC` | medication_id, drug_name, dosage, route |
| `VitalSigns` | `BFO:Quality` | heart_rate, blood_pressure, temperature, respiratory_rate, spo2 |
| `AcuityLevel` | `BFO:Quality` | esi_level (1--5) |

**Relations**:

| Relation | Domain | Range | Cardinality |
|----------|--------|-------|-------------|
| `hasEncounter` | Patient | Encounter | 1..* |
| `hasProvider` | Encounter | Provider | 1..1 |
| `hasDiagnosis` | Encounter | Diagnosis | 1..* |
| `hasMedication` | Encounter | Medication | 0..* |
| `hasVitalSigns` | Encounter | VitalSigns | 1..1 |
| `hasAcuity` | Encounter | AcuityLevel | 1..1 |
| `prescribedBy` | Medication | Provider | 1..1 |

**Axioms**:
- `Encounter.disposition ∈ {admission, discharge, transfer, observation}`
- `AcuityLevel.esi_level ∈ {1, 2, 3, 4, 5}`
- `Diagnosis.is_primary` is unique per Encounter (exactly one primary diagnosis)

## Step 3: SysMLv2 Model

The ontology maps to SysMLv2 block definitions:

```d2
direction: right

patient: part def Patient {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  attrs: "patient_id: String\ndate_of_birth: Date\ngender: String\naddress: String"
}

encounter: part def Encounter {
  style.fill: "#e8f4f8"
  style.stroke: "#1565c0"

  attrs: "encounter_id: String\nencounter_date: DateTime\npresenting_complaint: String\ndisposition: DispositionEnum"
  state: "state machine:\n  entry → triage → treatment\n  → disposition → closed"
}

diagnosis: part def Diagnosis {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  attrs: "diagnosis_id: String\nicd10_code: String\ndescription: String\nis_primary: Boolean"
}

medication: part def Medication {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  attrs: "medication_id: String\ndrug_name: String\ndosage: String\nroute: RouteEnum"
}

provider: part def Provider {
  style.fill: "#fff3e0"
  style.stroke: "#e65100"

  attrs: "provider_id: String\nname: String\nspecialty: String\nlicense_number: String"
}

patient -> encounter: "1..*\nhasEncounter" {style.stroke: "#1565c0"}
encounter -> diagnosis: "1..*\nhasDiagnosis" {style.stroke: "#6a1b9a"}
encounter -> medication: "0..*\nhasMedication" {style.stroke: "#6a1b9a"}
encounter -> provider: "1..1\nhasProvider" {style.stroke: "#e65100"}
medication -> provider: "1..1\nprescribedBy" {style.stroke: "#e65100"}
```

The SysMLv2 model adds lifecycle semantics (the Encounter state machine: entry → triage → treatment → disposition → closed) and formal constraints that the flat ontology fragment does not capture.

## Step 4: Python Data Objects

```python
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

class Disposition(Enum):
    ADMISSION = "admission"
    DISCHARGE = "discharge"
    TRANSFER = "transfer"
    OBSERVATION = "observation"

class Route(Enum):
    ORAL = "oral"
    IV = "intravenous"
    IM = "intramuscular"
    TOPICAL = "topical"
    INHALED = "inhaled"

@dataclass
class Patient:
    patient_id: str
    date_of_birth: date
    gender: str
    address: str

@dataclass
class Provider:
    provider_id: str
    name: str
    specialty: str
    license_number: str

@dataclass
class Encounter:
    encounter_id: str
    patient_id: str           # FK → Patient
    provider_id: str          # FK → Provider
    encounter_date: datetime
    presenting_complaint: str
    esi_level: int            # 1-5
    disposition: Disposition
    heart_rate: int
    blood_pressure: str
    temperature: float
    respiratory_rate: int
    spo2: int

@dataclass
class Diagnosis:
    diagnosis_id: str
    encounter_id: str         # FK → Encounter
    icd10_code: str
    description: str
    is_primary: bool

@dataclass
class Medication:
    medication_id: str
    encounter_id: str         # FK → Encounter
    prescribed_by: str        # FK → Provider
    drug_name: str
    dosage: str
    route: Route
```

Note that `VitalSigns` and `AcuityLevel` (BFO:Quality entities) have been denormalized into the `Encounter` table -- a deliberate schema variation that the model must learn to handle. In a different schema variant, these would be separate tables.

## Step 5: Relational Schema

```sql
CREATE TABLE patient (
    patient_id     VARCHAR(36) PRIMARY KEY,
    date_of_birth  DATE NOT NULL,
    gender         VARCHAR(10) NOT NULL,
    address        TEXT
);

CREATE TABLE provider (
    provider_id    VARCHAR(36) PRIMARY KEY,
    name           VARCHAR(100) NOT NULL,
    specialty      VARCHAR(50) NOT NULL,
    license_number VARCHAR(20) NOT NULL UNIQUE
);

CREATE TABLE encounter (
    encounter_id        VARCHAR(36) PRIMARY KEY,
    patient_id          VARCHAR(36) NOT NULL REFERENCES patient(patient_id),
    provider_id         VARCHAR(36) NOT NULL REFERENCES provider(provider_id),
    encounter_date      TIMESTAMP NOT NULL,
    presenting_complaint TEXT NOT NULL,
    esi_level           INTEGER NOT NULL CHECK (esi_level BETWEEN 1 AND 5),
    disposition         VARCHAR(20) NOT NULL
                        CHECK (disposition IN ('admission','discharge','transfer','observation')),
    heart_rate          INTEGER,
    blood_pressure      VARCHAR(10),
    temperature         NUMERIC(4,1),
    respiratory_rate    INTEGER,
    spo2                INTEGER CHECK (spo2 BETWEEN 0 AND 100)
);

CREATE TABLE diagnosis (
    diagnosis_id VARCHAR(36) PRIMARY KEY,
    encounter_id VARCHAR(36) NOT NULL REFERENCES encounter(encounter_id),
    icd10_code   VARCHAR(10) NOT NULL,
    description  TEXT,
    is_primary   BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (encounter_id, is_primary) -- at most one primary per encounter
);

CREATE TABLE medication (
    medication_id VARCHAR(36) PRIMARY KEY,
    encounter_id  VARCHAR(36) NOT NULL REFERENCES encounter(encounter_id),
    prescribed_by VARCHAR(36) NOT NULL REFERENCES provider(provider_id),
    drug_name     VARCHAR(100) NOT NULL,
    dosage        VARCHAR(50) NOT NULL,
    route         VARCHAR(20) NOT NULL
);
```

## Step 6: Synthetic Data

Sample rows from the populated tables:

**patient** (200 rows):

| patient_id | date_of_birth | gender | address |
|-----------|--------------|--------|---------|
| `a3f8c1d0-...` | 1987-03-15 | Female | 2847 Oak Ave, Portland, OR 97205 |
| `b7e2a4f1-...` | 1952-11-28 | Male | 156 Pine St, Austin, TX 78701 |
| `c9d0b3e2-...` | 2001-07-04 | Female | 4021 Maple Dr, Denver, CO 80202 |

**encounter** (1,400 rows, ~7 per patient):

| encounter_id | patient_id | provider_id | encounter_date | presenting_complaint | esi_level | disposition | heart_rate | blood_pressure | temperature |
|-------------|-----------|------------|---------------|---------------------|----------|------------|-----------|---------------|------------|
| `e1a2b3c4-...` | `a3f8c1d0-...` | `p001-...` | 2024-01-15 14:30 | Acute chest pain | 2 | admission | 98 | 145/92 | 98.6 |
| `e5f6a7b8-...` | `b7e2a4f1-...` | `p003-...` | 2024-02-03 09:15 | Laceration, left hand | 4 | discharge | 72 | 128/78 | 98.2 |

**diagnosis** (3,200 rows, ~2.3 per encounter):

| diagnosis_id | encounter_id | icd10_code | description | is_primary |
|-------------|-------------|-----------|-------------|-----------|
| `d100-...` | `e1a2b3c4-...` | I21.9 | Acute myocardial infarction, unspecified | true |
| `d101-...` | `e1a2b3c4-...` | I10 | Essential hypertension | false |
| `d200-...` | `e5f6a7b8-...` | S61.412A | Laceration without FB, left hand | true |

**medication** (2,100 rows):

| medication_id | encounter_id | prescribed_by | drug_name | dosage | route |
|-------------|-------------|-------------|-----------|--------|-------|
| `m100-...` | `e1a2b3c4-...` | `p001-...` | Aspirin | 325mg | oral |
| `m101-...` | `e1a2b3c4-...` | `p001-...` | Heparin | 5000 units | intravenous |
| `m200-...` | `e5f6a7b8-...` | `p003-...` | Lidocaine | 1% 5mL | topical |

## Step 7: Serialized Input

Aegir receives the tables as byte-serialized CSV data. Here's what the model actually sees (abbreviated):

```
patient_id,date_of_birth,gender,address
a3f8c1d0-7b2e-4a1f-9c3d-e5f6a7b8c9d0,1987-03-15,Female,"2847 Oak Ave, Portland, OR 97205"
b7e2a4f1-3c5d-4e6f-8a9b-c0d1e2f3a4b5,1952-11-28,Male,"156 Pine St, Austin, TX 78701"
c9d0b3e2-1a4f-4c7d-9e2b-f3a5b6c7d8e9,2001-07-04,Female,"4021 Maple Dr, Denver, CO 80202"
...
===TABLE_BOUNDARY===
encounter_id,patient_id,provider_id,encounter_date,presenting_complaint,esi_level,disposition,heart_rate,blood_pressure,temperature,respiratory_rate,spo2
e1a2b3c4-5d6e-4f7a-8b9c-0d1e2f3a4b5c,a3f8c1d0-7b2e-4a1f-9c3d-e5f6a7b8c9d0,p001-a2b3-c4d5,2024-01-15 14:30:00,Acute chest pain,2,admission,98,145/92,98.6,20,97
...
===TABLE_BOUNDARY===
diagnosis_id,encounter_id,icd10_code,description,is_primary
d100-e1f2-a3b4-c5d6,e1a2b3c4-5d6e-4f7a-8b9c-0d1e2f3a4b5c,I21.9,"Acute myocardial infarction, unspecified",true
...
```

The model sees raw bytes. No type annotations, no foreign key declarations, no semantic metadata -- just the patterns in the data itself.

## Step 8: Training Target

The expected predictions for this training example:

**CTA predictions** (per column):

| Table | Column | Expected Type | BFO Category |
|-------|--------|--------------|-------------|
| patient | patient_id | PersonIdentifier | GDC |
| patient | date_of_birth | BirthDate | Quality |
| patient | gender | BiologicalSex | Quality |
| encounter | encounter_id | EncounterIdentifier | GDC |
| encounter | patient_id | PersonIdentifier (FK) | GDC |
| encounter | esi_level | AcuityLevel | Quality |
| encounter | disposition | DispositionDecision | Quality |
| encounter | heart_rate | VitalSign | Quality |
| diagnosis | icd10_code | DiagnosisCode | GDC |
| diagnosis | is_primary | PrimaryIndicator | Quality |
| medication | drug_name | MedicationName | GDC |
| medication | dosage | Dosage | Quality |
| medication | route | AdministrationRoute | Quality |

**Data element predictions** (cross-table clusters):

| Data Element | Columns | Source Entity |
|-------------|---------|--------------|
| **PatientDemographics** | patient.patient_id, patient.date_of_birth, patient.gender, patient.address, encounter.patient_id | `Patient` |
| **ClinicalEncounter** | encounter.encounter_id, encounter.encounter_date, encounter.presenting_complaint, encounter.esi_level, encounter.disposition, encounter.heart_rate, encounter.blood_pressure, encounter.temperature | `Encounter` + `VitalSigns` + `AcuityLevel` |
| **DiagnosisRecord** | diagnosis.diagnosis_id, diagnosis.encounter_id, diagnosis.icd10_code, diagnosis.description, diagnosis.is_primary | `Diagnosis` |
| **MedicationOrder** | medication.medication_id, medication.encounter_id, medication.drug_name, medication.dosage, medication.route | `Medication` |
| **ClinicalProvider** | provider.provider_id, provider.name, provider.specialty, provider.license_number, encounter.provider_id, medication.prescribed_by | `Provider` |

Note that the **PatientDemographics** data element spans `patient.patient_id` *and* `encounter.patient_id` -- cross-table discovery. Similarly, **ClinicalProvider** spans columns in three tables (provider, encounter, medication). This is exactly the cross-table data element discovery that enterprise data governance requires.

## Step 9: Validation

The round-trip check confirms that predicted data elements map back to source ontological entities:

```d2
direction: right

predicted: Predicted Data Elements {
  style.fill: "#e8f8e8"
  style.stroke: "#2e7d32"

  pd: "PatientDemographics\n  patient.patient_id\n  patient.date_of_birth\n  patient.gender\n  encounter.patient_id"
  ce: "ClinicalEncounter\n  encounter.encounter_id\n  encounter.esi_level\n  encounter.heart_rate\n  ..."
  dr: "DiagnosisRecord\n  diagnosis.icd10_code\n  diagnosis.is_primary\n  ..."
  mo: "MedicationOrder\n  medication.drug_name\n  medication.dosage\n  ..."
  cp: "ClinicalProvider\n  provider.provider_id\n  encounter.provider_id\n  medication.prescribed_by"
}

source: Source Ontology Entities {
  style.fill: "#f0e8f8"
  style.stroke: "#6a1b9a"

  patient: "Patient ⊑ BFO:Object"
  encounter: "Encounter ⊑ BFO:Process"
  diagnosis: "Diagnosis ⊑ BFO:GDC"
  medication: "Medication ⊑ BFO:GDC"
  provider: "Provider ⊑ BFO:Role"
}

predicted.pd -> source.patient: "✓ match" {style.stroke: "#2e7d32"}
predicted.ce -> source.encounter: "✓ match" {style.stroke: "#2e7d32"}
predicted.dr -> source.diagnosis: "✓ match" {style.stroke: "#2e7d32"}
predicted.mo -> source.medication: "✓ match" {style.stroke: "#2e7d32"}
predicted.cp -> source.provider: "✓ match" {style.stroke: "#2e7d32"}
```

Every predicted data element corresponds to exactly one source ontology entity. The ClinicalEncounter element correctly groups encounter properties with the denormalized VitalSigns and AcuityLevel qualities -- demonstrating that the model learned to see through the denormalization to the underlying ontological structure.

This validation is **automatic and exact** because the generation pipeline preserves complete provenance. There is no human labeling, no ambiguity, and no annotation disagreement. The ground truth is a mathematical consequence of the generation process.

## What This Means in Practice

When this training process is applied at scale -- across hundreds of millions of passages spanning every domain in FineWeb-Edu -- the model learns:

1. **Column type recognition** that generalizes across naming conventions, data formats, and serialization styles
2. **Cross-table relationship discovery** that identifies semantically related columns regardless of which tables they appear in
3. **Ontological hierarchy** that connects specific types (ICD-10 codes) to general categories (information entities) through BFO's formal structure
4. **Confusable type resolution** by leveraging cross-column context (patient_id vs provider_id look identical in isolation but participate in different relationship patterns)

These capabilities transfer directly to real enterprise data warehouses, where the model encounters the same patterns -- just without the luxury of knowing the ontological provenance in advance.
