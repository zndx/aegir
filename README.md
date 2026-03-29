# Ægir (Aegir)

**Ægir: Hierarchical Sequence Modeling and Dynamic Chunking for Long-Context Efficiency**

Ægir is a fork of H-Net (arXiv:2507.07955) that replaces the main network with RWKV-8, leveraging ROSA for infinite-range lossless retrieval. Encoder and decoder stages retain Mamba-2 for high-resolution compression. The core objective is an efficient hierarchical recurrent model uniting adaptive chunking with RWKV’s linear-time inference and perfect recall.

## Key Features

* **RWKV-8 Backbone with ROSA**: Delivers linear-time inference and lossless long-range memory.
* **Dynamic Chunking from H-Net**: Adaptive, content-dependent hierarchical segmentation.
* **Mamba-2 Encoder/Decoder**: Efficient compression of high-resolution sequences.
* **Selective Context Management**: Optimized for retrieving and verifying relevant data in noisy environments.
* **Specialized Pretraining**: Seeded on FinePDFs-EDU and GLM-5 synthetic dialogue traces.

## Architecture Overview

Ægir employs Mamba-2 for initial encoding and final decoding. H-Net-style dynamic chunking constructs hierarchical representations, processed by an RWKV-8 core enhanced with ROSA. This enables end-to-end modeling of long sequences with constant memory footprint and superior recall.

## Why H-Net + RWKV-8 for Table Metadata Tasks

Wide tables in data warehouses contain many irrelevant columns. The architecture’s adaptive chunking and RWKV-8’s selective recurrent processing excel at context selection, aligning with the Retrieve-and-Verify paradigm (arXiv:2508.17203) for accurate Column Type Annotation (CTA) and Column Property Annotation (CPA).

## Use Cases

* **Primary**: Relational data warehouse metadata tagging, focusing on CTA and CPA for wide tables.
* Long-context document analysis and hierarchical sequence tasks requiring noise-resistant retrieval.

## Current Status

Pre-alpha prototype. Training in progress. Community contributions and feedback welcomed for accelerating development.

## Installation & Quickstart

```bash
git clone https://github.com/zndx/aegir.git
cd aegir
pip install -e .
```

