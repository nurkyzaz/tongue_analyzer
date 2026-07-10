# Design & Research — start here

Everything about how the system is designed, what it's grounded in, and how to improve it, in one
folder. Read in order, or jump to what you need.

| Doc | What's in it |
|---|---|
| **[01_USER_WORKFLOW.md](01_USER_WORKFLOW.md)** | The end-to-end user journey with a diagram, and — step by step — **which model or database produces each output**. Answers "from what database are the diagnoses selected and how are they identified?" **Start here.** |
| **[02_SYSTEM_DESIGN.md](02_SYSTEM_DESIGN.md)** | Architecture, components, every file's role, and the design principles. |
| **[03_KNOWLEDGE_SOURCES.md](03_KNOWLEDGE_SOURCES.md)** | The four reputable databases behind the "diagnoses" (WHO ICD-11, CCMQ, SymMap, Maciocia/Kirschbaum), the KB schema, and a worked example. |
| **[04_DATA_MODELS_RESEARCH.md](04_DATA_MODELS_RESEARCH.md)** | Datasets, model metrics, why outputs were "generic" + the fix, and the Phase-4 roadmap. |

## The one-paragraph summary
A user photographs their tongue. An **image model** (segmentation + a multi-task CNN) detects five
features — coating thickness, coating color, body color, cracks, tooth-marks — each with a **0–1
severity**. Those graded features are fed to a **knowledge layer** that reads an editable, grounded
database (`tcm_knowledge.json`): it describes each feature in **both the TCM term and plain language**,
then **votes (weighted by severity)** toward traditional **patterns** (the WHO-ICD-11-named, CCMQ
9-constitution set). The top pattern is shown with an honest confidence, plain-language associated
symptoms, and specific recommendations, plus optional **follow-up questions** (validated CCMQ items)
that refine confidence. Nothing is a black box: the "diagnosis" is transparent, auditable voting over a
file you can edit — and everything is framed as one tradition's perspective, not a medical diagnosis.

## To improve the system
- **Content / patterns / recommendations / questions** → edit
  [`tcm_knowledge.json`](../../stage2_interpretation/knowledge_base/tcm_knowledge.json) (no retraining). See 03.
- **Detect new features** (purple, red-dots, peeled coating…) → needs data + retraining. See 04 + [../PHASE4_TCM_TONGUE.md](../PHASE4_TCM_TONGUE.md).
- **Sensitivity / wording** → `severity_bands` and `points_to` weights in the KB. See 01 "How to improve it".

## Related deeper docs (repo root `docs/`)
[ARCHITECTURE.md](../ARCHITECTURE.md) · [TCM_RESEARCH.md](../TCM_RESEARCH.md) ·
[IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md) · [RESOURCES.md](../RESOURCES.md) · [PROGRESS.md](../PROGRESS.md)
