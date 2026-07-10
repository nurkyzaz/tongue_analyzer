# Benchmark Design — what we measure accuracy on, and why

_The goal: report **defensible, expert-grounded accuracy** for this tool. This doc records the decision
of what to benchmark (and what we deliberately do NOT claim). Results live in [BENCHMARK.md](BENCHMARK.md)._

## 1. What to identify — the decision

We considered three candidate targets (per the request): **diseases**, **TCM syndromes/patterns**, or
**TCM constitution (9 body types)**. The deciding criterion was: *for which do we have tongue photos
independently classified by experts?* (Anything else would be circular — grading our own rule-based
output against itself.)

| Candidate | Expert-labeled images we have | Verdict |
|---|---|---|
| Disease (diabetes, MASLD…) | none in our datasets (public sets exist but weren't obtainable) | ❌ can't ground |
| TCM syndrome / 9-constitution | none public & obtainable (the large constitution sets, e.g. 22,482 imgs, stay in-hospital) | ❌ can't ground (see note) |
| **Tongue feature / pathological categories** | **TonguExpert L1 manual (expert-verified) + TCM-Tongue 6,719 practitioner-verified** | ✅ **chosen** |

**Decision:** benchmark the **expert-annotated tongue categories** — the observable "conditions" of the
tongue that professionals actually labelled. This is exactly what the model detects, and it is what the
literature (SSC-Net, TCM-Tongue) reports, so our numbers are directly comparable.

> Why not constitution/disease, honestly: mapping tongue features → a TCM pattern or 9-constitution type
> is our **rule-based educational layer** (grounded in ICD-11 / CCMQ / Maciocia). We have no independent
> expert-labelled constitution/disease images, so reporting an "accuracy" for those would be
> scientifically invalid. Published tongue→constitution models reach ~0.71 accuracy ("junior-practitioner
> level") — informative context, but on private data we can't reproduce.

## 2. How many classes — a focused set

Rather than all 20+ categories (many rare), we benchmark a **focused set of clinically important,
well-supported conditions** — the ones with enough expert-labelled test examples to trust the number:

**Tier 1 — characteristics (expert gold, TonguExpert L1 manual):**
1. Coating color (white / light-yellow / yellow) — Cold vs Heat
2. Tongue body color (pale / normal / red-dark) — Deficiency vs Heat (**the pale→blood/iron axis**)
3. Fissures / cracks (none / light / severe) — Yin/fluid depletion
4. Tooth-marks (none / light / severe) — Spleen-qi deficiency

**Tier 2 — pathological categories (practitioner-labelled, TCM-Tongue) with adequate test support:**
5. Red tongue body — Heat
6. Red dots / prickles — Heat
7. Thin body — Blood/Yin deficiency

Classes with too few held-out positives to benchmark reliably (peeled n=1, grey-black n=3, purple n=18,
swollen, slippery) are **reported with their support count and flagged as low-confidence**, not headlined.

## 3. The test sets (expert-labelled, held out)
- **A. TonguExpert** `L1_Labels_Manual` — human-verified; we evaluate only on the held-out **test** images
  that carry a manual label (a pure expert-vs-model comparison).
- **B. TCM-Tongue** — licensed-practitioner annotations; the official held-out **553-image test split**
  (never seen in training).
Both are larger and more rigorous than hand-labelling "a small number" — they're ready-made expert sets.

## 4. Metrics
- Characteristics: **accuracy** + **macro-F1** + per-class precision/recall (multi-class).
- Pathological categories: **Average Precision** + precision/recall/F1 at the best threshold (multi-label),
  with **support (n positives)** always shown so low-n classes are transparent.

## 5. Literature anchors (for honest comparison)
- SSC-Net: ~0.85 F1 on the 5 characteristics (curated data).
- TCM-Tongue: YOLO detection benchmarks on the same 20 categories.
- Tongue→constitution models: ~0.71 accuracy, "junior-practitioner level" (private data) — context only.

## 6. Path to a constitution/disease benchmark (future)
To honestly report constitution or disease accuracy we would need expert-labelled images for those
targets. Options: partner with a clinic to label a small constitution set (using the validated **CCMQ** +
tongue photos), or obtain a public disease-tongue set (e.g. MASLD/diabetes). Until then, constitution is
presented as an **educational mapping**, not a benchmarked claim.
