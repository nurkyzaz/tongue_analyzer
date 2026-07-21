# Tongue Analyzer — Market, Accuracy & Regulatory Q&A (for the CEO)

_Updated 2026-07-21. Short answers, every external claim linked. A decision memo, not a launch commitment._

**Decision frame.** Include the tongue feature in Savor **iff**: it's accurate on what it actually claims,
it makes **no** medical/diagnosis claims, it rejects bad photos, and it plugs into
constitution/food/seasonality the TCM-correct way. Status below: **all four hold** for an
educational/wellness feature. The open question isn't "does anyone want this" (they do) — it's "does it lift
*Savor's* engagement," which only an in-app test answers.

---

## 1. Market — is the need real, and in which country?

**Yes, and it concentrates in TCM-affinity Asia; it's curiosity-grade in the West.**

- **China (Tier 1, strongest signal):** live commercial M&A and hospital roll-outs of AI tongue diagnosis,
  plus a state-backed "smart TCM" push
  ([market report](https://www.wiseguyreports.com/reports/tcm-tongue-diagnosi-system-market) ·
  [gov push, ecns.cn](https://www.ecns.cn/cns-wire/2026-06-18/detail-ihffptmh9488055.shtml)). Biggest market, hardest to enter.
- **Singapore / HK / Taiwan (Tier 1, best fit for us):** Aiteaic's **"Luli"** scans **face + tongue →
  constitution in <30 s**, API at **~US$0.01/scan**
  ([Taiwan News](https://www.taiwannews.com.tw/news/6392348) ·
  [PR Newswire](https://en.prnasia.com/releases/apac/singapore-healthtech-aiteaic-unveils-proprietary-tcm-wellness-system-pairing-ai-tongue-scanning-with-nanobubble-extraction-539215.shtml)).
  This is the closest analog to a Savor tongue feature and validates the **wellness (not medical)** framing.
- **Standing consumer app category:** e.g. [MyZenCheck](https://www.myzencheck.net/blog/myzencheck-user-demographics-november-2025/)
  (AI tongue reading, publishes global user demographics),
  [Tongue Analyzer](https://apps.apple.com/us/app/tongue-analyzer-ai-health/id6751777361) (US indie, thin traction),
  QiScan (symptoms + tongue → TCM pattern).
- **West:** interest is real but **curiosity-driven** (viral tongue content on Reddit/TikTok/小红书), not need-driven.

**Cheapest way to validate for Savor:** a fake-door / waitlist tile ("舌 tongue reading") vs. the existing
constitution quiz, run on **Singapore/HK users first**, and measure tap-through.

---

## 2. Accuracy — is it "reasonably accurate (>80%)," grounded in real TCM?

**Yes for feature *detection*; and we deliberately publish NO number for "diagnosis" (none is measurable).**

- **Our held-out benchmark** vs. expert/practitioner labels: coating-colour **0.92**, cracks **0.92**,
  body-colour **0.81**, tooth-marks **0.85** (mean core-4 **~0.87**), and **97% within one grade**
  (`docs/BENCHMARK.md`). Extra signs are **honestly validated** on 553 practitioner images and kept only
  where they work (red_dots AP 0.68, red_tongue 0.61…), with weak ones down-weighted and undetectable ones
  removed (`docs/VALIDATION_WORKLIST.md`).
- **Literature agrees this range is real** for phone tongue models — "Deep Tongue" reports **AUC ~0.90**
  across sub-tasks ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.02.02.526804v1.full)).
- **Why no "diagnosis %":** even human TCM experts reach ≥80% agreement on only **~17–19%** of tongue
  features ([PubMed 18564955](https://pubmed.ncbi.nlm.nih.gov/18564955/)) — there's no gold standard to
  score against. The flip side is our advantage: a standardized model can be **more consistent than the
  average practitioner**.
- **Grounded in real TCM:** interpretation is built on a knowledge graph from **licensed texts + WHO IST
  2022 + the validated CCMQ questionnaire + ICD-11**, with **282 book-cited feature→pattern edges, 0
  hallucinated** — every reading is traceable.

---

## 3. Are we making false disease / medical-condition claims? **No.**

We **never name a disease.** The output is (a) **visible features** (coating, colour, cracks, tooth-marks…)
and (b) **TCM "pattern/constitution leanings"** (证/体质 — e.g. "damp-heat leaning"), always framed
***"traditionally associated with…"***, never "you have X." "Damp-heat" is a traditional construct, not a
biomedical diagnosis; we make no claim about diabetes, anemia, thyroid, etc.

---

## 4. Does it reject bad images (blur / bad lighting)? **Yes — implemented.**

Two gates run *before* a reading is trusted (`deployment/api/service.py`):
1. **Framing gate** — no tongue / tongue too small → hard refusal, no reading.
2. **Capture-quality gate** — **blur** (variance-of-Laplacian) + **exposure/glare** (luma + clipping) →
   refuses "too dark / over-exposed / blurry, hold steady."

This matters because **lighting is the #1 phone-accuracy killer**
([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169260717308477); smartphone
reliability is only *moderate* even in controlled studies —
[JMIR mHealth](https://mhealth.jmir.org/2020/7/e16018)). Rejecting bad photos is the single biggest
reliability lever, and it also directly satisfies Apple (below).

---

## 5. Is there a real TCM method to combine tongue with constitution / food / seasonality? **Yes.**

The key idea: **constitution (体质) = your stable baseline; tongue (证) = your state *today*.**

- **Tongue is never used alone in TCM** — determination combines tongue + pulse + inquiry
  ([AIMIN](https://aimin.com.sg/insights/posts/eating-for-your-tcm-constitution-a-personalised-food-guide)).
  So the tongue photo is one **input** to the constitution estimate, not a standalone verdict — that *is*
  the method, not a limitation.
- **Wired today:** tongue features → pattern leaning → a **CCMQ 9-constitution crosswalk**
  (`interpret.py`) → feeds Savor's existing **constitution / food-compatibility / seasonality**. A **1–2
  question CCMQ follow-up** firms up the constitution (CCMQ is proven compressible by **−68.3%** of
  questions — [PMC7676967](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7676967/)), and constitution→diet
  is evidenced ([PMC12295300](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12295300/)).
- **The daily hook:** the pipeline emits **今日宜/忌 (do / avoid)** per reading — e.g. baseline says
  Qi-deficient, but today's greasy coating adds "go lighter on rich/fried food today." Tongue makes the
  existing food/seasonality features **update daily** instead of being a one-time quiz result.
- **Constraint from our own data:** pure constitution info tested **low-need**, so tongue must be the
  low-friction, shareable **on-ramp** to food/seasonality — not another constitution screen.

---

## 6. Apple — will it be rejected for medical/diagnosis advice? **No, because we make none.**

A diagnosis claim would get rejected and could trigger medical-device regulation. The
**educational/wellness** design we've built is the standard compliant path. Apple's
[App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/) **1.4.1** scrutinize
diagnostic apps and **reject unvalidated accuracy claims**; **5.1.3** governs health data. We stay clear by:
**no disease/diagnosis/treatment claims**, **rejecting bad images**, a **"consult a professional"
disclaimer** on every reading, and we **can disclose our methodology + benchmark** if asked (exactly what
1.4.1 wants). **The line we never cross — in the app or the App Store copy — is disease/diagnosis/treatment
language.**

---

## Two honest ship-blockers
- **Licensing:** the segmentation model was trained partly on **SM-Tongue (CC-BY-NC, non-commercial)** —
  retrain on commercially-clean data before charging money.
- **Copy discipline:** one careless "detects your liver problem" line in marketing undoes the whole
  compliance posture.

## Sources
Market: [WiseGuy report](https://www.wiseguyreports.com/reports/tcm-tongue-diagnosi-system-market) ·
[ecns.cn](https://www.ecns.cn/cns-wire/2026-06-18/detail-ihffptmh9488055.shtml) ·
[Taiwan News](https://www.taiwannews.com.tw/news/6392348) ·
[PR Newswire](https://en.prnasia.com/releases/apac/singapore-healthtech-aiteaic-unveils-proprietary-tcm-wellness-system-pairing-ai-tongue-scanning-with-nanobubble-extraction-539215.shtml) ·
[MyZenCheck](https://www.myzencheck.net/blog/myzencheck-user-demographics-november-2025/) ·
[Tongue Analyzer](https://apps.apple.com/us/app/tongue-analyzer-ai-health/id6751777361).
Accuracy: [Deep Tongue AUC~0.90 (bioRxiv)](https://www.biorxiv.org/content/10.1101/2023.02.02.526804v1.full) ·
[automated tongue dx / lighting (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169260717308477) ·
[expert-agreement study (PubMed 18564955)](https://pubmed.ncbi.nlm.nih.gov/18564955/) ·
[smartphone reliability (JMIR)](https://mhealth.jmir.org/2020/7/e16018).
Integration: [tongue+pulse+inquiry (AIMIN)](https://aimin.com.sg/insights/posts/eating-for-your-tcm-constitution-a-personalised-food-guide) ·
[CCMQ −68.3% (PMC7676967)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7676967/) ·
[constitution↔diet (PMC12295300)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12295300/).
Regulatory: [Apple App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/).
Internal: `docs/BENCHMARK.md`, `docs/VALIDATION_WORKLIST.md`, `docs/FEATURE_PATTERN_MAPPING.md`,
`deployment/api/service.py`, `stage2_interpretation/interpret.py`.
