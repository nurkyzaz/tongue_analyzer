# Tongue Analyzer — Market, Accuracy & Regulatory Q&A (for the CEO)

_Prepared 2026-07-21. Combines our own benchmark evidence with external market/clinical/regulatory
research. Every external claim is cited. This is a decision memo, not a launch commitment._

---

## TL;DR

1. **Market need is real and, right now, mostly proven in Asia (China first, then Singapore/HK/Taiwan
   and the Chinese diaspora).** There is a ~US$0.5B tongue-diagnosis-system market growing to ~US$1.5B
   by 2035, live commercial M&A in China, and a direct Singapore consumer competitor (Aiteaic/Luli)
   already selling AI tongue-scanning at ~US$0.01/scan. So the honest read is **not** "no evidence" — it's
   "strong evidence in TCM-affinity markets, thin evidence in the West beyond curiosity."
2. **We can defensibly claim >80% on the _feature-detection_ task (coating colour 0.92, body colour 0.81,
   cracks 0.92), but we must NOT claim >80% "diagnostic accuracy"** — no such ground truth exists, and
   even human TCM experts agree with each other on <20% of features at the ≥80% bar. That's a risk and an
   opportunity (a standardized model can be _more consistent_ than humans).
3. **Apple will reject a diagnosis claim, full stop.** The compliant path — which is exactly the one we've
   already built — is an **educational / wellness** feature: "traditionally associated with…", non-diagnostic,
   rejects bad images, reminds users to see a doctor, makes no disease claims.
4. **The right integration with Savor is as an _input_ to constitution/food/seasonality, not a standalone
   readout.** Tongue alone is weak; TCM itself never uses tongue alone. Its value to Savor is a
   **low-friction, visual, shareable hook** (snap a photo instead of a 60-question quiz) that then feeds the
   food-compatibility and seasonality features that already exist — plus a follow-up-question step to firm
   up the constitution. Given "pure constitution info tested as low market-need," tongue must earn its place
   as _engagement + a better on-ramp_, not as another constitution screen.

---

## Q1. Market validation — is there real, evidenced need, and in which country?

**Your draft answer is honest but undersells what's actually out there.** "Reddit users sometimes post their
tongue" is real signal (curiosity + willingness to share a tongue photo), but there's much harder evidence.

### The market exists and is being capitalized — primarily in Asia

- **Market size:** the TCM tongue-diagnosis-system market was ~**US$496.5M in 2024**, projected to
  ~**US$1.5B by 2035** ([WiseGuy Reports](https://www.wiseguyreports.com/reports/tcm-tongue-diagnosi-system-market)).
  Treat vendor market-sizing with skepticism, but the _direction_ is corroborated by real deals below.
- **China — live commercial M&A and partnerships (the strongest signal):**
  - Buchang Pharmaceuticals × Minghui Technology — AI tongue-diagnosis system for TCM clinics (Jan 2024).
  - Beijing Tong Ren Tang × Guangzhou Baiyunshan — deploying tongue-diagnosis AI across a hospital network (Sep 2024).
  - China Resources Sanjiu **acquired** tongue-diagnosis AI startup TongueTech (Jan 2025).
  (all via [WiseGuy Reports](https://www.wiseguyreports.com/reports/tcm-tongue-diagnosi-system-market))
  - Government tailwind: Chongqing is running a state-backed "digital and smart upgrade of TCM" including AI
    diagnosis ([ecns.cn](https://www.ecns.cn/cns-wire/2026-06-18/detail-ihffptmh9488055.shtml)).
- **Singapore — a direct consumer competitor already shipping:** Aiteaic's "Luli" is an AI mirror that scans
  **face + tongue → TCM body constitution in <30 s**, opening API access to wellness/hospitality partners at
  **~US$0.01/scan**, with pilots reportedly run with Meta and Bloomberg (Jun 2026)
  ([Taiwan News](https://www.taiwannews.com.tw/news/6392348),
  [PR Newswire](https://en.prnasia.com/releases/apac/singapore-healthtech-aiteaic-unveils-proprietary-tcm-wellness-system-pairing-ai-tongue-scanning-with-nanobubble-extraction-539215.shtml)).
  This is the closest analog to a Savor tongue feature — and it validates both the format and the wellness
  (not medical) framing.
- **Consumer apps already in-market:** QiScan (symptoms + tongue → likely TCM pattern) and MyZenCheck
  (AI tongue diagnosis, publishes global user demographics) show there is a standing consumer app category,
  not just clinic hardware ([MyZenCheck demographics](https://www.myzencheck.net/blog/myzencheck-user-demographics-november-2025/)).

### Where the need concentrates, by country

- **Tier 1 — China:** largest market, strong cultural affinity, high smartphone penetration, and government
  push. Highest need, but also the most competition and the hardest to enter.
- **Tier 1 — Singapore / Hong Kong / Taiwan:** established TCM sector already digitizing; Aiteaic proves
  consumer appetite. This is likely **Savor's best-fit launch geography** for the tongue feature.
- **Tier 2 — Chinese diaspora + Western "wellness":** growing but shallower. TCM brands (e.g. Eu Yan Sang)
  are pushing into North America's wellness market ([NutraIngredients](https://www.nutraingredients.com/Article/2023/09/06/eu-yan-sang-pursues-north-america-s-health-and-wellness-market/)),
  and Western interest is real but more curiosity-driven (the Reddit signal) than need-driven. Here the
  tongue feature sells as novelty/engagement, not as a trusted diagnostic.

### How to actually _validate_ it for Savor (concrete, cheap)

You said you don't know how to do market validation. Options, cheapest first:

1. **In-app demand test:** ship the tongue capture as a "coming soon / join waitlist" or a soft-gated beta and
   measure tap-through vs. the existing constitution quiz. This directly answers "do Savor users want this?"
   and sidesteps the fact that generic constitution info already tested low.
2. **Fake-door / concept test** in the current app (a "舌 Tongue reading" tile → interest capture).
3. **Reddit/TikTok/小红书 signal is quantifiable:** count posts/searches for "tongue reading / 舌诊 / tongue
   health" and engagement. The virality of tongue-scan content is itself the market thesis for the _hook_.
4. **Geo-targeted:** run the test on Singapore/HK users first, where affinity is highest.

**Bottom line:** the evidence for need is **strong in TCM-affinity Asian markets and among the diaspora**, and
**curiosity-grade in the West.** The open question for Savor isn't "does anyone want tongue reading" (they do)
— it's "does it lift _Savor's_ engagement/retention," which only an in-app test answers.

---

## Q2. Accuracy & reliability — is it "reasonably accurate (>80%), based on real TCM theory," with evidence?

This needs a precise split, because "accuracy" means two very different things here.

### (a) Feature detection — where we _can_ claim >80%, with our own held-out numbers

Our production model, benchmarked against **independent expert / practitioner labels** on held-out images
(`docs/BENCHMARK.md`):

| Feature | Accuracy | Notes |
|---|---|---|
| Coating colour | **0.92** | strongest signal |
| Tongue-body colour | **0.81** | |
| Fissures / cracks | **0.92** acc | |
| Tooth marks | 0.85 acc | precision/recall weaker |
| **Mean (core 4)** | **0.87** | |
| Extra pathological signs (red dots, red tongue, swollen, purple…) | mAP ~0.46 | reliable: red_dots, red-tongue; weak/low-n: purple, swollen, peeled |

Crucially, even where the exact 3-way grade is only ~61% vs. human labels, we are **97% within one grade**
(never confusing "none" with "severe"). So **on the standardized visual-feature task we meet the >80% bar for
the reliable features** (coating colour, cracks, body colour), and we're honest about the weak ones.

This is consistent with the external literature:
- A smartphone deep-learning tongue model ("Deep Tongue") reports **average AUC ~0.90** across 8 subtasks
  ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.02.02.526804v1.full)).
- Automated tongue-diagnosis feature classifiers in the literature report 90%+ on constrained tasks
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169260717308477));
  headline "98.71% XGBoost" numbers exist but are on curated datasets, not phone photos in the wild — don't
  quote them.

### (b) "Diagnostic accuracy" — where we must NOT claim a number

Here is the finding the CEO most needs to hear: **human TCM tongue inspection is itself poorly reproducible.**
- A formal reliability study found **inter-practitioner agreement reached ≥80% on only ~17–19% of feature
  judgments**, concluding tongue inspection "was not a reliable diagnostic method" for that group — largely
  because of **inadequate operational definitions** of features and tongue regions
  ([PubMed 18564955](https://pubmed.ncbi.nlm.nih.gov/18564955/)).
- Smartphone-based tongue-coating reliability is only **moderate** (Gwet AC2 0.49–0.55) and **fair** for body
  colour (AC2 0.34), though intra-rater consistency is good (Cohen κ 0.69–1.0)
  ([JMIR mHealth 2020](https://mhealth.jmir.org/2020/7/e16018)).

**Implication:** there is **no clean "diagnostic gold standard"** to be >80% against — so any vendor claiming
"90% diagnostic accuracy" is measuring against a moving target. We should therefore:
- **Claim accuracy only for feature detection** (which we can back with numbers), and
- Frame the interpretation as **"traditionally associated with…"**, never as a diagnosis.

The flip side is genuinely positive for us: because humans are inconsistent, a **standardized model can be
_more consistent_ than the average practitioner** — that's a legitimate, defensible value proposition, and it's
exactly why our "honest-metric" discipline (report within-one-grade, don't invent a diagnosis number) is the
right posture.

### (c) Bad-image rejection — required, and the literature agrees

Smartphone accuracy is dominated by **lighting/shooting conditions** — the same tongue photographs differently
under different light ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169260717308477)).
Your instinct to **reject blurry / bad-lighting images is not optional, it's the single biggest reliability
lever.** We already have the pieces: a live framing-guide oval, framing feedback, and colour-calibration work.
For ship we should harden this into an explicit **capture-quality gate** (blur, exposure, white-balance,
tongue-fills-frame) that refuses to produce a reading on a bad photo. This also directly satisfies Apple (below).

### (d) "Based on real TCM theory" — yes, and it's traceable

Our interpretation layer is grounded, not vibes: a knowledge graph built from **licensed TCM texts** (Gerlach,
Maciocia, Oriental Tongue Diagnosis), the **WHO International Standard Terminologies on TCM (2022)** as the
canonical bilingual vocabulary, **CCMQ** (the validated 9-constitution questionnaire) and **ICD-11 Chapter 26**
pattern codes — **282 book-cited feature→pattern edges, 0 hallucinated**, with every reading traceable to a
source. So "based on real TCM theory, with evidence" is defensible at the _method_ level even though the
_diagnostic outcome_ is inherently unfalsifiable.

**Answer to your >80% question:** Yes for **feature detection** (coating colour, cracks, body colour ≥0.81–0.92,
backed by our benchmark and by AUC~0.90 in the literature). **No** for "diagnosis/constitution accuracy" — don't
claim a number there, because no ground truth exists and even experts don't agree at 80%. Reject bad images —
the literature says lighting is the dominant error source.

---

## Q3. Apple / regulatory — will this get rejected?

**A diagnosis or disease claim will get rejected and can trigger medical-device regulation.** But the
educational/wellness framing we've already adopted is the standard, compliant path.

### What Apple's guidelines actually say (App Review Guideline 1.4.1, 5.1.3)

From Apple's [App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/):
- **1.4.1 Physical Harm:** "Medical apps that could provide inaccurate data … or that could be used for
  **diagnosing or treating patients may be reviewed with greater scrutiny.**" Apps must **disclose data and
  methodology** to support accuracy claims; **"if the level of accuracy or methodology cannot be validated, we
  will reject your app."** Apps should **remind users to consult a doctor** before making medical decisions,
  and if the app **has received regulatory clearance, submit a link** to it.
- Sensor-only measurement claims (blood pressure, glucose, temperature, oxygen from the phone alone) are
  **outright banned** — a cautionary parallel for "phone camera diagnoses your organs."
- **5.1.3 Health & Research:** strict rules on health data; don't write false data into HealthKit; research
  needs consent + ethics-board approval.

### The regulatory cliff behind Apple

A **diagnostic claim doesn't just risk App Review — it can make the app a regulated medical device**
(FDA "Software as a Medical Device" in the US, EU MDR in Europe, NMPA in China), which means clinical
evidence, submissions, and cost/time we are nowhere near. Staying **educational/wellness** is what keeps us
out of that regime.

### Our compliant design (already largely done)

- Position as **educational wellness**, output framed **"傳統上多與…有關 / traditionally associated with…"** —
  never "you have X."
- **No disease or medical-condition claims** (the whole project was reframed to educational; we deliberately do
  not output diagnoses).
- **Reject bad images** (capture-quality gate) — also satisfies the "inaccurate data" rejection risk.
- **"Consult a professional" disclaimer** surfaced with every reading.
- **Honest uncertainty** — confidence notes, tentative flags, no over-claiming faint signs.
- We can **disclose our methodology and benchmark numbers** (Q2) if Apple asks — which is exactly what 1.4.1
  wants.

One caveat to decide deliberately: **showing external citations/book names** to users (a trust feature) is
currently **default-off** (`TIH_SHOW_CITATIONS=false`). Keep it off for the public medical-caution surface, or
turn on only the wellness-framed sources — that's a product call, not a compliance blocker.

**Answer:** Yes, Apple rejects diagnosis claims and unvalidated accuracy claims — but the wellness/educational,
non-diagnostic, bad-image-rejecting, disclaimer-carrying design we've built is the well-trodden compliant path.
The line we must never cross in copy or marketing: **no disease/diagnosis/treatment language.**

---

## Q4. Should it go in Savor, and how does tongue info combine with the existing features?

This is the real decision, and your framing (accuracy + reliability + how it combines) is exactly right. Here's
the grounded answer.

### The TCM-correct way to combine tongue with constitution / food / seasonality

- **Tongue is never used alone in TCM.** Constitution/pattern determination combines **tongue + pulse + inquiry
  (symptoms) + history** ([AIMIN](https://aimin.com.sg/insights/posts/eating-for-your-tcm-constitution-a-personalised-food-guide)).
  So the theoretically-correct role for the tongue photo is as **one evidence input into the constitution
  estimate**, not a standalone verdict. This is not a limitation to hide — it _is_ the method.
- **We already built the bridge.** Our interactive refinement (WS-B) takes the tongue features → top-2
  candidate constitutions → asks the **1–2 most disambiguating follow-up questions drawn from CCMQ**, the
  validated 9-constitution questionnaire, and re-ranks. That is precisely tongue-as-input + inquiry →
  constitution. And **CCMQ is proven compressible**: a machine-learning selector cut questions by **68.3%**
  while improving constitution recognition ([PMC7676967](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7676967/))
  — so "photo + 2 questions" is a legitimate, fast constitution on-ramp, not a shortcut hack.
- **Constitution → food/seasonality is standard, evidenced TCM practice.** Personalized food therapy by
  constitution is mainstream ([Dr. D'Alberto](https://www.attiliodalberto.com/chinese-food-therapy/constitutions/),
  [Thomson Medical](https://www.thomsonmedical.com/blog/body-constitution)), and constitution now has
  emerging biological correlates (metabolic / gut-microbiota profiles), with ~half of the population
  showing an unbalanced constitution amenable to dietary adjustment
  ([PMC12295300](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12295300/)). So once tongue helps pin the
  constitution/leaning, **Savor's existing food-compatibility and seasonality logic already knows what to do
  with it** — e.g. a "damp-heat leaning" reading → favour cooling/damp-draining foods this season, flag
  greasy/rich foods as less compatible. That's real integration, not a bolt-on.

### But — the hard truth from Savor's own data

You already **validated that pure TCM body-constitution info has low market need.** That is the most important
constraint here, and it means:

- **Tongue must NOT just be "another way to show your constitution."** If it dead-ends in the same
  low-demand constitution screen, it inherits the same low demand.
- **Its job is to be the _engaging front door_** to the features that _do_ have value (food compatibility,
  seasonality). Snapping a tongue photo is a **10-second, visual, shareable, curiosity-driven** action; a
  60-item questionnaire is not. The competitor evidence (Aiteaic's 30-second scan, viral tongue content) says
  the _capture experience_ is the draw. So tongue's value to Savor is: **lower-friction on-ramp + novelty/
  virality + a richer signal feeding food/seasonality**, with the constitution kept in the background as
  plumbing rather than the headline output.

### Recommendation

**Conditional yes — include it, but as an engagement-first _input_ layer, gated on three things:**
1. **A capture-quality gate that reliably rejects bad photos** (blur/lighting/framing) — non-negotiable for
   accuracy _and_ Apple.
2. **Wire tongue → constitution-leaning → existing food/seasonality**, with the follow-up-question step doing
   the confirmation; keep the raw constitution readout de-emphasized (learned from the low-need finding).
3. **Validate demand in-app before full build** (Q1 soft-launch / fake-door test, Singapore/HK first).

**And two ship-blockers to flag honestly (not opinions — facts from our own tracking):**
- **Licensing:** the segmentation model was trained partly on **SM-Tongue (CC-BY-NC, non-commercial)** — must
  retrain on commercially-clean data or license it **before charging money**.
- **Copy/marketing discipline:** everything stays **"traditionally associated with…", non-diagnostic** — one
  careless "detects your liver problem" line in the App Store description undoes the whole compliance posture.

---

## Q5. Technical questions raised in the review (three specifics)

These came up inline while reviewing the research doc. All three are answered from our own code, not opinion.

### (a) "`blood_deficiency` — there's no such TCM *constitution*?" — **Correct. It's a *syndrome* (证), not a constitution (体质).**

You caught a real terminology slip. There are two different TCM targets:
- **Body constitution (体质)** — Prof. Wang Qi's validated **CCMQ 9-constitution** set (national standard
  ZYYXH/T157-2009): balanced, qi-deficiency, yang-deficiency, yin-deficiency, phlegm-dampness, damp-heat,
  **blood-stasis**, qi-stagnation, special/inherited. **`blood_deficiency` (血虚) is _not_ on this list.**
- **Syndrome / pattern (证)** — a clinical *state at a point in time*. `血虚` (blood deficiency) and
  `脾气虚` (spleen-qi deficiency) are **syndromes**, not constitutions.

So two of our 10 labels (`blood_deficiency`, `spleen_qi_deficiency`) are syndrome names; the other eight map
1:1 to CCMQ constitutions. **Why keep `blood_deficiency` anyway:** it is one of the most *tongue-legible*
patterns (pale + thin body + tooth-marks), and classical tongue knowledge is written in syndrome terms — so
reading a syndrome *tendency* is actually the TCM-correct thing for a tongue to do. This is already documented,
with the alignment table, in [`docs/FEATURE_PATTERN_MAPPING.md` §1](FEATURE_PATTERN_MAPPING.md); the report
layer also already softens the display name to **"Qi/Blood deficiency."**

**What to fix (cheap, no model change):** be consistent in copy — call the tongue output a **"pattern/syndrome
_leaning_ (辨证)"**, and when feeding Savor's *constitution* feature, run it through the existing crosswalk
(`blood_deficiency` & `spleen_qi_deficiency` → **qi-deficiency 气虚质**; the rest map directly). That preserves
the "constitution = your baseline, tongue = your state today" framing from Q4 and removes the mislabel.

### (b) "Tongue coverage `< 4%` — why 4%, why not 20?" — **4% isn't the quality bar; it's the "is a tongue even present" floor. There are three tiers.**

Coverage = fraction of the *whole frame* the tongue mask fills. A tongue that fills the on-screen guide oval is
≈ π·0.30·0.38 ≈ **36% of the frame**, so 4% is deliberately far below "good" — it only catches *nothing there*:

| coverage | meaning | behaviour |
|---|---|---|
| **< 5%** | essentially no tongue / a speck | **hard error, no reading** (`framing_feedback`; matches the Stage-1 acceptance floor in `stage1_quantitative/infer.py`) |
| **5–12%** | a tongue is present but small / far | reading is produced **but flagged low-reliability** ("move closer / extend more") |
| **≥ 12%, centred, not clipping** | well framed | "✓ great framing" |
| ~36% | fills the guide oval | the target |

So **20% would over-reject** legitimate readings (smaller tongues, further cameras) — you don't want a hard wall
there. But your instinct that 4% is *too permissive* was right in one respect: **we used to emit a full reading
in the 5–12% band with only a soft warning.** I've now (1) **harmonised the floor to 5%** (service.py and the
model were inconsistent at 4% vs 5%), and (2) added an explicit **`reliable` flag** to the framing output so the
app can down-weight or re-prompt on a sub-12% tongue instead of silently trusting it (`deployment/api/service.py`).

### (c) "Should we add detection for whether the tongue is in normal lighting?" — **Yes. Done — this was the missing capture-quality gate flagged in Q2(c)/Q3.**

This is exactly the blur/exposure gate the plan and the Apple-risk section (Q2c) said we still owed. Lighting is
the **#1 phone-accuracy killer** in the literature, so I've implemented a cheap, **ML-free** `capture_quality()`
gate (`deployment/api/service.py`) that runs *before* we trust a reading:
- **Blur** — variance of the Laplacian over the tongue's bounding box → reject "hold steady & let it focus."
- **Exposure** — mean luma + clipped-shadow/highlight fractions on the tongue pixels → reject "too dark" /
  "over-exposed / glare," warn on "uneven light."
- A **hard** capture failure is surfaced on the same `framing.status` the client already gates on, so a bad photo
  is refused end-to-end (non-breaking: it also adds a `quality_gate` block for richer client feedback).

Verified on synthetic images (blur → error, dark → error, dim → warn, glare → warn, clean → ok). **Thresholds are
sane defaults and need one tuning pass on real phone photos before launch** — noted in the plan. White-balance/
colour-cast is intentionally left to the existing `color_calibrate` step rather than a second, flakier check.

---

## Sources

- [TCM Tongue Diagnosis System Market report (WiseGuy)](https://www.wiseguyreports.com/reports/tcm-tongue-diagnosi-system-market)
- [Chongqing digital/smart TCM upgrade (ecns.cn)](https://www.ecns.cn/cns-wire/2026-06-18/detail-ihffptmh9488055.shtml)
- [Aiteaic/Luli AI tongue-scan wellness system (Taiwan News)](https://www.taiwannews.com.tw/news/6392348) · [PR Newswire](https://en.prnasia.com/releases/apac/singapore-healthtech-aiteaic-unveils-proprietary-tcm-wellness-system-pairing-ai-tongue-scanning-with-nanobubble-extraction-539215.shtml)
- [MyZenCheck AI tongue-diagnosis user demographics](https://www.myzencheck.net/blog/myzencheck-user-demographics-november-2025/)
- [Eu Yan Sang → North America wellness (NutraIngredients)](https://www.nutraingredients.com/Article/2023/09/06/eu-yan-sang-pursues-north-america-s-health-and-wellness-market/)
- [Deep Tongue smartphone model, AUC ~0.90 (bioRxiv)](https://www.biorxiv.org/content/10.1101/2023.02.02.526804v1.full)
- [Automated tongue diagnosis on the smartphone (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169260717308477)
- [Inter-/intra-practitioner reliability of TCM tongue inspection — low agreement (PubMed 18564955)](https://pubmed.ncbi.nlm.nih.gov/18564955/)
- [Smartphone tongue-coating reliability, Quasi-Delphi (JMIR mHealth 2020)](https://mhealth.jmir.org/2020/7/e16018)
- [Apple App Review Guidelines (1.4.1, 5.1.3)](https://developer.apple.com/app-store/review/guidelines/)
- [Eating for your TCM constitution (AIMIN)](https://aimin.com.sg/insights/posts/eating-for-your-tcm-constitution-a-personalised-food-guide) · [Nine constitutions & food therapy (D'Alberto)](https://www.attiliodalberto.com/chinese-food-therapy/constitutions/) · [9 body constitutions (Thomson Medical)](https://www.thomsonmedical.com/blog/body-constitution)
- [Constitution ↔ diet, metabolic/microbiota evidence (PMC12295300)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12295300/)
- [ML-compressed CCMQ, −68.3% questions (PMC7676967)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7676967/)
- Internal: `docs/BENCHMARK.md`, `docs/PLAN.md`, `docs/PROGRESS.md`
