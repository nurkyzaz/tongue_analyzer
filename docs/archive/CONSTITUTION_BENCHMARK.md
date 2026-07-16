# Constitution / Syndrome Benchmark — research & feasibility

_Can we benchmark the tool against TCM **constitution (体质)** or **syndrome (证型)** — even a text one?
Findings after surveying the Chinese literature, and a first working benchmark._

## What's out there (surveyed)

| Resource | Type | Has constitution/syndrome labels? | Public? | Fit for us |
|---|---|---|---|---|
| Hospital constitution sets (e.g. 22,482 tongue imgs, doctor-labelled 9 types) | image→constitution | ✅ | ❌ in-hospital | would be ideal — **not obtainable** |
| **CCMQ** (Wang Qi 9-constitution questionnaire) | text questionnaire | ✅ (scoring is a public standard) | scoring yes, response data no | our follow-up-question basis |
| **ZhongJing-OMNI** ([github](https://github.com/pariskang/ZhongJing-OMNI), [HF](https://huggingface.co/datasets/CMLM/ZhongJing-OMNI)) | **multimodal** — tongue images + Q&A | Q&A/VQA | ⚠️ **announced, not released** | data dir empty on both GitHub & HF (README + 1 `demo.png` only) |
| **TCMEval-SDT** ([Nature SciData](https://www.nature.com/articles/s41597-025-04772-9), [github](https://github.com/zhuyan166/TCMEval)) | **text** cases | ✅ gold syndrome (MCQ) | ✅ **CC-BY 4.0** | **usable** (see below) |
| **TCM-BEST4SDT** ([github](https://github.com/DYJG-research/TCM-BEST4SDT)) | text, 257 syndromes | ✅ | ✅ | syndrome differentiation for LLMs |
| **TCMBench / TCM-ED** | text, 5,473 licensing-exam Qs | exam answers | ✅ | LLM TCM knowledge |

**Conclusion on the target:** there is **no public tongue-image→constitution dataset** we can obtain, so a
true image→constitution accuracy number isn't possible right now. But **public text syndrome-differentiation
benchmarks exist** — and their cases contain tongue findings + a gold expert syndrome. That lets us
benchmark our **knowledge-base reasoning** (tongue signs → pattern), which is a real, grounded check.

## What we built: KB-reasoning vs expert syndrome (TCMEval-SDT)

`evaluation/benchmark_syndrome.py`. Data: TCMEval-SDT train split — the only publicly-labelled split
(the Validation/Test splits ship with the gold answers blanked for leaderboard submission). Of its
**200 cases, 167 (84%) contain a tongue description** (舌…) and 150 carry a gold syndrome; **109 are
scorable** (parseable tongue features + gold). For each:
1. **Isolate the tongue clause.** Cases read `…舌{body}{coating}，{symptoms}，脉{pulse}`; we scan from
   the first 舌 over the bounded tongue charset and stop at the first non-tongue character, so colour
   words in later symptom text (小便赤 dark urine, 面色苍白 pale face) can't leak in.
2. **Parse** to our feature values (舌红→red body + Heat detector, 苔黄→yellow coating, 苔腻→greasy,
   齿痕→tooth-marks, 裂→cracks, 舌淡→pale, 少苔→scanty/Yin, …).
3. Run our **KB feature→pattern** mapping.
4. Reduce both our pattern and the gold syndrome to a coarse **TCM axis**
   (Heat / Cold / Damp / Qi-def / Blood-def / Yin-def / Stasis).
5. Measure **directional consistency**: is our tongue-derived axis among the expert syndrome's axes?

**Result (TCMEval-SDT train, 109 scorable cases):**

| | Directional consistency |
|---|---|
| Naïve first-pass parser | 52/118 = **44.1%** |
| Fidelity-corrected parser (current) | 76/109 = **69.7%** |

Per expert-axis recall (corrected parser): **Heat 95%** (58/61), **Damp 81%** (43/53),
**Yin-def 69%** (11/16), **Cold 60%** (6/10), **Stasis 57%** (12/21), **Blood-def 20%** (1/5),
**Qi-def 12%** (1/8).

### Two parser-fidelity fixes drove the jump (44% → 70%) — and are real KB findings
- **Red tongue was misread as blood-stasis.** The naïve parse sent 舌红/舌绛 only to the 3-class body
  head's coarse "dark" bucket, which the KB weights toward `blood_stasis (0.8)` — so textbook 血热
  (blood-heat) red tongues scored as *stasis* (Heat recall 59%). The product actually runs a dedicated
  **`red_tongue` Heat detector** alongside the body-colour head; firing it (as it does on real photos)
  is the faithful representation. Red tongues are the majority here (fired in 62/109 cases) → Heat
  recall 59% → **95%**. *KB takeaway: the coarse "dark" body-colour class leans stasis; the
  `red_tongue`/`purple_body` detectors are essential to separate heat-red from stasis-purple.*
- **淡红 (normal light-red) was misread as "pale".** Substring matching let 舌淡/质淡 fire inside
  舌质淡**红**, tagging the *healthiest* colour as blood-deficient. Fixed by scoping 淡红→normal.
- Plus **红 + 少苔 (scanty coating) → Yin-deficiency** (红少苔 is the classic yin-heat tongue), which
  lifted Yin-def recall 38% → 69%.

The **low Blood-def / Qi-def recall is the honest ceiling, not a bug**: a pale tongue (舌淡) is common
to blood-def, qi-def *and* yang-def, and the tongue alone cannot separate them — the expert used pulse
and symptoms. These stay misses by design, and mark where the tongue is genuinely uninformative.

### Reproduce
Data is CC-BY 4.0 from [github.com/zhuyan166/TCMEval](https://github.com/zhuyan166/TCMEval) (gitignored
under `data/external/`, not vendored here). Then:
```
python evaluation/benchmark_syndrome.py \
  --data data/external/TCMEval/evaluation/TCMEval-SDT/data/Train_TCM_Data_v1.json
```

### Why "consistency", not "accuracy" (honest framing)
The **tongue alone does not determine a full syndrome** — the expert also used pulse + symptoms, and the
options are fine-grained (10 per case, sometimes multi-label). So exact syndrome match is the wrong bar.
Directional consistency answers the fair question: *does our tongue-reading point in the same
thermal/deficiency direction the expert concluded?* This validates the **knowledge base** (does it reason
like the literature), independent of the image model.

## What this does and doesn't tell us
- ✅ Checks our **KB reasoning** against real expert cases (public, citable, CC-BY).
- ✅ Surfaces where our tongue→pattern rules disagree with experts → concrete KB fixes.
- ❌ Not an image-model benchmark (uses the cases' text tongue descriptions, not photos).
- ❌ Not a constitution-accuracy claim.

## Multimodal image→syndrome: harness built, dataset unreleased

`evaluation/benchmark_multimodal.py` runs our **real production vision pipeline** (seg_combined + v5
characteristics + extra features) on tongue photos and scores it against expert answers — the closest
thing to a true image→syndrome check. It uses the ZhongJing-OMNI `TongueDiagnosis/` layout
(`images/<id>.png` + `answers/<id>_answer.txt`) and reuses the **same Chinese parser** as the text
benchmark for the gold side, reporting (a) feature agreement (does our vision read the same
body-colour/coating/marks the expert described?) and (b) axis consistency.

**Blocker (honest):** ZhongJing-OMNI's multimodal tongue data is **announced but never released** — its
GitHub and HuggingFace repos both ship only a README and a single `demo.png` (263 kB, "1 row"), and the
citation URL is a `yourusername` placeholder. Worse, that lone `demo.png` is **a screenshot of a chat
UI**, not a diagnostic photo, and its gold is self-contradictory (README caption: "swollen + tooth-marks
→ Qi deficiency"; the answer *inside* the same screenshot: "no swelling or tooth-marks, normal
pale-pink"). Cropping the ~93 px tongue thumbnail out of the screenshot, the harness runs cleanly
end-to-end (our read: regular/white/greasy → Damp), but at that resolution/domain and with a
contradictory label this is a **plumbing verification, n=1 — not a score**. The harness is ready to
produce a real number the day the images are released (or on any image+expert-answer set in this layout).

## Other paths to a real constitution/image benchmark (future)
1. **LLM-reasoning score**: if we add a Stage-2 LLM, report its score on TCMEval-SDT / TCM-BEST4SDT MCQ
   (a standard, citable "TCM reasoning" number for the reasoning layer).
2. **Collect a small constitution set**: pair tongue photos with a validated **CCMQ** assessment (clinic
   partnership) — the only route to a true tongue→constitution accuracy.
3. **Chase a released multimodal set**: e-mail the ZhongJing-OMNI authors (ylkan21@m.fudan.edu.cn) for
   the tongue images, or substitute another image+expert-answer set into the same harness.
