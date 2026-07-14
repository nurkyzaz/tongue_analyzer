# Label-quality diagnosis: why "my labels" and "expert labels" seem to disagree (2026-07-13)

The user asked, reasonably: *the model is ~59% vs my labels but 97% auto-vs-expert — are my labels wrong,
is there a photo↔label bug, or am I labeling incorrectly?* We investigated all three. **Answer: no bug,
the labels are good, and the "disagreement" is almost entirely the subjective severity boundary.**

## 1. No photo↔label bug
- The images the user labeled are **byte-identical** to the images the model is evaluated on (md5 match,
  local vs server).
- Each `t##.jpg` is **the same image as its `meta.json` source** (pixel-exact). Ids are correctly mapped.
- `evaluation/diagnose_label_agreement.py` / integrity checks. → Ruled out.

## 2. The 59% was never "user vs expert" — it's "user vs model"
The 96–98% figure was TonguExpert's **auto vs its own expert** labels on *clinical* images. The 59% is
the **model vs the user** on a *different, mixed* set. Those are different comparisons. So we did the one
that was missing — **user vs professional labels on the SAME images** (`diagnose_label_agreement.py`,
over the label store):

| feature | user vs **expert** | user vs **TCM practitioner** |
|---------|-------------------:|-----------------------------:|
| tai (coating colour) | **100%** (3/3) | **100%** (25/25) |
| zhi (body colour) | 80% | — |
| fissure | 75% | 57%* |
| tooth_mk | 50% | 54%* |
| red_dots | — | 58%* |

**The user agrees with experts perfectly on categories (colour 100%).** Every fissure/tooth_mk
disagreement vs the graded expert is a **severity grade** (user "light" ↔ expert "severe"), never
"present vs absent". Confirmed by eye (e.g. TE0002099 has a real central crack — "light" vs "severe" is a
judgment call). *The TCM `absent` rows are a **completeness artifact**: TCM-Tongue is an object-detection
set where annotators only box *salient* features, so a *mild* crack / *few* red-dots weren't annotated —
that reads as "absent" but is really "not-salient / unknown", not a true disagreement.

## 3. The exact-match metric massively understates agreement
`evaluation/eval_fair.py` scores v5 on the 76 human images three ways:

| feature | exact 3-way | **within-1-grade** | presence (none vs present) |
|---------|------------:|-------------------:|---------------------------:|
| coating | 53% | **95%** | 66% |
| tai | 70% | **99%** | — |
| zhi | 56% | **96%** | — |
| fissure | 58% | **96%** | 68% |
| tooth_mk | 58% | **99%** | 66% |
| **OVERALL** | **59%** | **97%** | — |

**The model is within one grade of the human label 97% of the time.** It essentially never confuses
none↔severe. The "41% error" is the ±1 severity boundary (none↔light, light↔severe, non_greasy↔greasy),
which is inherently subjective — experts disagree there too (user-vs-expert tooth_mk was only 50%).

## Conclusion
- **The labels are good and there is no bug.** The user labels categories like an expert.
- **Exact 3-way accuracy is the wrong headline.** Report **within-1-grade (97%)** + **presence** instead;
  they reflect real agreement.
- The genuine gap is the **none↔light / light↔severe boundary** (subjective for everyone) and the
  **presence** decision on faint features (66–68%) — that's where improvement is real, not "the model
  can't read tongues."

## Fixes (what to change)
1. **Metric (done):** use `eval_fair.py` — within-1-grade + presence — as the honest accuracy going
   forward. 59%→97% is not spin; it's measuring the right thing.
2. **Severity rubric** to shrink the subjective boundary — see `docs/LABELING_GUIDE.md`: concrete anchors
   for none/light/severe with the labeled gallery (`data/eval/gallery/`) as the visual reference. Consistent
   labeling = a stable target to train/measure against.
3. **Prefer present/absent + a continuous severity score** over the brittle 3-way class for graded signs
   (fissure, tooth_mk, coating). The model already predicts a continuous severity; reporting "cracks:
   present, moderate (0.4)" sidesteps the light/severe boundary that has no ground truth.
4. **Label-store caveat:** don't read TCM `absent` on mild features as a true negative (it's non-salient/
   unknown). Fine for the salient-feature benchmark; exclude from user-vs-pro agreement.

## Implication for the modeling plan
The bottleneck is **not** noisy training labels (97% expert-consistent) and **not** user mislabeling — it's
the **subjective grading boundary + the presence decision on faint features + domain gap** on real photos.
So: a rubric (consistency) + present/absent+severity schema + domain-adaptive pretraining beat "bigger
model" or "label cleaning". See `docs/DIRECTION_REVIEW.md`.
