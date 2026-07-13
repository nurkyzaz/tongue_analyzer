# User feedback round — human-40 labels + new features (2026-07-13)

The user hand-labeled 40 stratified images (`data/eval/human40`, ids t00–t39) and gave qualitative
feedback. This is the trustworthy, human-labeled eval set the accuracy work has been missing.

## 1. Honest accuracy on 38 human-labeled images

`evaluation/human40_labels.json` (t15, t22 excluded — see below; t05 has no body-colour label).
Scored with `evaluation/eval_human_labels.py`:

| model | coating | tai (coat colour) | zhi (body colour) | fissure | tooth-mark | **overall** |
|-------|--------:|------------------:|------------------:|--------:|-----------:|------------:|
| **v5 (production)** | 55% | 66% | 62% | 58% | 66% | **61%** |
| v7  | 29% | 68% | 54% | 58% | 66% | 55% |
| v7b | 37% | 63% | 62% | 50% | 68% | 56% |

**v5 stays production.** This confirms the earlier 10-image finding on 4× the data: the auto-label
benchmark (0.87) is optimistic; against a human the model is ~61%. v7's non-greasy expansion *hurt*
coating badly (55%→29%) — over-corrected, as suspected.

### Where v5 fails (per-image confusion)
- **Coating (weakest, 55%).** Two distinct failure modes: (a) over-calls `greasy` on genuinely
  `non_greasy` tongues — 8 cases (t04, t21, t23, t25, t26, t32, t33, t36); (b) misses thick coating —
  reads `greasy_thick` as `non_greasy`/`greasy` (t05, t13, t19, t28). The `greasy`↔`greasy_thick`
  granularity the user flagged is real (t30, t31).
- **Body colour (zhi, 62%).** Mostly ±1-bin (light↔regular↔dark). Several are the *zoned-colour*
  problem below — a red tip + pale centre averaged into one wrong "regular".
- **Coating colour (tai, 66%).** Almost entirely adjacent-bin white↔light_yellow↔yellow — subjective
  thresholds, low priority.
- **Fissures / tooth-marks.** Threshold/subjectivity (model under-calls faint cracks; none-vs-light
  disagreements). Our expert-manual set had *zero* "none" examples, so this set finally measures it.

## 2. Data-quality flags (from the user)
Recorded in `evaluation/human40_notes.json`:
- **t15** — not a usable tongue image → excluded.
- **t22** — two tongues in frame → excluded (segmentation undefined).
- **t24** — rotated → kept as an orientation stress case.
- **t05** — coating too greasy to read body colour → `zhi` left blank (scored N/A).

## 3. New measured feature: **red tip** (`stage1_quantitative/zoning.py`)

The user's core observation: many tongues are pale/white in the CENTRE (coating) but far REDDER at the
TIP, and a single whole-tongue colour read conflates them. This is also a classic TCM sign (red tip =
heart / upper-jiao heat).

Built a training-free **zoned colour analyzer**: PCA on the seg mask → long axis (rotation-robust,
survives t24), split into tip/middle/root by width (the tongue tapers, so the narrow end is the tip),
plus a centre-core vs. edge-ring split via distance transform. Measures CIELAB L*/a*/b* per zone and
derives `tip_redness_delta = a*(tip) − a*(whole)`.

**Validated** on the 40 images (`evaluation/eval_zoning.py`) and visually: it cleanly flags a red tip
on 13/38 tongues (t05, t10, t12, t13, t18, t26, t27, t29, t35, t36…), confirmed by eye (e.g. t29:
+7.4, tip visibly redder; t23: −3.9, uniform). **Wired into the pipeline**: `Stage1Output` now has a
`zoned_analysis` block with a `red_tip` {present/absent, severity} signal (threshold +2.0).

### Honest negative finding
The "pale coated centre" geometric proxy (`center_coating` = L* centre − edge) does **not**
discriminate coating thickness — the centre is lighter on *every* tongue (~+8.5 for all three coating
classes), because tongues are geometrically/optically lighter in the middle regardless of coating. So
this measurement is reported but is **not** a fix for the greasy/greasy_thick problem; that remains a
model/labeling issue (see §4).

## 4. Other feedback → status
- **Red dots / prickles.** Already have a `red_dots` extra feature (from TCM-Tongue; noisy, AP ~0.68).
  It's in `extra_characteristics`. Next round's `red_dots` human label will give it a real eval.
- **Surface pattern/texture confounding greasiness.** This is papillae texture read as coating — the
  likely cause of the 8 `non_greasy`→`greasy` errors. Needs a texture-aware signal or targeted labels;
  proposed `surface_pattern` label field.
- **Tip shape vs. tooth-mark ambiguity.** Segmentation/anatomy ambiguity; proposed
  `tip_shape_ambiguous` flag so we can measure how often it bites.

### Next labeling round
`human40_notes.json` proposes 5 new yes/no-or-scale fields (`red_tip`, `red_dots`, `surface_pattern`,
`coating_obscures_body`, `tip_shape_ambiguous`). `red_tip` labels will let us tune the +2.0 threshold
and turn the measurement into a benchmarked feature.

**Labeling tool:** `evaluation/label_human40.html` — a self-contained page (open in a browser via a
static server rooted at the repo, e.g. `python3 -m http.server`, then
`/evaluation/label_human40.html`). Shows each image, click-to-pick each field, autosaves to
localStorage, "Export JSON" downloads `human40_extra_labels.json` in the exact eval format. Model
predictions are deliberately NOT shown, to keep the labeler unbiased.

## 5. Do we already have professional labels? (`evaluation/find_professional_labels.py`)

**Yes, but partial and heterogeneous — not a substitute for the user's own labels.** The 40 images
come from three sources with different annotation coverage:

| source | n | professional labels held |
|--------|--:|--------------------------|
| TonguExpert (TE) | 14 | 8 have **manual expert grading** (`*_manual` cols in `manifest.csv`) — but SPARSE: each image has only some of {tai, zhi, fissure, tooth_mk}, and there is **no coating grade at all**. |
| TCM-Tongue (TCM) | 12 | all 12 have **practitioner YOLO annotations** over 20 categories (`shezhenv3-txt/…/labels/*.txt`), incl. **red dots** (`hongdianshe`), fissures, tooth-marks, coating colour, red/purple/swollen tongue, and organ-zoning boxes. |
| SM-Tongue (SM) | 14 | none — segmentation masks only. |

So **20 / 40** carry some professional label; **no single image is fully labeled** on our 5-char scale.
Coating *thickness* — our weakest and most-wanted axis — is covered by **neither** source in a usable
form. That's exactly why the user's hand-labeling is the load-bearing eval.

**Where professional and user labels overlap, agreement is high** (validates the user's labels):
fissure *presence* agrees 9/9 (severe-vs-light differs on t04, t33); tooth-mark presence 6/6; coating
colour 12/12; body colour 2/3 (t26 dark vs regular).

**Key nuance for the coating problem:** TCM annotations separate coating **thickness** (薄 *thin* =
`botaishe`, 厚 *thick*) from **texture** (腻 *greasy* / 滑 *slippery* = `huataishe`). Our single
`coating` axis (`non_greasy`/`greasy`/`greasy_thick`) conflates the two, which is likely part of why
it's the hardest axis. A future schema could split coating into thickness × texture, matching how
professionals actually grade it. (Also note the pre-existing `build_tcm_tongue_labels.py` bug: it maps
`botaishe`→`peeled_coating`; `botaishe` is *thin coating*, not peeled — tracked as task_d164e8d5.)
