# Improving labels at scale (then retrain) — research & plan (2026-07-13)

Goal: improve the labels on the **existing thousands** of images and train on them; if that's infeasible,
go self-supervised. This researches both and recommends a path.

## Reframe: the problem is CONVENTION, not noise
We measured it: TonguExpert auto labels are **96–98% consistent with its own experts** — they're *clean*,
just in **TonguExpert's convention**, which differs from the target/user convention (the model over-calls
`yellow` coating and `pale` body vs the user; `LABEL_QUALITY_DIAGNOSIS.md`). So:
- Confident-learning / Cleanlab (find *noisy* labels) has **limited value** here — the labels aren't noisy.
- Hand-relabeling ~15k images is infeasible.
- "Improve the labels" therefore means **re-align the categorical labels to the target convention at
  scale**, using objective data + a small human anchor set. This is a **weak-supervision** problem.

## What we have
- **TonguExpert (~6k):** images + auto labels + **~1,300 continuous phenotypes** — incl. OBJECTIVE colour
  measurements per characteristic: `P21_Tai_Color` (coating L*/a*/b*/HSV), `P31_Zhi_Color` (body colour),
  `P11_Tg_Color`, plus shape/texture for fissure/tooth-mark. **This is the key asset.**
- **TCM-Tongue (~6.7k):** practitioner category boxes (no phenotypes).
- **SM-Tongue (~2.2k):** real photos, **no characteristic labels** (the domain-gap set).
- **Human anchors:** 76 labeled (target convention) + a 60-image batch prepped for labeling → ~136.

## The four paths (researched)

### Path A — Phenotype-anchored relabeling  ★ RECOMMENDED, directly "improves the labels"
Re-derive the categorical colour labels for the ~6k TonguExpert images by **thresholding their objective
colour phenotypes**, with thresholds **calibrated to the target convention on the human anchors**:
- `tai` white/light-yellow/yellow ← coating **b\*** (yellowness) cut-points tuned so anchor-white↔low-b*,
  anchor-yellow↔high-b*.
- `zhi` pale/regular/dark ← body **L\*** (lightness) + **a\*** (redness) cut-points tuned to the anchors.
- Extend to fissure/tooth-mark via their shape phenotypes later.
Output: cleaner, **target-aligned** labels for thousands of images with **no hand-labeling**, directly
attacking the over-yellow/over-pale errors. Then retrain (no WB-aug). *Honest caveat:* choosing thresholds
to match the user IS a calibration step — but it calibrates the **labels/ground-truth** from objective
measurements, not an inference-time hack like v5. This is exactly "improve labels → train". Grounded in
weak supervision (Snorkel-style labeling functions + a label model). [Snorkel; confident-learning refs below]

### Path B — Semi-supervised self-training (extend to the rest)  ★ complement
For images WITHOUT phenotypes (TCM, SM real photos): **Noisy-Student / FixMatch self-training** — anchor a
teacher on the improved + human labels, **pseudo-label** the full ~15k with **confidence filtering +
strong augmentation**, train a student, iterate. Propagates the target convention to non-TE and real-photo
images without hand-labeling. [Noisy Student; FixMatch]

### Path C — Self-supervised pretraining (the fallback; domain gap)
**DINOv2 fine-tune or MAE** pretraining on all ~15k **unlabeled** tongues → then fine-tune on the improved
labels. Best for the **clinical→phone domain gap** (SM real photos). Evidence: DINOv2 fine-tune beats
ImageNet pretraining on medical classification; MAE is robust under low labels. **Still needs labels at the
end** — so it complements A/B (which supply them), it doesn't replace them. [DINOv2; MAE refs]

### Path D — Confident-learning audit (cheap sanity, low expected value)
Run **Cleanlab** to flag images where the model confidently disagrees with the auto label. Since labels
aren't noisy, expect it mostly surfaces the systematic convention shift — useful as an audit / to prioritise
anchor labeling, not a fix on its own.

## Recommendation & sequence
1. **Path A first** — phenotype-anchored relabel of `tai`/`zhi` on TonguExpert, thresholds fit on the
   anchors; retrain (`--init v5 --no-wb`); eval on **human40b** (independent). This is the fastest, most
   direct "improve the labels for the thousands", uses data we already hold, and targets the exact
   over-yellow/over-pale errors. *Decision gate:* does it reduce the over-calls + beat v5 on human40b?
2. **Path B** — self-training to carry the convention to TCM + real photos.
3. **Path C (SSL)** — only if A/B leave a domain gap on real photos; it's the longer play and still ends in
   a fine-tune on the improved labels.
4. **Grow anchors** — label the 60-image batch (→ ~136); anchors are the calibration target for A/B and the
   honest eval. More anchors = tighter thresholds + more trustworthy validation.

## Honest bottom line
- **Yes, we can improve labels at scale** — Path A (re-derive from objective phenotypes, anchored to the
  target convention). That is the answer to "can we improve labels for the thousands."
- **Self-supervised is the fallback / domain layer, not the first move**, and it *still needs labels* — so
  it doesn't remove the need for the anchor set.
- Either way the ~136 human anchors are essential (as the convention target and the eval); the 60-image
  batch (`evaluation/label_human_train.html`) is the next concrete step regardless.

## Sources
- Confident learning / Cleanlab — https://l7.curtisnorthcutt.com/confident-learning ; https://github.com/cleanlab/cleanlab
- Weak supervision / Snorkel label model — https://snorkel.ai/data-centric-ai/weak-supervision/
- FixMatch (semi-supervised) — https://papers.nips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f733-Paper.pdf
- Noisy Student self-training — https://arxiv.org/pdf/1911.04252
- DINOv2 vs ImageNet for medical — https://arxiv.org/pdf/2402.07595 ; MAE low-label robustness — https://arxiv.org/html/2604.22854
