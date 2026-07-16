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

### UPDATE (user: "trust the expert labels, use their definitions, no calibration")
Measured v5 directly against the **expert (L1 manual) labels** on the held-out test set (n=250):
**tai 94% · zhi 78% · fissure 96% · tooth_mk 82%.** So *by the expert standard*:
- The labels are **already expert-standard at scale** (auto labels are 97% expert-consistent; TonguExpert
  has no published threshold "definition" — the labels ARE the experts' judgments), and **v5 already
  matches the experts at 78–96%.**
- The big "errors" you saw are mostly **you-vs-expert** disagreement (you're not a TCM expert, as you
  noted) — the model-vs-expert gap is much smaller. So relabeling has **little headroom on tai/fissure**
  (already 94–96%); some remains on **zhi (78%) / tooth_mk (82%)**.
- Where the model DOES slip is **real photos**: coating-colour on the SM (phone-like) set was 57% vs 74%
  on clinical — a **domain gap**, not a label problem.

**Revised recommendation (no calibration, trust experts):**
1. **Adopt the EXPERT labels as the accuracy standard + training target** (drop chasing your convention).
   By this standard v5 is the baseline at 78–96%. Report accuracy vs experts, not vs your labels.
2. Relabeling from phenotypes (below) is **de-prioritised** — the color/coating labels are already
   expert-standard; it can't beat 94–96%. (Kept for reference / for zhi if we ever get more expert zhi
   labels.) v8 already gold-weighted the sparse expert labels and did NOT beat v5, consistent with this.
3. **The real remaining lever is the real-photo DOMAIN gap → Path C (self-supervised / domain-adaptive
   pretraining)**, which is exactly the SSL path. This is now the evidence-based next step.

### Path A (original) — Phenotype-anchored relabeling  [de-prioritised per the update above]
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

### Path C — Self-supervised pretraining (the fallback; domain gap)  ❌ TRIED, DID NOT HELP (2026-07-13)
Built `ssl_pretrain.py`: SimCLR-style, **colour-preserving** aug, on ~13.5k unlabeled tongues (loss
3.3→1.69). Fine-tuned the multitask heads from that encoder, **--no-wb** (per user, no WB-aug). Result:
- **vs EXPERT test: a WASH** (tai 94=94, zhi 81=81, fissure 95→90, tooth_mk 82→86).
- **By source (the whole point): WORSE** — TCM tai 75→39%, zhi 54→39%; SM zhi 60→45%. The domain gap
  *widened*.
Why: (1) the SSL pool is **clinical-dominated** (5992 TE vs 2175 SM), so the encoder stays clinical-biased —
SSL can't manufacture real-photo robustness from data it doesn't have; (2) **removing WB-aug** (the
`--no-wb` the user asked for) took away the one thing that WAS giving v5 real-photo colour robustness, and
SSL didn't replace it. Net: val improved (0.73→0.78 on the TE val) but **generalisation to TCM/SM got
worse** — better on the train domain, worse elsewhere. **v5 stays production.**
**Real conclusion:** the real-photo colour gap can't be closed by SSL on the *existing* (clinical-heavy)
images. It needs **actual real phone photos** — to pretrain on and/or fine-tune on. That is the true
blocker (WS3a), not the method. The `ssl_resnet34` encoder + `ssl_pretrain.py` are kept and become useful
*once* we have a real-photo corpus to pretrain on. [DINOv2; MAE refs]

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
