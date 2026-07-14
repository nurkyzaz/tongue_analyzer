# Output redesign plan — from demo-trial feedback (2026-07-13)

## Does "present/absent + severity" improve model accuracy? — Honest answer: NO.
It's a **reporting/schema** change, not a modelling one — it doesn't make the network read tongues
better. What it does is **show only the parts that are reliable**: presence + prominence (which the model
gets ~66–97% right) instead of the exact 3-way grade (~59%, dominated by the subjective light/severe
boundary — see `LABEL_QUALITY_DIAGNOSIS.md`). So it improves *trustworthiness of the output*, not raw
accuracy.

**The user's own idea is better and we adopt it:** *don't display feature grades — name the features that
are PRESENT, rank them by prominence, and let the RAG+LLM reason about which conditions they match
(possibly several) and what to do.* This leans on the reliable signal (presence + prominence) and offloads
the fuzzy categorical→pattern mapping to grounded reasoning. It also naturally handles "several
conditions" and produces actionable, plain output. **This is the new North Star for Stage-2 output.**

## Grounded diagnosis of each comment
| # | Comment | What actually happens (verified) | Root cause |
|---|---------|----------------------------------|-----------|
| 1 | t00 "features didn't show it's red" | model DID detect `zhi=dark` and shows a "body color: dark" card | **presentation** — "dark" isn't communicated as "red"; a flat card, no plain-text |
| 2 | t12 "idk if it saw it's pale" + "show only present features" | model DID detect `zhi=light`; but 7 cards shown incl. spurious `red_tip`, `wet` | **too many + noisy** cards; no plain findings text |
| 3 | t08 "cracks are because the coating is so thick it cracks" | model called `fissure=severe` → voted `yin_deficiency` | model can't tell **coating-crack from body-fissure** → spurious yin |
| 4 | share card "not insightful, remove it" | "Tongue Type" archetype + share/download | wrong value prop for a health app |
| 5 | "add a recommendation card / findings-in-text / rank features → conditions → action" | only per-feature cards + pattern cards today | missing a **plain, actionable** synthesis |
| 6 | "why is every condition 'poor digestion'?" | greasy coating is common → spleen-qi/phlegm patterns → repeated "poor digestion"; symptom lists are short/samey | **pattern over-firing + thin symptom content** |
| 7 | "add text about what symptoms the tongue indicates" | only follow-up questions carry symptoms | no **symptom narrative** |

## The plan (sequenced; each item notes: Frontend / Interpret / KB / Model)

### S1 — "Salient findings" as the core Stage-2 object  *(Interpret)*
Build a ranked list of the tongue's **notable present signs**, scored by prominence
= severity × distinctiveness × detector-reliability. Each carries a plain phrase ("a red/dark body", "a
greasy coating", "cracks", "scalloped edges"). Drop normal + low-reliability-noise signs. This list drives
everything below.

### S2 — Plain-text "What we found" description  *(Interpret + Frontend)*
A 1–3 sentence natural-language description of the salient findings, most-prominent first
("Your tongue looks **red/dark** with a **greasy coating** and some **cracks**…"). Directly fixes #1, #2
(t00 redness, t12 pale now stated in words). Render at the top of results.

### S3 — Replace the share card with a **Recommendation card**  *(Frontend + Interpret)*
Remove the "Tongue Type"/share card (#4). New card:
1. **Findings, ranked** (from S1) — the 2–3 most prominent signs in plain words.
2. **What it may point to** — the likely condition(s), *several allowed* ("leans damp + a touch of heat").
3. **Do this** — 1–2 **specific, actionable** recommendations tied to the findings ("looks dry → drink
   more water & rest"; "pale → favour iron-rich foods like red meat & greens"; "greasy → lighter, warm,
   cooked meals"). Optionally the card is *recommendation-only* (most shareable/useful part). (#5)

### S4 — Show only NOTABLE features; fold the rest away  *(Frontend + Interpret)*
Display feature cards only for **present/prominent** signs; summarise the normal ones in one line; suppress
low-reliability noise (e.g. a weak `red_tip`/`wet`). Fixes #2. Keep the bars for the ones we show.

### S5 — Coating-crack vs body-fissure hedge  *(KB combination rule)*
Add a rule: `coat_texture=greasy` + `coat_thickness=thick` + `fissure present` → **note the cracks may be
in the thick coating rather than the tongue body, and down-weight the yin-deficiency vote**. Near-term fix
for #3 / t08 (removes the spurious yin). Long-term: a detector that separates them (needs labels — parked).

### S6 — Symptom narrative + richer, varied symptoms  *(KB + RAG + Interpret)*
- Add a **"symptoms this tongue may go with"** text block (from the pattern's `associated_symptoms` +
  the RAG symptom cards we already authored) — the user wants symptom *text*, not just questions (#7).
- Expand + vary `associated_symptoms` per pattern so it isn't always "poor digestion" (#6).

### S7 — Reduce pattern over-firing toward digestion  *(Interpret/KB)*
Greasy coating (common) dominates votes toward spleen-qi/phlegm ("poor digestion"). Re-check the vote
weights / distinctiveness so a common greasy coat doesn't always lead digestion; let body-colour and other
signs carry more when present. Validate on `mapping_testset.json` (keep 12/12) + eyeball variety across the
gallery. (#6)

## Honest note on scope
S2–S7 are **Stage-2 (interpretation/UX)** — no retraining, fast, high-impact on *perceived* quality, which
is most of the feedback. They do **not** raise feature-detection accuracy (that's the separate
domain-adaptation / rubric track in `DIRECTION_REVIEW.md`). The two tracks are complementary: this one
makes the output trustworthy and actionable *given* the model; that one slowly improves the model.

## Status
- ✅ **S2** findings_text — plain "Your tongue shows … a red or dark body …" (fixes t00 redness, t12 pale).
- ✅ **S3** Recommendation card replaces the share/Tongue-Type card — findings → likely condition(s) →
  specific actions ("cracks → drink water & rest"; "pale → iron-rich foods").
- ✅ **S4** show only NOTABLE signs (reliability-weighted + clinical priority tiers so body-colour/coating
  lead and marginal red_tip/moisture sink), with a one-line "within a typical range: …" summary.
- ✅ **S5** coating-crack hedge — thick greasy + cracks → phrase "cracks (possibly in the thick coating)",
  suppress the spurious Yin vote (KB rule `thick_greasy_cracks_maybe_coating`), and skip the dryness
  action (t08 now leads phlegm/damp, not yin).
- ⬜ **S1** salient-findings object — done as `findings`/`build_findings` (prominence = rel × reliability,
  tiered).
- ⬜ **S6** symptom narrative + varied symptoms · ⬜ **S7** reduce digestion over-firing — NEXT.

## Suggested order (done: S2+S3+S4+S5) → next S6/S7 (content depth), then re-demo on the gallery.
