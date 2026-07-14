# Labeling guide / severity rubric

Purpose: shrink the subjective boundary that causes most label disagreement (see
`LABEL_QUALITY_DIAGNOSIS.md` — the model is 97% within one grade of the user; the gap is the fuzzy
none↔light↔severe boundary). Use the **labeled gallery** as the visual reference while labeling: open
`evaluation/gallery.html` (or browse `data/eval/gallery/`, filenames encode the labels). Label the tongue
**body/surface itself**, ignoring lips and lighting glare.

## The golden rules
1. **Separate WHAT from HOW MUCH.** First decide if a sign is present at all (this is the reliable part),
   then grade it. Presence matters more than the grade.
2. **When torn between two grades, pick the LOWER** (none over light, light over severe) — consistently.
   A consistent rule beats a "correct" but variable one.
3. **Colour is a category, not a spectrum** — don't overthink it; the gallery anchors white vs light-yellow
   vs yellow.

## Per-feature anchors

### Coating — greasiness/thickness (`coating`)  *[the hardest — use the greasy test]*
- **non_greasy** — thin; you can SEE the tongue's own surface and its little dots (papillae) through it.
- **greasy** — a moist, sticky, pasty film that HIDES the surface/dots (like a smear of curd), but not piled up.
- **greasy_thick** — that pasty film is heavy/caked, especially thick in the centre.
- *Test:* can you see the surface texture/dots? yes → non_greasy · partly hidden → greasy · fully caked → greasy_thick.

### Coating colour (`tai`)
- **white** · **light_yellow** (a faint warm tint) · **yellow** (clearly yellow). Judge the CENTRE/root
  where coating is thickest. (This is your most reliable axis — 100% vs experts.)

### Body colour (`zhi`) — the tongue body itself, ignore the coating
- **light** = paler than a healthy pink (washed-out) · **regular** = normal pink-red · **dark** = clearly
  red or dusky/deep. If the tip is redder than the body, judge the BODY (mid/sides), not the tip.

### Fissures / cracks (`fissure`)
- **none** — no visible cracks.
- **light** — one shallow/short crack, OR a single central line, OR a couple of faint cracks.
- **severe** — a deep/wide crack, OR many cracks / a network, OR cracks with obvious depth.
- *(TE0002099 = a single moderate central crack → borderline; by rule-2, call it light.)*

### Tooth marks (`tooth_mk`)
- **none** — smooth edges.
- **light** — faint scalloping on part of an edge.
- **severe** — clear, deep indentations along most of the edge(s).

### Extras
- **red_tip** none/mild/strong — is the TIP redder than the rest of the body? (strong = obviously, only-the-tip).
- **red_dots** none/few/many — raised red spots/prickles on the surface.
- **surface_pattern** — a papillae texture strong enough that you can't judge greasiness from it.
- **coating_obscures_body** — coating so thick you can't read the body colour underneath.
- **tip_shape_ambiguous** — odd tip shape that could be a tooth-mark or just anatomy.

## Better still (recommended change)
For fissure / tooth_mk / coating, the light↔severe boundary has no true answer. The model already outputs
a **continuous severity 0–1**, so the product should report **present/absent + a severity score** ("cracks:
present, moderate") instead of a brittle 3-way class. For *labeling*, that means the most valuable label is
**present vs absent** (high agreement) plus an optional rough severity — not agonising over light vs severe.

## If two people label
Label ~20 images independently and compare — the disagreement rate is your **irreducible floor** (you
cannot expect the model to beat it). Expect high agreement on presence/colour, lower on severity — that's
normal and it's the number to target, not 100% exact-match.
