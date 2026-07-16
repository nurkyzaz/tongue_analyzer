# Feature spec — 舌診 Tongue check-in

A tongue photo becomes a second reading of the user's **constitution (體質)**, feeding the
same 9-type CCMQ engine the quiz already uses. Non-diagnostic, gentle, Cantonese.

## Where it lives

- **Home / 今日 tab:** a tongue card sits among the daily-habit cards — "今日舌診 · 睇下今日
  條脷" with a soft cinnabar tongue mark. Suggested cadence: weekly, or "when you feel off."
- **Constitution profile:** tongue readings appear as evidence next to the quiz result
  ("你的舌象 supports 濕熱 leaning"). Photo + quiz agree → higher confidence; disagree → a
  gentle "worth re-checking."
- Optional **宜/忌 widget tie-in:** today's tongue leaning nudges the day's 宜/忌 items.

## Flow (5 screens)

### 1 · Entry card (on 今日)
Small card in the daily rhythm. Copy: **今日舌診** / "30 秒，睇下你把脷想講咩". CTA: **開始**.
Shows last check date + a tiny trend dot if history exists.

### 2 · Capture
- Live camera, **oval framing guide**, real-time framing feedback (the pipeline already does
  this: oval + "framing feedback").
- Gentle guidance chips, one at a time: 「自然光最好」 「脷放鬆、伸出少少」 「唔使太用力」.
- Big soft shutter. No countdown pressure. Retake is one tap.

### 3 · Reading it (loading)
Calm, ~2–3 s. Copy: 「望聞問切… 而家望緊你把脷」 (playful nod to the four TCM examinations).
No spinner-anxiety; a slow ink-wash / breathing animation.

### 4 · Your reading (results) — the core screen
Maps directly to pipeline output:

| UI element | From pipeline |
|-----------|---------------|
| Tongue photo with 2–4 subtle **feature pins** | segmentation mask + detected features |
| **Headline** (lead sign + overall lean) | `interpret.headline` |
| **Findings** almanac list, each with a five-flavour colour tag | `build_findings` (ranked by prominence × reliability × tier) |
| Graded **severity bars** (gentle, /100) | graded severity (fissure/toothmk/coating thickness/red tip) |
| **Constitution leaning** chip (e.g. 濕熱 Damp-heat) + "traditionally associated with" | KB pattern voting → CCMQ 9-type |
| **宜 / 忌** two-column card | `build_recommendation` {actions} in almanac form |
| **你可能亦會留意** symptom chips | `build_symptoms` (top-2 patterns) |
| **Distinctiveness** line ("比約 N% 嘅人明顯") | reference_stats percentile hooks |
| **Confidence note** (honest snapshot caveat) | `confidence_note` + per-sign reliability tags |

Each finding names the sign in plain Cantonese first, TCM term second, e.g.
**脷苔偏厚膩** · thick greasy coating · 傳統上多與「濕」有關.

Features currently reliable enough to surface (from benchmarks): coating **thickness**,
body colour, fissures, toothmarks, red tip (strong only), red dots. Texture (greasy/smooth)
shown but flagged "worth confirming."

### 5 · History / trend
- Timeline of past tongues (thumbnail + date + leaning).
- One gentle metric line (e.g. coating thickness over weeks) — habit-tracking register, not
  a medical chart. Ties into the app's morning/afternoon/evening rhythm.

## Non-negotiables

- **Educational, non-diagnostic.** No disease names as conclusions. "傳統上多與…有關".
- **Honest uncertainty.** Show the confidence note; don't over-claim faint features.
- **Pressure-free.** Missing a day is fine; no streaks-guilt.
- Result is **shareable** as a card (the tool already builds a PNG share card) — optional.
