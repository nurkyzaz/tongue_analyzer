# Paste-in prompt for Claude design (English)

> Copy everything in the fenced block below into Claude (design / artifact mode). If you've
> filled `BRAND.md` with your real Canva colours/fonts/logo, paste that in too, replacing the
> placeholder palette in the "BRAND TOKENS" section.

```
Design the UI for a new "Tongue check-in" feature inside a Traditional Chinese Medicine
wellness app in the style of "Savor" (計五味). Produce a set of mobile phone-screen mockups
as a single self-contained, responsive HTML artifact — light + dark theme, no external
assets. Use ENGLISH for all UI copy.

THE APP IT LIVES IN
- Savor (計五味, "measure the five flavours") is a gentle, pressure-free TCM wellness app.
  It profiles the user's body constitution using the CCMQ questionnaire, tracks daily
  habits, and shows a "do / avoid" almanac widget. The voice is warm, friendly, and
  low-pressure — "See what your tongue is telling you today," never clinical or alarming.

THE FEATURE
- The user photographs their tongue; the app reads visual features and maps them to their
  TCM constitution — a SECOND input alongside the questionnaire. It is EDUCATIONAL and
  NON-DIAGNOSTIC: every reading is phrased "traditionally associated with…", never a
  diagnosis or a disease name. It complements, and cross-checks, the quiz result.

CORE CONCEPT (use this — don't default to a generic wellness look)
- The app is literally about the FIVE FLAVOURS, and the tongue is the organ that tastes
  them, so the tongue check-in is the app's most on-brand feature. Lean into an ALMANAC
  register (like a modern Chinese almanac / "do & avoid" list): findings shown as clean
  vertical entries, a single cinnabar "seal / stamp" accent used sparingly, and each finding
  tagged to its five-flavour / five-element colour. This colour system IS the brand, not
  decoration. Avoid the overused "cream background + serif headline + terracotta accent"
  wellness template.

BRAND TOKENS  (placeholder five-flavour palette — replace with real Canva values if you have them)
- paper #F6F1E7, raised surface #FCF9F2, ink #26221D, soft ink #7A7166, hairline #E4DBCB
- cinnabar seal accent #C0453A (use sparingly, like a stamp)
- five-flavour tag colours: Sour/Wood celadon #5E7F6E · Bitter/Fire cinnabar #C0453A ·
  Sweet/Earth amber #D6A24E · Pungent/Metal warm-grey #B9B2A6 · Salty/Water slate #3C4A57
- dark theme (warm, NOT a plain inversion): paper #1E1B17, raised #262119, ink #EDE6D8,
  hairline #38322A
- Type: a serif display face for the almanac voice + a clean humanist sans for body/UI.
  Warm, hue-biased neutrals — never pure grey. Use tabular numerals for scores and dates.

SCREENS TO DESIGN (render as phone frames, arranged side by side)
1. Home / Today: a "Today's tongue check — see what your tongue is telling you" card sitting
   among the daily-habit cards. Shows last-checked date and a "Start" button.
2. Capture: a live-camera screen with an OVAL framing guide, one gentle guidance chip at a
   time ("Natural light works best", "Relax and gently stick out your tongue", "No need to
   strain"), a live "position looks good" confirmation, and a soft shutter — no countdown,
   no pressure.
3. Reading it: a calm ~2-3 second loading state with a soft ink-wash / breathing animation
   and copy like "Reading your tongue…".
4. Your reading (THE core screen), containing:
   - the tongue photo with 2-4 subtle feature pins
   - a headline (the lead sign + the overall lean), e.g. "Thick, greasy coating with a
     reddish body — leans toward damp-heat"
   - a FINDINGS list; each row = a plain-English sign name, a short "traditionally
     associated with…" gloss, a five-flavour colour tag, and a gentle 0-100 severity bar.
     Example rows: "Thick, greasy coating — associated with dampness"; "Reddish body colour
     — associated with heat"; "Redder tip — mild, worth watching".
   - a constitution-leaning chip (e.g. "Damp-heat") with a note that it agrees with / differs
     from the quiz result
   - a "Today: do / avoid" two-column card in almanac form (e.g. do: light meals, mung bean,
     earlier nights / avoid: fried & greasy food, cold drinks, late nights)
   - a "You might also notice" row of symptom chips
   - a distinctiveness line ("thicker coating than about two-thirds of tongues") and an
     honest confidence note ("A tongue is a snapshot and shifts with food and sleep. This is
     for education, not medical diagnosis.")
5. History / trend: a timeline of past tongue readings (thumbnail + date + leaning) plus one
   gentle metric line (a habit-tracker style sparkline, e.g. "coating thickness over time"),
   NOT a clinical medical chart.

CONSTRAINTS
- Light + dark theme via CSS custom properties (support prefers-color-scheme AND a
  data-theme override on the root, both directions).
- Real English copy, no lorem. Tabular numerals for scores and dates.
- Tone: gentle, pressure-free, non-diagnostic. The cinnabar is a BRAND accent; use separate,
  quieter colours for state — nothing should read as an alarming error.
- Commit to the almanac / five-flavour system above; do not fall back to a generic template.
```

## Tips for iterating
- Ask for one screen at a time if you want more detail per screen.
- Paste `FEATURE_SPEC.md` when you want the copy/data to exactly match the analysis pipeline.
- If you have real tongue photos, ask it to swap the placeholder tongue illustration.
