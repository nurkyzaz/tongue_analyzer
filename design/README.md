# 舌診 · Tongue feature design pack

Design proposal for adding a **tongue check-in (舌診)** to a *Savor / 計五味*-style TCM
wellness app. Built to sit beside the app's CCMQ constitution quiz, daily-habit tracking,
and 宜/忌 almanac widget — the tongue scan is a **second input into the same constitution
engine**, in the same gentle Cantonese voice.

## What's in here

| File | What it is |
|------|-----------|
| `DESIGN_PROMPT.md` | **The paste-in prompt.** Drop this into Claude (design / artifact mode) to generate the UI. Self-contained. |
| `BRAND.md` | Design tokens — palette, type, logo. Has **`<<FILL FROM CANVA>>` slots** for your exact brand values. |
| `FEATURE_SPEC.md` | The feature: screen-by-screen flow, real copy, and how each screen maps to the tongue-analysis pipeline output. |
| `mockup.html` | A working visual mockup of the screens (open in a browser, or view the published artifact). |

## How to use it

1. **Fill `BRAND.md`** with your real Canva colours / fonts / logo (I couldn't read the
   Canva link — it's login-gated). Everything below assumes a five-flavour placeholder palette.
2. **Paste `DESIGN_PROMPT.md`** into Claude design. If you filled `BRAND.md`, paste that too
   (or the prompt tells Claude to ask for it).
3. Iterate screen-by-screen using `FEATURE_SPEC.md` as the source of truth for content.

## The one idea

Savor is literally *計五味* — "measure the five flavours." The **tongue is the organ that
tastes them.** So the tongue check-in isn't a bolt-on medical scanner; it's the most
on-brand feature the app could have. The design leans into that: an almanac register,
findings tagged by their five-flavour / five-element colour, cinnabar seal used like a stamp,
and the same pressure-free tone ("睇下今日條脷" rather than "Diagnostic scan").

Framing stays **educational, non-diagnostic** — every reading is "traditionally associated
with…", matching the underlying tool's stance.
