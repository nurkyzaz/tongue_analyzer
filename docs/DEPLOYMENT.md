# Deployment plan — tongue check-in → the app (v1)

Status: **proposal for team sign-off.** Decision leaning: **self-host on a cheap CPU box.**
This is the durable engineering record; the team-facing brief is the published artifact.

## TL;DR

- Serving inference needs **no GPU**. Full pipeline is **~0.34 s/image on CPU**.
- Ship the **deterministic report** (rule engine) at launch. It's already the WS7-redesigned,
  honest, graded output and the code's graceful-fallback path — $0 extra, auditable, no data leaves the box.
- The **RAG + LLM narrative is a fast-follow**, served *off* the cheap box, gated behind a
  hallucination-rate check. It is an enhancement, not a launch dependency.
- One **hard blocker before we charge money**: the seg model is trained partly on SM-Tongue
  (CC-BY-**NC** — non-commercial). Retrain without it or license it.

## What ships (the pipeline)

| Component | What it is | Cost to serve |
|---|---|---|
| Segmentation | U-Net++ (resnet34), `checkpoints/seg_combined` — real-photo Dice ~0.975 | CPU |
| Characteristics | mask-guided CNN, `checkpoints/multitask_v5` (5 core signs + severity) | CPU |
| Extra features | 2nd CNN, `checkpoints/extra_features` (red-dots etc.) | CPU |
| Geometry | `zoning.py` — red-tip, moisture, zoning (pure numpy/opencv, no model) | CPU |
| Interpretation | `interpret.py` + `tcm_knowledge.json` — distinctiveness-weighted votes + combination rules | CPU, deterministic |
| Narrative (optional) | RAG (faiss + nomic embeds) + LLM re-expression of grounded facts | **off-box, Phase 2** |

Three small CNNs, **~300 MB of weights total**, loaded once into a lazy singleton (`TongueService`).

**Honesty we carry into the product:** against the user's human labels the model is ~61% *exact
3-way*, but **within-1-grade 97%** and presence is strong. The app surfaces only benchmark-reliable
signs — coating **thickness**, body colour, fissures, toothmarks, strong red-tip, red-dots — and flags
texture (greasy/smooth) as "worth confirming." Everything stays **"traditionally associated with…"**,
non-diagnostic.

## The box

Stateless `POST /analyze` → JSON. No database needed for inference (history lives in the app backend
or on-device). At 0.34 s/image a single 2-vCPU box serves thousands of scans/day; scale is horizontal
and trivial when we need it.

| Option | Spec | Ballpark |
|---|---|---|
| **Hetzner CX22 (recommended)** | 2 vCPU / 4 GB / 40 GB | ~€4–5 / mo |
| DigitalOcean / Vultr | 2 vCPU / 4 GB | ~$18–24 / mo |
| Fly.io shared-cpu-2x | 2 vCPU / 4 GB | usage-based, ~$5–15 / mo |

4 GB RAM gives comfortable headroom over the ~1.5–2 GB resident footprint (torch-CPU + 3 models).
1 GB boxes are too tight for the torch import.

## The one open decision — the narrative

The demo currently writes the narrative with **Ollama gemma3:4B**. On a *cheap CPU* box that's
~10–30 s/report and 4–6 GB+ RAM — it breaks both "cheap" and the calm ~2–3 s UX. So:

- **A · Launch template-only (recommended).** The deterministic `interpret.py` report is already
  good (findings text, 宜/忌 recommendation, symptom chips, distinctiveness hooks, confidence note).
  It's the fallback the code already ships. Fast, free, auditable, **all image data stays on our box.**
- **B · Add the LLM narrative as a fast-follow, off-box.** Either (i) call **casper's GPU** as a
  narrative microservice, or (ii) a **hosted small-LLM API** (pennies/call). The rule engine stays the
  auditable backbone; the LLM only re-expresses grounded facts. **Gate on a hallucination-rate check**
  before defaulting it on (WS-D gate prerequisite). Note (ii) sends features/image off our box → revisit
  privacy copy + consent.

**Recommendation:** ship A now, do B as a measured fast-follow. This matches our discipline: don't
default the LLM on until we've shown it doesn't add unstated claims.

## Architecture

```
  Savor app (native; built from design/ pack)
        │  HTTPS, X-API-Key
        ▼
  cheap box ─ caddy/nginx (TLS) ─ uvicorn/gunicorn ─ FastAPI (deployment/api/app.py)
                                                        │  API key + CORS locked to app origin (both already in app.py)
                                                        ▼
                                            TongueService (lazy singleton) ─ FullPipeline
                                              seg → multitask_v5 → extra → zoning → interpret (rules)
                                                        │
                                            [Phase 2] → narrative microservice (casper GPU or hosted)
```
The existing web demo (`deployment/api/static/index.html`) doubles as the **PWA/reference client** and
the **internal QA harness** (run it against `data/eval/gallery/` — known answers).

## Container (WS-E — built & smoke-tested 2026-07-16)

`deployment/Dockerfile` (multi-stage: a `build-essential` builder compiles the few source wheels —
e.g. `stringzilla`, an albumentations dep — so the final `python:3.11-slim` image carries **no gcc**),
`deployment/requirements.txt` (CPU torch wheels, pinned to the casper env; **excludes** faiss /
scikit-learn / pandas — only the off-box RAG narrative needs those), `.dockerignore` (never bakes
`tongue_lit/` or datasets; re-includes only the 3 production checkpoints). Image **1.71 GB**.

**Default serving = template + CPU graph-RAG ensemble, NO LLM** (`TIH_WSC_ENSEMBLE=1`, no `TIH_LLM_*`):
the matcher/ensemble fall back to the pure-graph ranking, so patterns still carry **book citations +
confidence %** with zero LLM cost and no image data leaving the box. The LLM narrative + matcher `why`
are the off-box Phase-2 toggle (uncomment `TIH_LLM_*` in `docker-compose.yml`).

```bash
docker compose -f deployment/docker-compose.yml up -d --build     # build + run
curl -fsS localhost:7860/health
# smoke-tested: POST /analyze on a real photo -> full reading in ~1.45 s CPU, ensemble applied
#   (graph-rag-fallback), patterns cited + %-scored, WS-B follow-up questions present.
```

## Public surface has NO external attribution (`TIH_SHOW_CITATIONS`, default false)

Book/author/database names, URLs, and section codes ground the reasoning **internally** but must not
reach the client. `TIH_SHOW_CITATIONS=false` (the default) strips them from the response JSON right
before it's returned — the KG, matcher, and refinement engine still build **with** citations; only the
outbound surface is redacted (`interpret.strip_external`):
- pattern cards drop `citations` / `evidence`; rule `reasoning[].cite` codes are blanked; top-level
  `sources` (the Sources-sheet payload) is emptied.
- all free text (LLM narrative, `regions` notes, `why`, findings, advice) is passed through a recursive
  redactor that rewrites any source name/URL/§-code to generic "traditional texts" phrasing.
- the UI (`static/index.html`) hides the **Sources** button + citation rows when `show_citations` is
  false and shows a generic "grounded in traditional tongue-diagnosis knowledge" note; the hard-coded
  demo data carries no source names either.
- kept for the client: `patterns` (+ `confidence_pct`), `headline`, `recommendation`/advice, `followup`
  (WS-B questions), `report` — all attribution-free.

Set `TIH_SHOW_CITATIONS=true` only for internal QA/debug (returns the full cited payload). **Set it
`false` (or leave unset) in every public/cloud environment and on the casper demo.** Logs are clean by
construction: the response is stripped before return, and uvicorn access logs carry no bodies.
Verified: `POST /analyze` on a real photo returns zero external names / codes / URLs.

## Rollout

1. **Containerize — DONE.** Dockerfile + pinned requirements + baked 3 checkpoints (see above). `/health`
   works; `/analyze` verified in-container. Remaining ops (need the box): push image to a registry.
2. **Provision** the cheap box + domain + TLS. Set `TIH_API_KEY` and `TIH_CORS_ORIGINS=<app origin>`.
3. **Deploy + smoke-test** `/health` and `/analyze` on gallery images (known answers) before wiring the app.
4. **App team wires** the tongue feature (design pack) to `POST /analyze` → renders the reading screen.
5. **Soft launch** → this is also how we finally collect **real phone photos**, which settles the
   pending color-calibration and real-world-accuracy items.

## Before we charge money (risks / must-clear)

- **Licensing (hard blocker for paid ship):** `seg_combined` is trained partly on **SM-Tongue
  (CC-BY-NC-4.0, non-commercial)**. Fix = retrain seg on TonguExpert-only (we have it) or license
  SM-Tongue. (DINOv3 was gated → we used a DINOv2 fallback, but only as an offline auto-labeler — not shipped.)
- **Non-diagnostic framing + disclaimers** on every screen (design already does this). Educational
  wellness tool, not a medical device.
- **Privacy:** template-only path keeps all image data on our box — a real selling point for a
  health-adjacent app. Adding a hosted LLM (option B-ii) changes that; revisit consent/copy first.
- **Real-phone eval** still pending (blocked on real photos) — soft launch closes it.
