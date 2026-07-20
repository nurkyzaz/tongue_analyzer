# Handoff — TongueInsight (TIH) · state & next steps (2026-07-16)

Read this first, then **[PLAN.md](PLAN.md)** — the single source of truth for remaining work (Stage 2
RAG overhaul). Deeper history is in `docs/PROGRESS.md`, `docs/archive/`, and `git log`.

**Current focus:** the Stage 2 overhaul — a macro-micro **knowledge graph** built from newly-licensed
TCM books (`tongue_lit/`, git-ignored/copyrighted), a grounded cite-or-abstain matcher, interactive
symptom-evidence refinement, and a RAGAS eval gate. Stage 1 is **frozen** (v5). See PLAN.md §3.

## What the project is
An **educational** wellness app: photo of a tongue → detected visual features → a grounded
traditional-Chinese-medicine (TCM) reading. **Not a medical device**; every output is framed as "in this
tradition, … is associated with …", never a diagnosis. Two stages:

- **Stage 1 (image → features)** — `stage1_quantitative/`. Segmentation (U-Net++), a mask-guided
  multi-task CNN reading the 5 core signs + a second CNN for 8 extra features, plus training-free
  **geometry** (`zoning.py`: red-tip, moisture, zoning). Emits a structured JSON (`schema.py`).
- **Stage 2 (features → reading)** — `stage2_interpretation/`. A **rule engine** (`interpret.py` +
  `knowledge_base/tcm_knowledge.json`) computes the pattern via distinctiveness-weighted votes +
  **combination rules**; then a **grounded RAG+LLM** layer writes the narrative (falls back to a
  deterministic template). This split is deliberate: rules are the auditable/testable backbone; the LLM
  only re-expresses grounded facts.

## Where it runs (IMPORTANT — most work is server-side)
- **Server `casper` = `192.168.1.184`**, GPU0 (2× RTX 3090). `ssh nurkyz@192.168.1.184`. Project at
  `~/tongue/`; python is `envs/tih/bin/python`. **Data, checkpoints, datasets live on the server, not in
  git.** Code is synced local↔server with `rsync` (edit locally, `rsync -az -e ssh <file> nurkyz@192.168.1.184:~/tongue/<path>`).
- **Inference is CPU-fast: 0.34s** full pipeline (no GPU needed to *serve*).
- **GitHub:** `github.com/nurkyzaz/tongue_analyzer`. Push: `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519" git push origin main`.
  Commit/push only when the user asks; branch off main isn't required here (solo repo, user pushes to main).

## Production models (do not silently swap)
- Seg: `checkpoints/seg_combined/best.pt` (real-photo Dice ~0.975).
- Characteristics: **`checkpoints/multitask_v5/best.pt`** — v6/v7/v7b/**v8 all FAILED to beat it on the
  honest human metric** (bigger models & rebalancing fit noisy auto-labels). Extra features:
  `checkpoints/extra_features/best.pt`.

## The one thing to internalize: the honest metric
Auto-labels (TonguExpert) are optimistic (~0.87). Against the user's **human labels the model is ~61%**
(`evaluation/eval_model.py --source human`, on `evaluation/human40_labels.json` / `data/eval/human40`).
Coating (esp. greasy-vs-thick) is the weakest axis. **Only promote a model/KB change if it beats v5 on
the human eval or the mapping test — not on val/auto metrics.** This discipline is why v5 is still prod.

## Demo (live)
FastAPI + web page (`deployment/api/`), `uvicorn ... :7860`, public HTTPS via a cloudflared tunnel.
**Currently launched WITH the local LLM (qwen2.5:14b) + WS-C ensemble + citations HIDDEN.** Restart
recipe that works (ssh returns 124/"Terminated" but the detached process launches fine):
```
ssh nurkyz@192.168.1.184 'pkill -9 -f "uvicorn deployment.api.app"; sleep 2'
ssh nurkyz@192.168.1.184 'cd ~/tongue && source envs/tih/bin/activate && setsid nohup env \
  TIH_SHOW_CITATIONS=false TIH_WSC_ENSEMBLE=1 TIH_LLM_BACKEND=openai \
  TIH_LLM_BASE_URL=http://localhost:11434/v1 TIH_LLM_MODEL=qwen2.5:14b-instruct TIH_LLM_API_KEY=ollama \
  uvicorn deployment.api.app:app --host 0.0.0.0 --port 7860 > logs/demo.log 2>&1 </dev/null & disown'
```
LLM needs **Ollama** on `:11434` (no auth). Without the LLM env the demo serves the deterministic
template + CPU graph-RAG ensemble (always works). **`TIH_SHOW_CITATIONS=false` (default) keeps all
external references (book/author/DB names, URLs, § codes) OUT of the response, UI, and logs** — internal
grounding is untouched; set `true` only for internal QA. See DEPLOYMENT.md § "Public surface has no
external attribution". `pgrep "uvicorn deployment.api.app"` also matches your own ssh cmd string — check
port 7860 / `ss -ltn` instead.

## What's new (2026-07-16) — Stage 2 KG overhaul started
Consolidated all planning into **PLAN.md** (v2, single source of truth); archived 17 superseded docs →
`docs/archive/`, 11 one-off scripts → `evaluation/archive/`. **WS-A knowledge graph built**
(`stage2_interpretation/kg/`):
- `graph.py` — typed `KnowledgeGraph` with forward `feature→pattern` **and inverse `symptom/question→
  pattern`** edges (the WS-B lever). `build_kg.py --verify` seeds it from `tcm_knowledge.json` and asserts
  **strict superset parity** (enabling the KG changes nothing day-1). `parse_book.py` parsed Gerlach's 184
  sections → `book_sections.json` (macro layer; metadata+offsets only, never the copyrighted text). Graph =
  359 nodes / 427 edges / 10 rules.
- `micro_extract.py` — offline LLM triplet extractor (casper GPU 0, Ollama, **free**), cite-or-abstain.
  `normalize.py` maps extracted triplets → our canonical vocab (Gerlach Latin → our keys) as mapped /
  candidate / junk. **Model comparison (Gerlach ch.2): qwen2.5:14b chosen** over gemma3:4b — more faithful,
  uses our keys, 0 junk. `review_triplets.py` builds a local QA page.
- **Next:** full chapters 2–4 extraction with qwen14b → write `build_kg.add_micro_layer` → WS-C grounded
  matcher (shadow) + WS-B refinement engine. A **design bundle** for the app is in
  `prompt-execution-request/` (Savor 舌 tab; maps 1:1 onto WS-B/WS-C/Sources — see PLAN.md §5).

Prior rounds (pushed): coating split thickness×texture, red-tip/moisture, combination rules
(`eval_mapping.py` 12/12), hybrid RAG+LLM over a cited corpus (`knowledge_cards.json`, retrieval hit@4
96%), labeled gallery, human40+human40b eval sets. Detail in `docs/archive/` + `git log`.

## Blocked on the user (do these when data arrives)
1. ✅ **human40b labeled** (2026-07-13) — folded into the eval (75 human images now). Confirmed v5 ~59%,
   coating-split thickness 82%, red_dots; tempered red_tip. See ACCURACY_INVESTIGATION §human40b.
2. **Real phone photos** across skin tones/lighting → settles color calibration (`eval_color_calib.py`)
   and measures true real-world accuracy. STILL NEEDED.

## Advised next steps (per PLAN.md sequence)
1. **WS-A — substantially done.** Micro layer folds **3 books** (Gerlach ch.2–7 + Oriental + Maciocia,
   `--mode title` for the un-numbered ones) = **282 cited edges, 0 junk**; **WHO-IST ontology spine**
   (`who_terms.py`→`who_spine.json`) tags 25 nodes with code + 中文 + pīnyīn. Graph 605 nodes / 1245 edges
   / 282 snippets, parity OK. Remaining polish: broaden `normalize.py` aliases for Gerlach's Latin
   candidates (ira/ventus/repletio — held on purpose, no honest map); Chinese textbooks blocked (no
   source files). To add a book: `parse_book.py [--mode title]` → `micro_extract.py --book-id X
   --chapters all` on casper → `build_kg.py --verify` (globs all `book_triplets_*.json`).
2. **WS-C — built & gated.** Matcher (`kg/matcher.py`) + shadow run (0 halluc, 0.50 top-1 vs rules) →
   **ensemble** (`kg/ensemble.py`, rule prior + cited matcher evidence, α=0.35) wired into `interpret.py`
   behind `TIH_WSC_ENSEMBLE` (**default OFF**). **α sweep picked α=0.2** (default): stability-vs-rule
   0.85, lead-cited 0.90, 0 hallucination, **WS-D faithfulness 0.929** (≈ rule-only 0.936) — vs α=0.35
   (0.75 / 0.925 / 0.868). Citations attach independent of α, so α=0.2 keeps grounding while recovering
   faithfulness → the tradeoff is gone. Raw `confidence_pct` now on every card + in the UI. **Promoted default-ON & LIVE** (casper demo, qwen2.5:14b) — verified end-to-end. **WS-B
   refinement engine** is now the active build.
3. **WS-B — DONE (2026-07-16).** `kg/refine_engine.py`: two-pass reading — `select_questions` (info-gain
   over the KG's `probes` edges, disambiguates the top-2, covers both) + `rescore` (folds answers over
   answer→pattern edges, re-ranks the whole set). Wired into `interpret._followup_block` + `/refine`
   pass-2 + the frontend. Verified live (t12: spleen overtakes phlegm to 74% after answers). Log-odds
   `refine()` is the interim fallback.
4. **WS-E — containerized (2026-07-16).** `deployment/Dockerfile` (multi-stage CPU torch, pinned
   `requirements.txt`, baked 3 checkpoints, `.dockerignore` keeps books/datasets out), `docker-compose.yml`.
   Image 1.71 GB, built + smoke-tested: `docker compose -f deployment/docker-compose.yml up -d --build`
   → `/analyze` ~1.45 s CPU, template + graph-RAG ensemble (cited, LLM-less), WS-B questions. LLM narrative
   = off-box toggle. **Remaining (needs the box): provision cheap CPU host + TLS + API key/CORS, push image,
   app wires `POST /analyze`.** Then the seg SM-Tongue (CC-BY-NC) licensing blocker before charging money.
4. **WS-D eval gate** — adopt **RAGAS** (faithfulness) as the hallucination gate before defaulting the LLM
   ON; expand `eval_mapping.py`; keep TCMEval-SDT (rules 69.7%) and `eval_rag.py`.
5. **WS-E deploy** — containerize FastAPI on a cheap CPU box (0.34s/img); narrator off-box; template is the
   always-on fallback. **Licensing before ship:** seg uses SM-Tongue (CC-BY-NC) → retrain without it or
   license; DINOv3 gated (DINOv2 fallback); surface book snippets to users only if the grant allows.
6. **WS-F output design — phone demo BUILT (2026-07-16).** `deployment/api/static/index.html` now
   recreates the `prompt-execution-request/project/TongueInsight.dc.html` design as a live phone UI
   (Capture → Analysing → Reading → Refine → Sources), wired to `/analyze` + `/refine`, with a `?demo=1`
   offline fixture. See PROGRESS.md Phase 4. Remaining design polish: 十日之間 trend (needs app-side
   history storage — intentionally omitted rather than fabricated), citation+snippet licensing toggle.

**Plan additions (2026-07-16):** community-tool framing + a usable-sources list in
[`INTERNET_RESOURCES.md`](INTERNET_RESOURCES.md); an **evaluated enhancement backlog** (the A–E
recommendations, each scored through a *cheap-mobile-deploy* lens) in [`PLAN.md`](PLAN.md) §7 — accept all
offline KB work + graph-RAG + JSON-mode + feedback loop; defer per-request-heavy items (HyDE,
32B serving model, self-consistency, full Stage-1 retrain).

**Stage-2 RAG progress (2026-07-16):**
- WS-A: KG micro layer now covers **3 books** — Gerlach ch.2–7 + Oriental Tongue Diagnosis + Maciocia
  (**282 cited edges / 282 snippets, 0 junk**); **WHO-IST 2022 ontology spine** tags 25 nodes with
  canonical code + 中文 + pīnyīn (`who_terms.py`, `who_spine.json`). Graph 605 nodes / 1245 edges. §7-A
  weight recalibration applied **in the graph-RAG layer** (`kg/retrieval.py`) — seed weights untouched.
- WS-C: **graph-RAG retrieval** (`kg/retrieval.py`, `graph.neighborhood`) + **grounded cite-or-abstain
  matcher** (`kg/matcher.py`, JSON-mode). **Shadow run on real human40 (40 imgs):** hallucination-rate
  **0.0**, top-1 agreement **0.50** vs rules, Jaccard 0.48, disagreements within-family. **Verdict:
  ensemble (cited evidence + prior), don't replace the rule ranker.**
- WS-D: **RAGAS-style faithfulness gate** (`evaluation/eval_faithfulness.py`) — local claim-grounding
  judge over the LLM narrative. **faithfulness 0.936 (12 imgs) → GATE PASS** (`<0.85 ⇒ template only`,
  env `TIH_FAITHFULNESS_MIN`).
- Rule engine remains **production**; the matcher runs in shadow. Promote on the numbers.

## Map of the important files
- **Plan (SoT):** `docs/PLAN.md`. Architecture: `docs/ARCHITECTURE.md`. Status board: `docs/PROGRESS.md`.
- Knowledge graph (WS-A): `stage2_interpretation/kg/` (`graph.py`, `build_kg.py`, `parse_book.py`,
  `micro_extract.py`, `normalize.py`, `who_terms.py`, `retrieval.py`, `matcher.py`, `ensemble.py` (WS-C),
  `refine_engine.py` (WS-B), `README.md`).
  Curated data: `knowledge_base/{book_sections.json, who_spine.json}` (tracked); `book_triplets_*.json`,
  `who_terms.json`, `kg_graph.json` are git-ignored rebuildable artifacts.
- Living mapping refs: `docs/FEATURE_MAPPING_REFERENCE.md`, `docs/FEATURE_PATTERN_MAPPING.md`,
  `docs/LABEL_STORE.md`, `docs/BENCHMARK.md`, `docs/LABELING_GUIDE.md`.
- Historical detail (accuracy/RAG/coating/color investigations): `docs/archive/`.
- Living eval harnesses: `evaluation/{eval_model,eval_mapping,eval_rag,eval_coat_axes,eval_extra_features,
  eval_seg,eval_stage1,benchmark,benchmark_syndrome}.py` (one-offs archived under `evaluation/archive/`).
- App design bundle: `prompt-execution-request/` (Claude Design handoff — the Savor 舌 tab comps).

## Working style the user expects
Rigorous and honest: measure before promoting, report negatives plainly (v8/color-calib were kept OFF on
evidence), keep the rule engine auditable, ground the LLM, flag licensing. The user labels data and cares
about accuracy + genuinely-insightful (not generic) output.
