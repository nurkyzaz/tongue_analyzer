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
**Currently launched WITH the local LLM + RAG** (Ollama). Restart recipe that works (ssh returns
124/"Terminated" but the detached process launches fine):
```
ssh nurkyz@192.168.1.184 'pkill -f "uvicorn deployment.api.app"; sleep 2'
ssh nurkyz@192.168.1.184 'cd ~/tongue && source envs/tih/bin/activate && setsid nohup env \
  CUDA_VISIBLE_DEVICES=0 TIH_LLM_BACKEND=openai TIH_LLM_BASE_URL=http://localhost:11434/v1 \
  TIH_LLM_MODEL=gemma3:latest TIH_LLM_API_KEY=ollama \
  uvicorn deployment.api.app:app --host 0.0.0.0 --port 7860 >> logs/api.log 2>&1 </dev/null & disown'
```
LLM/RAG needs **Ollama** on `:11434` (models: `gemma3:latest` chat, `nomic-embed-text` embeddings — both
local, no auth). The shared vLLM `:8000` needs a key we don't have. Without the env vars, the demo serves
the deterministic template (always works).

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
1. **Finish WS-A:** run the full Gerlach chapters 2–4 micro-extraction with **qwen2.5:14b** on casper
   (free, ~15 min), expand `normalize.py`'s alias map to catch its candidates, then write
   `build_kg.add_micro_layer` (validated against real triplets) so the graph gets cited book edges.
   (Later: WHO-terminology ontology spine; parse Maciocia too.)
2. **WS-C grounded matcher** in *shadow mode* — LLM cite-or-abstain over the KG, logged alongside the rule
   engine on the gallery; promote on the numbers. Add raw confidence % back to the output (quick win).
3. **WS-B refinement engine** — symptom-evidence re-scoring + information-gain question selection over the
   KG's `evidence_for` edges (UI-agnostic; pairs with the design's Refine flow).
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

## Map of the important files
- **Plan (SoT):** `docs/PLAN.md`. Architecture: `docs/ARCHITECTURE.md`. Status board: `docs/PROGRESS.md`.
- Knowledge graph (WS-A): `stage2_interpretation/kg/` (`graph.py`, `build_kg.py`, `parse_book.py`,
  `micro_extract.py`, `normalize.py`, `review_triplets.py`, `README.md`).
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
