# Handoff — TongueInsight (TIH) · state & next steps (2026-07-13)

Read this first. It's the self-contained "start here" for a fresh agent. Deeper detail is in the docs
linked below and in `git log`.

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

## What was done this session (all pushed)
Stage-1 signals: **coating split into thickness×texture** (COATING_SPLIT.md — thickness 82% vs conflated
55% on human), **red-tip voting** (validated), **moisture** (wet-only, honest). Stage-2 mapping:
**combination rules** (context: swelling flips yang↔damp-heat by colour/moisture; `mapping_testset.json`
+ `eval_mapping.py` = 12/12). Interpretation: **headline + distinctiveness + per-sign confidence**
honesty; **hybrid RAG+LLM** — grounded narrative (rule backbone intact) + a **true vector RAG** (faiss +
nomic embeddings + TF-IDF hybrid) over a **102-chunk cited corpus** (`knowledge_cards.json`:
constitutions, ICD-11 patterns, disambiguations, symptoms, combos, food therapy), retrieval **hit@4 96%**
(`eval_rag.py`). Data/testing: **labeled gallery** (`build_gallery.py` → `data/eval/gallery/` +
`gallery.html`), **fresh human40b** set (40 imgs, not yet labeled).

## Blocked on the user (do these when data arrives)
1. ✅ **human40b labeled** (2026-07-13) — folded into the eval (75 human images now). Confirmed v5 ~59%,
   coating-split thickness 82%, red_dots; tempered red_tip. See ACCURACY_INVESTIGATION §human40b.
2. **Real phone photos** across skin tones/lighting → settles color calibration (`eval_color_calib.py`)
   and measures true real-world accuracy. STILL NEEDED.

## Advised next steps (roughly by value)
1. **Grow the human eval** (label human40b) — it gates everything; then re-check v5 vs recent changes.
2. **Training-signal fix** needs *cleaner/more coating labels*, not loss re-weighting (v8 proved that).
   Consider harvesting expert-graded greasy-vs-thick data or per-characteristic fine-tuning.
3. **Keep growing the RAG corpus** (`knowledge_cards.json` → `build_corpus.py` → `rag.py --build` →
   `eval_rag.py`) with open-access (CC-BY) literature + remaining disambiguation pairs. Corpus quality =
   the RAG's ceiling. NEVER vendor copyrighted texts (Maciocia etc.).
4. **Before defaulting the LLM ON in production:** run a hallucination-rate check (does the narrative
   ever add a sign/claim not in the grounding?) and consider a stronger model. Rule backbone stays.
5. **Deployment:** easiest = containerize the FastAPI app on a cheap CPU box (0.34s/img), app = thin
   client or PWA of the existing demo. On-device (Core ML/TFLite) is the v2 privacy story.
6. **Licensing before commercial ship:** seg model trained partly on SM-Tongue (CC-BY-NC) — retrain seg
   without it or license. DINOv3 gated (using DINOv2 fallback).

## Map of the important files
- Plan & status: `docs/DIRECTION_REVIEW.md` (workstreams WS1-6, statuses), `docs/PROGRESS.md`.
- Accuracy story: `docs/ACCURACY_INVESTIGATION.md`, `docs/BENCHMARK.md`, `docs/LABEL_STORE.md`.
- Mapping: `docs/FEATURE_MAPPING_REFERENCE.md`, `docs/FEATURE_PATTERN_MAPPING.md`, `docs/COATING_SPLIT.md`.
- Interpretation/RAG: `docs/RAG_LLM_INTERPRETATION.md`.
- Color: `docs/COLOR_CALIBRATION.md`. Feedback round: `docs/FEATURE_FEEDBACK_2026-07-13.md`.
- Eval harnesses: `evaluation/{eval_model,eval_mapping,eval_rag,eval_color_calib,eval_coat_axes,eval_human_labels}.py`.

## Working style the user expects
Rigorous and honest: measure before promoting, report negatives plainly (v8/color-calib were kept OFF on
evidence), keep the rule engine auditable, ground the LLM, flag licensing. The user labels data and cares
about accuracy + genuinely-insightful (not generic) output.
