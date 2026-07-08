Implementation Plan

### **Phase 0: Project Setup & Research** 

**Goal**: Establish foundation, reduce technical risk, and secure access to all key datasets and codebases.

#### **0.1 General Setup**

* Finalize system architecture document.  
* Create GitHub repository with modular structure (stage1\_quantitative/, stage2\_interpretation/, data/, evaluation/, deployment/).  
* Set up development environment (Python 3.10+, PyTorch, Hugging Face, LangChain, etc.).  
* Create shared documentation (Notion / Google Docs) for architecture decisions and progress tracking.

#### **0.2 Access Datasets and Code from Source Papers**

**TonguExpert (2025) – Primary dataset for rich phenotypes**

* Visit the official platform: [https://www.biosino.org/TonguExpert](https://www.biosino.org/TonguExpert)  
* Register for an account (free platform).  
* Download the **largest publicly released tongue image dataset** (5,992 images) along with the 773 phenotype labels.  
* Check the paper’s supplementary materials or the platform for the exact download link and license.  
* Review their methodology section for implementation details on SAM-based segmentation and feature extraction.

**SSC-Net (2025)**

* Read the full paper (Digital Health journal).  
* Look for the **“Code is available here”** link in the paper (usually in the abstract or methods section).  
* If no direct link, check the authors’ institutional pages or email the corresponding authors (Xiaopeng Sha or Zhaojun Chen) politely requesting the code.  
* Download the BUCM dataset if it is made available with the code.

**RTDS (2026)**

* The paper explicitly mentions the GitHub repository: [**https://github.com/byzhang811/RTDS-Tongue\_Analysis**](https://github.com/byzhang811/RTDS-Tongue_Analysis)  
* Clone the repository immediately.  
* Download or request access to their **2,100 real outpatient tongue images** (the most realistic dataset for commercial robustness).  
* Study their two-stage pipeline, U-Net++ segmentation code, and Focal Loss implementation.

**MMIR-TCM (July 2026\)**

* Paper GitHub link (stated in the paper): [**https://github.com/jw-chae/MMIR-TCM**](https://github.com/jw-chae/MMIR-TCM)  
* Clone the full repository.  
* Study the RAG implementation, prompt templates, and structured output format.  
* Check for access instructions to the **MedTCM dataset** (images \+ expert annotations \+ clinical records). If not publicly downloadable, follow the paper’s instructions to request access from the authors.

**Additional Useful Resources**

* Search for any updated links or supplementary materials on arXiv / journal pages.  
* Create a shared spreadsheet to track:  
  * Dataset download status  
  * Code repository links  
  * Access requests sent (with dates)  
  * Licenses and usage restrictions

#### **0.3 Documentation & Risk Management**

* Document all dataset licenses and citation requirements.  
* Create a risk register (data access delays, licensing issues, compute requirements).  
* Set up experiment tracking (Weights & Biases or MLflow).

**Deliverables of Phase 0**:

* Fully set up GitHub repository  
* All four main codebases cloned  
* TonguExpert and RTDS datasets downloaded (or access requested)  
* MMIR-TCM MedTCM dataset access initiated  
* Architecture document v1.0  
* Risk register

### **Phase 0: Project Setup & Research** 

**Goal**: Establish foundation and reduce technical risk.

* Finalize system architecture document (use the detailed description I provided earlier).  
* Set up repository structure (modular: stage1\_quantitative/, stage2\_interpretation/, data/, evaluation/, deployment/).  
* Choose tech stack:  
  * PyTorch \+ Hugging Face Transformers  
  * LangChain \+ FAISS (for RAG in Stage 2\)  
  * ONNX / TensorRT for optimization  
* Review source code/papers in detail:  
  * TonguExpert (SAM \+ feature extraction)  
  * SSC-Net (multi-task architecture)  
  * RTDS (two-stage \+ Focal Loss)  
  * MMIR-TCM (RAG \+ structured output)  
* Create data usage agreements / ethics checklist (especially for clinical data).

**Deliverable**: Architecture spec \+ GitHub repo skeleton \+ risk register.

---

### **Phase 1: Data Pipeline & Curation** 

**Goal**: Build a robust, combined training dataset.

**Key Tasks**:

* Download and explore **TonguExpert dataset** (5,992 images \+ 773 phenotypes) — primary source.  
* Obtain/access **RTDS 2,100 real outpatient images** for robustness.  
* Explore **MMIR-TCM’s MedTCM dataset** for interpretation training.  
* Build unified data loader with:  
  * Patient-level splits (prevent leakage)  
  * Real-world augmentations (lighting, angles, phone artifacts)  
  * Quality filtering (blur, exposure, partial tongue)  
* Create paired dataset for Stage 2: (Stage 1 structured output \+ expert interpretations)  
* Implement data versioning (DVC or similar).

**Deliverable**: Versioned dataset \+ data pipeline code \+ statistics report.

---

### **Phase 2: Stage 1 – Quantitative Scoring Layer** 

**Goal**: Build the hybrid quantitative engine.

**Sub-phases**:

**2.1 Segmentation Module (3–4 weeks)**

* Implement TonguExpert’s SAM-based segmentation as baseline.  
* Fine-tune/adapt using RTDS’s U-Net++ training protocol on mixed data.  
* Add robustness techniques from RTDS (real noisy images).

**2.2 Multi-Task Feature Extraction & Scoring (6–8 weeks)**

* Build shared encoder backbone (start with ResNet-34 \+ lightweight Swin block from RTDS).  
* Implement SSC-Net style components:  
  * Segmentation mask → feature masking  
  * ROI extraction \+ bottom-up fusion  
  * Multi-label \+ regression heads for the 5 key characteristics  
* Add TonguExpert-style rich phenotype extraction head (global \+ local features).  
* Integrate Focal Loss (RTDS) for imbalance.

**2.3 Training & Validation**

* Pretrain on TonguExpert dataset.  
* Fine-tune on RTDS real data.  
* Multi-task loss combination (Dice \+ classification \+ regression \+ consistency losses).

**Deliverable**: Trained Stage 1 model \+ inference script that outputs structured JSON.

---

### **Phase 3: Stage 2 – Clinical Interpretation Layer** 

**Goal**: Adapt and enhance MMIR-TCM for enriched input.

**Key Tasks**:

* Clone and set up MMIR-TCM codebase.  
* Modify the pipeline to accept **rich structured JSON** from Stage 1 instead of (or in addition to) raw image \+ basic description.  
* Enhance the RAG component:  
  * Expand knowledge base with more TCM patterns and clinical cases.  
  * Improve prompt templates for richer input.  
* Fine-tune (LoRA) the Qwen3-based components on enriched data.  
* Implement TDEU-style evaluation or custom clinical accuracy metrics.

**Deliverable**: Stage 2 module that takes Stage 1 JSON \+ metadata and outputs natural-language clinical report.

---

### **Phase 4: Full Pipeline Integration** 

**Goal**: Connect both stages into one seamless system.

**Tasks**:

* Build orchestration layer (e.g., run\_full\_analysis(image, metadata)).  
* Define clean JSON interface between Stage 1 and Stage 2\.  
* Add optional user metadata input (chief complaint, symptoms, history).  
* Implement confidence scoring and quality gates (e.g., reject very poor images).  
* Add basic explainability (Grad-CAM from Stage 1 \+ RAG citations from Stage 2).  
* Create simple web demo (Streamlit or Gradio) for internal testing.

**Deliverable**: End-to-end working prototype.

---

### **Phase 5: Training, Optimization & Efficiency (Ongoing, parallel with Phases 2–4)**

**Goal**: Make the model commercially viable (fast \+ cheap).

**Tasks**:

* Model optimization: Quantization (INT8), pruning, ONNX export.  
* Distillation (optional): Distill larger components into smaller models.  
* Inference optimization: vLLM or similar for Stage 2 LLM part.  
* Caching strategy for common RAG results.  
* Cost analysis (tokens per inference, GPU/CPU requirements).

**Deliverable**: Optimized inference pipeline with performance benchmarks.

---

### **Phase 6: Evaluation, Validation & Safety** 

**Goal**: Rigorously test accuracy, robustness, and clinical usefulness.

**Tasks**:

* Quantitative metrics:  
  * Stage 1: mAP, F1, regression error, segmentation Dice/IoU (on held-out sets).  
  * Stage 2: TDEU or adapted clinical accuracy \+ human evaluation.  
* Robustness testing on real smartphone photos (diverse lighting, angles, skin tones, partial tongue).  
* Clinical validation:  
  * Expert TCM practitioners review interpretation quality (blind comparison).  
  * Safety review (hallucination rate, contraindication handling).  
* User testing (internal beta).

**Deliverable**: Evaluation report \+ safety checklist.

---

### **Phase 7: Deployment, API & Commercialization** 

**Goal**: Make the system usable in production.

**Tasks**:

* Build REST API (FastAPI) with:  
  * Image upload endpoint  
  * Structured JSON output \+ natural language report  
  * Optional metadata input  
* Add authentication, rate limiting, and usage tracking.  
* Deploy options:  
  * Cloud (AWS/GCP) with spot instances  
  * On-device / edge option for Stage 1 (privacy)  
* Create documentation and SDK.  
* Basic frontend (web app or integrate into existing wellness/telehealth platform).  
* Compliance: Add strong disclaimers, audit logging, and human-in-the-loop options for high-risk outputs.

**Deliverable**: Production-ready API \+ demo application.

---

### **Phase 8: Iteration, Monitoring & Scaling (Ongoing)**

* Collect real usage data and feedback.  
* Continuously improve with new labeled data.  
* Expand knowledge base in RAG.  
* Add new capabilities (e.g., longitudinal tracking, multi-language support).  
* Explore partnerships for clinical validation or larger datasets.

