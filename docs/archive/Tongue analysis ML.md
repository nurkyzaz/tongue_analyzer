### **Model Architecture**

**TongueInsight Hybrid (TIH)** — A modular, two-stage system for precise quantitative tongue analysis \+ clinically grounded interpretation, optimized for commercial use on real user photos.

### **High-Level Architecture**

The system follows a clean **two-stage pipeline**:

1. **Stage 1: Quantitative Scoring Layer** (Hybrid of TonguExpert \+ SSC-Net \+ RTDS) → Produces rich, structured quantitative output (scores \+ hundreds of phenotypes \+ mask).  
2. **Stage 2: Clinical Interpretation Layer** (Based on MMIR-TCM) → Takes the structured output from Stage 1 \+ optional patient metadata and generates natural-language clinical reports with reasoning, grounded via RAG.

This design maximizes **accuracy** (best components from each paper), **richness** of quantitative data, **robustness** to real smartphone photos, and **clinical usefulness** of the final output.

---

### **Stage 1: Quantitative Scoring Layer (Hybrid)**

**Goal**: Extract the most accurate and richest possible quantitative representation of the tongue from real-world photos.

**Why Hybrid?**

No single paper has everything we need:

* TonguExpert → Richest feature set (773 phenotypes)  
* SSC-Net → Best multi-task synergy on clinically critical characteristics  
* RTDS → Best robustness to noisy real outpatient images

We combine them into an enhanced two-stage pipeline.

#### **1.1 Segmentation & ROI Extraction**

**Primary Source**: TonguExpert (2025) \+ RTDS (2026)

* **What to take**:  
  * TonguExpert’s **SAM-based segmentation** (Segmentation Anything Model) as the foundation — it delivers state-of-the-art generalization and fine-grained masks.  
  * RTDS’s **U-Net++** training recipe and robustness techniques (trained on real noisy clinical images with glare, clutter, and pose variation).  
* **Why**:  
  * TonguExpert’s SAM approach is more modern and generalizes better than older U-Net variants.  
  * RTDS proves superior performance on actual outpatient photos (the exact distribution we face in commercial use).  
* **Implementation**:  
  * Use TonguExpert’s SAM pipeline as the core.  
  * Fine-tune/adapt it using RTDS’s real 2,100-image dataset and training protocol (ImageNet-pretrained ResNet-34 encoder, Dice loss, etc.).  
  * Output: High-quality binary mask \+ clean ROI image (black background).

**Links**:

* TonguExpert paper & platform: [https://www.biosino.org/TonguExpert](https://www.biosino.org/TonguExpert)  
* RTDS paper: [https://link.springer.com/article/10.1007/s11760-026-05316-3](https://link.springer.com/article/10.1007/s11760-026-05316-3) (GitHub mentioned in paper: [https://github.com/byzhang811/RTDS-Tongue\_Analysis](https://github.com/byzhang811/RTDS-Tongue_Analysis))

#### **1.2 Multi-Task Feature Extraction & Scoring**

**Primary Sources**: SSC-Net (2025) \+ TonguExpert (2025) \+ RTDS (2026)

* **What to take**:  
  * **SSC-Net’s multi-task architecture**:  
    * Shared encoder  
    * Use segmentation mask to mask redundant background features (key innovation)  
    * ROI extraction module \+ bottom-up feature fusion  
    * Fine-grained multi-label classification module  
  * **TonguExpert’s rich phenotype extraction**:  
    * Extract the full set of **773 phenotypes** (global \+ local features for color, shape, texture, fissures, tooth marks, subregions, etc.)  
    * Add regression heads for continuous severity measures (e.g., crack depth, coating coverage %)  
  * **RTDS elements**:  
    * Hybrid backbone (ResNet-34 \+ Swin Transformer block)  
    * Focal Loss for class imbalance and label ambiguity  
    * Training on real noisy data  
* **Why this combination is best**:  
  * SSC-Net shows that joint segmentation \+ classification with feature masking significantly boosts performance on the 5 clinically most important characteristics.  
  * TonguExpert provides by far the richest quantitative output (hundreds of features), which is extremely valuable for the downstream interpretation layer.  
  * RTDS adds the robustness and imbalance handling needed for commercial photos.  
* **Backbone Recommendation**:  
  * Start with a lightweight hybrid (ResNet-34 \+ lightweight Swin block, as in RTDS) or a more modern efficient hybrid (e.g., ConvNeXt or distilled TransNeXt-style).  
  * Keep total parameters reasonable (\~30–50M) for commercial efficiency.  
* **Output of Stage 1** (Structured JSON):  
  JSON

{  
  "segmentation\_mask": "...",  
  "key\_characteristics": {  
    "body\_color": {"value": "pale\_red", "confidence": 0.87},  
    "coating\_thickness": {"value": "thick", "regression\_score": 0.72},  
    "cracks": {"present": true, "severity": 2.4, "locations": \[***...***\]},  
    "tooth\_marks": {"present": true, "severity": 1.8}  
  },  
  "rich\_phenotypes": \[773***\-dimensional*** ***feature*** ***vector*** ***or*** ***structured*** ***dict***\],  
  "overall\_quality\_score": 0.91

* }

**Links**:

* SSC-Net paper: [https://journals.sagepub.com/doi/10.1177/20552076251343696](https://journals.sagepub.com/doi/10.1177/20552076251343696) (Code available per paper)  
* TonguExpert: [https://www.biosino.org/TonguExpert](https://www.biosino.org/TonguExpert) (dataset \+ detailed methodology)

---

### **Stage 2: Clinical Interpretation Layer**

**Primary Source**: **MMIR-TCM** (arXiv July 2026\) — with minor adaptations.

* **What to take (almost everything)**:  
  * The overall three-stage philosophy and **RAG-enhanced reasoning** engine.  
  * Qwen3-based RAG component over clinical case bank \+ LiuJing Theory.  
  * Structured output format (Diagnosis, Dialectical analysis, Reason for diagnosis).  
  * TDEU evaluation metric (or adapted version).  
  * Training-free Memory-SAM idea (already used in Stage 1).  
  * Prompt templates and overall workflow that emulates expert TCM reasoning.  
* **Adaptations for our hybrid**:  
  * Replace or heavily enrich MMIR-TCM’s internal “Tongue Diagnosis Generator” (Qwen3-VL stage) with the **rich structured output from our Stage 1**.  
  * Feed the full JSON (key characteristics \+ rich phenotypes vector) \+ optional patient metadata (chief complaint, history, pulse) directly into the RAG stage.  
  * This gives the interpretation layer much higher-quality and more detailed quantitative input than the original MMIR-TCM had.  
* **Why MMIR-TCM is the best choice here**:  
  * It is currently the strongest system for generating **clinically meaningful natural-language interpretation** grounded in real cases and TCM theory via RAG.  
  * It directly solves the “black-box scores → meaningful insights” problem.  
  * Highly modular — easy to plug in better quantitative features from Stage 1\.

**Links**:

* MMIR-TCM paper: [https://arxiv.org/abs/2607.01814](https://arxiv.org/abs/2607.01814)  
* GitHub (per paper): [https://github.com/jw-chae/MMIR-TCM](https://github.com/jw-chae/MMIR-TCM)

---

### **Data Strategy**

* **Primary training data for Stage 1**:  
  * TonguExpert’s 5,992-image dataset (largest \+ richest labels) — main pretraining source.  
  * RTDS’s 2,100 real outpatient images — for robustness fine-tuning.  
  * BUCM dataset from SSC-Net (if accessible).  
* **For Stage 2 (Interpretation)**:  
  * Use/enhance MMIR-TCM’s MedTCM dataset.  
  * Optionally add paired (Stage 1 outputs \+ expert-written interpretations) for fine-tuning the RAG component.  
* **General**:  
  * Heavy real-world augmentation (lighting, angles, phone camera effects).  
  * Patient-level splits to avoid leakage.  
  * Mix of controlled \+ noisy images for best generalization.

---

### **Training Strategy**

* **Stage 1 (Quantitative)**:  
  * Pretrain on TonguExpert dataset using multi-task loss (SSC-Net style).  
  * Fine-tune with RTDS real data \+ Focal Loss.  
  * Use knowledge distillation if needed for efficiency.  
* **Stage 2 (Interpretation)**:  
  * Start with MMIR-TCM weights.  
  * Fine-tune the RAG component with enriched inputs from our Stage 1\.  
  * Use LoRA for efficiency.  
* **Overall**:  
  * Staged training (Stage 1 first, then Stage 2).  
  * Multi-task \+ consistency losses where applicable.

---

### **Final Output Example (What the User Sees)**

**Quantitative Summary** (can be shown or hidden):

* Body color: Pale red (87%)  
* Coating: Thick, white-yellow, greasy  
* Cracks: Moderate severity (score 2.4)  
* Tooth marks: Present, mild

**Clinical Interpretation** (main output):

"Your tongue shows a pale-red body with moderate swelling and tooth marks, along with a thick, greasy white-yellow coating. This pattern is commonly associated with **Spleen Qi deficiency with dampness** in Traditional Chinese Medicine. It may relate to symptoms such as fatigue, digestive sluggishness, or fluid retention. In Western terms, similar tongue features can sometimes correlate with signs of poor circulation or nutritional considerations. Consider consulting a healthcare professional for personalized advice."

---

### **Efficiency & Commercial Considerations**

* Stage 1: Aim for \<50M parameters \+ quantization/ONNX export for fast inference (mobile or cheap cloud).  
* Stage 2: Use LoRA \+ smaller distilled LLM where possible; cache common RAG results.  
* Modular design: Users can run Stage 1 locally (privacy) and call Stage 2 via API.  
* Cost: Very low per analysis after optimization.

---

### **Summary: What Comes From Where**

| Component | Main Source | Why | Secondary Sources |
| ----- | ----- | ----- | ----- |
| Segmentation | TonguExpert (SAM) | Best generalization | RTDS (U-Net++ robustness) |
| Multi-task learning & key characteristics | SSC-Net | Best synergy on clinical features | — |
| Rich phenotype extraction (700+ features) | TonguExpert | Richest quantitative output | — |
| Real-world robustness & Focal Loss | RTDS | Critical for commercial photos | — |
| Clinical interpretation \+ RAG reasoning | MMIR-TCM | Strongest grounded clinical output | — |

This hybrid represents the current state-of-the-art synthesis as of July 2026\.

