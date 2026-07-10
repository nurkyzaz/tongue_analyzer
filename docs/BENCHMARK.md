# Accuracy Benchmark — model vs. expert labels

_Grounded in the categories professionals actually annotated. We report where we have **independent expert labels**; we do NOT report disease/constitution accuracy (no such ground-truth in our data — that would be circular)._

Models: `checkpoints/multitask_v5/best.pt` (characteristics), `checkpoints/extra_features/best.pt` (extra features).

## A. Key characteristics vs TonguExpert **expert (L1 manual)** gold — held-out test
| characteristic | n (expert-labeled) | accuracy | precision | recall | macro-F1 |
|---|---|---|---|---|---|
| coating color | 107 | 0.916 | 0.907 | 0.926 | 0.913 |
| tongue body color | 115 | 0.809 | 0.828 | 0.806 | 0.805 |
| fissures / cracks | 161 | 0.919 | 0.629 | 0.613 | 0.621 |
| tooth marks | 199 | 0.849 | 0.591 | 0.576 | 0.581 |
| **mean** |  | **0.873** |  |  | **0.730** |

## B. Pathological categories vs TCM-Tongue **practitioner** labels — held-out 553-img test
| category | n positives | Average Precision | precision | recall | F1 |
|---|---|---|---|---|---|
| peeled / mirror coating | 1 | 1.000 | 0.077 | 1.000 | 0.143 |
| red tongue body | 147 | 0.610 | 0.551 | 0.667 | 0.603 |
| purple / dusky body | 18 | 0.169 | 0.200 | 0.222 | 0.211 |
| swollen / enlarged body | 34 | 0.189 | 0.172 | 0.765 | 0.281 |
| thin body | 30 | 0.582 | 0.465 | 0.667 | 0.548 |
| red dots / prickles | 80 | 0.676 | 0.630 | 0.725 | 0.674 |
| grey-black coating | 3 | 0.051 | 0.077 | 0.667 | 0.138 |
| wet / slippery coating | 12 | 0.383 | 0.222 | 0.500 | 0.308 |
| **mean (mAP)** |  | **0.457** |  |  |  |

## Method & grounding
- **Expert labels:** (A) TonguExpert `L1_Labels_Manual` (human-verified); (B) TCM-Tongue licensed-practitioner annotations, held-out test split.
- **Held-out:** models never saw these test images in training.
- **Literature anchors:** SSC-Net reports ~0.85 F1 on the 5 characteristics; TCM-Tongue's own YOLO benchmarks and published constitution models (~0.71 acc, 'junior-practitioner level') are the comparison points. Numbers here are directly comparable for the characteristic/feature task.
- **Not benchmarked (honest scope):** TCM pattern / 9-constitution / disease outputs are a rule-based educational mapping (grounded in ICD-11 / CCMQ / Maciocia); we have no independent expert-labeled images for them, so we do not claim an accuracy number.
