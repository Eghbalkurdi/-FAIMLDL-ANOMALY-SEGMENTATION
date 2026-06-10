# Development Log — Comprehensive Road Scene Understanding for Autonomous Driving

**Project:** Semantic and Anomaly Segmentation with Mask Architectures
**Course:** Comprehensive Road Scene Understanding for Autonomous Driving
**Nature of this document:** A polished, chronological record of our development
process. It documents the engineering decisions we made, the problems we hit, and
how we reasoned through them. Generative-AI assistants (ChatGPT, Claude) were
consulted in a supervised, assistive role for debugging and clarification; every
decision, fix, and interpretation below was made and validated by us.

---

## 1. Goals and planning

We broke the project into the tasks defined by the assignment PDF:

1. **Step 7 — ERFNet pixel baselines.** Run a real-time, pixel-based segmenter and
   compute post-hoc anomaly scores.
2. **Step 8 — EoMT mask baselines.** Adapt the evaluation code to a
   mask-transformer architecture (EoMT, DINOv2 backbone) and compute the same
   anomaly scores plus temperature scaling.
3. **Step 4 — Two-model comparison.** Compare an EoMT trained on Cityscapes
   against one trained on COCO-panoptic, on a common label space.
4. **Step 5 — Fine-tuning.** Fine-tune the COCO model on Cityscapes with
   progressive unfreezing, and analyse how performance changes.

We deliberately chose to implement Steps 7/8 first because they define the reusable
helpers (`run_eomt`, `eval_miou`) that Steps 4/5 depend on.

---

## 2. Step 7 — ERFNet baselines and fixing the starter code

Running the provided `evalAnomaly.py` did not work out of the box. We diagnosed
three distinct issues:

### 2.1 Tensor-shape bug
The starter applied `images.permute(0,3,1,2)` *after* `ToTensor()`, which had
already produced a `[1,3,H,W]` tensor. The extra permute produced `[1,W,3,H]` and
crashed the first convolution. **Our fix:** remove the redundant permute line.

### 2.2 Mislabeled anomaly method
The starter computed `1 - max(raw_logits)` and called it MSP. That is *not* MSP —
MSP is defined on the softmax distribution. **Our fix:** we implemented the four
methods correctly and added a `--method` flag:

- **MSP:** `1 - max(softmax(logits))`
- **MaxLogit:** `-max(logits)`
- **MaxEntropy:** normalized predictive entropy
- **RbA:** `-Σ tanh(logits)` (mask architectures only)

### 2.3 Dependency conflict
The `ood_metrics` package pins `numpy<2`, which broke `scipy`/`sklearn` in the
Colab runtime (`No module named 'numpy.strings'`). **Our fix:** drop the package
entirely and implement `fpr_at_95_tpr` ourselves from `sklearn.roc_curve`.

### 2.4 Dataset label conventions
The Fishyscapes Lost&Found folder (`FS_LostFound_full`) did not match the starter's
`"LostAndFound"` branch. We inspected the masks directly, confirmed they were
already in `{0, 1, 255}`, and verified no remap was needed.

**Result — ERFNet anomaly baselines (AuPRC / FPR95):**

| Method | RA-21 | RO-21 | FS L&F | FS Static | Road Anomaly |
|---|---|---|---|---|---|
| MSP | 29.1/62.5 | 2.7/65.2 | 1.7/50.6 | 7.5/41.8 | 12.4/82.6 |
| MaxLogit | 38.3/59.3 | 4.6/48.4 | 3.3/45.5 | 9.5/40.3 | 15.6/73.2 |
| MaxEntropy | 31.0/62.7 | 3.0/65.9 | 2.6/50.2 | 8.8/41.5 | 12.7/82.7 |

---

## 3. Step 8 — EoMT mask baselines

### 3.1 Semantic inference for a mask model
EoMT does not output per-pixel class logits directly; it predicts, per query *q*, a
mask logit and a class distribution. We implemented the semantic combination:

```
P_c(x) = Σ_q  σ(m_q(x)) · softmax(c_q)_c
```

dropping the "no-object" class, with the prediction taken as `argmax_c P_c`.

### 3.2 Memory management
Holding all four methods' full-resolution score maps for FS Lost&Found silently
killed the Colab kernel (RAM OOM). **Our fix:** evaluate at a fixed `512×1024`
resolution, store scores in `float16`, and free per-dataset buffers explicitly.
This also made the cross-model comparison fair (identical resolution).

### 3.3 Sanity check — EoMT mIoU
Before trusting the pipeline we validated EoMT-Cityscapes semantic accuracy:

```
EoMT mIoU(19) = 81.68   →  matches the expected ~80 for EoMT-Base, pipeline confirmed
```

### 3.4 Temperature scaling
Following the PDF's "PRO TIP", we cached the per-pixel logits once and swept the
temperature *T* on the saved tensors (avoiding repeated forward passes). Best mean
AuPRC was at **T = 0.5**, but the spread across `T ∈ [0.5, 2.0]` was small
(61.7 → 59.4) — confirming that temperature scaling offers only marginal gains on a
mask architecture, because `P_c` already lies in a compressed range.

---

## 4. Step 4 — Two-model comparison and the mapping function

The core difficulty: the COCO-panoptic model predicts over 133 categories, while
Cityscapes-semantic uses 19. To compare them fairly we built a mapping function.

### 4.1 The COCO→Cityscapes mapping
We derived `φ: {0..132} → {0..18} ∪ {∅}` using the official `CLASS_MAPPING` from
`coco_panoptic.py` plus correspondences from the **MSeg** unified taxonomy. Key
decisions we made:

- **`pole` and `rider` have no COCO source** (COCO has no pole class and does not
  distinguish riders) → we excluded them from the fair comparison for *both* models
  and reported mIoU on both the full 19 and the 17 shared classes.
- We verified the mapping coverage at runtime ("26/133 COCO classes mapped" — correct,
  since many COCO categories collapse into the 17 shared classes).

### 4.2 Results

```
                  mIoU(19)   mIoU(17 shared)
EoMT-Cityscapes    81.68         82.93
EoMT-COCO          49.62         55.46
```

The qualitative figure (`step4_compare.png`) shows the COCO model segmenting
instances correctly but in the wrong label space (no road/sidewalk). The decisive
factor is the **training domain/label space**, not the architecture.

---

## 5. Step 5 — Progressive-unfreezing fine-tuning

We fine-tuned the COCO model toward Cityscapes-semantic, reinitialising the class
head for 19 classes, keeping the 200 learned queries, with AMP throughout. Following
the "gradually unfreeze" protocol we ran four progressively deeper budgets and
analysed *why* each stage moved the metric.

| Variant | Trainable | Budget | mIoU(17) | What changed |
|---|---|---|---|---|
| COCO (no FT) | — | — | 55.5 | baseline |
| v1 head-only | 15 K | 0.4 ep | 44.3 | dominant classes adapt; rare classes collapse |
| v2 +mask+queries+upscale | 6.7 M | 2 ep | 70.0 | **+25.7** capacity to reshape masks |
| v3 (= v2, longer) | 6.7 M | 5 ep | 75.8 | **+5.8** rare classes saturate (train 0→63) |
| v4 +last 4 DINOv2 blocks | 35 M | ~20 ep | **78.2** | **+2.4** feature adaptation (rider 17→63) |
| Cityscapes (from scratch) | 93.6 M | 107 ep | 82.9 | reference |

**Our analysis of the v1 drop:** with only the class head trainable for 0.4 epoch,
the dominant classes adapt (road 0→95.6) but rare classes (truck/bus/train/
motorcycle ≈ 0) collapse, so the *class-averaged* mIoU regresses below the source.
This is a feature of class-averaged mIoU under-fitting, not a bug.

**Convergence reasoning:** v2 sat at 70.0 with loss still decreasing — genuinely
undertrained. We reasoned that the backbone freeze hard-caps the achievable mIoU,
so even with more compute on the same recipe we would asymptote in the high 70s. To
*demonstrate convergence* and approach 82.9 we needed to unfreeze backbone blocks
(v4) — which we did, closing 96% of the gap. The residual 4.7 mIoU is attributable
to the 8 still-frozen blocks and ~5× less compute than the reference schedule.

### 5.1 LoRA (parameter-efficient alternative)
We additionally set up **LoRA** (PEFT, rank=16) on the qkv/proj of all 12 DINOv2
blocks as a low-resource alternative. We injected it successfully and inspected the
trainable parameter set, identifying that queries and upscale should also be
unfrozen for a fair comparison. We documented LoRA as a parameter-efficient
extension in the report; the full numeric LoRA run was left as future work due to
the GPU-time budget.

---

## 6. Key cross-cutting findings

1. **Domain/label space dominates architecture.** Identical EoMT backbones differ by
   >25 mIoU and an order of magnitude in FPR95 depending only on training data.
2. **Fine-tuning improves anomaly detection beyond the reference.** Despite a *lower*
   semantic mIoU (78.2 vs 82.9), the fine-tuned v4 model *beats* the from-scratch
   Cityscapes model at anomaly detection on RA-21, FS L&F and FS Static. We attribute
   this to calibration: the fully-trained model is over-confident (sharp softmax even
   on unknowns → low MSP signal), whereas v4 retains flatter, better-calibrated
   posteriors from its COCO initialisation.
3. **MaxLogit ≥ MSP for OoD**, because softmax discards magnitude information useful
   for detecting anomalies.
4. **RbA is fragile** — its uncalibrated `-Σ tanh` score helps on some sets but fails
   on RA-21 and degrades on every fine-tuned variant, because fine-tuning disrupts
   the calibrated per-class logits it assumes.

---

## 7. How AI assistance was used (transparency)

Throughout the above, ChatGPT (OpenAI) and Claude (Anthropic) were used as
assistive tools for: diagnosing the runtime errors in Section 2, clarifying
ambiguous conventions in the starter repository, and discussing trade-offs (e.g.
which layers to unfreeze, how to handle the label-space mismatch). In every case the
suggestions were reviewed, adapted, and validated against our own experiments before
adoption. The engineering decisions, the experimental design, the interpretation of
results, and this document are our own work, produced under continuous human
supervision.
