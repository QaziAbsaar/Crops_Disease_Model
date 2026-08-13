# Crop Disease Detection — Dataset & Training Strategy (Pakistan-focused)

Date: 2026-08-12

## Goal

Train a mobile, on-device crop disease classifier covering the 6 crops most
relevant to Pakistan's agriculture: Wheat, Cotton, Rice, Sugarcane, Citrus
(Kinnow), Mango. Training happens on Google Colab / Kaggle (no local
dataset downloads). Priority: real-world field robustness over clean
benchmark accuracy, since inference target is farmer phone photos in
uncontrolled field conditions.

## 1. Dataset → crop mapping

| Crop | Primary dataset | Supplement |
|---|---|---|
| Wheat | Wheat Leaf Disease Dataset (Kaggle) | Wheat subset of Plant Disease Dataset (71 Classes) |
| Cotton | Cotton Leaf Disease Detection Dataset (Kaggle, 2,137 orig / 7,000 aug, 7 classes) | Cotton subset of 71-Classes |
| Rice | **Gap in original 10 — add external source**: "Rice Leaf Diseases Dataset" (Kaggle, ~5,932 img, 4 classes: Bacterial Blight, Blast, Brown Spot, Tungro) | Rice subset of 71-Classes; optionally "Rice Diseases Image Dataset" (~120 field-shot images) for extra field-realistic val samples |
| Sugarcane | Sugarcane Plant Diseases Dataset (Kaggle, 19,926 img, 6 classes) | — |
| Citrus | Citrus Plant Disease Image Dataset (Kaggle, 7 categories) | — |
| Mango | Mango Leaf Diseases Dataset (Kaggle, 4,000 img, 7 diseases) | — |

Auxiliary-only (never used as final validation, clean lab backgrounds):
PlantVillage, New Plant Diseases Dataset, Crop Leaf Disease Dataset
(45 classes — verify crop overlap with the 6 target crops before use).

Field-domain pretrain pool: LeafNet (186k in-situ images, 22 species, 62
diseases, Hugging Face) — used for domain adaptation, not final classes.

## 2. Domain-adaptation training strategy

Problem: most Kaggle sets above are clean/uniform-background lab shots.
A model trained only on these degrades badly on real farmer photos
(clutter, glare, shadow, motion blur, varied phone cameras).

Three-stage transfer:
1. ImageNet-pretrained backbone (standard init).
2. Finetune on LeafNet field images — teaches real-world leaf/background/
   lighting distribution even though species differ from target 6 crops.
3. Finetune on the 6-7 crop-specific datasets (stage-1 + stage-2 heads,
   see architecture below).

Clean lab datasets get synthetic-background augmentation (segment leaf,
paste onto random field backgrounds sampled from LeafNet) before use in
training. They are never used for validation/test.

**Validation/test rule:** carve val/test splits only from in-situ /
field-shot images (LeafNet subset + any field-shot images within the
Kaggle sets). Clean white-background validation accuracy is not trusted
as a proxy for real-world performance.

## 3. Model architecture & mobile export

- **Teacher model:** EfficientNetV2-S or ConvNeXt-Tiny. Trained through
  the full 3-stage pipeline on Colab/Kaggle GPU, no size constraint.
- **Student model:** MobileNetV3-Large or EfficientNet-Lite2, trained via
  knowledge distillation from the teacher to recover accuracy at small
  size.
- **Two-stage hierarchical classification:**
  - Stage 1: crop identifier, 6-way (Wheat/Cotton/Rice/Sugarcane/Citrus/
    Mango), small model (~2-3MB quantized).
  - Stage 2: per-crop disease classifier, 6 separate heads (5-8 classes
    each), lazy-loaded on-device based on Stage-1 output (~4-6MB per
    head).
- **Export:** TFLite with INT8 post-training quantization (fall back to
  QAT if accuracy drop is too large). Target total footprint <15MB for
  Stage-1 + one active Stage-2 head.
- Validate quantized model on an actual mid-range Android device, not
  just emulator/desktop — quantization can silently tank accuracy.

## 4. Class imbalance & augmentation

- Focal loss or class-weighted cross-entropy — disease class distribution
  is uneven across sources (e.g. wheat blast likely underrepresented vs
  common rust).
- Oversample rare classes at the batch-sampling level, not loss weighting
  alone.
- Augmentation: color/white-balance jitter (simulate different phone
  cameras), random blur, synthetic glare/shadow, mixup/cutmix, plus the
  background-replacement augmentation from section 2 for lab-shot images.

## 5. Colab/Kaggle training pipeline

1. **Notebook 1 — ingest:** pull all 7 dataset sources directly into the
   Colab/Kaggle runtime (never to local disk), normalize into a unified
   `crop/disease/image.jpg` folder structure.
2. **Notebook 2 — Stage-1 training:** crop classifier, backbone +
   LeafNet domain-pretrain step.
3. **Notebook 3 (×6) — Stage-2 training:** one per-crop disease head,
   reusing the Stage-1 backbone as a finetuned feature extractor.
4. **Notebook 4 — distillation:** teacher → MobileNetV3 student, per
   stage (Stage-1 and each Stage-2 head).
5. **Notebook 5 — export & device test:** TFLite conversion, INT8
   quantization, on-device sanity test on a real phone.

## Open items / decisions deferred

- Exact class taxonomy per crop (which disease labels to merge/keep) —
  decide when inspecting each dataset's actual label set in Notebook 1.
- Whether "Crop Leaf Disease Dataset (45 classes)" has any of the 6
  target crops — verify before including as auxiliary data.
- No Pakistan-specific field photo set currently identified — v1 relies
  on LeafNet's general in-situ images for field-domain adaptation; a
  future v2 could collect real Pakistani farm photos for a proper
  in-domain validation set.
