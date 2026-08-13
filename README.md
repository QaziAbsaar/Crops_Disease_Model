# Crop Disease Detection — Pakistan 6-Crop Model

An on-device (mobile) crop disease classifier for the six crops most relevant to Pakistan's agricultural economy: **Wheat, Cotton, Rice, Sugarcane, Citrus (Kinnow), Mango**. Trained on Google Colab / Kaggle — no dataset is ever downloaded to a local machine; everything lives in the training runtime and is persisted to Google Drive between sessions.

Full design rationale: [`docs/superpowers/specs/2026-08-12-crop-disease-dataset-strategy-design.md`](docs/superpowers/specs/2026-08-12-crop-disease-dataset-strategy-design.md)

## Why these crops

Wheat, cotton, rice, sugarcane, citrus, and mango together represent the bulk of Pakistan's crop economy — wheat is the largest staple crop, cotton and rice are major exports, and Kinnow (citrus) and mango are significant export/cash crops. The model is deliberately scoped to these six rather than a broader generic plant-disease set, trading class-count breadth for higher per-crop accuracy on the crops that actually matter locally.

## Datasets

Seven Kaggle sources (one per target crop, plus one multi-crop supplement) and one Hugging Face source (field-domain pretraining) were combined and deduplicated into a single unified manifest.

| Source | Platform | Slug | Role | Confirmed classes found | Raw image count |
|---|---|---|---|---|---|
| Wheat Leaf Disease Dataset | Kaggle | `khanaamer/wheat-leaf-disease-dataset` | Primary — wheat | LeafBlight, WheatBlast, HealthyLeaf, BlackPoint, FusariumFootRot (5) | nested under Train/Test/Validation splits |
| Cotton Leaf Disease Dataset | Kaggle | `seroshkarim/cotton-leaf-disease-dataset` | Primary — cotton | fussarium_wilt, curl_virus, healthy, bacterial_blight (4) | ~1,700 |
| Rice Leaf Disease Images | Kaggle | `nirmalsankalana/rice-leaf-disease-image` | Primary — rice | Tungro, Bacterialblight, Blast, Brownspot (4) | 5,932 |
| Sugarcane Plant Diseases Dataset | Kaggle | `akilesh253/sugarcane-plant-diseases-dataset` | Primary — sugarcane | Yellow, Mosaic, BacterialBlights, Healthy, RedRot, Rust (6) | 19,926 |
| Citrus Leaf Disease Image | Kaggle | `myprojectdictionary/citrus-leaf-disease-image` | Primary — citrus (**unverified fit** — see note below) | Black spot, Canker, Melanose, Healthy, Greening (5) | ~600 |
| Mango Leaf Disease Dataset (MangoLeafBD) | Kaggle | `aryashah2k/mango-leaf-disease-dataset` | Primary — mango | Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould, Healthy (8) | 4,000 |
| 20k+ Multi-Class Crop Disease Images | Kaggle | `jawadali1045/20k-multi-class-crop-disease-images` | Supplement — wheat, cotton, rice, sugarcane | selected classes only (e.g. Wheat___Yellow_Rust, Anthracnose on Cotton, Cotton Aphid, Wheat mite, RedRust sugarcane); Maize/Corn classes intentionally excluded (not a target crop) | ~2,300 (of the mapped classes) |
| LeafNet | Hugging Face | `enalis/LeafNet` | Field-domain pretraining pool (not part of the crop/disease manifest) | 22 species, ~97 species×disease combinations, image+caption pairs | 121,337 (train split) |

**Note on citrus:** the original project brief listed disease names ("Ash Weevil," "Dry Root Rot") that don't match any citrus dataset found on Kaggle during sourcing. The dataset above was selected as the closest available match (Pakistan/Kinnow-relevant, similar class count) but its actual disease labels (Black spot, Canker, Melanose, Healthy, Greening) differ from the original brief — flagged for future replacement if a better-matching source turns up.

**Note on the "71-classes" dataset:** the original brief also referenced a 71-class, 20-species, ~28,500-image dataset that could not be located on Kaggle under that description during sourcing. It was substituted with the multi-class supplement dataset above.

### Final unified dataset (post-ingest, post-dedup)

Produced by Notebook 1 (`notebooks/Notebook1_Ingest_Unify.ipynb`): sources normalized into a single `crop/disease/image.jpg` layout, corrupt images dropped, exact duplicates removed by content hash.

| Crop | Images |
|---|---|
| Citrus | 607 |
| Cotton | 1,887 |
| Mango | 3,979 |
| Rice | 4,794 |
| Sugarcane | 19,160 |
| Wheat | 5,137 |
| **Total** | **35,564** |

Class distribution is heavily imbalanced (sugarcane ~35x cotton) — handled via class-weighted loss during training rather than by dropping data.

## Approach

1. **Domain-adaptation pretraining**: most source datasets are clean, lab-shot images (uniform backgrounds), which don't generalize well to real farmer phone photos (clutter, glare, shadow). The backbone is pretrained on ImageNet, then finetuned on LeafNet's 121k in-situ field images before ever seeing the target crop data, to bias the model toward real-world visual conditions.
2. **Two-stage hierarchical classification**: a small Stage-1 model identifies the crop (6-way), then a per-crop Stage-2 model classifies the specific disease. Two-stage keeps each classifier's label space small and easier to get right, and keeps the on-device footprint smaller than one large flat classifier.
3. **Teacher/student distillation for mobile**: a larger teacher backbone (EfficientNetV2-S) is trained first for accuracy, then distilled into a MobileNetV3-sized student for on-device (offline, no internet required) inference on a farmer's phone.

## Repository structure

```
docs/superpowers/specs/    Design spec (dataset strategy, architecture rationale)
docs/superpowers/plans/    Step-by-step implementation plans per notebook
notebooks/                 Colab notebooks, run in order
```

## Notebooks

| Notebook | Status | Purpose |
|---|---|---|
| `Notebook1_Ingest_Unify.ipynb` | Done | Downloads all 8 sources, normalizes into unified `crop/disease/` layout, builds manifest CSV, integrity-checks (corrupt/duplicate removal), persists to Google Drive as a single zip |
| `Notebook2_Stage1_CropClassifier.ipynb` | Done | LeafNet domain-pretrain (EfficientNetV2-S, 3 epochs, 98% train acc) → Stage-1 crop classifier finetune (6-way, class-weighted). Val accuracy ~100% (crop *types* are visually very distinct — expected result, not a leakage red flag) |
| `Notebook3_Stage2_DiseaseHeads.ipynb` | Done | Trains 6 per-crop disease classifiers, each warm-started from Stage-1's backbone. Uses a group-aware train/val split (images grouped by inferred source photo, namespaced by disease) to prevent augmentation duplicates from leaking between train and val — see results below |
| Notebook 4 (distillation) | Planned | Teacher → MobileNetV3 student, per stage |
| Notebook 5 (export) | Planned | TFLite conversion, INT8 quantization, on-device test |

### Stage-2 disease classifier results (post-leakage-fix)

Several source datasets turned out to be pre-augmented (rotated/zoomed/cropped copies of the same base leaf photo under different filenames). A naive random train/val split lets those siblings land on both sides, inflating validation accuracy. Fixed by grouping images by an inferred source-photo key (stripped of known augmentation-prefix patterns, namespaced per disease) before splitting, so siblings always stay together.

| Crop | Val accuracy | Duplicates found | Notes |
|---|---|---|---|
| Citrus | 94.6% | 0 / 607 | Smallest dataset (one class, Melanose, has only 2 images) — expect noisy numbers here regardless |
| Cotton | ~100% (99.65% final epoch) | 2 / 1,887 | Disease classes are visually distinct (curl virus vs. wilt vs. blight) |
| Mango | 100% | 0 / 3,979 (by filename) | Unresolved caveat — see below |
| Rice | 100% | 0 / 4,794 (by filename) | Unresolved caveat — see below |
| Sugarcane | 95.3% | **15,765 / 19,160 (82%)** | Confirms the leakage was real — accuracy dropped from a pre-fix 96.8% to a more believable 95.3% once duplicates were properly held out |
| Wheat | 100% | 0 / 5,137 | — |

**Mango and rice caveat**: the duplicate-detection heuristic is filename-pattern-based and found nothing for these two, yet both still score 100%. This could mean the classes are genuinely easy to separate (MangoLeafBD in particular is documented in published work as achieving ~99-100% test accuracy even under proper random splits — its 8 disease classes are visually very distinct), or it could mean these datasets' augmentation duplicates use a naming convention the heuristic doesn't recognize. Not resolved — worth a manual sanity check (e.g. eyeballing whether visually similar images appear in both train and val) before fully trusting these two numbers.

## Running the notebooks

Everything runs on Google Colab (or Kaggle Notebooks, with adaptation) with a GPU runtime — no local downloads.

**Required Colab secrets** (left sidebar → key icon):
- `KAGGLE_API_TOKEN` — from `kaggle.com/settings` → API → Create New Token
- `HF_TOKEN` — from `huggingface.co/settings/tokens`

**One manual step required before Notebook 2 works**: LeafNet is a gated Hugging Face dataset. Visit `huggingface.co/datasets/enalis/LeafNet` while logged in and click "Agree and access repository" once — a valid token alone is not sufficient.

**Known gotcha**: persisting the unified dataset to Google Drive as ~35,000 individual files hits Google's per-user API request quota (regardless of how much free storage you have). Notebook 1 zips the dataset into a single file before copying to Drive to avoid this — don't revert to a plain folder copy.

Progress checkpoints (`leafnet_pretrain.pt`, `stage1_crop_classifier.pt`, etc.) are saved to `Google Drive/crop_disease/` after every epoch and are resume-safe across Colab disconnects or GPU-quota interruptions.
