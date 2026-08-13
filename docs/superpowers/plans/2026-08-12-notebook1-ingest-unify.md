# Notebook 1: Ingest & Unify Dataset Structure — Implementation Plan

> **For agentic workers:** This plan targets a Google Colab / Kaggle notebook, not a local repo. There is no git repo and no local pytest suite — "tests" are notebook cells that print counts and run `assert` checks. Execute cell-by-cell, in order, checking each cell's output before moving to the next.

**Goal:** Pull the 8 dataset sources for the 6-crop Pakistan disease model into a Colab/Kaggle runtime, normalize them into one folder layout (`crop/disease/image.jpg`), and produce a single manifest CSV describing every image — without ever downloading anything to the local machine.

**Architecture:** One Colab notebook, run top-to-bottom. Config cell holds dataset identifiers (filled in by the user from actual Kaggle/HF pages — this plan does not fabricate dataset slugs, since guessing them risks pointing at the wrong dataset). Download cell pulls each source into `/content/raw/<key>/`. An explore cell prints each raw dataset's folder tree, because every Kaggle dataset has a different internal layout — normalization mappings can only be written after seeing real structure. Per-dataset normalizer functions copy/symlink files into `/content/data/<crop>/<disease>/`. A manifest builder walks that unified tree into a pandas DataFrame. Integrity checks catch corrupt/duplicate images. Final cell persists `/content/data` + manifest to Google Drive (Colab's `/content` is ephemeral and wiped on disconnect).

**Tech Stack:** Python 3, `kagglehub`, `huggingface_hub`/`datasets`, `pandas`, `Pillow`, Google Colab, Google Drive mount.

## Global Constraints

- No dataset is ever downloaded to the local machine — Colab/Kaggle runtime storage only.
- Do not hardcode Kaggle dataset slugs from guesswork — the config cell takes them as user-supplied values copied from the actual dataset page URL (`kaggle.com/datasets/<owner>/<name>` → slug is `<owner>/<name>`).
- 6 target crops: wheat, cotton, rice, sugarcane, citrus, mango.
- Field-realism priority: LeafNet (in-situ) images get a `is_field=True` manifest flag; clean/lab-shot datasets get `is_field=False`, per the dataset strategy spec (`docs/superpowers/specs/2026-08-12-crop-disease-dataset-strategy-design.md`).

---

### Task 1: Environment setup + Kaggle auth

**Files:**
- Create: `Notebook1_Ingest_Unify.ipynb` (Colab notebook, cells described below — build this in Colab directly, or as a local `.ipynb`/`.py` cell script you paste in)

**Interfaces:**
- Produces: working `kagglehub` and `kaggle` CLI auth in the runtime, used by every later download cell in this task.

- [ ] **Step 1: Install dependencies cell**

```python
!pip install -q kagglehub huggingface_hub datasets pillow pandas
```

- [ ] **Step 2: Kaggle auth cell**

```python
import os
from google.colab import files

# Upload kaggle.json (from kaggle.com/settings -> API -> Create New Token)
# if not already present.
if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
    uploaded = files.upload()  # select kaggle.json
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    for fname in uploaded:
        os.rename(fname, os.path.expanduser(f"~/.kaggle/{fname}"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

print("Kaggle auth file present:", os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")))
```

- [ ] **Step 3: Verify — run cell, check output**

Expected output: `Kaggle auth file present: True`

- [ ] **Step 4: Mount Google Drive (for final persistence in Task 8)**

```python
from google.colab import drive
drive.mount('/content/drive')
```

Expected: prompts auth flow, then `Mounted at /content/drive`.

---

### Task 2: Dataset registry config

**Files:**
- Modify: `Notebook1_Ingest_Unify.ipynb` (new cell)

**Interfaces:**
- Produces: `DATASETS` dict — `{key: {"type": "kaggle"|"hf", "slug": str, "crop": str|None}}`, consumed by Task 3's download loop and Task 4's explore loop.

- [ ] **Step 1: Write config cell — fill in real slugs before running**

```python
# Fill each "slug" with the real identifier from the dataset's page:
#   Kaggle: kaggle.com/datasets/<owner>/<name>  -> slug = "<owner>/<name>"
#   Hugging Face: huggingface.co/datasets/<org>/<name> -> slug = "<org>/<name>"
DATASETS = {
    "wheat_primary":     {"type": "kaggle", "slug": "FILL_ME", "crop": "wheat"},
    "cotton_primary":    {"type": "kaggle", "slug": "FILL_ME", "crop": "cotton"},
    "rice_primary":      {"type": "kaggle", "slug": "FILL_ME", "crop": "rice"},
    "sugarcane_primary": {"type": "kaggle", "slug": "FILL_ME", "crop": "sugarcane"},
    "citrus_primary":    {"type": "kaggle", "slug": "FILL_ME", "crop": "citrus"},
    "mango_primary":     {"type": "kaggle", "slug": "FILL_ME", "crop": "mango"},
    "multi_71classes":   {"type": "kaggle", "slug": "FILL_ME", "crop": None},  # multi-crop supplement
    "leafnet_field":     {"type": "hf",     "slug": "FILL_ME", "crop": None},  # field-domain pretrain pool
}

missing = [k for k, v in DATASETS.items() if v["slug"] == "FILL_ME"]
assert not missing, f"Fill in slugs before proceeding: {missing}"
print(f"{len(DATASETS)} datasets configured.")
```

- [ ] **Step 2: Verify — run cell after filling in real slugs**

Expected: `AssertionError` if any slug still `"FILL_ME"` (this is the intended guard — do not skip past it by deleting the assert). Once filled: `8 datasets configured.`

---

### Task 3: Download all sources into runtime storage

**Files:**
- Modify: `Notebook1_Ingest_Unify.ipynb` (new cell)

**Interfaces:**
- Consumes: `DATASETS` dict from Task 2.
- Produces: `RAW_PATHS` dict — `{key: local_path_str}`, consumed by Task 4 (explore) and Task 5 (normalize).

- [ ] **Step 1: Write download cell**

```python
import kagglehub
from huggingface_hub import snapshot_download

RAW_PATHS = {}
for key, cfg in DATASETS.items():
    print(f"Downloading {key} ({cfg['slug']}) ...")
    if cfg["type"] == "kaggle":
        path = kagglehub.dataset_download(cfg["slug"])
    elif cfg["type"] == "hf":
        path = snapshot_download(repo_id=cfg["slug"], repo_type="dataset")
    else:
        raise ValueError(f"Unknown type for {key}: {cfg['type']}")
    RAW_PATHS[key] = path
    print(f"  -> {path}")

assert len(RAW_PATHS) == len(DATASETS)
print("All downloads complete.")
```

- [ ] **Step 2: Verify — run cell**

Expected: one `Downloading ... -> <path>` line per dataset, ending with `All downloads complete.` If any download 404s, the slug in Task 2 is wrong — fix it there, not here.

---

### Task 4: Explore raw folder structure per dataset

**Files:**
- Modify: `Notebook1_Ingest_Unify.ipynb` (new cell)

**Interfaces:**
- Consumes: `RAW_PATHS` from Task 3.
- Produces: printed folder trees — human reads these to write Task 5's per-dataset normalizer mapping (there is no way to know each dataset's internal class-folder naming without looking).

- [ ] **Step 1: Write tree-printing cell**

```python
import os

def print_tree(path, max_depth=2, max_entries=15):
    for root, dirs, filenames in os.walk(path):
        depth = root[len(path):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or root}/")
        for f in (filenames[:max_entries] if depth == max_depth else []):
            print(f"{indent}  {f}")
        if len(filenames) > max_entries and depth == max_depth:
            print(f"{indent}  ... ({len(filenames)} files total)")

for key, path in RAW_PATHS.items():
    print(f"\n=== {key} ({path}) ===")
    print_tree(path)
```

- [ ] **Step 2: Run cell, read output**

No assertion — this is a human-in-the-loop inspection step. Record, for each dataset, which subfolder names correspond to which disease label. This mapping feeds directly into Task 5.

---

### Task 5: Per-dataset normalizers → unified `crop/disease/` layout

**Files:**
- Modify: `Notebook1_Ingest_Unify.ipynb` (new cell)

**Interfaces:**
- Consumes: `RAW_PATHS` (Task 3), the folder-name → disease-label mapping read from Task 4's output.
- Produces: `/content/data/<crop>/<disease>/<file>` unified tree, consumed by Task 6 (manifest) and Task 7 (integrity checks).

- [ ] **Step 1: Write normalizer cell — one mapping table per dataset, filled from Task 4 findings**

```python
import shutil

UNIFIED_ROOT = "/content/data"

# Fill in real subfolder -> (crop, disease) mappings after inspecting
# Task 4's printed trees. Example shape shown for wheat_primary —
# replace folder names with what Task 4 actually printed.
LABEL_MAPS = {
    "wheat_primary": {
        # "<raw_subfolder_name>": ("wheat", "<disease_label>"),
    },
    "cotton_primary": {},
    "rice_primary": {},
    "sugarcane_primary": {},
    "citrus_primary": {},
    "mango_primary": {},
    "multi_71classes": {
        # only entries whose crop is one of the 6 targets get included
    },
}

def normalize_dataset(key, raw_path, label_map):
    count = 0
    for subfolder, (crop, disease) in label_map.items():
        src_dir = os.path.join(raw_path, subfolder)
        if not os.path.isdir(src_dir):
            print(f"  WARNING: {src_dir} not found, skipping")
            continue
        dst_dir = os.path.join(UNIFIED_ROOT, crop, disease)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            if not os.path.isfile(src_file):
                continue
            dst_file = os.path.join(dst_dir, f"{key}_{fname}")
            shutil.copyfile(src_file, dst_file)
            count += 1
    return count

for key in ["wheat_primary", "cotton_primary", "rice_primary",
            "sugarcane_primary", "citrus_primary", "mango_primary",
            "multi_71classes"]:
    n = normalize_dataset(key, RAW_PATHS[key], LABEL_MAPS[key])
    print(f"{key}: copied {n} files")
```

- [ ] **Step 2: Verify — run cell**

Expected: nonzero copy count per dataset once `LABEL_MAPS` is filled in from Task 4's tree output. A `0` count for any dataset means its `LABEL_MAPS` entry is still empty or wrong — fix before proceeding.

Note: LeafNet (`leafnet_field`) is intentionally excluded here — it feeds the domain-pretrain step (Notebook 2), not the per-crop manifest. Keep it referenced only via `RAW_PATHS["leafnet_field"]`.

---

### Task 6: Build the manifest CSV

**Files:**
- Modify: `Notebook1_Ingest_Unify.ipynb` (new cell)

**Interfaces:**
- Consumes: `/content/data/<crop>/<disease>/` tree from Task 5.
- Produces: `manifest` DataFrame with columns `filepath, crop, disease, is_field`, saved to `/content/data/manifest.csv`, consumed by Task 7 and by Notebook 2.

- [ ] **Step 1: Write manifest-builder cell**

```python
import pandas as pd

rows = []
for crop in os.listdir(UNIFIED_ROOT):
    crop_dir = os.path.join(UNIFIED_ROOT, crop)
    if not os.path.isdir(crop_dir):
        continue
    for disease in os.listdir(crop_dir):
        disease_dir = os.path.join(crop_dir, disease)
        if not os.path.isdir(disease_dir):
            continue
        for fname in os.listdir(disease_dir):
            fpath = os.path.join(disease_dir, fname)
            if os.path.isfile(fpath):
                rows.append({
                    "filepath": fpath,
                    "crop": crop,
                    "disease": disease,
                    "is_field": False,  # lab-shot Kaggle sources; flip per-source if known field-shot
                })

manifest = pd.DataFrame(rows)
manifest.to_csv(os.path.join(UNIFIED_ROOT, "manifest.csv"), index=False)
print(manifest.shape)
print(manifest.groupby(["crop", "disease"]).size())
```

- [ ] **Step 2: Verify — run cell**

Expected: `manifest.shape` has >0 rows, and the groupby printout shows all 6 crops present with at least one disease class each. `assert manifest["crop"].nunique() == 6` as an explicit check:

```python
assert manifest["crop"].nunique() == 6, f"Expected 6 crops, got {manifest['crop'].nunique()}: {manifest['crop'].unique()}"
print("All 6 crops present.")
```

---

### Task 7: Integrity checks — corrupt images, duplicates, class balance report

**Files:**
- Modify: `Notebook1_Ingest_Unify.ipynb` (new cell)

**Interfaces:**
- Consumes: `manifest` DataFrame from Task 6.
- Produces: cleaned `manifest` (corrupt/duplicate rows dropped), re-saved to `/content/data/manifest.csv`.

- [ ] **Step 1: Write integrity-check cell**

```python
import hashlib
from PIL import Image

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

manifest["valid"] = manifest["filepath"].apply(is_valid_image)
corrupt_count = (~manifest["valid"]).sum()
print(f"Corrupt/unreadable images: {corrupt_count}")

manifest = manifest[manifest["valid"]].drop(columns=["valid"])

manifest["hash"] = manifest["filepath"].apply(file_hash)
dup_count = manifest.duplicated(subset="hash").sum()
print(f"Duplicate images (by content hash): {dup_count}")

manifest = manifest.drop_duplicates(subset="hash").drop(columns=["hash"])

manifest.to_csv(os.path.join(UNIFIED_ROOT, "manifest.csv"), index=False)
print(f"Final manifest: {manifest.shape[0]} images across {manifest['crop'].nunique()} crops")
print(manifest.groupby(["crop", "disease"]).size())
```

- [ ] **Step 2: Verify — run cell**

Expected: printed corrupt/duplicate counts (0 is fine, just confirms the check ran), final manifest row count less than or equal to Task 6's raw count, all 6 crops still present in the final groupby.

- [ ] **Step 3: Flag thin classes**

```python
counts = manifest.groupby(["crop", "disease"]).size()
thin = counts[counts < 50]
if len(thin):
    print("WARNING — classes with <50 images (may need oversampling or more sourcing):")
    print(thin)
else:
    print("No classes below 50 images.")
```

Expected: either a clean pass or a printed list to revisit — this is diagnostic, not a hard failure.

---

### Task 8: Persist to Google Drive

**Files:**
- Modify: `Notebook1_Ingest_Unify.ipynb` (new cell, final)

**Interfaces:**
- Consumes: `/content/data/` tree + `manifest.csv` from Tasks 5-7.
- Produces: `/content/drive/MyDrive/crop_disease/data/` — persisted copy that survives Colab runtime disconnects, consumed by Notebook 2 (Stage-1 training).

- [ ] **Step 1: Write persistence cell**

```python
DRIVE_DEST = "/content/drive/MyDrive/crop_disease/data"
os.makedirs(DRIVE_DEST, exist_ok=True)

# Copy tree (not symlink — Drive doesn't preserve symlinks reliably)
shutil.copytree(UNIFIED_ROOT, DRIVE_DEST, dirs_exist_ok=True)
print(f"Persisted to {DRIVE_DEST}")
```

- [ ] **Step 2: Verify — run cell, then confirm in a fresh cell**

```python
saved_manifest = pd.read_csv(os.path.join(DRIVE_DEST, "manifest.csv"))
assert len(saved_manifest) == len(manifest), "Row count mismatch after copy to Drive"
print("Drive copy verified:", saved_manifest.shape)
```

Expected: `Drive copy verified: (<same row count>, 4)`. This is the deliverable Notebook 2 (Stage-1 crop classifier + LeafNet domain pretrain) will load from.

---

## Self-Review Notes

- **Spec coverage:** Task 1-3 cover dataset sourcing (spec section 1), Task 5's `is_field` groundwork + LeafNet being kept separate covers domain-adaptation staging (spec section 2 — full 3-stage training itself is Notebook 2, out of this plan's scope). Architecture/mobile export (spec sections 3-4) and remaining notebooks (spec section 5, Notebooks 2-5) are follow-on plans, not part of Notebook 1.
- **Placeholder scan:** `"FILL_ME"` and empty `LABEL_MAPS` entries are intentional, guarded by asserts that fail loudly rather than silently proceeding — not left as vague prose.
- **Scope:** This plan is Notebook 1 only, matching the decomposition of the 5-notebook pipeline in the spec. Notebook 2 (Stage-1 + LeafNet pretrain) gets its own plan once this one is done and its manifest output is real.
