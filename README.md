# Multimodal Sentiment Analysis with Hybrid MCDM–Neural Fusion

An interpretable **multimodal sentiment analysis** framework that fuses text and image
signals for sentiment classification using two complementary strategies:

1. **Conventional neural fusion** — soft (weighted) averaging and stacked
   meta-classification over text and image model outputs.
2. **Multi-Criteria Decision Making (MCDM)** — treating each modality's class
   probabilities as *decision criteria* and ranking sentiment classes with
   classical MCDM algorithms (TOPSIS, RAFSI, TODIM, MARCOS, EDAS).

The goal is to combine the transparency of criteria-based decision theory with
the predictive power of deep learning, so that fusion decisions are not just
accurate but explainable in terms of *which modality contributed how much*.

---

## Why hybrid fusion?

Most multimodal sentiment models either:
- concatenate/average modality embeddings and let a neural net learn the
  weighting implicitly (opaque), or
- use fixed heuristic rules (rigid, not data-driven).

This project instead scores each sentiment class against both modalities using
MCDM methods borrowed from operations research, then compares that against
standard neural fusion (soft-weighted averaging and logistic-regression
stacking) to see which approach — or combination — generalizes best across
datasets and label granularities (binary / tertiary).

---

## Datasets

| Dataset | Modality | Labels | Link |
|---|---|---|---|
| **MVSA** (Single & Multiple) | Tweet text + attached image | Positive / Negative / Neutral | [Google Drive](https://drive.google.com/file/d/1UYaPJWZd4NvnLj_A41awkmP5oPc4SP3K/view?usp=sharing) |
| **MOSI** | Spoken utterance (text/audio-derived) + visual | Binary / Tertiary sentiment | [Google Drive folder](https://drive.google.com/drive/folders/1u7zquWeM9qw-iYzzyybdaniJLxHlXQzK) |

> Datasets are **not bundled** in this repo (see [Data Setup](#data-setup) below).

---

## Repository structure

| File | Purpose |
|---|---|
| `AMVSA-Single-Binary.py` | Binary (positive/negative) sentiment pipeline on the **MVSA-Single** subset. |
| `AMVSA-MULTIPLE-binary.py` | Binary sentiment pipeline on the **MVSA-Multiple** subset — builds a cleaned CSV from raw label files, fine-tunes RoBERTa-large (text) and a CLIP ViT-B/32 + linear head (image), then compares soft fusion, stacking, and MCDM fusion. |
| `Mosi-Binary.py` | Binary sentiment classification on the **MOSI** dataset. |
| `mosi-Tertiary.py` | Three-class (positive/negative/neutral) sentiment classification on **MOSI**. |
| `RoBERT+VGG.Net+MCDM-Single.py` | Alternative image encoder variant using **VGG16** (instead of CLIP) combined with RoBERTa text features, fused via MCDM, on the single-label MVSA setup. |
| `MCDM.py` | Standalone implementations of the MCDM ranking algorithms (TOPSIS, RAFSI, TODIM, MARCOS, EDAS) used across the other scripts. |
| `Single-modality-Acc.py` | Baseline accuracy evaluation for text-only and image-only models, used as a reference point against fusion results. |

---

## Method overview

### 1. Feature extraction
- **Text:** `roberta-large` fine-tuned for sequence classification (2 or 3 classes depending on script).
- **Image:** either
  - CLIP ViT-B/32 visual encoder + a lightweight linear classification head, or
  - VGG16 convolutional features (in the RoBERTa+VGG variant).

### 2. Neural fusion baselines
- **Soft fusion:** weighted average of text/image softmax probabilities, with
  the weight `w` tuned on the validation set over a grid (e.g. `0.5–0.95`).
- **Stacking:** a `LogisticRegressionCV` meta-classifier trained on the
  concatenation of both modalities' logits plus **entropy** and **decision
  margin** as additional uncertainty features.

### 3. MCDM fusion
Each sample's text and image class probabilities form a small decision
matrix (`classes × modalities`), which is ranked using:
- **TOPSIS** — distance to ideal/nadir solutions
- **RAFSI** — rank-based aggregation
- **TODIM** – prospect-theory-inspired pairwise dominance
- **MARCOS** — measurement of alternatives with ratio to compromise solution
- **EDAS** — evaluation based on distance from average solution

Modality weights, temperature scaling (for probability calibration), and
method-specific hyperparameters (e.g. TODIM's `theta`) are swept on the
validation set, and the best configuration per method is applied to the test
set for a fair, like-for-like comparison against the neural baselines.

### 4. Evaluation
All scripts report **accuracy** and **F1-score** for:
- text-only
- image-only
- soft fusion
- stacking
- each MCDM method (best validated configuration)

---

## Getting started

### Requirements
```bash
pip install torch torchvision transformers scikit-learn pandas numpy pillow tqdm
```
A CUDA-capable GPU is strongly recommended — RoBERTa-large fine-tuning on CPU
will be very slow.

### Data setup
1. Download the dataset(s) you want to use from the links above.
2. Extract MVSA data so that the folder structure matches:
   ```
   extracted_multiple/MVSA/
     ├── labelResultAll.txt
     └── data/
         ├── 1.txt
         ├── 1.jpg
         ├── 2.txt
         ├── 2.jpg
         └── ...
   ```
3. For MOSI, follow the same convention used in `Mosi-Binary.py` /
   `mosi-Tertiary.py` (adjust the `SOURCE_DIR` / path variables at the top of
   each script to point at your local copy).

### Running a pipeline
```bash
# Binary sentiment on MVSA-Multiple (text + image fusion + MCDM comparison)
python AMVSA-MULTIPLE-binary.py

# Binary sentiment on MVSA-Single
python AMVSA-Single-Binary.py

# MOSI binary / tertiary sentiment
python Mosi-Binary.py
python mosi-Tertiary.py

# VGG16-based variant
python "RoBERT+VGG.Net+MCDM-Single.py"
```

Each script will:
1. Build/clean a CSV from the raw dataset (cached after first run).
2. Fine-tune the text and image models.
3. Print validation and test accuracy/F1 for every fusion strategy, including
   a ranked table of the top-performing MCDM configurations.

### Config knobs
Most tunable settings live as constants near the top of each script, e.g.:
- `SEED`, `MAX_LEN`, `TEXT_EPOCHS`, `IMG_EPOCHS`, `TEXT_BS`, `IMG_BS`
- `LR_TEXT`, `LR_IMG`, `WD`
- `SOFT_W_GRID` (soft-fusion weight search space)
- MCDM weight/theta/temperature candidate grids near the fusion section

---

## Results

_Add your latest accuracy/F1 numbers here per dataset and fusion method once
you've run the scripts, e.g.:_

| Dataset | Text-only | Image-only | Soft Fusion | Stacking | Best MCDM |
|---|---|---|---|---|---|
| MVSA-Single |  |  |  |  |  |
| MVSA-Multiple |  |  |  |  |  |
| MOSI (Binary) |  |  |  |  |  |
| MOSI (Tertiary) |  |  |  |  |  |

---

## Citation

If you use this code in your research, please cite this repository:

```bibtex
@misc{punetha_multimodal_sentiment_hybrid,
  author = {Neha Punetha},
  title  = {Multimodal Sentiment Analysis with Hybrid MCDM--Neural Fusion},
  year   = {2026},
  url    = {https://github.com/NehaPunetha/Multimodel-Sentiment-analysis-Hybrid}
}
```

## License

_No license file is currently included — add one (e.g. MIT, Apache-2.0) if you
intend others to reuse this code._
