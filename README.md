# HM²F-Net: Hybrid Multimodal Multi-Criteria Fusion for Sentiment Analysis

Official code for the paper:

> **Disentangling Multimodal Sentiment: Interpretable Multi-Criteria Decision Fusion for Cross-Modal Robustness**
> Neha Punetha (IIT Kharagpur) and Vinayak Abrol (IIIT Delhi)

HM²F-Net reformulates multimodal fusion as a **structured decision-making problem**. Instead of
concatenating modality embeddings or relying on unconstrained cross-attention, it:

1. extracts **multi-level unimodal sentiment evidence** (fine-grained textual, visual, and optional acoustic criteria),
2. organizes that evidence into a **class-specific decision matrix**,
3. aggregates the matrix with a classical **Multi-Criteria Decision-Making (MCDM)** operator (primarily TOPSIS), and
4. combines the resulting interpretable decision profile with a **neural late-fusion prior** through a calibrated hybrid layer.

The goal is fusion that is accurate, robust when a modality is noisy or missing, and explainable in terms of
*which criterion contributed how much* to each prediction.

---

## Why decision-level fusion?

Feature-level fusion collapses rich multi-level signals (sentence vs. segment text; global vs. object vs. scene
imagery) into a single latent vector, so a corrupted modality can dominate the joint representation. HM²F-Net
keeps each criterion in its own column of the decision matrix. Under TOPSIS, a perturbation of one criterion
enters the weighted distance to the ideal/anti-ideal profiles through a single orthogonal coordinate scaled by
its weight, which limits how far that noise propagates (Proposition 3.1 in the paper; a theoretical motivation,
not a formal guarantee).

Because MCDM aggregation alone cannot model highly non-linear decision boundaries, its output is blended with a
neural late-fusion prior. Intermediate blending weights consistently outperform both the pure rule-based and the
pure neural extremes on all three main benchmarks.

---

## Method overview

![HM²F-Net pipeline](images/pipelines-of-the-methodology.png)

### 1. Multi-level unimodal evidence (criteria)

Criteria of the same modality share one encoder and differ only in input granularity and prediction head. Every
head outputs a class distribution in the probability simplex Δ^(C−1).

| Criterion | Modality | Extraction |
|---|---|---|
| `s_sent` | Text | Classification head on the pooled representation of the full utterance |
| `s_seg` | Text | Split at sentence punctuation (or 8-token windows, stride 4); segment logits mean-pooled before softmax |
| `v_glb` | Image | Linear head on the image-level CLIP embedding |
| `v_obj` | Image | Linear head on the mean CLIP embedding of the top-5 region proposals of a pre-trained detector |
| `v_scn` | Image | Linear head on the embedding of a scene classifier |
| `a_aud` | Audio (optional) | Head on utterance-level acoustic features (COVAREP on CMU-MOSI) |

### 2. Decision matrix

For each sample, a matrix **D** ∈ ℝ^(C×K) is built, where `d[i, j]` is the probability of class *i* under
criterion *j*, together with a criterion weight vector **w** (Σ w = 1).

| Setting | Matrix shape | Columns |
|---|---|---|
| MVSA (ternary, headline) | 3 × 5 | 2 text + 3 visual criteria |
| CMU-MOSI (binary, headline) | 2 × 9 | 6 fine-grained criteria + 3 whole-modality posteriors `p_T, p_I, p_A` |
| CMU-MOSI (ternary) | 3 × 9 | as above |

Binary evaluation excludes the neutral class (2 × K); ternary evaluation uses 3 × K.

### 3. MCDM aggregation

`s_MCDM = Φ(D, w)`, where Φ is one of the operators implemented in `MCDM.py`:

- **TOPSIS** — distance to ideal / anti-ideal solutions (default)
- **SAW** — simple additive weighting
- **RAFSI** — ranking of alternatives through functional mapping of criterion sub-intervals
- **TODIM** — prospect-theory-based pairwise dominance
- **MARCOS** — ratio to the compromise solution
- **EDAS** — distance from the average solution

### 4. Hybrid calibration and final decision

```
p_NN = Σ_m α_m · p^(m)                 # neural late-fusion prior over unimodal posteriors
f    = (1 − λ) · s_MCDM + λ · p_NN     # hybrid score, λ ∈ [0, 1]
ŷ    = argmax_i f_i
```

A temperature `T` is applied to each unimodal posterior *before* MCDM aggregation,
`p̃ = softmax(log p / T)`. Because TOPSIS normalizes each criterion column, `T` can change the aggregated ranking,
so it is selected jointly with **w**. The weights **w**, `α_m`, `λ` (step 0.1), and `T` are all grid-searched on
the **validation split only**.

---

## Datasets

| Dataset | Modalities | Task | Split | Link |
|---|---|---|---|---|
| **CMU-MOSI** | Text, video, audio | Binary / ternary; regression in [−3, +3] | Official 1,284 / 229 / 686 | [Google Drive](https://drive.google.com/drive/folders/1u7zquWeM9qw-iYzzyybdaniJLxHlXQzK) |
| **MVSA-Single** | Tweet text + image | Pos / Neu / Neg (4,511 filtered samples) | Fixed stratified split (indices released) | [Google Drive](https://drive.google.com/file/d/1UYaPJWZd4NvnLj_A41awkmP5oPc4SP3K/view?usp=sharing) |
| **MVSA-Multiple** | Tweet text + image | Pos / Neu / Neg (16,779 processed samples) | Fixed stratified split (indices released) | [Google Drive](https://drive.google.com/file/d/1UYaPJWZd4NvnLj_A41awkmP5oPc4SP3K/view?usp=sharing) |
| **CMU-MOSEI** | Text, video, audio | Binary; regression in [−3, +3] | Official 16,326 / 1,871 / 4,659 | https://www.kaggle.com/datasets/samarwarsi/cmu-mosei |
| **Twitter-2015 / 2017** | Text + image (target-oriented) | 3-class | Official | https://www.kaggle.com/datasets/mocmeo/data-twitter-2015-2017 |

> Datasets are **not bundled** with this repository (see [Data setup](#data-setup)).

---

## Repository structure

| File | Purpose |
|---|---|
| `MCDM.py` | Implementations of the MCDM operators (TOPSIS, SAW, RAFSI, TODIM, MARCOS, EDAS) used by all pipelines. |
| `AMVSA-Single-Binary.py` | MVSA-Single pipeline: fine-tunes RoBERTa-large and CLIP ViT-B/32, then compares soft fusion, stacking, and every MCDM operator ± neural calibration. |
| `AMVSA-MULTIPLE-binary.py` | MVSA-Multiple pipeline: builds a cleaned CSV from the raw label files, fine-tunes the text and image models, and runs the same fusion comparison. |
| `Mosi-Binary.py` | CMU-MOSI binary pipeline (Acc2 / F1 / MAE / Corr) on pre-computed SDK features. |
| `mosi-Tertiary.py` | CMU-MOSI three-class pipeline (Acc3 / F1). |
| `RoBERT+VGG.Net+MCDM-Single.py` | RoBERTa + VGG16 variant, including the naive (un-normalized concatenation + MLP) fusion control. |
| `Single-modality-Acc.py` | Unimodal text-only and image-only reference accuracies. |

---

## Implementation details

Hardware used: one NVIDIA RTX 3090 (24 GB), 256 GB RAM, PyTorch.

| Hyperparameter | CMU-MOSI | MVSA-Single | MVSA-Multiple |
|---|---|---|---|
| Text encoder | Pre-computed embeddings | RoBERTa-large (fine-tuned) + TextMLP | RoBERTa-large (fine-tuned, sequence classifier) |
| Image encoder | Pre-computed features | CLIP ViT-B/32 + linear head | CLIP ViT-B/32 + linear head |
| Batch size | 64 | 128 | 16 (text), 64 (image) |
| Optimizer / weight decay | AdamW / 1e-4 | AdamW / 1e-4 | AdamW / 1e-4 |
| Learning rates | 8e-4 (fusion) | 2e-5 (RoBERTa), 3e-4 (TextMLP), 8e-4 (image head) | 2e-5 (RoBERTa/text head), 8e-4 (image head) |
| Dropout | 0.35 | 0.20 (CLIP), 0.30 (TextMLP) | 0.20 (CLIP) |
| Loss | CE, label smoothing 0.05 | Focal, 0.02 | CE |
| Epochs / patience | 50 / 7 | 20 / 6 | 4 / 2 (text); 10 / 3 (image) |
| MCDM operator | TOPSIS | TOPSIS | TOPSIS |
| Temperature grid `T` | — | {0.8, 1.0, 1.2, 1.5, 2.0} | {0.8, 1.0, 1.2, 1.5, 2.0} |
| Text–image weight grid | — | 0.98 … 0.50 | 0.98 … 0.50 |
| Selected `λ` | 0.4 | 0.6 | 0.6 |

**Protocol.** All hyperparameters and early stopping are selected on the validation split; each configuration is
evaluated once on test. Results are averaged over five seeds: **42, 123, 256, 512, 1024**.

---

## Getting started

### Requirements

```bash
pip install torch torchvision transformers scikit-learn scipy pandas numpy pillow tqdm
```

A CUDA-capable GPU is strongly recommended; RoBERTa-large fine-tuning on CPU is very slow.

### Data setup

1. Download the dataset(s) from the links above.
2. Extract MVSA so the folder structure matches:
   ```
   extracted_multiple/MVSA/
     ├── labelResultAll.txt
     └── data/
         ├── 1.txt
         ├── 1.jpg
         └── ...
   ```
3. For CMU-MOSI, point the `SOURCE_DIR` / path variables at the top of `Mosi-Binary.py` and `mosi-Tertiary.py`
   to your local copy of the CMU-Multimodal SDK features.

### Running

```bash
python AMVSA-Single-Binary.py           # MVSA-Single
python AMVSA-MULTIPLE-binary.py         # MVSA-Multiple
python Mosi-Binary.py                   # CMU-MOSI binary
python mosi-Tertiary.py                 # CMU-MOSI ternary
python "RoBERT+VGG.Net+MCDM-Single.py"  # VGG16 variant + naive fusion control
python Single-modality-Acc.py           # unimodal references
```

Each script builds and caches a cleaned CSV, fine-tunes the unimodal models, sweeps the fusion
hyperparameters on validation, and prints test accuracy / F1 for every fusion strategy, including a ranked table
of the best MCDM configurations.

### Config knobs

Constants near the top of each script: `SEED`, `MAX_LEN`, `TEXT_EPOCHS`, `IMG_EPOCHS`, `TEXT_BS`, `IMG_BS`,
`LR_TEXT`, `LR_IMG`, `WD`, `SOFT_W_GRID`, plus the MCDM weight, temperature, and TODIM `theta` grids in the fusion
section.

---

## Results

Numbers below are as reported in the submitted manuscript (mean over five seeds).

### Main results

| Dataset | Metric | HM²F-Net |
|---|---|---|
| CMU-MOSI (binary) | Acc2 / F1 / MAE / Corr | **87.14** / 85.89 / 0.548 / 0.887 |
| CMU-MOSI (ternary) | Acc3 / F1 | 76.64 / 76.38 |
| MVSA-Single | Acc3 / macro-F1 | **88.96** / 93.84 |
| MVSA-Multiple | Acc3 / macro-F1 | **88.60** / 89.80 |

On CMU-MOSI, HM²F-Net has the best Acc2 among the compared methods that do not use cross-utterance context; it is
**not** best on every metric (MCL-MCF leads on F1; MMA leads on MAE and Corr). On MVSA, the quoted baselines use
weaker encoders, so part of the margin may come from the backbones; same-backbone controls are being added in the
revision (see below).

### Effect of the hybrid layer (CMU-MOSI binary, Acc2)

| Fusion | Acc2 |
|---|---|
| Text only (NN) | 80.95 |
| Naive concatenation + MLP (un-normalized, untuned control) | 43.81 |
| SAW + NN | 69.52 |
| MARCOS + NN | 72.40 |
| **TOPSIS + NN (HM²F-Net)** | **87.14** |

The naive control is deliberately untuned; it isolates the effect of removing the decision-matrix structure and
should not be read as representative of tuned neural fusion.

### Robustness (bimodal text + audio reproduction, CMU-MOSI)

| Condition | Naive fusion | HM²F-Net |
|---|---|---|
| Clean | 84.49 | 83.23 |
| Acoustic noise σ = 0.5 | 53.48 | **77.53** |

---

## Revision experiments (in progress)

The TKDE revision extends the evaluation with:

- **Same-pipeline baselines** with identical splits, backbones, tuning budget, and seeds: late fusion
  (averaged / learned), concat + MLP (naive and LayerNorm-tuned), GMU, LMF, a cross-modal transformer, and
  TFN, MulT, MISA, Self-MM, MMIM, ALMT (via M-SENA/MMSA), MMML, and DLF.
- **Extended benchmarks:** CMU-MOSEI and Twitter-2015/2017.
- **Multimodal LLMs:** GPT-4o, Qwen2.5-VL-7B, LLaVA-OneVision-7B, Qwen2-Audio (zero-shot, 8-shot, LoRA), plus an
  *MLLM-as-extractor* variant that feeds MLLM criterion probabilities into the TOPSIS decision matrix.
- **Operator ablation:** the same decision matrix fed to TOPSIS, SAW, a linear layer, and an MLP.
- **Bias–variance analysis:** Domingos' 0-1 decomposition and error correlation at λ ∈ {0, λ*, 1}.

Scripts for these experiments will be added to this repository as they are completed.

---

## Citation

If you use this code, please cite:

```bibtex
@article{punetha2026hm2fnet,
  author  = {Punetha, Neha and Abrol, Vinayak},
  title   = {Disentangling Multimodal Sentiment: Interpretable Multi-Criteria
             Decision Fusion for Cross-Modal Robustness},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## License

No license file is currently included. Add one (e.g., MIT or Apache-2.0) if you intend others to reuse this code.

## Contact

Neha Punetha — nehapunetha80@gmail.com
