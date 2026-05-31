# Psychiatric Medication Sentiment Analysis
[https://colab.research.google.com/github/dacamas/Psychiatric-Medication-Sentiment-Analysis
](https://colab.research.google.com/github/dacamas/Psychiatric-Medication-Sentiment-Analysis/blob/main/Psychiatric_Medication_Sentiment_Analysis_(Personal_Project).ipynb)

Binary sentiment classifier for patient-authored psychiatric medication reviews, fine-tuned on 40k+ real-world reviews from Drugs.com.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 93.9% |
| F1 (weighted) | 0.9387 |
| F1 (positive class) | 0.9595 |
| F1 (negative class) | 0.8796 |
| AUC-ROC | 0.9771 |
| Training time | ~21 min (T4 GPU) |

The negative class is harder to classify (recall 0.85 vs 0.97 for positive), reflecting the real-world distribution of the data — satisfied patients post more reviews, and negative reviews tend to contain mixed sentiment that resists clean binary labels.

---

## Overview

Patient reviews of psychiatric medications contain dense clinical signal — side-effect language, adherence patterns, outcome descriptions — that is difficult to capture with general-purpose NLP models. This pipeline uses **PubMedBERT**, pre-trained on biomedical literature, combined with eight engineered auxiliary features to classify reviews as positive or negative sentiment.

The dataset is the [UCI Drug Review Dataset](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) (215k reviews from Drugs.com). Filtering to psychiatric conditions yields ~40k reviews after dropping ambiguous middle-band ratings (5–7).

---

## Model Architecture

```
Review text ──► PubMedBERT encoder ──► [CLS] token (768-d)
                                                │
8 aux features ──► AuxMLP (8 → 16-d) ──────────┘
                                        Concat (784-d)
                                     ──► Dropout
                                     ──► Linear(128) + GELU
                                     ──► Dropout
                                     ──► Linear(2)
                                     ──► logits
```

**Why PubMedBERT over general BERT?** General BERT treats "akathisia", "tardive dyskinesia", and "titration" as low-frequency noise. PubMedBERT was pre-trained on PubMed abstracts and full-text biomedical articles, so these terms carry meaningful representations.

**Why auxiliary features?** Structured signals like adherence language ("stopped taking", "still on it") and side-effect density are strong sentiment predictors that text embeddings alone can underweight. Concatenating them onto the [CLS] representation gives the classifier a direct path to these features.

---

## Auxiliary Features

| Feature | Description |
|---|---|
| `side_effect_density` | SIDER/MedDRA term hits per 100 words |
| `negation_score` | Fraction of negation words ("didn't", "never", "not") |
| `adherence_flag` | +1 still taking / −1 stopped / 0 neutral |
| `outcome_score` | (positive outcome terms − negative) / word count |
| `review_length` | Word count |
| `useful_count` | Community helpfulness votes |
| `drug_freq_enc` | Drug name frequency encoding |
| `condition_freq_enc` | Condition frequency encoding |

---

## Pipeline Stages

| Stage | File | Description |
|---|---|---|
| 1 | `preprocess.py` | Filter psych conditions, clean HTML, engineer features, binarise labels, save parquet splits |
| 2 | `dataset.py` | PyTorch Dataset — tokenise text + pack aux features into float tensor |
| 3 | `model.py` | PubMedBERT + AuxMLP hybrid classifier |
| 4 | `train.py` | AdamW + linear warmup, fp16, early stopping on val F1, MLflow logging |
| 5 | `evaluate.py` | Classification report, confusion matrix, ROC/AUC, SHAP on aux features |
| 6 | `inference.py` | `MedSentimentPredictor` class for single and batch prediction |

---

## Quick Start

### Option A — Google Colab (recommended)
Open `psych_medication_sentiment.ipynb` in Colab with a T4 GPU runtime. Set `USE_MOCK_DATA = False` and run all cells. The dataset downloads automatically (~112MB), no Kaggle account required.

### Option B — Local

```bash
git clone <your-repo>
cd sentiment_pipeline
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py

# Or step by step
python preprocess.py
python train.py
python evaluate.py
```

### Inference

```python
from inference import MedSentimentPredictor

predictor = MedSentimentPredictor("models/checkpoints/best_model.pt")

result = predictor.predict(
    "This medication changed my life. Finally stable after 3 years. Still taking it.",
    drug_name="Sertraline",
    condition="depression",
)
# {"label": "positive", "confidence": 0.94, "probs": {...}}
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Base model | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Frozen BERT layers | Bottom 6 of 12 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 256 tokens |
| Warmup ratio | 10% |
| Precision | fp16 |
| Early stopping | Patience 3 on val F1 |
| Positive label threshold | Rating ≥ 7 |
| Negative label threshold | Rating ≤ 5 |

Early stopping triggered at epoch 4 when validation loss began increasing, with the epoch 3 checkpoint (best val F1 = 0.9632) loaded for final evaluation.

---

## Limitations

**Class imbalance.** The dataset is ~75% positive reviews. The model is consequently better calibrated on positive sentiment — negative recall (0.85) lags positive recall (0.97) by 12 points. Addressing this would require class reweighting or oversampling, validated on a held-out set before re-evaluating on test.

**Selection bias.** Drugs.com reviewers skew toward strong experiences and digitally active, English-speaking patients. The model learns the distribution of *reviewers*, not the full patient population. Populations with the highest psychiatric medication burden are largely absent from this data.

**Binary collapse.** Ratings 5–7 are dropped as ambiguous. A review rating a drug 6/10 for tolerability but 9/10 for efficacy contains clinical signal that binary classification discards entirely.

**Not clinical decision support.** This model is appropriate for population-level signal detection — flagging emerging side-effect patterns, informing pharmacovigilance pipelines — not for influencing individual prescribing decisions.

---

## MLOps

Experiment tracking via **MLflow** — all hyperparameters, per-step validation metrics, and model artifacts are logged automatically. Data and model versioning via **DVC**.

```bash
# View experiment dashboard
mlflow ui --backend-store-uri logs/mlruns
```

---

## Tech Stack

`PyTorch` · `HuggingFace Transformers` · `PubMedBERT` · `MLflow` · `DVC` · `SHAP` · `scikit-learn` · `pandas`
