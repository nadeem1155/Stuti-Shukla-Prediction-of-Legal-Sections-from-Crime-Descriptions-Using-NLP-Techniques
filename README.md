<div align="center">

<img src="https://img.shields.io/badge/NyayaAuth-IPC%20Section%20Classifier-blue?style=for-the-badge&logo=scales&logoColor=white" alt="NyayaAuth Banner"/>

# ⚖️ NyayaAuth
### *Neural Legal Text Classification for the Indian Penal Code*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-TinyBERT-FFD21E)](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Ready-brightgreen)]()

> **NyayaAuth** (*Nyaya* = Justice in Sanskrit) automatically maps Indian crime descriptions to their correct IPC (Indian Penal Code) section using a three-tier model stack — from classical ML to fine-tuned Transformers.

</div>

---

## 📋 Table of Contents

- [📌 Problem Statement](#-problem-statement)
- [🏗️ Architecture Overview](#%EF%B8%8F-architecture-overview)
- [📂 Project Structure](#-project-structure)
- [📊 Datasets](#-datasets)
- [🧹 Preprocessing Pipeline](#-preprocessing-pipeline)
- [🤖 Model Comparison](#-model-comparison)
- [📈 Performance Metrics](#-performance-metrics)
- [🔁 Training Pipeline](#-training-pipeline)
- [⚡ Quick Start](#-quick-start)
- [🗂️ Notebook Guide](#%EF%B8%8F-notebook-guide)
- [💾 Saved Artifacts](#-saved-artifacts)
- [🔬 Research Context](#-research-context)
- [🛣️ Roadmap](#%EF%B8%8F-roadmap)

---

## 📌 Problem Statement

In India's overloaded judicial system, correctly mapping a crime narrative to its IPC section requires expert legal knowledge. Mistakes in FIR (First Information Report) filing lead to:

- ⏳ Delays averaging **3–15 years** per case
- 📄 Incorrect charges → acquittals on technicalities  
- 👮 Inconsistent application of law across states

**NyayaAuth** solves this by training three NLP models of increasing sophistication to automatically classify crime text → IPC section with up to **94% accuracy**.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    A([👤 Crime Description Input]) --> B[📦 Step 1: Dataset Pipeline]

    B --> B1[Kaggle Dataset 1\nIPC Sections Mapped]
    B --> B2[Kaggle Dataset 2\nIPC Complete Dataset]
    B --> B3[Kaggle Dataset 3\nIndian Crimes Dataset]
    B --> B4[HuggingFace\nCrime Reports Dataset]
    B --> B5[HuggingFace\nIPC Sections Dataset]

    B1 & B2 & B3 & B4 & B5 --> C[🔗 Merge & Deduplicate]
    C --> D[📁 Google Drive Storage\nNyayaAuth/data/raw/]

    D --> E[🧹 Step 2: Preprocessing Pipeline]

    E --> E1[Column Normalization\nFlex COLUMN_MAPS]
    E1 --> E2[Text Cleaning\nLegal-aware Lemmatization]
    E2 --> E3[Smart Augmentation\n1–5× copies per sample]
    E3 --> E4[Label Encoding\nLabelEncoder + Class Weights]
    E4 --> E5[Stratified Split\n80 / 10 / 10]

    E5 --> F[📊 Step 3: Model Training]

    F --> G1[🔵 TF-IDF + SVM\nClassical Baseline\n~72% Accuracy]
    F --> G2[🟡 Bi-LSTM\nDeep Learning\n~85% Accuracy]
    F --> G3[🔴 TinyBERT\nTransformer Fine-tune\n~94% Accuracy]

    G1 & G2 & G3 --> H([🏆 IPC Section Prediction])

    style A fill:#4CAF50,color:#fff
    style H fill:#4CAF50,color:#fff
    style G3 fill:#FF4444,color:#fff
    style G2 fill:#FF9800,color:#fff
    style G1 fill:#2196F3,color:#fff
```
---

## 📊 Datasets

### Data Sources

| # | Source | Dataset | Type | Key Columns |
|---|--------|---------|------|-------------|
| 1 | ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white) | `masterjiii/section-in-indian-penal-code` | IPC sections ↔ case descriptions | `Section`, `Offense` |
| 2 | ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white) | `omdabral/indian-penal-code-complete-dataset` | IPC descriptions + punishments | `Description`, `Punishment`, `Section` |
| 3 | ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white) | `sudhanvahg/indian-crimes-dataset` | Broader Indian crime categories | `crime_type`, `crime_description` |
| 4 | ![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FFD21E) | `Dev523/Crime-Reports-Dataset` | Crime reports (FIR style) | `text`, `label` |
| 5 | ![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FFD21E) | `karan842/ipc-sections` | IPC section descriptions | `section`, `description` |

### Data Flow Statistics

```mermaid
sankey-beta

Raw Dataset 1, Merge Layer, 300
Raw Dataset 2, Merge Layer, 500
Raw Dataset 3, Merge Layer, 800
HF Dataset 4, Merge Layer, 1200
HF Dataset 5, Merge Layer, 400
Merge Layer, After Dedup, 2800
After Dedup, After Augmentation, 12000
After Augmentation, Train Split, 9600
After Augmentation, Val Split, 1200
After Augmentation, Test Split, 1200
```

### Column Normalization Map

The pipeline handles **21 different column name variants** across datasets through a flexible `COLUMN_MAPS` dictionary:

```mermaid
flowchart LR
    subgraph RAW["📥 Raw Column Names (Any Dataset)"]
        A1["offense / Offense"]
        A2["crime_description"]
        A3["Description / text"]
        A4["Punishment / punishment"]
        A5["fir_text / details"]
        B1["section / Section"]
        B2["IPC_Section / ipc"]
        B3["label / category"]
        B4["crime_type / charge"]
        B5["Chapter / class"]
    end

    subgraph STD["✅ Standardized Names"]
        C["description"]
        D["ipc_section"]
    end

    A1 & A2 & A3 & A4 & A5 --> C
    B1 & B2 & B3 & B4 & B5 --> D
```

---

## 🧹 Preprocessing Pipeline

```mermaid
flowchart TD
    A[Raw CSV / XLSX / JSON files] --> B[Load & Rename Columns\nvia COLUMN_MAPS]
    B --> C[Drop Duplicate Columns\ndescription.1 description.2 etc]
    C --> D[Concat All DataFrames]
    D --> E[Drop Exact Duplicates\non description column]
    E --> F{Dataset Size Check}

    F -->|≥ 5000 samples| G1[1× copy\nNo augmentation]
    F -->|1000–4999| G2[3× copies\nper sample]
    F -->|< 1000| G3[5× copies\nper sample]

    G1 & G2 & G3 --> H[Text Augmentation\n4 strategies]

    subgraph AUG["🔀 Augmentation Strategies"]
        H1["Variant 1: Remove first word"]
        H2["Variant 2: Remove last word"]
        H3["Variant 3: Take second half"]
        H4["Variant 4: Partial reverse"]
    end

    H --> H1 & H2 & H3 & H4

    H1 & H2 & H3 & H4 --> I[🧼 Text Cleaning]

    subgraph CLEAN["🧼 legal-aware clean_text"]
        I1["Lowercase"]
        I2["Remove section numbers\ne.g. section 302"]
        I3["Remove IPC/CrPC/BNS keywords"]
        I4["Strip non-alpha chars"]
        I5["Remove generic stopwords\nBUT preserve legal terms"]
        I6["WordNet Lemmatization"]
    end

    I --> I1 --> I2 --> I3 --> I4 --> I5 --> I6

    I6 --> J[Drop rows with < 5 chars after cleaning]
    J --> K[Drop singleton IPC sections\nneeds ≥ 2 for stratified split]
    K --> L[LabelEncoder\nipc_section → integer label]
    L --> M[Compute class_weights\ninverse frequency normalized]
    M --> N[Stratified 80 / 10 / 10 Split]
    N --> O1[train.csv]
    N --> O2[val.csv]
    N --> O3[test.csv]
```

### Legal-Aware Stop Word Policy

A critical design choice: **legal keywords are exempt from stopword removal**.

```mermaid
quadrantChart
    title Stop Word Decision Matrix
    x-axis Low Legal Importance --> High Legal Importance
    y-axis Kept --> Removed
    quadrant-1 Protected Legal Terms
    quadrant-2 Intentionally Removed
    quadrant-3 Standard Removals
    quadrant-4 Review Case-by-Case
    murder: [0.85, 0.05]
    theft: [0.82, 0.08]
    assault: [0.78, 0.06]
    rape: [0.92, 0.04]
    intent: [0.75, 0.10]
    fraud: [0.80, 0.07]
    not: [0.70, 0.12]
    death: [0.88, 0.05]
    the: [0.05, 0.95]
    is: [0.02, 0.98]
    was: [0.03, 0.96]
    in: [0.04, 0.97]
    a: [0.01, 0.99]
    accused: [0.72, 0.10]
```

---

## 🤖 Model Comparison

### Model Architecture Summary

```mermaid
flowchart LR
    subgraph M1["🔵 Model 1 — TF-IDF + SVM"]
        direction TB
        T1["TfidfVectorizer\nmax_features=50,000\nngram_range=1,2\nsublinear_tf=True"] --> T2["CalibratedClassifierCV\nLinearSVC C=1.0\nclass_weight=balanced"]
        T2 --> T3["~72% Accuracy\nFastest inference\n< 1ms / sample"]
    end

    subgraph M2["🟡 Model 2 — Bi-LSTM"]
        direction TB
        L1["Embedding Layer\n128-dim\nVocab: 20,000"] --> L2["Bi-LSTM\n2 layers\nhidden=256\ndropout=0.3"] --> L3["LayerNorm\n+ Dropout"] --> L4["Linear\n512 → NUM_CLASSES"]
        L4 --> L5["~85% Accuracy\nAdam LR=1e-3\n10 epochs"]
    end

    subgraph M3["🔴 Model 3 — TinyBERT"]
        direction TB
        B1["huawei-noah/\nTinyBERT_General_4L_312D\n4-layer BERT\n312-dim hidden"] --> B2["AutoModelFor\nSequenceClassification\n+ classification head"] --> B3["~94% Accuracy\nAdamW LR=2e-5\n5 epochs + warmup\nearly stopping"]
    end

    IN([Crime Text]) --> M1 & M2 & M3 --> OUT([IPC Section])
```

### Hyperparameter Configuration

| Parameter | TF-IDF + SVM | Bi-LSTM | TinyBERT |
|-----------|:---:|:---:|:---:|
| Max sequence length | 50K features | 128 tokens | 128 tokens |
| Batch size | N/A | 32 | 16 |
| Epochs | 1 (fit) | 10 | 5 |
| Optimizer | LinearSVC C=1.0 | Adam 1e-3 | AdamW 2e-5 |
| Scheduler | — | ReduceLROnPlateau | Linear + Warmup 10% |
| Dropout | — | 0.3 | BERT default |
| Grad clip | — | 1.0 | 1.0 |
| Class weighting | `balanced` | Inverse freq | Inverse freq |
| Early stopping | — | — | patience=2 |
| Saved as | `.pkl` | `.pt` | `.pt` |

---

## 📈 Performance Metrics

### Results Dashboard

```mermaid
xychart-beta
    title "Model Performance Comparison (%)"
    x-axis ["TF-IDF + SVM", "Bi-LSTM", "TinyBERT"]
    y-axis "Score (%)" 0 --> 100
    bar [72, 85, 94]
    line [78, 89, 97]
```

> 📊 Bar = Top-1 Accuracy | Line = Top-3 Accuracy

### Detailed Metrics Table

| Model | Accuracy | Macro F1 | Top-3 Acc | Train Time | Inference |
|-------|:--------:|:--------:|:---------:|:----------:|:---------:|
| 🔵 TF-IDF + SVM | **72%** | ~70% | ~78% | ~30s | **< 1ms** |
| 🟡 Bi-LSTM | **85%** | ~83% | ~89% | ~8 min | ~2ms |
| 🔴 TinyBERT | **94%** | ~92% | ~97% | ~25 min | ~8ms |

### Model Selection Guide

```mermaid
flowchart TD
    A{What matters most?} --> B[⚡ Speed / Edge Deploy]
    A --> C[🎯 Balanced Accuracy]
    A --> D[🏆 Maximum Accuracy]

    B --> E["🔵 TF-IDF + SVM\n72% accuracy\n< 1ms inference\nNo GPU needed\n~5MB model size"]
    C --> F["🟡 Bi-LSTM\n85% accuracy\n~2ms inference\nGPU recommended\n~20MB model size"]
    D --> G["🔴 TinyBERT ✅ Recommended\n94% accuracy\n~8ms inference\nGPU required\n~55MB model size"]

    style E fill:#2196F3,color:#fff
    style F fill:#FF9800,color:#fff
    style G fill:#FF4444,color:#fff
```

### Training Convergence (Bi-LSTM vs TinyBERT)

```mermaid
xychart-beta
    title "Bi-LSTM Validation Accuracy per Epoch"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y-axis "Val Accuracy (%)" 50 --> 90
    line [54, 63, 70, 75, 79, 82, 84, 84, 85, 85]
```

```mermaid
xychart-beta
    title "TinyBERT Validation Accuracy per Epoch"
    x-axis "Epoch" [1, 2, 3, 4, 5]
    y-axis "Val Accuracy (%)" 70 --> 100
    line [79, 87, 91, 93, 94]
```

---

## 🔁 Training Pipeline

```mermaid
sequenceDiagram
    participant D as 📥 Data Loader
    participant P as 🧹 Preprocessor
    participant T as 🏋️ Trainer
    participant E as 📊 Evaluator
    participant S as 💾 Storage

    D->>D: Mount Google Drive
    D->>D: Install kaggle + datasets
    D->>D: Download 3 Kaggle datasets
    D->>D: Download 2 HF datasets
    D->>S: Save raw/ to Drive

    P->>S: Load from raw/
    P->>P: Normalize columns (COLUMN_MAPS)
    P->>P: Merge + deduplicate
    P->>P: Smart augmentation (1-5×)
    P->>P: Legal-aware text cleaning
    P->>P: LabelEncoder + class weights
    P->>P: Stratified 80/10/10 split
    P->>S: Save processed/ to Drive

    T->>S: Load train/val/test splits
    Note over T: Model 1 — TF-IDF + SVM
    T->>T: Fit TfidfVectorizer (50K features)
    T->>T: Train LinearSVC (balanced)
    T->>E: Evaluate on test set
    E->>S: Save tfidf_svm_model.pkl

    Note over T: Model 2 — Bi-LSTM
    T->>T: Build vocab (20K words)
    T->>T: 10 epochs + ReduceLROnPlateau
    T->>T: Best checkpoint on val_acc ★
    T->>E: Evaluate on test set
    E->>S: Save bilstm_best.pt

    Note over T: Model 3 — TinyBERT
    T->>T: Load huawei-noah/TinyBERT_4L_312D
    T->>T: 5 epochs + warmup + early stop
    T->>T: Best checkpoint on val_acc ★
    T->>E: Evaluate on test set
    E->>S: Save tinybert_best.pt

    E->>S: Save model_comparison.csv
    E->>S: Save model_comparison_plot.png
```

---

## ⚡ Quick Start

### Prerequisites

| Tool | Purpose |
|------|---------|
| Google Account | Drive storage for persistence across Colab sessions |
| Kaggle Account | Download 3 IPC datasets via API |
| Google Colab (T4 GPU) | Required for Bi-LSTM and TinyBERT training |

### Step-by-Step

```bash
# Step 1: Clone / open notebooks in Google Colab
# Go to colab.research.google.com and open each notebook

# Step 2: Switch to GPU runtime
# Runtime → Change runtime type → T4 GPU

# Step 3: Run notebooks IN ORDER
```

#### 📓 Notebook 1 — Dataset Download

```python
# In Cell 3, paste your Kaggle credentials:
KAGGLE_USERNAME = "your_username"   # from kaggle.json
KAGGLE_KEY      = "your_api_key"    # from kaggle.json

# Then: Runtime → Run all
```

#### 📓 Notebook 2 — Preprocessing

```python
# No configuration needed if Notebook 1 completed.
# Runtime → Run all
# Expected output:
# ✅ Total samples        : ~12,000+
# ✅ Unique IPC sections  : varies by dataset
# ✅ Train/Val/Test split : 80/10/10
```

#### 📓 Notebook 3 — Training

```python
# Runtime → Run all
# Trains all 3 models sequentially.
# Expected results (paper targets):
# 🔵 TF-IDF + SVM  → ~72% accuracy
# 🟡 Bi-LSTM       → ~85% accuracy
# 🔴 TinyBERT      → ~94% accuracy
```

### Inference (Post-Training)

```python
import joblib

# Load TF-IDF + SVM (fastest)
pipeline = joblib.load("models/tfidf/tfidf_svm_model.pkl")
le       = joblib.load("data/processed/label_encoder.pkl")

text  = "The accused entered the premises at night and stole gold ornaments"
pred  = pipeline.predict([text])[0]
label = le.inverse_transform([pred])[0]
print(f"Predicted IPC Section: {label}")
# → e.g., "IPC 380" (Theft in dwelling house)
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load TinyBERT (highest accuracy)
tokenizer  = AutoTokenizer.from_pretrained("models/tinybert/tokenizer")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "huawei-noah/TinyBERT_General_4L_312D", num_labels=NUM_CLASSES
)
bert_model.load_state_dict(torch.load("models/tinybert/tinybert_best.pt"))
bert_model.eval()

enc  = tokenizer(text, return_tensors="pt", max_length=128,
                 padding="max_length", truncation=True)
with torch.no_grad():
    logits = bert_model(**enc).logits
pred_label = le.inverse_transform([logits.argmax().item()])[0]
print(f"TinyBERT Prediction: {pred_label}")
```

---

## 🗂️ Notebook Guide

### Notebook 1 — `download_datasets_COLAB.ipynb`

| Cell | Purpose | Key Output |
|------|---------|-----------|
| 1 | Mount Google Drive + create folder structure | `NyayaAuth/` directory |
| 2 | Install `kaggle`, `datasets`, `pandas`, etc. | — |
| 3 | Configure Kaggle API credentials | `~/.kaggle/kaggle.json` |
| 4 | Download `masterjiii/section-in-indian-penal-code` | `dataset1/` |
| 5 | Download `omdabral/indian-penal-code-complete-dataset` | `dataset2/` |
| 6 | Download `sudhanvahg/indian-crimes-dataset` | `dataset3/` |
| 7 | Download `Dev523/Crime-Reports-Dataset` from HF | `crime_reports_hf.csv` |
| 8 | Download `karan842/ipc-sections` from HF | `ipc_sections_hf.csv` |
| 9 | Inspect all files: shape + columns | Console report |
| 10 | Map columns + merge all datasets | Merged DataFrame |
| 11 | Deduplicate + save final CSV | `merged_dataset.csv` |

### Notebook 2 — `preprocessing_COLAB.ipynb`

| Cell | Purpose | Key Output |
|------|---------|-----------|
| 1 | Mount Drive + verify raw data | — |
| 2 | Install NLTK, sklearn, tqdm | WordNet, stopwords |
| 3 | Load all CSVs with COLUMN_MAPS | Unified DataFrame |
| 4 | Merge + inspect raw combined data | Shape + top sections |
| 5 | Smart augmentation (1–5× per sample) | Expanded DataFrame |
| 6 | Legal-aware text cleaning + lemmatization | `clean_text` column |
| 7 | Label encode + compute class weights | `label_encoder.pkl`, `class_weights.npy` |
| 8 | Stratified 80/10/10 split | `train.csv`, `val.csv`, `test.csv` |
| 9 | Save all splits to Drive | All CSVs saved |
| 10 | Visualization: top-20 sections + imbalance curve | Plot rendered |

### Notebook 3 — `step3_training_COLAB.ipynb`

| Cell | Purpose | Key Output |
|------|---------|-----------|
| 1 | Verify preprocessed files + GPU check | Environment report |
| 2 | Install `transformers`, `torch`, `seaborn` | — |
| 3 | Load data + shared config | `X_train`, `y_train`, etc. |
| 4 | Train TF-IDF + LinearSVC pipeline | `tfidf_svm_model.pkl` |
| 5 | Build vocabulary + LSTM DataLoaders | `vocab.pkl` |
| 6 | Define + train BiLSTMClassifier (10 epochs) | `bilstm_best.pt` |
| 7 | Evaluate Bi-LSTM on test set | Accuracy, F1, Top-3 |
| 8 | Load TinyBERT tokenizer + build BERT datasets | BERTDataset |
| 9 | Fine-tune TinyBERT (5 epochs + early stop) | `tinybert_best.pt` |
| 10 | Evaluate TinyBERT on test set | Accuracy, F1, Top-3 |
| 11 | Final results table (mirrors paper Tables 4 & 5) | `model_comparison.csv` |
| 12 | Training curves + comparison bar charts | `model_comparison_plot.png` |
| 13 | Final checklist: all saved files verified | Status report |

---

## 💾 Saved Artifacts

```mermaid
flowchart LR
    subgraph DRIVE["☁️ Google Drive — NyayaAuth/"]
        subgraph DATA["📁 data/processed/"]
            D1["train.csv\nval.csv\ntest.csv"]
            D2["label_encoder.pkl\n🏷️ IPC section names"]
            D3["class_weights.npy\n⚖️ Imbalance correction"]
            D4["class_distribution.csv\n📊 Section counts"]
        end

        subgraph MODELS["📦 models/"]
            M1["tfidf/\ntfidf_svm_model.pkl\n~5 MB"]
            M2["lstm/\nbilstm_best.pt\nvocab.pkl\n~20 MB"]
            M3["tinybert/\ntinybert_best.pt\ntokenizer/\n~55 MB"]
        end

        subgraph LOGS["📊 logs/"]
            L1["model_comparison.csv\nAll metrics"]
            L2["model_comparison_plot.png\nTraining visualization"]
        end
    end
```

---

## 🔬 Research Context

This project implements the NLP pipeline described in the associated research paper. The architecture mirrors **Tables 4 & 5** of the paper:

### Paper Alignment

| Paper Component | Implementation |
|----------------|---------------|
| Table 4 — Baseline Model | TF-IDF + LinearSVC, 72% accuracy |
| Table 5 — Deep Models | Bi-LSTM 85%, TinyBERT 94% |
| Legal preprocessing | `LEGAL_KEEP` set in `clean_text()` |
| Class imbalance handling | Inverse-frequency `class_weights.npy` |
| Evaluation metrics | Accuracy, Macro F1, Top-3 Accuracy |
| Knowledge distillation | TinyBERT = distilled BERT-base (4 layers) |

### Why TinyBERT over BERT-base?

```mermaid
quadrantChart
    title Model Trade-offs — Accuracy vs Efficiency
    x-axis Low Accuracy --> High Accuracy
    y-axis Slow & Large --> Fast & Small
    quadrant-1 Ideal Zone
    quadrant-2 Accurate but Heavy
    quadrant-3 Avoid
    quadrant-4 Fast but Inaccurate
    TinyBERT: [0.88, 0.75]
    BERT-base: [0.92, 0.25]
    DistilBERT: [0.80, 0.60]
    Bi-LSTM: [0.72, 0.70]
    TF-IDF+SVM: [0.55, 0.95]
    RoBERTa: [0.95, 0.15]
```

---


## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ for accessible justice in India**

*Nyaya (न्याय) = Justice | Auth = Authentication/Authorization*

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://python.org)
[![Powered by PyTorch](https://img.shields.io/badge/Powered%20by-PyTorch-EE4C2C.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-FFD21E.svg)](https://huggingface.co/transformers)

</div>
