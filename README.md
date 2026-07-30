# 🛡️ Toxic Comment Classification using DistilBERT

A Natural Language Processing (NLP) project that automatically identifies whether an online comment is **toxic** or **non-toxic** by comparing a traditional machine learning model with a transformer-based deep learning model.

---

## 📌 Overview

Online communities generate millions of comments every day, making manual moderation difficult. Automated toxicity detection helps improve online safety by identifying offensive or harmful content before it spreads.

This project evaluates two different text classification approaches:

- **Logistic Regression** using TF-IDF features
- **Fine-Tuned DistilBERT** using Hugging Face Transformers

The goal is to measure how contextual language models improve toxic comment detection compared to traditional NLP techniques.

---

## 🎯 Objectives

- Detect toxic comments automatically.
- Build a classical machine learning baseline.
- Fine-tune a transformer model for binary classification.
- Compare model performance using multiple evaluation metrics.
- Demonstrate the advantages of contextual embeddings.

---

# 📚 Dataset

This project uses a publicly available **Toxic Comment Dataset** from Hugging Face.

### Target Classes

| Label | Meaning |
|-------|---------|
| 0 | Non-toxic |
| 1 | Toxic |

### Dataset Characteristics

- Binary text classification
- Large class imbalance
- Short online comments
- Stratified train/validation/test split

---

# 🧹 Data Preparation

The preprocessing pipeline includes:

- Text cleaning
- Dataset splitting
- Stratified sampling
- Tokenization
- Attention mask generation
- Maximum sequence length of **128 tokens**

---

# 🤖 Models

## Logistic Regression

A lightweight baseline model built using:

- TF-IDF Vectorizer
- Unigram features
- Bigram features

---

## DistilBERT

A compressed transformer model based on BERT.

Training configuration:

| Parameter | Value |
|-----------|------:|
| Model | distilbert-base-uncased |
| Epochs | 4 |
| Learning Rate | 2 × 10⁻⁵ |
| Optimizer | AdamW |
| Early Stopping | Yes |
| Max Length | 128 |

---

# 🛠️ Software & Libraries

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

# 🔄 Project Workflow

```text
Raw Comments
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Text Cleaning
      │
      ▼
Train/Test Split
      │
      ▼
TF-IDF Features ─────────► Logistic Regression
      │
      │
      ▼
DistilBERT Tokenizer
      │
      ▼
Fine-Tune DistilBERT
      │
      ▼
Evaluation
```

---

# 📈 Experimental Results

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 89.17% | 91.92% | 89.17% | 90.28% |
| DistilBERT | **94.20%** | **94.01%** | **94.20%** | **94.10%** |

---

# 📊 Key Observations

- Logistic Regression provides a reliable baseline with strong TF-IDF features.
- DistilBERT significantly improves overall classification performance.
- Transformer embeddings capture sentence meaning rather than relying on keyword frequency.
- Context-aware representations reduce classification errors.

---

# 📂 Repository Structure

```
Toxic-Comment-Classification/

│── LLM_Gnanesh_.ipynb
│── README.md
│── requirements.txt
│── report.pdf

├── data/
│
├── models/
│
├── figures/
│     ├── class_distribution.png
│     ├── confusion_matrix.png
│     ├── training_metrics.png
│     └── evaluation_results.png
│
└── outputs/
      ├── predictions.csv
      └── classification_report.csv
```

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/toxic-comment-classification.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
LLM_Gnanesh_.ipynb
```

---

# 📦 Dependencies

```
transformers
datasets
torch
evaluate
accelerate
numpy
pandas
matplotlib
scikit-learn
```

---

# 💬 Prediction Examples

| Comment | Prediction |
|----------|------------|
| Thank you for your contribution! | Non-toxic |
| You are completely useless. | Toxic |
| I appreciate your help. | Non-toxic |
| Nobody wants you here. | Toxic |

---

# 🚀 Future Improvements

- Multi-label toxicity classification
- Multilingual toxicity detection
- Data augmentation
- Larger transformer models (RoBERTa, DeBERTa)
- Explainable AI (SHAP/LIME)
- Real-time moderation API

---

# 📖 References

- DistilBERT: Smaller, Faster, Cheaper and Lighter
- BERT: Bidirectional Transformers for Language Understanding
- Hugging Face Transformers
- Deep Learning Based Text Classification Survey
- Scikit-learn Documentation

---

# 👤 Author

**Gnanesh Cheedarla**

MSc Data Science

University of Hertfordshire

---

# 📄 License

This project is submitted for academic and educational purposes.
