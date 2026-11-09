# 🐦 Production-Grade Twitter Sentiment Analysis with NLP & Deep Learning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.9-green?style=for-the-badge)](https://www.nltk.org/)
[![Dataset: Sentiment140](https://img.shields.io/badge/Dataset-Sentiment140-blueviolet?style=for-the-badge)](http://help.sentiment140.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

An end-to-end, production-ready Natural Language Processing (NLP) system comparing **Classical Statistical Benchmarks (N-gram TF-IDF + Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Classifier)** against a **Bidirectional LSTM Deep Neural Architecture** on the Stanford **Sentiment140** Twitter benchmark.

---

## 📌 Table of Contents
* [Executive Summary & Business Context](#-executive-summary--business-context)
* [System Architecture & Pipeline](#-system-architecture--pipeline)
* [Linguistic Cleaning Innovations](#-linguistic-cleaning-innovations)
* [Model Benchmark Leaderboard](#-model-benchmark-leaderboard)
* [Visual Analytics Gallery](#-visual-analytics-gallery)
* [Project Directory Structure](#-project-directory-structure)
* [Quickstart & Usage](#-quickstart--usage)
  * [1. Jupyter Notebook](#1-interactive-jupyter-notebook)
  * [2. Production CLI Inference](#2-production-cli-inference)
  * [3. Python API Integration](#3-python-api-integration)
* [Error Analysis & Qualitative Diagnostics](#-error-analysis--qualitative-diagnostics)
* [License & Acknowledgments](#-license--acknowledgments)

---

## 📖 Executive Summary & Business Context

Understanding consumer opinion, public reaction, and brand health in real-time is a foundational challenge in modern enterprise NLP. Microblog data (Twitter / X) exhibits unique linguistic properties:
1. **Severe Brevity & Slang**: Tweets are constrained in length, laden with abbreviations, contractions, and irregular orthography.
2. **Dense Emotional Signals**: Emoticons (`:)`, `:-(`, `<3`) and hashtags (`#loveit`, `#hateit`) carry direct emotional sentiment that naive filters often destroy.
3. **The Negation Danger**: Sentences like *"The camera is not good"* or *"Surprisingly not bad at all"* depend entirely on negations (`not`, `no`, `never`, `cannot`). Standard stopword removal removes these words, reversing the semantic meaning.
4. **Exaggerated Repetitions**: Users emphasize sentiment by repeating characters (`"sooooo coooool"`). Naive regex replacement corrupts legitimate words (`"good"` $\rightarrow$ `"god"`, `"cool"` $\rightarrow$ `"col"`, `"school"` $\rightarrow$ `"schol"`).

This repository implements a **linguistically-sound, high-throughput pipeline** that handles these challenges and benchmarks multiple modeling paradigms.

---

## 🏗 System Architecture & Pipeline

```
                      ┌────────────────────────────────────────┐
                      │    Stanford Sentiment140 Dataset       │
                      │       (1.6M Raw Twitter Corpus)        │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │  High-Performance Stratified Ingestion │
                      │       (Balanced 50,000 Tweets)         │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │  Linguistically-Sound NLP Cleaning     │
                      │  • HTML unescaping & URL removal       │
                      │  • Mentions & Hashtag parsing          │
                      │  • Contraction expansion (can't->cannot)
                      │  • Emoticon tokenization (:)->__pos__  │
                      │  • Controlled repeat squeeze (3+ -> 2) │
                      │  • Negation-preserving stopword filter │
                      │  • WordNet Lemmatization               │
                      └──────────────────┬─────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
        ┌───────────────────────┐                 ┌───────────────────────┐
        │  Representation A:    │                 │  Representation B:    │
        │  Sublinear TF-IDF     │                 │  Dense Sequential     │
        │  N-grams (1, 2)       │                 │  Token Sequences      │
        │  (25,000 Features)    │                 │  (Length=50, V=20,000)│
        └───────────┬───────────┘                 └───────────┬───────────┘
                    │                                         │
         ┌──────────┴──────────┐                              │
         ▼                     ▼                              ▼
┌─────────────────┐   ┌─────────────────┐           ┌───────────────────┐
│ Multinomial NB  │   │   Logistic Reg  │           │ Bidirectional     │
│  (Laplace α)    │   │ (L2 Regularized)│           │ LSTM Network      │
└────────┬────────┘   └────────┬────────┘           └─────────┬─────────┘
         │                     │                              │
         └───────────┬─────────┴──────────────────────────────┘
                     ▼
         ┌──────────────────────────────────────┐
         │ Comprehensive Benchmarking Suite     │
         │ Accuracy, Precision, Recall, F1, ROC │
         │ Side-by-side Confusion Matrices      │
         │ Error Diagnostics & Sarcasm Analysis │
         └──────────────────┬───────────────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │ Production Inference API & Artifacts │
         └──────────────────────────────────────┘
```

---

## 🔬 Linguistic Cleaning Innovations

| Challenge | Naive Preprocessing Flaw | Our Engineered Solution |
| :--- | :--- | :--- |
| **Negation Flipping** | Removes `"not"`, `"no"`, `"never"`. *"Not good"* becomes *"good"*. | **Negation-Preserving Filter**: Keeps 24 critical negation operators in vocabulary. |
| **Emoticons** | Strips punctuation, discarding `:)`, `:-(`, `<3`. | **Emoticon Mapping**: Translates symbols into semantic tokens `__pos_emoticon__` / `__neg_emoticon__`. |
| **Character Elongation** | `(.)\1+ -> \1` breaks `"good"` to `"god"`, `"school"` to `"schol"`. | **Controlled Squeeze**: `(.)\1{2,} -> \1\1` reduces 3+ repetitions to 2 (`"coooool"` $\rightarrow$ `"cool"`). |
| **Contractions** | Leaves messy apostrophes (`can't`, `it's`). | **Contextual Contraction Expansion** (`can't` $\rightarrow$ `cannot`, `won't` $\rightarrow$ `will not`). |
| **Root Normalization** | Porter Stemmer creates non-words (`"besti"`, `"uandpet"`). | **WordNet Lemmatizer**: Syntactically valid English lemma extraction. |

---

## 🏆 Model Benchmark Leaderboard

Evaluated on a strictly held-out stratified test set of **10,000 tweets** (5,000 Positive, 5,000 Negative):

| Rank | Model Architecture | Feature Representation | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Logistic Regression** | **TF-IDF (1-2 N-grams)** | **76.47%** | **0.7648** | **0.7646** | **0.7647** | **0.8496** | **0.16s** |
| 🥈 | **Linear SVC (Calibrated)** | TF-IDF (1-2 N-grams) | 75.76% | 0.7589 | 0.7550 | 0.7570 | 0.8415 | 1.48s |
| 🥉 | **Bidirectional LSTM** | Sequential Embeddings (d=64) | 76.04% | 0.7696 | 0.7434 | 0.7563 | 0.8335 | 29.50s |
| 4 | **Multinomial Naive Bayes** | TF-IDF (1-2 N-grams) | 75.15% | 0.7618 | 0.7318 | 0.7465 | 0.8354 | 0.01s |

> **Architectural Insight**: While the Bidirectional LSTM captures long-range bidirectional semantics, **Logistic Regression with N-gram TF-IDF and linguistically-sound preprocessing** achieved the highest overall test accuracy (76.47%) and ROC-AUC (0.8496) while training **180x faster** and offering sub-millisecond inference latency with full interpretability.


## 📁 Project Directory Structure

```
Twitter-Sentiment-Analysis/
├── figures/                            # High-resolution publication plots
│   ├── eda_distributions.png           # Target & token distributions
│   ├── wordclouds.png                  # Dual negative/positive wordclouds
│   ├── top_bigrams.png                 # Salient bigram bar charts
│   ├── bilstm_history.png              # Loss & accuracy epoch curves
│   ├── confusion_matrices.png          # 4-panel side-by-side heatmaps
│   └── roc_curves.png                  # Combined ROC-AUC curves
├── models/                             # Serialized production artifacts
│   ├── best_sentiment_model_lr.joblib  # Trained Logistic Regression classifier
│   ├── tfidf_vectorizer.joblib         # Fitted 25k-feature TF-IDF matrix
│   ├── keras_tokenizer.joblib          # Token ID tokenizer for Keras
│   └── bilstm_sentiment_model.keras    # Saved deep learning BiLSTM weights
├── data.csv                            # Stanford Sentiment140 CSV (1.6M tweets)
├── project 1.ipynb                     # Master Interactive Jupyter Notebook
├── twitter_sentiment_pipeline.py       # Production inference module & CLI
├── requirements.txt                    # Pinned environment dependencies
└── README.md                           # Comprehensive documentation
```

---

## 🚀 Quickstart & Usage

### Setup Environment
```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Interactive Jupyter Notebook
Launch Jupyter to explore all experiments, inline plots, and diagnostic breakdowns:
```bash
jupyter notebook "project 1.ipynb"
```

### 2. Production CLI Inference
Analyze arbitrary tweet text directly from your terminal:
```bash
# Positive Tweet Example
python twitter_sentiment_pipeline.py --predict "The product quality is absolutely outstanding and fast delivery! :)"

# Negative Tweet Example
python twitter_sentiment_pipeline.py --predict "Worst flight ever, delayed 6 hours and lost all my luggage! Never flying again."

# Run Built-in Benchmark Suite
python twitter_sentiment_pipeline.py --eval
```

### 3. Python API Integration
```python
from twitter_sentiment_pipeline import SentimentClassifier

# Initialize inference container
clf = SentimentClassifier()

# Single prediction
result = clf.predict("I was skeptical at first, but this tool is surprisingly not bad at all!")
print(f"Sentiment:  {result['sentiment']}")
print(f"Confidence: {result['confidence']}%")
print(f"Probability: {result['positive_probability']}")
# Output:
# Sentiment:  POSITIVE
# Confidence: 78.4%
```

---

## 🔍 Error Analysis & Qualitative Diagnostics

To uncover model limitations, we inspected the most confident misclassifications:

| Category | Example Tweet | Target | Predicted | Underlying Cause |
| :--- | :--- | :---: | :---: | :--- |
| **Sarcasm** | *"Great, love waiting another hour for my delayed flight yay"* | 0 (Neg) | 1 (Pos) | Sarcastic surface tokens (`great`, `love`, `yay`) overpower the situational negative context. |
| **Dark Humor / Slang** | *"That joke was so funny it literally killed me, dying laughing"* | 1 (Pos) | 0 (Neg) | Hyperbolic negative tokens (`killed`, `dying`) interpreted literally by Bag-of-Words features. |
| **Mixed Valence** | *"The screen is gorgeous, but the battery life is unacceptable"* | 0 (Neg) | 1 (Pos) | Equal balance of strong positive and negative clauses requires syntax-level constituency parsing. |

---

## 📜 License & Acknowledgments
* **License**: Distributed under the MIT License.
* **Corpus**: Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*. Stanford University Technical Report.
