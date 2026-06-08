# Twitter Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-BERT%20%7C%20TF--IDF-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

> Twitter sentiment analysis pipeline: tweet preprocessing → TF-IDF and BERT embeddings → multi-class classification (positive / neutral / negative) → visualisation dashboards.

## Pipeline

```
Raw Tweets (Twitter API / dataset)
        ↓
Preprocessing
  ├── Remove URLs, hashtags, @mentions
  ├── Lowercase, strip punctuation
  ├── Tokenisation + stop-word removal
  └── Lemmatisation (spaCy)
        ↓
Feature Extraction
  ├── TF-IDF (baseline)
  └── BERT sentence embeddings (bert-base-uncased)
        ↓
Classification
  ├── Logistic Regression (TF-IDF baseline)
  ├── SVM with RBF kernel
  └── Fine-tuned DistilBERT (best)
        ↓
Evaluation + Visualisation
  ├── Confusion matrix
  ├── Word clouds by sentiment
  └── Sentiment trend over time
```

## Model Results

| Model | Accuracy | F1 (macro) |
|---|---|---|
| TF-IDF + Logistic Regression | 78.3% | 0.771 |
| TF-IDF + SVM | 81.2% | 0.806 |
| DistilBERT (fine-tuned) | 89.7% | 0.891 |

## Setup
```bash
git clone https://github.com/jaiminbabariya7/Twitter-sentimental-analysis
cd Twitter-sentimental-analysis
pip install transformers scikit-learn pandas nltk spacy
python main.py --model distilbert --data data/tweets.csv
```

## Skills Demonstrated
`Python` · `NLP` · `BERT` · `TF-IDF` · `scikit-learn` · `Sentiment Analysis` · `Text Preprocessing` · `Visualisation`
