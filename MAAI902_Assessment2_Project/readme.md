# Automated Sentiment Analysis for Content Moderation

**Pitch:** This project designs and prototypes an NLP pipeline that automatically classifies incoming user-generated movie reviews as Positive or Negative, enabling scalable automated content moderation and platform analytics.

## Dataset
* **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Size:** 49,582 unique reviews (perfectly balanced 50/50 split).

## Pipeline Architecture
```text
[Raw Review] --> [Regex Cleaning] --> [WordPiece Tokenizer (Max 512)] --> [DistilBERT Classifier] --> [Sentiment + Confidence Score]
```

## Results Comparison
| Model | Macro-F1 Score | Accuracy |
| :--- | :---: | :---: |
| **Baseline:** TF-IDF + LinearSVC | 0.90 | 90.00% |
| **Final:** DistilBERT (uncased) | **0.93** | **93.28%** |

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt` (requires `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`).
3. Set up your Kaggle API credentials (`kaggle.json`) or environment variables to download the dataset.
4. Run the Jupyter Notebook `maai902_sentiment_analysis.ipynb` top-to-bottom. *(Note: A T4 GPU or better is recommended for the DistilBERT training cell).*

## Ethics and Limitations
* **Sequence Limits:** The model is capped at 512 tokens. Highly detailed, long-form reviews that place their final verdict at the very end may be truncated, leading to misclassification.
* **Sarcasm and Mixed Sentiment:** The model occasionally struggles with reviews that change tone mid-paragraph or employ heavy sarcasm.
* **Human-in-the-Loop:** This tool is designed as a routing and triage aid, not an autonomous gatekeeper. Predictions with ambiguous confidence scores (e.g., 0.50 - 0.65) should always be routed to human moderators to prevent automation bias and unfair censorship.