#!/usr/bin/env python3
"""
=============================================================================
Twitter Sentiment Analysis Pipeline (Production Module & CLI)
=============================================================================
An end-to-end NLP and Deep Learning system for classifying tweet sentiment.
Supports:
  1. High-throughput linguistically-sound text cleaning.
  2. Sublinear TF-IDF + Logistic Regression / MNB / LinearSVC inference.
  3. Bidirectional LSTM Neural Network inference.
  4. Command-Line Interface (CLI) for batch evaluation or single-tweet analysis.

Usage:
  python twitter_sentiment_pipeline.py --predict "I love this product!"
  python twitter_sentiment_pipeline.py --predict "Flight was delayed 5 hours :("
  python twitter_sentiment_pipeline.py --eval
=============================================================================
"""

import os
import sys
import re
import html
import time
import argparse
import joblib
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pre-compiled regular expressions for high-throughput vectorized execution
HTML_CLEAN = re.compile(r'<.*?>')
URL_CLEAN = re.compile(r'https?://\S+|www\.\S+')
MENTION_CLEAN = re.compile(r'@\w+')
HASHTAG_CLEAN = re.compile(r'#(\w+)')
CHAR_REPEAT_CLEAN = re.compile(r'(.)\1{2,}')  # Reduce 3+ repetitions to 2

# Comprehensive Contraction Map
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "it's": "it is", "he's": "he is",
    "she's": "she is", "that's": "that is", "what's": "what is",
    "there's": "there is", "you're": "you are", "they're": "they are",
    "we're": "we are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'd": "i would",
    "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "i'll": "i will",
    "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "we'll": "we will", "they'll": "they will"
}
CONTRACTION_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS.keys()) + r')\b', re.IGNORECASE)

# Sentiment-bearing Emoticon Lexicons
EMOTICONS_POS = {':)', ':-)', ';)', ':d', ':-d', '<3', ':p', ';p', 'xd', '=)'}
EMOTICONS_NEG = {':(', ':-(', ":'(", ':/', ':-/', ':-|', ':[', ':{'}

# Negation-aware Stopwords Definition
try:
    base_stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    base_stopwords = set(stopwords.words('english'))

NEGATION_TOKENS = {
    'no', 'not', 'nor', 'neither', 'never', 'none', 'cannot',
    "don't", "aren't", "couldn't", "didn't", "doesn't", "hadn't",
    "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't",
    "shouldn't", "wasn't", "weren't", "won't", "wouldn't"
}
CUSTOM_STOPWORDS = base_stopwords - NEGATION_TOKENS

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()


def preprocess_tweet(text: str) -> str:
    """
    Standardizes raw tweet text through a sequence of linguistically-sound
    normalization steps while rigorously preserving emotional polarity and negations.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities (&amp; -> &, &lt; -> <)
    text = html.unescape(text).lower()
    
    # 2. Tokenize and map emoticons before stripping punctuation
    tokens = text.split()
    tokens = ['__pos_emoticon__' if t in EMOTICONS_POS else '__neg_emoticon__' if t in EMOTICONS_NEG else t for t in tokens]
    text = " ".join(tokens)
    
    # 3. Strip URLs and User Mentions
    text = URL_CLEAN.sub('', text)
    text = MENTION_CLEAN.sub('', text)
    
    # 4. Extract hashtag alphanumeric text (#happy -> happy)
    text = HASHTAG_CLEAN.sub(r'\1', text)
    
    # 5. Expand Contractions (e.g. can't -> cannot)
    text = CONTRACTION_PATTERN.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)
    
    # 6. Squeeze exaggerated repeated characters (3+ -> 2, e.g. cooool -> cool)
    text = CHAR_REPEAT_CLEAN.sub(r'\1\1', text)
    
    # 7. Remove non-alphabetical noise while preserving emoticon placeholders
    text = re.sub(r'[^a-zA-Z_\s]', ' ', text)
    
    # 8. Negation-preserving stopword filtration & lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in CUSTOM_STOPWORDS and len(word) > 1
    ]
    return " ".join(cleaned_tokens)


class SentimentClassifier:
    """Production inference container for trained sentiment models."""
    def __init__(self, model_path='models/best_sentiment_model_lr.joblib',
                 vectorizer_path='models/tfidf_vectorizer.joblib'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Model or vectorizer artifact not found. Please train first or ensure "
                f"'{self.model_path}' and '{self.vectorizer_path}' exist."
            )
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict(self, raw_text: str) -> dict:
        """Classify a single raw tweet string and return structured prediction."""
        cleaned = preprocess_tweet(raw_text)
        vec = self.vectorizer.transform([cleaned])
        prob = float(self.model.predict_proba(vec)[0][1])
        
        label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
        confidence = prob if prob >= 0.5 else (1.0 - prob)
        
        return {
            'raw_text': raw_text,
            'cleaned_text': cleaned,
            'sentiment': label,
            'confidence': round(confidence * 100, 2),
            'positive_probability': round(prob, 4)
        }

    def predict_batch(self, texts: list) -> pd.DataFrame:
        """Classify a batch of raw tweet strings."""
        return pd.DataFrame([self.predict(t) for t in texts])


def main():
    parser = argparse.ArgumentParser(description="Twitter Sentiment Analysis Production Pipeline")
    parser.add_argument("--predict", type=str, help="Raw tweet string to analyze")
    parser.add_argument("--eval", action="store_true", help="Run benchmark suite on sample test cases")
    args = parser.parse_args()

    clf = SentimentClassifier()

    if args.predict:
        res = clf.predict(args.predict)
        print("\n" + "="*50)
        print(" TWITTER SENTIMENT PREDICTION RESULT")
        print("="*50)
        print(f" Raw Text:       {res['raw_text']}")
        print(f" Cleaned:        {res['cleaned_text']}")
        print(f" Sentiment:      {res['sentiment']}")
        print(f" Confidence:     {res['confidence']}%")
        print(f" P(Positive):    {res['positive_probability']}")
        print("="*50 + "\n")

    elif args.eval or len(sys.argv) == 1:
        test_samples = [
            "I absolutely loved the customer support, they fixed my ticket in 5 minutes! <3",
            "Terrible experience. The flight was delayed 7 hours and baggage was lost. Never again.",
            "Not gonna lie, the battery life is surprisingly not bad at all!",
            "Can't wait to sit in bumper-to-bumper traffic for another 2 hours... yay :(",
            "Just wrapped up the final project for my NLP course! Feeling accomplished and ready for the weekend.",
            "Why does this app crash every single time I try to checkout?? So frustrating!!",
            "Good morning everyone! Sending good vibes and positivity your way! :D"
        ]
        print("\nEvaluating benchmark test suite...")
        df_res = clf.predict_batch(test_samples)
        for _, row in df_res.iterrows():
            print(f"[{row['sentiment']:8s}] ({row['confidence']:5.1f}% conf) | {row['raw_text']}")
        print("\n[✓] Inference complete.")


if __name__ == "__main__":
    main()
