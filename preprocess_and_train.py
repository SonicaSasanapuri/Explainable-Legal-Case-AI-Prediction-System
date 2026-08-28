"""
STEP 1 — Preprocess + Train + Build FAISS Index
Improved preprocessing for better prediction accuracy.

Run on Google Colab with GPU for faster training.
Usage: python preprocess_and_train.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import faiss
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import normalize

import xgboost as xgb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
DATA_PATH       = "data/final_dataset.csv"
MODELS_DIR      = "models"
EMBEDDING_MODEL = "law-ai/InLegalBERT"
TFIDF_MAX_FEAT  = 20000   # increased from 15k for better coverage
BATCH_SIZE      = 32

os.makedirs(MODELS_DIR, exist_ok=True)

# Legal stopwords — very common words that don't help prediction
LEGAL_STOPWORDS = {
    "the", "a", "an", "in", "of", "to", "and", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "that", "this", "these", "those", "it", "its", "on", "at", "by",
    "for", "with", "as", "from", "or", "but", "not", "no", "so",
    "if", "then", "than", "when", "where", "which", "who", "whom",
    "what", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "also", "said", "one", "two", "three",
}


# ─────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────
def load_data():
    print("=" * 60)
    print("  STEP 1: Loading data")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df = df.convert_dtypes(dtype_backend="numpy_nullable")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    print(f"  Total cases    : {len(df):,}")
    print(f"  Label dist     : {Counter(df['label'])}")
    accepted_pct = (df["label"] == 1).mean() * 100
    print(f"  Accepted %     : {accepted_pct:.1f}%")
    return df


# ─────────────────────────────────────────────────────
# 2. IMPROVED TEXT CLEANING
# ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove IndianKanoon URL artifacts
    text = re.sub(r"indiankanoon\.org\S*", "", text)

    # Remove citation numbers like (2014) 4 SCC 769
    text = re.sub(r"\(\d{4}\)\s+\d+\s+\w+\s+\d+", "", text)

    # Remove dates
    text = re.sub(r"\b\d{1,2}\s+\w+\s+\d{4}\b", "", text)

    # Remove pure numbers
    text = re.sub(r"\b\d+\b", "", text)

    # Remove special characters but keep legal punctuation
    text = re.sub(r"[^\w\s\.\,\;\:\(\)\-\/]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def extract_legal_content(text: str) -> str:
    """
    Extract the most legally relevant parts of long case texts.
    Focus on: facts, arguments, held/order sections.
    """
    text_lower = text.lower()

    # Try to find key legal sections
    sections   = []
    markers    = [
        "facts", "held", "order", "judgment", "decision",
        "argued", "submitted", "contended", "appeal",
        "petition", "respondent", "appellant"
    ]

    lines = text.split(".")
    for line in lines:
        line_lower = line.lower()
        if any(marker in line_lower for marker in markers):
            sections.append(line.strip())

    # If we found sections, use them; otherwise use full text
    if len(sections) >= 5:
        return ". ".join(sections[:50])  # top 50 relevant sentences
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  STEP 2: Preprocessing text...")

    df = df.copy()

    # Extract legal content first, then clean
    print("  Extracting legal content...")
    df["legal_text"] = df["text"].apply(extract_legal_content)

    print("  Cleaning text...")
    df["clean_text"] = df["legal_text"].apply(clean_text)

    # Remove too-short texts
    df = df[df["clean_text"].str.len() > 200].reset_index(drop=True)

    print(f"  After cleaning : {len(df):,} cases")
    print(f"  Avg text length: {df['clean_text'].str.len().mean():.0f} chars")

    return df


# ─────────────────────────────────────────────────────
# 3. TRAIN TF-IDF + XGBOOST
# ─────────────────────────────────────────────────────
def train_prediction_model(df: pd.DataFrame):
    print("\n  STEP 3: Training TF-IDF + XGBoost...")

    X = df["clean_text"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Improved TF-IDF — bigrams + trigrams, legal stopwords removed
    print("  Fitting TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features   = TFIDF_MAX_FEAT,
        ngram_range    = (1, 3),        # unigrams, bigrams, trigrams
        min_df         = 3,
        max_df         = 0.85,
        stop_words     = list(LEGAL_STOPWORDS),
        sublinear_tf   = True,          # log normalization
        analyzer       = "word",
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # Balanced class weights
    n_neg            = int(np.sum(y_train == 0))
    n_pos            = int(np.sum(y_train == 1))
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  Rejected: {n_neg:,} | Accepted: {n_pos:,}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # XGBoost — tuned for legal text classification
    print("  Training XGBoost (this takes ~5 mins on Colab GPU)...")
    model = xgb.XGBClassifier(
        n_estimators      = 500,        # more trees for better accuracy
        max_depth         = 7,
        learning_rate     = 0.05,       # lower = more careful learning
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        min_child_weight  = 3,          # avoids overfitting
        gamma             = 0.1,        # regularization
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
        early_stopping_rounds = 30,     # stop if no improvement
    )

    model.fit(
        X_train_vec, y_train,
        eval_set    = [(X_test_vec, y_test)],
        verbose     = 50,
    )

    # Evaluate
    y_pred      = model.predict(X_test_vec)
    y_proba     = model.predict_proba(X_test_vec)[:, 1]
    acc         = accuracy_score(y_test, y_pred)

    # Also test with calibrated threshold (0.38)
    y_pred_cal  = (y_proba >= 0.38).astype(int)
    acc_cal     = accuracy_score(y_test, y_pred_cal)

    print(f"\n  Default threshold accuracy : {acc*100:.2f}%")
    print(f"  Calibrated (0.38) accuracy : {acc_cal*100:.2f}%")
    print("\n  Report (calibrated threshold):")
    print(classification_report(y_test, y_pred_cal,
          target_names=["Rejected", "Accepted"]))

    # Save
    with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(f"{MODELS_DIR}/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("  Saved: tfidf_vectorizer.pkl + xgboost_model.pkl")
    return tfidf, model


# ─────────────────────────────────────────────────────
# 4. BUILD FAISS INDEX
# ─────────────────────────────────────────────────────
def build_faiss_index(df: pd.DataFrame):
    print("\n  STEP 4: Building FAISS index with InLegalBERT embeddings...")
    print(f"  Model: {EMBEDDING_MODEL}")

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Use clean_text for embeddings too
    texts = [t[:1500] for t in df["clean_text"].astype(str).tolist()]

    print(f"  Encoding {len(texts):,} cases (batch={BATCH_SIZE})...")
    embeddings = embedder.encode(
        texts,
        batch_size       = BATCH_SIZE,
        show_progress_bar= True,
        convert_to_numpy = True,
    )

    embeddings = normalize(embeddings, norm="l2")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    print(f"  FAISS index: {index.ntotal:,} vectors, dim={dim}")

    faiss.write_index(index, f"{MODELS_DIR}/faiss_index.bin")

    # Store original text for display (not clean_text)
    case_store = df[["text", "label"]].copy()
    case_store["clean_text"] = df["clean_text"]
    case_store = case_store.reset_index(drop=True)

    with open(f"{MODELS_DIR}/case_store.pkl", "wb") as f:
        pickle.dump(case_store, f)

    print("  Saved: faiss_index.bin + case_store.pkl")
    return index, case_store


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Explainable Legal Case Outcome Prediction")
    print("  Improved Training Pipeline v2")
    print("=" * 60 + "\n")

    df                = load_data()
    df                = preprocess(df)
    tfidf, model      = train_prediction_model(df)
    index, case_store = build_faiss_index(df)

    print("\n" + "=" * 60)
    print("  ALL DONE! Models saved to models/")
    print("  Next: .venv\\Scripts\\python -m uvicorn backend.main:app --reload --port 8000")
    print("=" * 60 + "\n")
