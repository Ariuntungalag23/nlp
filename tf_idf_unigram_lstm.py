import ast
import os
import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ======================
# CONFIG
# ======================
METHOD = "unigram"   # "unigram" эсвэл "ngram"

DATA_CSV = "/Users/ariuntungalag/Desktop/Natural_Lan/data/imdb_preprocessed.csv"
OUTPUT_CSV = f"/Users/ariuntungalag/Desktop/Natural_Lan/src/experiments/tfidf_{METHOD}_lstm_20_results.csv"
LOG_FILE = f"logs/tfidf_{METHOD}_lstm.log"

# ======================
# CREATE DIRS
# ======================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ======================
# LOGGING
# ======================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ======================
# LOAD DATA
# ======================
logger.info("Loading data")
df = pd.read_csv(DATA_CSV)
df["tokens"] = df["tokens"].apply(ast.literal_eval)

df = df.sample(15000, random_state=42)

texts = df["tokens"].apply(lambda x: " ".join(x))
y = df["label"].values

# ======================
# TF-IDF
# ======================
logger.info(f"Building TF-IDF features: {METHOD}")
if METHOD == "unigram":
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=20000,
        norm="l2"
    )
else:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        norm="l2"
    )

X = vectorizer.fit_transform(texts).toarray()

# ======================
# LSTM PREP
# ======================
X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))

lstm_units = [
    4, 8, 12, 16, 20,
    24, 28, 32, 36, 40,
    44, 48, 52, 56, 60,
    64, 72, 80, 96, 128
]

results = []

# ======================
# LSTM TRAINING
# ======================
for units in lstm_units:
    logger.info(f"Training LSTM: units={units}")

    model = Sequential([
        LSTM(units, input_shape=(1, X.shape[1])),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy"
    )

    model.fit(
        X_lstm, y,
        epochs=2,
        batch_size=64,
        verbose=0
    )

    # ======================
    # PREDICTION + METRICS
    # ======================
    y_prob = model.predict(X_lstm, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    logger.info(
        f"LSTM units={units} | "
        f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}"
    )

    results.append({
        "embedding": f"TFIDF_{METHOD.upper()}",
        "model": "LSTM",
        "params": f"units={units}, epochs=2",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

# ======================
# SAVE RESULTS
# ======================
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
logger.info("LSTM-only experiments finished")

print("LSTM-only experiments finished!")
print("Saved to:", OUTPUT_CSV)