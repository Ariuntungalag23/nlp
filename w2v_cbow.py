import ast
import os
import logging
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ======================
# PATHS
# ======================
DATA_CSV = "/Users/ariuntungalag/Desktop/Natural_Lan/data/imdb_preprocessed.csv"
W2V_MODEL = "/Users/ariuntungalag/Desktop/Natural_Lan/data/w2v_cbow.model"
OUTPUT_CSV = "/Users/ariuntungalag/Desktop/Natural_Lan/src/experiments/w2v_cbow.csv"
LOG_FILE = "logs/w2v_cbow.log"

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
y = df["label"].values

# ======================
# LOAD WORD2VEC
# ======================
logger.info("Loading Word2Vec CBOW model")
w2v = Word2Vec.load(W2V_MODEL)
VECTOR_SIZE = w2v.vector_size

def sentence_vector(tokens):
    vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(VECTOR_SIZE)

X = np.array([sentence_vector(t) for t in df["tokens"]])

results = []

# ======================
# SCORING (CLASSICAL)
# ======================
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

# ======================
# 1️⃣ LOGISTIC REGRESSION (20)
# ======================
logreg_C = [
    0.0001, 0.001, 0.005, 0.01, 0.05,
    0.1, 0.5, 1, 2, 5,
    10, 20, 50, 100, 200,
    300, 500, 700, 900, 1000
]

for C in logreg_C:
    logger.info(f"LogisticRegression: C={C}")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=300))
    ])

    scores = cross_validate(model, X, y, cv=3, scoring=scoring)

    results.append({
        "embedding": "W2V_CBOW",
        "model": "LogisticRegression",
        "params": f"C={C}",
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean()
    })

# ======================
# 2️⃣ RANDOM FOREST (20)
# ======================
rf_params = [
    (50,5),(50,10),(50,15),(50,20),
    (100,5),(100,10),(100,15),(100,20),
    (150,5),(150,10),(150,15),(150,20),
    (200,5),(200,10),(200,15),(200,20),
    (300,5),(300,10),(300,15),(300,20)
]

for n_est, depth in rf_params:
    logger.info(f"RandomForest: n_estimators={n_est}, depth={depth}")
    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        n_jobs=-1
    )

    scores = cross_validate(model, X, y, cv=3, scoring=scoring)

    results.append({
        "embedding": "W2V_CBOW",
        "model": "RandomForest",
        "params": f"n_estimators={n_est}, depth={depth}",
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean()
    })

# ======================
# 3️⃣ ADABOOST (20)
# ======================
ada_params = [
    (50,0.01),(50,0.05),(50,0.1),(50,0.5),
    (100,0.01),(100,0.05),(100,0.1),(100,0.5),
    (150,0.01),(150,0.05),(150,0.1),(150,0.5),
    (200,0.01),(200,0.05),(200,0.1),(200,0.5),
    (300,0.01),(300,0.05),(300,0.1),(300,0.5)
]

for n_est, lr in ada_params:
    logger.info(f"AdaBoost: n_estimators={n_est}, lr={lr}")
    model = AdaBoostClassifier(
        n_estimators=n_est,
        learning_rate=lr
    )

    scores = cross_validate(model, X, y, cv=3, scoring=scoring)

    results.append({
        "embedding": "W2V_CBOW",
        "model": "AdaBoost",
        "params": f"n_estimators={n_est}, lr={lr}",
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean()
    })

# ======================
# 4️⃣ LSTM (20) – SEPARATE METRICS
# ======================
X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))

lstm_units = [
    4, 8, 12, 16, 20,
    24, 28, 32, 36, 40,
    44, 48, 52, 56, 60,
    64, 72, 80, 96, 128
]

for units in lstm_units:
    logger.info(f"LSTM: units={units}")
    model = Sequential([
        LSTM(units, input_shape=(1, VECTOR_SIZE)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy"
    )

    model.fit(X_lstm, y, epochs=2, batch_size=64, verbose=0)

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
        "embedding": "W2V_CBOW",
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
logger.info("80 CBOW experiments finished")

print("80 CBOW experiments finished!")
print("Saved to:", OUTPUT_CSV)