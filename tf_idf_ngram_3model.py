import ast
import logging
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    filename="logs/tfidf_ngram_3models.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DATA_CSV = "/Users/ariuntungalag/Desktop/Natural_Lan/data/imdb_preprocessed.csv"
OUTPUT_CSV = "/Users/ariuntungalag/Desktop/Natural_Lan/src/experiments/tfidf_ngram_60_3ev_results.csv"

logger.info("Loading data")
df = pd.read_csv(DATA_CSV)
df["tokens"] = df["tokens"].apply(ast.literal_eval)

df = df.sample(15000, random_state=42)

texts = df["tokens"].apply(lambda x: " ".join(x))
y = df["label"].values

logger.info("Building TF-IDF N-gram features")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000,
    min_df=2,
    norm="l2"
)
X = vectorizer.fit_transform(texts)

results = []

logreg_C = [
    0.0001, 0.001, 0.005, 0.01, 0.05,
    0.1, 0.5, 1, 2, 5,
    10, 20, 50, 100, 200,
    300, 500, 700, 900, 1000
]

rf_params = [
    (50,5),(50,10),(50,15),(50,20),
    (100,5),(100,10),(100,15),(100,20),
    (150,5),(150,10),(150,15),(150,20),
    (200,5),(200,10),(200,15),(200,20),
    (300,5),(300,10),(300,15),(300,20)
]

ada_params = [
    (50,0.01),(50,0.05),(50,0.1),(50,0.5),
    (100,0.01),(100,0.05),(100,0.1),(100,0.5),
    (150,0.01),(150,0.05),(150,0.1),(150,0.5),
    (200,0.01),(200,0.05),(200,0.1),(200,0.5),
    (300,0.01),(300,0.05),(300,0.1),(300,0.5)
]

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

for C in logreg_C:
    logger.info(f"LogReg training: C={C}")
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(C=C, max_iter=300))
    ])

    scores = cross_validate(model, X, y, cv=3, scoring=scoring)

    results.append({
        "embedding": "TFIDF_NGRAM",
        "model": "LogisticRegression",
        "params": f"C={C}",
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean()
    })

for n_est, depth in rf_params:
    logger.info(f"RandomForest: n_estimators={n_est}, depth={depth}")
    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        n_jobs=-1
    )

    scores = cross_validate(model, X, y, cv=3, scoring=scoring)

    results.append({
        "embedding": "TFIDF_NGRAM",
        "model": "RandomForest",
        "params": f"n_estimators={n_est}, depth={depth}",
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean()
    })

for n_est, lr in ada_params:
    logger.info(f"AdaBoost: n_estimators={n_est}, lr={lr}")
    model = AdaBoostClassifier(
        n_estimators=n_est,
        learning_rate=lr
    )

    scores = cross_validate(model, X, y, cv=3, scoring=scoring)

    results.append({
        "embedding": "TFIDF_NGRAM",
        "model": "AdaBoost",
        "params": f"n_estimators={n_est}, lr={lr}",
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean()
    })

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
logger.info("TF-IDF NGRAM (3 models) finished")
print(" TF-IDF NGRAM (3 models) finished")