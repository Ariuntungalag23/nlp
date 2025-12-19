# TF-IDF Unigram config
TFIDF_UNI = {
    "max_features": 10000,
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 0.95,
    "norm": "l2"
}

# TF-IDF N-gram config
TFIDF_NGRAM = {
    "max_features": 20000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "norm": "l2"
}

# Paths
INPUT_CSV = "data/imdb_preprocessed.csv"
OUTPUT_DIR = "data/"