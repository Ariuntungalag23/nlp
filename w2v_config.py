# ===============================
# WORD2VEC CONFIG
# ===============================

# ---- CBOW ----
W2V_CBOW = {
    "vector_size": 100,
    "window": 5,
    "min_count": 5,
    "sg": 0,          # 0 = CBOW
    "workers": 4
}

# ---- Skip-gram ----
W2V_SKIPGRAM = {
    "vector_size": 100,
    "window": 5,
    "min_count": 5,
    "sg": 1,          # 1 = Skip-gram
    "workers": 4
}

import os

# ===== PROJECT ROOT =====
BASE_DIR = "/Users/ariuntungalag/Desktop/Natural_Lan"

# ===== INPUT DATA =====
INPUT_CSV = os.path.join(
    BASE_DIR,
    "data",
    "imdb_preprocessed.csv"
)

# ===== OUTPUT =====
OUTPUT_DIR = os.path.join(BASE_DIR, "data")