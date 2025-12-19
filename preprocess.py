import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

def preprocess_and_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)       
    text = re.sub(r"[^a-z\s]", "", text)    
    tokens = text.split()                  
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return tokens

def build_csv(dataset_path: str):
    rows = []

    for split in ["train", "test"]:
        for label_name in ["pos", "neg"]:
            label = 1 if label_name == "pos" else 0
            folder = os.path.join(dataset_path, split, label_name)

            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    tokens = preprocess_and_tokenize(f.read())
                    rows.append((tokens, label))

    return pd.DataFrame(rows, columns=["tokens", "label"])


if __name__ == "__main__":
    DATASET_DIR = "aclImdb"
    OUTPUT_CSV = "imdb_preprocessed.csv"

    print("Preprocessing + tokenizing...")
    df = build_csv(DATASET_DIR)

    df.to_csv(OUTPUT_CSV, index=False)

    print("Done!")
    print("Samples:", len(df))