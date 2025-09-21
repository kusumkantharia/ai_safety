import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text) # keep only letters and spaces
    return text.strip()

def load_and_preprocess(path="data/raw/jigsaw_toxic/train.csv"):
    df = pd.read_csv(path)
    
    # Clean text
    df["clean_text"] = df["comment_text"].astype(str).apply(clean_text)

    # Labels (multi-label: toxic, obscene, insult, identity_hate, etc.)
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    X = df["clean_text"].values
    y = df[label_cols].values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, label_cols
