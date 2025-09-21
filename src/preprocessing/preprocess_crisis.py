import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Basic cleaning: lowercase, remove URLs, punctuation, numbers, etc."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess(path="data/raw/crisis/suicide_data.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    if "text" not in df.columns or "class" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'class' columns")

    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    X = df["clean_text"]
    y = df["class"].apply(lambda x: 1 if x.strip().lower() == "suicide" else 0)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
