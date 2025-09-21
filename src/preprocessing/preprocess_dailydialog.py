import pandas as pd
import ast
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_emotion_string(emotion_str):
    """Fixes format like [0 0 6 0] into [0,0,6,0]."""
    cleaned = emotion_str.strip()
    cleaned = cleaned.replace(" ", ",")
    try:
        emotions = ast.literal_eval(cleaned)
    except Exception as e:
        print("❌ Error parsing:", emotion_str)
        raise e
    return emotions

def load_and_preprocess(path="data/raw/escalation/train.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    texts, labels = [], []

    for _, row in df.iterrows():
        # Parse dialog (string → list)
        dialog = ast.literal_eval(row["dialog"])
        dialog = [clean_text(u) for u in dialog]
        conversation = " ".join(dialog)

        # Parse emotions (string → list[int])
        emotions = parse_emotion_string(row["emotion"])

        # For multi-class, let’s take the **most frequent emotion** in conversation
        if len(emotions) > 0:
            label = max(set(emotions), key=emotions.count)
        else:
            label = 0  # fallback to neutral

        texts.append(conversation)
        labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    return X_train, X_test, y_train, y_test
