import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_train(path="data/raw/content_filter/train.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    toxic_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # Binary: any toxic column = 1 → Unsafe
    df["is_unsafe"] = df[toxic_labels].max(axis=1)

    X = df["comment_text"].astype(str)
    y = df["is_unsafe"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def load_and_preprocess_test(test_path="data/raw/content_filter/test.csv",
                             labels_path="data/raw/content_filter/test_labels.csv"):
    df_test = pd.read_csv(test_path)
    df_labels = pd.read_csv(labels_path)

    df = df_test.merge(df_labels, on="id")

    toxic_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # 🚨 drop rows with -1 (not annotated)
    df = df[df[toxic_labels].min(axis=1) >= 0]

    df["is_unsafe"] = df[toxic_labels].max(axis=1)

    X = df["comment_text"].astype(str)
    y = df["is_unsafe"]

    return X, y
