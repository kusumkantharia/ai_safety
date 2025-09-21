import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Import preprocess loaders
from src.preprocessing.preprocess_jigsaw import load_and_preprocess as load_jigsaw
from src.preprocessing.preprocess_dailydialog import load_and_preprocess as load_dailydialog
from src.preprocessing.preprocess_crisis import load_and_preprocess as load_crisis
from src.preprocessing.preprocess_content import load_and_preprocess_train, load_and_preprocess_test

REPORT_DIR = "reports/metrics"
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------- Utility Functions ----------
def save_classification_report(y_true, y_pred, labels, title, filename):
    """Save classification report as heatmap (jpg)."""
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.iloc[:-3, :-1], annot=True, cmap="YlGnBu", fmt=".2f", cbar=False)
    plt.title(title)
    plt.ylabel("Classes")
    plt.xlabel("Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), format="jpg")
    plt.close()

def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Save confusion matrix as heatmap (jpg)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), format="jpg")
    plt.close()

# ---------- 1. Abuse Detection ----------
def evaluate_abuse():
    print("\nEvaluating Abuse Detection...")
    model = joblib.load("models/jigsaw_pipeline.pkl")
    _, X_val, _, y_val, label_cols = load_jigsaw()
    y_pred = model.predict(X_val)

    # Multi-label report
    save_classification_report(y_val, y_pred, label_cols,
                               "Abuse Detection - Precision/Recall/F1",
                               "abuse_report.jpg")

    # Per-class confusion matrices
    for i, label in enumerate(label_cols):
        save_confusion_matrix(y_val[:, i], y_pred[:, i],
                              ["Not " + label, label],
                              f"Abuse Detection - {label}",
                              f"abuse_cm_{label}.jpg")

# ---------- 2. Escalation Recognition (DailyDialog emotions) ----------
def evaluate_escalation():
    print("\nEvaluating Escalation Recognition...")
    model = joblib.load("models/dailydialog_emotion_model.pkl")
    X_train, X_test, y_train, y_test = load_dailydialog()

    y_pred = model.predict(X_test)
    emotion_labels = ["Neutral", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]

    save_classification_report(y_test, y_pred, emotion_labels,
                               "Escalation Recognition (Emotions)",
                               "escalation_report.jpg")

    save_confusion_matrix(y_test, y_pred, emotion_labels,
                          "Escalation Recognition (Emotions)",
                          "escalation_cm.jpg")

# ---------- 3. Crisis Detection ----------
def evaluate_crisis():
    print("\nEvaluating Crisis Detection...")
    model = joblib.load("models/crisis_model.pkl")
    X_train, X_test, y_train, y_test = load_crisis()

    y_pred = model.predict(X_test)
    crisis_labels = ["Non-SUICIDE", "SUICIDE"]

    save_classification_report(y_test, y_pred, crisis_labels,
                               "Crisis Detection",
                               "crisis_report.jpg")

    save_confusion_matrix(y_test, y_pred, crisis_labels,
                          "Crisis Detection",
                          "crisis_cm.jpg")

# ---------- 4. Content Filtering ----------
def evaluate_content():
    print("\nEvaluating Content Filtering...")
    model = joblib.load("models/content_filter_model.pkl")
    _, _, _, _ = load_and_preprocess_train()
    X_test, y_test = load_and_preprocess_test()

    y_pred = model.predict(X_test)
    labels = ["Safe", "Unsafe"]

    save_classification_report(y_test, y_pred, labels,
                               "Content Filtering",
                               "content_report.jpg")

    save_confusion_matrix(y_test, y_pred, labels,
                          "Content Filtering",
                          "content_cm.jpg")

# ---------- Run All ----------
if __name__ == "__main__":
    evaluate_abuse()
    evaluate_escalation()
    evaluate_crisis()
    evaluate_content()
    print(f"\n✅ All reports saved in {REPORT_DIR}")
