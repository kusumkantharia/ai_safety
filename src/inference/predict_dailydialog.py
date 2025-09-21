import pandas as pd
import joblib
import ast
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

# ---------------------------
# Prepare data for inference
# ---------------------------
def prepare_data(path):
    df = pd.read_csv(path)

    # Convert string of dialogues to plain text
    def clean_dialog(dialog):
        try:
            utterances = ast.literal_eval(dialog)
            return " ".join(utterances)
        except Exception:
            return str(dialog)

    df["dialog"] = df["dialog"].apply(clean_dialog)

    # Extract labels
    def extract_emotion(row):
        try:
            emotions = ast.literal_eval(row["emotion"])
            if isinstance(emotions, list) and len(emotions) > 0:
                return emotions[-1]  # last utterance emotion
            return 0
        except Exception:
            return 0

    df["label"] = df.apply(extract_emotion, axis=1)

    return df["dialog"], df["label"]


# ---------------------------
# Predict & Evaluate
# ---------------------------
def predict_and_evaluate(model_path="models/dailydialog_emotion_model.pkl",
                         test_path="data/raw/escalation/test.csv"):
    # Load model
    clf = joblib.load(model_path)

    # Load test set
    X_test, y_test = prepare_data(test_path)

    # Predict
    y_pred = clf.predict(X_test)

    # Detect labels present
    labels_present = unique_labels(y_test, y_pred)
    emotion_names = ["Neutral", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
    target_names = [emotion_names[i] for i in labels_present]

    # Report
    print("\n📊 Evaluation on:", test_path)
    print(classification_report(
        y_test, y_pred,
        labels=labels_present,
        target_names=target_names,
        zero_division=0
    ))


# ---------------------------
# Predict a single dialogue
# ---------------------------
def predict_single(dialog, model_path="models/dailydialog_emotion_model.pkl"):
    clf = joblib.load(model_path)
    emotion_names = ["Neutral", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]

    pred = clf.predict([dialog])[0]
    return emotion_names[pred]


# ---------------------------
# Run script
# ---------------------------
if __name__ == "__main__":
    # Run evaluation on test + validation
    predict_and_evaluate(test_path="data/raw/escalation/test.csv")
    predict_and_evaluate(test_path="data/raw/escalation/validation.csv")

    # Interactive predictions
    print("\n💬 Type a dialogue to predict emotion (or 'exit' to quit):")
    while True:
        user_inp = input(">> ")
        if user_inp.lower() in ["exit", "quit"]:
            break
        emotion = predict_single(user_inp)
        print(f"Predicted emotion: {emotion}")
