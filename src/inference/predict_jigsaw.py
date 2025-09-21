import joblib

# Load model + vectorizer
model = joblib.load("models/jigsaw_baseline_model.pkl")
vectorizer = joblib.load("models/jigsaw_tfidf_vectorizer.pkl")

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict(text):
    X = vectorizer.transform([text])
    preds = model.predict(X)[0]
    return {label: bool(pred) for label, pred in zip(label_cols, preds)}

if __name__ == "__main__":
    test_text = "I hate you, you are the worst!"
    print("🔎 Input:", test_text)
    print("🚦 Prediction:", predict(test_text))
