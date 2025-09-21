import joblib
import pandas as pd
from sklearn.metrics import classification_report

def prepare_data(path):
    df = pd.read_csv(path)
    if "text" not in df.columns or "class" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'class' columns")

    X = df["text"].astype(str)
    y = df["class"].apply(lambda x: 1 if x.strip().lower() == "suicide" else 0)
    return X, y

def predict_and_evaluate(model_path="models/crisis_model.pkl", test_path="data/raw/crisis/suicide_data.csv"):
    clf = joblib.load(model_path)
    X, y = prepare_data(test_path)
    y_pred = clf.predict(X)

    print("📊 Evaluation on:", test_path)
    print(classification_report(y, y_pred, target_names=["Non-Suicide", "Suicide"]))

def predict_single(text, model_path="models/crisis_model.pkl"):
    clf = joblib.load(model_path)
    pred = clf.predict([text])[0]
    return "Suicide" if pred == 1 else "Non-Suicide"

if __name__ == "__main__":
    # Evaluate on full dataset
    predict_and_evaluate()

    # Interactive predictions
    print("\n💬 Type a message to check if it's a crisis (or 'exit' to quit):")
    while True:
        user_inp = input(">> ")
        if user_inp.lower() in ["exit", "quit"]:
            break
        print("Prediction:", predict_single(user_inp))
