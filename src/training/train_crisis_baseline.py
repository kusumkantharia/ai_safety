import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from src.preprocessing.preprocess_crisis import load_and_preprocess

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Pipeline: TF-IDF + Logistic Regression
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Suicide", "Suicide"]))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(clf, "models/crisis_model.pkl")
    print("✅ Crisis model saved to models/crisis_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
