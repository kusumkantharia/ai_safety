import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from src.preprocessing.preprocess_jigsaw import load_and_preprocess

def train_and_evaluate():
    # Load preprocessed data
    X_train, X_val, y_train, y_val, label_cols = load_and_preprocess()

    # Create pipeline
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=300)))
    ])

    # ✅ Train pipeline
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    print("📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_cols))

    # ✅ Save fitted pipeline
    joblib.dump(clf, "models/jigsaw_pipeline.pkl")
    print("✅ Abuse detection pipeline saved!")

if __name__ == "__main__":
    train_and_evaluate()
