import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from src.preprocessing.preprocess_content import load_and_preprocess_train, load_and_preprocess_test

def train_and_evaluate():
    # Train/validation split
    X_train, X_val, y_train, y_val = load_and_preprocess_train()

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

    clf.fit(X_train, y_train)

    # Validation report
    y_val_pred = clf.predict(X_val)
    print("\n📊 Validation Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Safe", "Unsafe"]))
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

    # Kaggle test set
    X_test, y_test = load_and_preprocess_test()
    y_test_pred = clf.predict(X_test)
    print("\n📊 Kaggle Test Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Safe", "Unsafe"]))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

    # Save
    joblib.dump(clf, "models/content_filter_model.pkl")
    print("\n✅ Content filter model saved to models/content_filter_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
