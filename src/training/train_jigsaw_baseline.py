import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
#from preprocessing.preprocess_jigsaw import load_and_preprocess
from src.preprocessing.preprocess_jigsaw import load_and_preprocess



def train_baseline():
    # Load data
    X_train, X_val, y_train, y_val, label_cols = load_and_preprocess()

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # Train model (multi-label with OneVsRest)
    model = OneVsRestClassifier(LogisticRegression(max_iter=300))
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_val_vec)
    print("📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_cols))

    # Save model + vectorizer
    joblib.dump(model, "models/jigsaw_baseline_model.pkl")
    joblib.dump(vectorizer, "models/jigsaw_tfidf_vectorizer.pkl")

    print("✅ Model and vectorizer saved!")

if __name__ == "__main__":
    train_baseline()
