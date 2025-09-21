import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from src.preprocessing.preprocess_dailydialog import load_and_preprocess

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=300, class_weight="balanced", multi_class="ovr"))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("📊 Classification Report (Multi-class Emotions)")
    print(classification_report(
        y_test, y_pred, 
        target_names=["Neutral","Anger","Disgust","Fear","Joy","Sadness","Surprise"]
    ))

    joblib.dump(clf, "models/dailydialog_emotion_model.pkl")
    print("✅ Multi-class emotion model saved!")

if __name__ == "__main__":
    train_and_evaluate()
