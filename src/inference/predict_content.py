import joblib

# --------------------------
# Load Model
# --------------------------
def load_model(model_path="models/content_filter_model.pkl"):
    return joblib.load(model_path)

# --------------------------
# Predict content safety
# --------------------------
def predict_single(text, clf):
    pred = clf.predict([text])[0]
    return "Unsafe" if pred == 1 else "Safe"

# --------------------------
# Apply age policy
# --------------------------
def apply_age_policy(prediction, age_group):
    """
    prediction: "Safe" or "Unsafe"
    age_group: "kid", "teen", "adult"
    """
    if prediction == "Safe":
        return "✅ Allowed (Safe for all)"

    if prediction == "Unsafe":
        if age_group == "kid":
            return "❌ Blocked (Unsafe for Kids)"
        elif age_group == "teen":
            return "⚠️ Warning: Unsafe Content (Teens supervised)"
        elif age_group == "adult":
            return "✅ Allowed (Unsafe but permitted for Adults)"
    
    return prediction

# --------------------------
# Interactive usage
# --------------------------
if __name__ == "__main__":
    clf = load_model()

    print("\n💬 Type a message to check if it's safe (or 'exit').")
    print("   Choose profile: kid / teen / adult\n")

    while True:
        age_group = input("Age group (kid/teen/adult): ").strip().lower()
        if age_group not in ["kid", "teen", "adult"]:
            print("⚠️ Please enter kid / teen / adult")
            continue

        user_inp = input("Message >> ")
        if user_inp.lower() in ["exit", "quit"]:
            break

        pred = predict_single(user_inp, clf)
        decision = apply_age_policy(pred, age_group)
        print(f"Prediction: {pred} → {decision}\n")
