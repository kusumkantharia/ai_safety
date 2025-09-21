import os
import joblib
import streamlit as st

MODEL_DIR = "models"

# --- Load fitted pipelines ---
abuse_model = joblib.load(os.path.join(MODEL_DIR, "jigsaw_pipeline.pkl"))   # ✅ pipeline
emotion_model = joblib.load(os.path.join(MODEL_DIR, "dailydialog_emotion_model.pkl"))  # ✅ pipeline
crisis_model = joblib.load(os.path.join(MODEL_DIR, "crisis_model.pkl"))  # ✅ pipeline
content_model = joblib.load(os.path.join(MODEL_DIR, "content_filter_model.pkl"))  # ✅ pipeline

# --- Moderation functions ---
def check_abuse(text: str):
    """Check for toxic/abusive language"""
    preds = abuse_model.predict([text])[0]
    labels = abuse_model.classes_
    flagged = any(preds)
    categories = [labels[i] for i, p in enumerate(preds) if p == 1]
    return {"flagged": flagged, "categories": categories}

def check_emotion(text: str):
    """Detect user emotion (DailyDialog)"""
    pred = emotion_model.predict([text])[0]
    return emotion_model.classes_[pred]

def check_crisis(text: str):
    """Detect self-harm or crisis intent"""
    pred = crisis_model.predict([text])[0]
    return "Crisis" if pred == 1 else "Not Crisis"

def check_content_filter(text: str, age_group: str):
    """Filter unsafe content by age profile"""
    pred = content_model.predict([text])[0]
    if pred == 1:  # Unsafe
        return {"status": "Unsafe", "decision": f"❌ Blocked ({age_group.capitalize()})"}
    else:
        return {"status": "Safe", "decision": "✅ Allowed"}

def moderate_message(text: str, age_group: str):
    """Run all safety checks and return moderation summary"""
    abuse = check_abuse(text)
    emotion = check_emotion(text)
    crisis = check_crisis(text)
    content = check_content_filter(text, age_group)

    # Final action logic
    if crisis == "Crisis":
        final_action = "🚨 Escalate to human moderator (Crisis detected)"
    elif abuse["flagged"]:
        final_action = "⚠️ Abuse detected → Warn/Block depending on policy"
    elif content["status"] == "Unsafe":
        final_action = "❌ Block due to content filtering"
    else:
        final_action = "✅ Allow message"

    return {
        "abuse": abuse,
        "emotion": emotion,
        "crisis": crisis,
        "content": content,
        "final_action": final_action
    }

# --- Streamlit UI ---
st.title("🤖 AI Safety Chatbot (POC)")

age_group = st.radio("Select user profile:", ["kid", "teen", "adult"])

user_input = st.text_input("User >>")

if user_input:
    moderation = moderate_message(user_input, age_group)

    st.subheader("🔍 Moderation Results")
    st.json(moderation)
