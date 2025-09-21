import joblib
import random
from colorama import Fore, Style, init

# Initialize colorama for colored console output
init(autoreset=True)

# ----------------------------
# Load trained models
# ----------------------------
abuse_model = joblib.load("models/jigsaw_baseline_model.pkl")
abuse_vectorizer = joblib.load("models/jigsaw_tfidf_vectorizer.pkl")
escalation_model = joblib.load("models/dailydialog_emotion_model.pkl")
crisis_model = joblib.load("models/crisis_model.pkl")
content_filter_model = joblib.load("models/content_filter_model.pkl")

# Label sets
abuse_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
emotion_labels = ["Neutral", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]

# ----------------------------
# Moderation functions
# ----------------------------
def check_abuse(text: str):
    X = abuse_vectorizer.transform([text])
    pred = abuse_model.predict(X)[0]
    if 1 in pred:
        return True, [l for l, p in zip(abuse_labels, pred) if p == 1]
    return False, []

def check_escalation(text: str):
    pred = escalation_model.predict([text])[0]
    return emotion_labels[pred]

def check_crisis(text: str):
    pred = crisis_model.predict([text])[0]
    return "SUI Risk" if pred == 1 else "Not Crisis"

def check_content_filter(text: str, age_group="kid"):
    pred = content_filter_model.predict([text])[0]
    status = "Unsafe" if pred == 1 else "Safe"

    if status == "Safe":
        decision = "✅ Allowed"
    else:
        if age_group == "kid":
            decision = "❌ Blocked (Kids)"
        elif age_group == "teen":
            decision = "⚠️ Warning (Teens)"
        else:
            decision = "✅ Allowed (Adults)"
    return status, decision

# ----------------------------
# Combine results
# ----------------------------
def moderate_message(text: str, age_group="kid"):
    results = {}

    abuse_flag, abuse_types = check_abuse(text)
    results["abuse"] = {"flagged": abuse_flag, "categories": abuse_types}
    results["emotion"] = check_escalation(text)
    results["crisis"] = check_crisis(text)

    cf_status, cf_decision = check_content_filter(text, age_group)
    results["content_filter"] = {"status": cf_status, "decision": cf_decision}

    if results["crisis"] == "SUI Risk":
        results["action"] = "🚨 Escalate to human moderator (SUI risk detected)"
    elif results["abuse"]["flagged"]:
        results["action"] = "⚠️ Abuse detected → Warn/Block depending on policy"
    elif results["content_filter"]["status"] == "Unsafe" and age_group == "kid":
        results["action"] = "❌ Block (unsafe for kids)"
    else:
        results["action"] = "✅ Allow message"

    return results

# ----------------------------
# Bot reply logic
# ----------------------------
def generate_response(user_msg: str, moderation: dict, age_group: str):
    if moderation["crisis"] == "SUI Risk":
        return f"{Fore.RED}🚨 I'm really concerned about what you said. A moderator will help you soon."

    if moderation["abuse"]["flagged"]:
        return f"{Fore.YELLOW}⚠️ Please avoid abusive language. Let's keep the chat respectful."

    if moderation["content_filter"]["status"] == "Unsafe":
        if age_group == "kid":
            return f"{Fore.RED}❌ This content is blocked for kids."
        elif age_group == "teen":
            return f"{Fore.YELLOW}⚠️ Warning: This may not be appropriate."
        else:
            return f"{Fore.GREEN}✅ Allowed, but flagged as unsafe."

    safe_replies = [
        "That's interesting! Tell me more.",
        "I hear you 👍",
        "Cool! What else is on your mind?",
        "Got it 🙂",
        "Thanks for sharing!"
    ]
    return random.choice(safe_replies)

# ----------------------------
# Run chatbot
# ----------------------------
if __name__ == "__main__":
    print("\n🤖 AI Safety Chatbot (type 'exit' to quit)")
    age_group = input("Enter user profile (kid/teen/adult): ").strip().lower()

    while True:
        msg = input(Fore.CYAN + "\nYou >> " + Style.RESET_ALL)
        if msg.lower() in ["exit", "quit"]:
            print(Fore.MAGENTA + "\n👋 Chat ended.\n")
            break

        moderation = moderate_message(msg, age_group)
        bot_reply = generate_response(msg, moderation, age_group)

        print(Fore.GREEN + f"Bot >> {bot_reply}")
        print(Fore.LIGHTBLACK_EX + f"[Debug: {moderation}]")
