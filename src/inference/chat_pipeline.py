import joblib
import os
import csv
from colorama import Fore, Style, init

# init colorama for colored console output
init(autoreset=True)

# ----------------------------
# Load models
# ----------------------------
abuse_model = joblib.load("models/jigsaw_baseline_model.pkl")
abuse_vectorizer = joblib.load("models/jigsaw_tfidf_vectorizer.pkl")
escalation_model = joblib.load("models/dailydialog_emotion_model.pkl")
crisis_model = joblib.load("models/crisis_model.pkl")
content_filter_model = joblib.load("models/content_filter_model.pkl")

# Labels
abuse_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
emotion_labels = ["Neutral","Anger","Disgust","Fear","Joy","Sadness","Surprise"]

# ----------------------------
# Predict functions
# ----------------------------
def check_abuse(text):
    X = abuse_vectorizer.transform([text])
    pred = abuse_model.predict(X)[0]
    if 1 in pred:
        detected = [l for l, p in zip(abuse_labels, pred) if p == 1]
        return True, detected
    return False, []

def check_escalation(text):
    pred = escalation_model.predict([text])[0]
    return emotion_labels[pred]

def check_crisis(text):
    pred = crisis_model.predict([text])[0]
    return "Crisis" if pred == 1 else "Not Crisis"

def check_content_filter(text, age_group="kid"):
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
# Chat Moderation Pipeline
# ----------------------------
def moderate_message(text, age_group="kid"):
    results = {}

    abuse_flag, abuse_types = check_abuse(text)
    results["abuse"] = {"flagged": abuse_flag, "categories": abuse_types}

    results["emotion"] = check_escalation(text)
    results["crisis"] = check_crisis(text)

    cf_status, cf_decision = check_content_filter(text, age_group)
    results["content_filter"] = {"status": cf_status, "decision": cf_decision}

    # Final moderation decision
    if results["crisis"] == "Crisis":
        results["action"] = f"{Fore.RED + Style.BRIGHT}🚨 Escalate to human moderator (Crisis detected)"
    elif results["abuse"]["flagged"]:
        results["action"] = f"{Fore.YELLOW}⚠️ Abuse detected → Warn/Block depending on policy"
    elif results["content_filter"]["status"] == "Unsafe" and age_group == "kid":
        results["action"] = f"{Fore.RED}❌ Block (unsafe for kids)"
    else:
        results["action"] = f"{Fore.GREEN}✅ Allow message"

    return results

# ----------------------------
# Save logs
# ----------------------------
def log_message(user_msg, results, log_file="logs/chat_log.csv"):
    os.makedirs("logs", exist_ok=True)
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            user_msg,
            results["abuse"],
            results["emotion"],
            results["crisis"],
            results["content_filter"],
            results["action"]
        ])

# ----------------------------
# Interactive Chat Session
# ----------------------------
if __name__ == "__main__":
    print("\n💬 AI Safety Chat Pipeline (type 'exit' to quit)")
    age_group = input("Enter user profile (kid/teen/adult): ").strip().lower()

    while True:
        msg = input(Fore.CYAN + "\nUser >> " + Style.RESET_ALL)
        if msg.lower() in ["exit", "quit"]:
            print(Fore.MAGENTA + "\n👋 Session ended. Logs saved in logs/chat_log.csv\n")
            break

        output = moderate_message(msg, age_group)

        print("\n--- Moderation Results ---")
        print(f"Abuse: {output['abuse']}")
        print(f"Emotion: {output['emotion']}")
        print(f"Crisis: {output['crisis']}")
        print(f"Content Filter: {output['content_filter']}")
        print(f"Final Action: {output['action']}")
        print("--------------------------")

        # Save to logs
        log_message(msg, output)
