🤖 AI Safety Models Proof of Concept (POC)

This repository contains the implementation of an AI Safety pipeline designed to enhance user safety in conversational AI platforms (e.g., chatbots, messaging systems, or social media).

The system integrates four machine learning models to handle harmful content in real-time:

Abuse Language Detection

Escalation Pattern Recognition

Crisis Intervention

Content Filtering (Age-Appropriate Moderation)

📌 Project Overview

The goal of this project is to build a Proof of Concept (POC) that demonstrates:

Detecting abusive or harmful language in real-time.

Recognizing escalation patterns in conversations.

Identifying crisis situations (self-harm/suicidal ideation).

Filtering content based on age profiles (kid, teen, adult).

The POC integrates these models into a single moderation pipeline that can be used in a chat simulation (CLI) or a web-based Streamlit app.

📂 Repository Structure

ai_safety_poc/
│── data/                           # Datasets (raw and processed)
│   ├── raw/
│   │   ├── jigsaw_toxic/           # Jigsaw dataset for abuse detection
│   │   ├── escalation/             # DailyDialog for escalation recognition
│   │   ├── crisis/                 # Suicide detection dataset
│   │   └── content_filter/         # Safe vs Unsafe dataset
│── models/                         # Saved trained models (.pkl files)
│   ├── jigsaw_pipeline.pkl
│   ├── dailydialog_emotion_model.pkl
│   ├── crisis_model.pkl
│   └── content_filter_model.pkl
│── src/                            # Source code
│   ├── preprocessing/              # Preprocessing scripts
│   │   ├── preprocess_jigsaw.py
│   │   ├── preprocess_dailydialog.py
│   │   ├── preprocess_crisis.py
│   │   └── preprocess_content.py
│   ├── training/                   # Model training scripts
│   │   ├── train1_jigsaw_baseline.py
│   │   ├── train_dailydialog_baseline.py
│   │   ├── train_crisis_baseline.py
│   │   └── train_content_baseline.py
│   └── inference/                  # Inference / chatbot integration
│       ├── chat_pipeline.py        # Unified moderation pipeline
│       └── chatbot_app.py          # Streamlit chatbot app
│── requirements.txt                # Python dependencies
│── README.md                       # Project documentation


⚙️ Setup & Installation

1. Clone the repository
git clone <your_repo_url>
cd ai_safety_poc

2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

📊 Datasets Used
Model	Dataset Used	Source
Abuse Detection	Jigsaw Toxic Comment Classification	Kaggle
Escalation Recognition	DailyDialog (emotion labels)	HuggingFace
Crisis Intervention	Suicide Detection Dataset	Kaggle / Reddit-based
Content Filtering	Jigsaw (repurposed for Safe/Unsafe)	Kaggle

⚠️ Note: All datasets are publicly available and anonymized.

🚀 Training the Models

Train each model individually:

1. Abuse Detection
python -m src.training.train1_jigsaw_baseline

2. Escalation Recognition
python -m src.training.train_dailydialog_baseline

3. Crisis Intervention
python -m src.training.train_crisis_baseline

4. Content Filtering
python -m src.training.train_content_baseline


Trained models are stored in the models/ folder.

🔎 Running the Moderation Pipeline

Run the CLI chat pipeline:

python -m src.inference.chat_pipeline


Example:

AI Safety Chat Pipeline (type 'exit' to quit)
Enter user profile (kid/teen/adult): kid

User >> You are an idiot
--- Moderation Results ---
Abuse: toxic, insult
Emotion: Neutral
Crisis: Not Crisis
Content Filter: Unsafe for kids
Final Action: ⚠️ Block message

💬 Running the Chatbot (Web UI)

Start the Streamlit chatbot app:

streamlit run src/inference/chatbot_app.py


Open in browser: http://localhost:8501

Select user profile (kid/teen/adult).

Enter chat messages and see moderation in action.

📈 Model Performance
Model	Metric	Score
Abuse Detection	Weighted F1	~0.66
Escalation Recognition	Macro F1	~0.23 (imbalanced)
Crisis Intervention	Accuracy	~0.90
Content Filtering	Accuracy	~0.92
🏗️ High-Level Architecture
User Input → Preprocessing → 4 Models
     │
     ├─ Abuse Detector → toxic/insult/etc.
     ├─ Escalation → anger/fear/joy/etc.
     ├─ Crisis → sui / non-sui
     ├─ Content Filter → safe / unsafe
     ↓
Decision Engine → Final Action
     (Allow / Warn / Block / Escalate)

⚖️ Ethical Considerations

Bias Mitigation: Models trained on public datasets, but may underperform on slang/dialects.

Privacy: No personal/private user data was used.

Explainability: Chose Logistic Regression + TF-IDF for transparency.

🔮 Future Improvements

Upgrade models to BERT/RoBERTa for higher accuracy.

Add multilingual support (non-English text).

Deploy as a REST API or integrate with real chat apps.

Continuous monitoring for fairness and bias.

👨‍💻 Author

Machine Learning Candidate - AI Safety POC

Built for demonstrating end-to-end ML solution design, training, evaluation, and integration.

📌 Deliverables in this Repo:

✅ Source code for preprocessing, training, inference.

✅ Trained models (.pkl).

✅ README.md (this document).

✅ Scripts for evaluation & reports.