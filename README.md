🤖 AI Safety Models – Proof of Concept (POC)
📌 Overview

This project is a Proof of Concept (POC) for AI Safety Models, developed as part of Solulab’s Machine Learning assignment.

The goal is to demonstrate how machine learning can be applied to enhance user safety in chat platforms by detecting and handling unsafe or harmful messages in real time.

The POC integrates four models into a moderation pipeline:

Abuse Language Detection → Flags toxic, obscene, insulting, or threatening text.

Escalation Pattern Recognition → Detects when conversations become emotionally dangerous.

Crisis Intervention → Identifies severe distress or SUI-related messages for human escalation.

Content Filtering → Ensures content is age-appropriate based on user profile (kid/teen/adult).

📂 Project Structure

ai_safety_poc/
│
├── data/
│   └── raw/
│       ├── jigsaw_toxic/              # Abuse dataset (train.csv, test.csv)
│       ├── escalation/                # DailyDialog / EmpatheticDialogues
│       │   ├── train.csv
│       │   ├── test.csv
│       │   └── validation.csv
│       ├── crisis/                    # Crisis/SUI dataset
│       │   └── Suicide_Detection.csv
│       └── content_filter/            # Content filter dataset
│           ├── train.csv
│           ├── test.csv
│           └── test_labels.csv
│
├── models/                            # Trained models (saved as .pkl)
│   ├── jigsaw_pipeline.pkl            # Abuse detection pipeline
│   ├── dailydialog_emotion_model.pkl  # Escalation (emotions)
│   ├── crisis_model.pkl               # Crisis detection
│   └── content_filter_model.pkl       # Content filtering
│   
│
├── src/
│   ├── preprocessing/                 # Preprocessing scripts
│   │   ├── preprocess_jigsaw.py
│   │   ├── preprocess_dailydialog.py
│   │   ├── preprocess_crisis.py
│   │   └── preprocess_content.py
│   │
│   ├── training/                      # Model training scripts
│   │   ├── train1_jigsaw_baseline.py
│   │   ├── train_dailydialog_baseline.py
│   │   ├── train_crisis_baseline.py
│   │   └── train_content_baseline.py
│   │
│   ├── inference/                     # Prediction & chatbot pipeline
│   │   ├── predict_jigsaw.py
│   │   ├── predict_dailydialog.py
│   │   ├── predict_crisis.py
│   │   ├── predict_content.py
│   │   ├── chat_pipeline.py           # Command-line moderation pipeline
│   │   └── chatbot_app.py             # Streamlit chatbot app
│   │
│   └── evaluation/                    # Model evaluation + reports
│       └── eval_reports.py
│
├── reports/                           # Generated outputs
│   ├── abuse_report.jpg
│   ├── escalation_report.jpg
│   ├── crisis_report.jpg
│   ├── content_report.jpg
│   └── confusion_matrices/            # Optional sub-folder
│
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── save_pipeline_diagram.py           # Optional: ASCII/Graphviz diagram
└── report.pdf                         # Final technical report (submission)


⚙️ Setup Instructions
1. Clone Repository
git clone https://github.com/<your-username>/ai_safety.git
cd ai_safety_poc

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt

📊 Datasets

We used publicly available, anonymized datasets:

Jigsaw Toxic Comment
 → Abuse detection

DailyDialog
 → Escalation recognition

Suicide Detection Dataset
 → Crisis intervention

Repurposed Safe/Unsafe labels from Jigsaw → Content filtering

👉 Note: Full datasets are not uploaded (to avoid large files). Please download via Kaggle/HuggingFace and place them inside data/raw/<task_name>/.

🚀 Running the POC
1. Train Models

Each model has a training script under src/training/. Example:

python -m src.training.train_jigsaw_baseline
python -m src.training.train_dailydialog_baseline
python -m src.training.train_crisis_baseline
python -m src.training.train_content_baseline


Models will be saved in models/.

2. Run Moderation Pipeline

Use the chat pipeline to test integrated models:

python -m src.inference.chat_pipeline


You can type sample inputs, and the system will return:

Abuse check results

Emotion/escalation check

Crisis check

Content filter decision

Final moderation action

📈 Evaluation

Abuse Detection → F1 ≈ 0.66 (toxic/insult strong)

Escalation Recognition → Accuracy ≈ 88%

Crisis Intervention → Accuracy ≈ 90%

Content Filtering → Accuracy ≈ 92%

🧭 Ethical Considerations

Only public, anonymized datasets are used.

Models are balanced to reduce bias.

Crisis cases always escalate to human moderators, never automated responses.

🔮 Future Improvements

Upgrade to transformer-based models (BERT, DistilBERT).

Add multilingual support.

Build moderator dashboard with human feedback loop.

Deploy as microservices for real-time scaling.