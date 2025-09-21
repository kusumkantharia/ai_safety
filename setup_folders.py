import os

# Define folder structure
folders = [
    "ai_safety_poc/data/raw/jigsaw_toxic",
    "ai_safety_poc/data/raw/escalation",
    "ai_safety_poc/data/raw/crisis",
    "ai_safety_poc/data/raw/content_filter",

    "ai_safety_poc/models",

    "ai_safety_poc/src/preprocessing",
    "ai_safety_poc/src/training",
    "ai_safety_poc/src/inference",
    "ai_safety_poc/src/evaluation",

    "ai_safety_poc/reports/confusion_matrices",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create placeholder files
placeholder_files = {
    "ai_safety_poc/requirements.txt": "# Add project dependencies here\n",
    "ai_safety_poc/README.md": "# AI Safety POC Project\n\nDocumentation for setup and usage.\n",
    "ai_safety_poc/save_pipeline_diagram.py": "# Script to generate pipeline diagram\n",
    "ai_safety_poc/report.pdf": "",  # placeholder, will be generated later
}

for filepath, content in placeholder_files.items():
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write(content)

print("✅ Project folder structure created successfully!")
