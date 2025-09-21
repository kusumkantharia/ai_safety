# save as setup_folders.py and run: python setup_folders.py
import os

folders = [
    "data/raw",
    "data/processed",
    "models",
    "notebooks",
    "src/preprocessing",
    "src/training",
    "src/inference",
    "src/utils",
    "scripts",
    "docs",
    "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(f"# {folder}\n")

print("✅ Folder structure created!")
