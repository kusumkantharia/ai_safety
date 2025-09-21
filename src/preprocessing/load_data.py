import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/jigsaw_toxic/train.csv")

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())


