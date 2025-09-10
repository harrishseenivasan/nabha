import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# ===========================
# Step 1: Load dataset
# ===========================
df = pd.read_csv("data.csv")
X = df["description"]
y = df["category"]

# ===========================
# Step 2: Build ML pipeline
# ===========================
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# ===========================
# Step 3: Train model
# ===========================
model.fit(X, y)

# ===========================
# Step 4: Save model
# ===========================
joblib.dump(model, "symptom_model.pkl")
print("✅ Model trained and saved as symptom_model.pkl")
