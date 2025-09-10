import os
import pandas as pd
import joblib

# get absolute path
BASE_DIR = os.path.dirname(__file__)

# Load model
model = joblib.load(os.path.join(BASE_DIR, "symptom_model.pkl"))

# Load doctors.csv safely
doctor_df = pd.read_csv(os.path.join(BASE_DIR, "doctors.csv"))

def predict_line(text):
    category = model.predict([text])[0]
    doctors = doctor_df[doctor_df["specialization"] == category]["doctor_name"].tolist()
    return category, doctors
