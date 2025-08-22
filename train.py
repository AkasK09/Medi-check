# train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ========== Load Data ==========
symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")

# Some rows may have NaN symptoms → fill with empty
symptom_cols = [col for col in symptom_df.columns if col.startswith("Symptom")]
symptom_df[symptom_cols] = symptom_df[symptom_cols].fillna("")

# Collect all symptoms per row into a list
symptom_df["SymptomList"] = symptom_df[symptom_cols].values.tolist()
symptom_df["SymptomList"] = symptom_df["SymptomList"].apply(
    lambda x: [s.strip().lower().replace(" ", "_") for s in x if s != ""]
)

# ========== Encode Symptoms ==========
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptom_df["SymptomList"])

# ========== Encode Diseases ==========
le = LabelEncoder()
y = le.fit_transform(symptom_df["Disease"])

# ========== Train Model ==========
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

# ========== Save Artifacts ==========
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/disease_model.pkl")
joblib.dump(mlb, "model/symptom_encoder.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("✅ Model training complete. Saved in /model folder.")
