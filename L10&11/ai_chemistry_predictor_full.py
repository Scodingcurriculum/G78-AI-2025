"""
AI Chemistry Predictor 
This script:
1. Loads a dataset of chemical combinations.
2. Encodes the output labels.
3. Trains a Naive Bayes classifier.
4. Saves the trained model.
5. Launches a tkinter GUI for live predictions.

Requirements:
- pandas
- scikit-learn
- joblib
- tkinter (pre-installed with Python)
"""

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import messagebox

# ========== STEP 1: Load and Prepare Dataset ==========
print("📦 Loading chemistry dataset...")

df = pd.read_csv("D:/ChromeDownload/chemistry_dataset_500.csv")  # Path must match the CSV file location
df["reaction"] = df["Chemical1"].str.strip().str.lower() + " + " + df["Chemical2"].str.strip().str.lower()

X = df["reaction"]
y = df["Result"]

# ========== STEP 2: Encode Labels ==========
print("🔢 Encoding result labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ========== STEP 3: Train Naive Bayes Model ==========
print("🧠 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(pipeline, "chemistry_pipeline.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("💾 Model and label encoder saved.")

# ========== STEP 4: GUI Prediction ==========       # ========== ****** ONLY FUNCTION TO BE WRITTEN BY STUDENTS. ******* ==========

def predict():
    chem1 = entry1.get().strip().lower()
    chem2 = entry2.get().strip().lower()

    if not chem1 or not chem2:
        messagebox.showwarning("Input Error", "Please enter both chemicals.")
        return

    try:
        model = joblib.load("chemistry_pipeline.pkl")
        le = joblib.load("label_encoder.pkl")
        input_text = chem1 + " + " + chem2
        prediction = model.predict([input_text])[0]
        result = le.inverse_transform([prediction])[0]
        result_label.config(text=f"⚗️ Predicted Result: {result}")
    except Exception as e:
        result_label.config(text=f"❌ Error: {str(e)}")

# GUI setup
window = tk.Tk()
window.title("AI Chemistry Predictor")
window.geometry("500x300")
window.configure(bg="#f0faff")

tk.Label(window, text="🔬 Enter Two Chemicals:", bg="#f0faff", font=("Arial", 14)).pack(pady=10)

entry1 = tk.Entry(window, width=40, font=("Arial", 12))
entry1.pack(pady=5)
entry1.insert(0, "e.g. Hydrochloric Acid")

entry2 = tk.Entry(window, width=40, font=("Arial", 12))
entry2.pack(pady=5)
entry2.insert(0, "e.g. Sodium Hydroxide")

tk.Button(window, text="Predict Reaction", font=("Arial", 12), command=predict).pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 14), bg="#f0faff")
result_label.pack(pady=20)

window.mainloop()
