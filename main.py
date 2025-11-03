# ==========================
# Machine Learning–Based Intrusion Detection System for Wireless Networks
# Author: ZHN (Trodev)
# ==========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ---------- Step 1: Load Dataset ----------
# Replace 'your_dataset.csv' with your actual file
dataset_path = "your_dataset.csv"
df = pd.read_csv(dataset_path)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------- Step 2: Handle Missing Values ----------
df = df.dropna()
print("✅ Missing values removed. New shape:", df.shape)

# ---------- Step 3: Encode Categorical Columns ----------
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("✅ Categorical columns encoded.")

# ---------- Step 4: Split Features and Target ----------
# Replace 'target' with the actual name of your label column
target_column = 'target'   # e.g., 'class', 'label', or 'attack_type'
X = df.drop(columns=[target_column])
y = df[target_column]

# ---------- Step 5: Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------- Step 6: Feature Scaling ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- Step 7: Train the ML Model ----------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training completed.")

# ---------- Step 8: Prediction ----------
y_pred = model.predict(X_test)

# ---------- Step 9: Evaluation ----------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n📊 Evaluation Results:")
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-Score: {:.4f}".format(f1))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))
