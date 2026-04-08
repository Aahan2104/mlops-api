# ==============================
# Expt 1: Train ML Model and Save to .pkl using Pickle
# ==============================

import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# --------------------------
# 1. Load Dataset
# --------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Dataset loaded successfully!")
print(f"Shape: {X.shape}")
print(X.head())

# --------------------------
# 2. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# --------------------------
# 3. Scale the Features
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --------------------------
# 4. Train the Model
# --------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("\nModel training complete!")

# --------------------------
# 5. Evaluate the Model
# --------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# --------------------------
# 6. Save Model using Pickle
# --------------------------
os.makedirs('models', exist_ok=True)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ model.pkl saved successfully!")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ scaler.pkl saved successfully!")

# --------------------------
# 7. Verify: Reload and Predict
# --------------------------
with open('models/model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

print("\n--- Verification ---")
print("Model loaded from .pkl ✅")
print("Scaler loaded from .pkl ✅")

# Test with a new sample input
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)

print(f"\nSample Input: {sample[0]}")
print(f"Predicted Class: {iris.target_names[prediction[0]]}")
print("\n✅ Expt 1 Complete — Model deployed to .pkl file successfully!")