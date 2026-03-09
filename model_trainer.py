import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

X = np.array([
    [0],
    [1],
    [2],
])

y = np.array([0, 1, 2])

model = LogisticRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/reply_model.pkl")

print("Model saved successfully.")
