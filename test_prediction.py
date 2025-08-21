# test_prediction.py
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

# -------------------------------
# Step 1: Load the trained model
# -------------------------------
model = load_model("artifacts/model.h5")
print("✅ Model loaded successfully!")

# -------------------------------
# Step 2: Prepare new customer data
# -------------------------------
# Feature order MUST match training data exactly
X_new = pd.DataFrame([
    # Replace these values with actual new customer data
    [25, 12, 15, 2, 0, 500, 3, 1, 0, 1, 0, 0, 1, 0, 1],  # Sample 1
    #[40, 24, 5, 1, 1, 1200, 5, 0, 1, 0, 1, 0, 0, 1, 0]   # Sample 2
], columns=[
    "Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", 
    "Total Spend", "Last Interaction",
    "Gender_Male", "Gender_Female",
    "Subscription Type_Premium", "Subscription Type_Standard",
    "Contract Length_Monthly", "Contract Length_Quarterly",
    "ExtraFeature1", "ExtraFeature2"   # Replace with actual feature names from X_train
])

print("✅ New data prepared:")
print(X_new)

# -------------------------------
# Step 3: Predict churn
# -------------------------------
y_pred = model.predict(X_new)

# Convert probabilities to 0 or 1 (binary classification)
y_pred_class = (y_pred > 0.5).astype("int32")

# -------------------------------
# Step 4: Display results
# -------------------------------
for i, pred in enumerate(y_pred_class):
    status = "Churn" if pred[0] == 1 else "No Churn"
    print(f"Customer {i+1}: {status}")
