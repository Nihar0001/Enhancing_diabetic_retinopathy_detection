import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# Set the path to the directory where your NPY files are located (e.g., your 'data' folder)
DATA_DIR = "./data" 
# Set the path where the scaler will be saved (e.g., your 'backend/models' folder)
MODELS_DIR = "./backend/models" 
TEST_SIZE = 0.2 # Use the same split size you used for training
RANDOM_STATE = 42 # Use the same random state if possible

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

try:
    # 1. Load the original unscaled feature data
    X = np.load(os.path.join(DATA_DIR, "X_features.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))
    
    print(f"Loaded X_features.npy with shape: {X.shape}")

    # 2. Re-run the train/test split on the original unscaled data
    # This ensures the scaler is only fit on the training portion, 
    # just like it was done during the original model training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Create, Fit, and Transform the StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train) # ***THIS IS THE CRITICAL STEP***
    
    # Optional: Verify the scaled training data matches your saved file 
    # X_train_scaled_verify = scaler.transform(X_train) 
    # print(f"New scaled X_train shape: {X_train_scaled_verify.shape}")

    # 4. Save the fitted scaler object for deployment
    scaler_filepath = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_filepath)
    
    print("\n--- SUCCESS ---")
    print(f"Scaler successfully created and saved to: {scaler_filepath}")
    print("You can now restart your Flask API and it will load the scaler.")

except FileNotFoundError:
    print("\n--- ERROR: FILE NOT FOUND ---")
    print(f"Make sure 'X_features.npy' and 'y_labels.npy' are in the {DATA_DIR} directory.")
except Exception as e:
    print(f"\n--- FATAL ERROR ---")
    print(f"An unexpected error occurred: {e}")
