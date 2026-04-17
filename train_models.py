"""
Quick Model Retraining Script

Loads real images from flat directories, matches labels from CSV, extracts features, and trains models.
Works with real data in data/train_images/ and data/test_images/

Data structure:
  data/train_images/ - flat directory with PNG files
  data/train.csv - maps id_code to diagnosis (0-4)
  data/test_images/ - flat directory with PNG files
  data/test.csv - contains id_code (unlabeled)

NOTE: For first-time setup, use the Jupyter notebook:
  notebooks/hybrid_dr_detection_fixed.ipynb
"""

import numpy as np
import joblib
import sys
import os
import pandas as pd
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Add scripts to path
sys.path.insert(0, "scripts")

# Import custom modules (dynamic path)
try:
    from preprocessing import preprocess_image  # type: ignore
    from feature_extraction import extract_features  # type: ignore
except ImportError:
    preprocess_image = None  # type: ignore
    extract_features = None  # type: ignore
    print("⚠️ Warning: Custom modules not found. Will use fallback feature extraction.")

print("\n" + "="*60)
print("REAL DATA MODEL TRAINING")
print("="*60)

# Define paths
train_images_dir = Path("data/train_images")
test_images_dir = Path("data/test_images")

# Load CSV files
train_csv = pd.read_csv("data/train.csv")
test_csv = pd.read_csv("data/test.csv")

print(f"\n✓ Loaded train.csv: {len(train_csv)} images")
print(f"✓ Loaded test.csv: {len(test_csv)} images")

# Check if pre-extracted features exist
features_exist = (
    Path("data/X_train.npy").exists() and
    Path("data/X_test.npy").exists() and
    Path("data/y_train.npy").exists() and
    Path("data/y_test.npy").exists()
)

if features_exist:
    print("\n✓ Loading pre-extracted features from data/")
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    X_train_scaled = np.load("data/X_train_scaled.npy")
    X_test_scaled = np.load("data/X_test_scaled.npy")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
else:
    print("\n✓ Extracting features from real images...")
    
    if not train_images_dir.exists():
        print(f"\n✗ ERROR: {train_images_dir} not found!")
        print("  Please download data from:")
        print("  https://www.kaggle.com/competitions/diabetic-retinopathy-detection")
        exit(1)
    
    # Load training images
    print("  Loading training images...")
    X_train_list = []
    y_train_list = []
    
    for idx, row in train_csv.iterrows():
        if idx % 500 == 0:
            print(f"    Processing {idx}/{len(train_csv)}...")
        
        img_id = row['id_code']
        diagnosis = row['diagnosis']
        img_path = train_images_dir / f"{img_id}.png"
        
        if not img_path.exists():
            continue
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (256, 256))
            features = img_resized.flatten()[:595]
            if len(features) < 595:
                features = np.pad(features, (0, 595 - len(features)))
            X_train_list.append(features.astype(np.float32))
            y_train_list.append(diagnosis)
        except Exception as e:
            print(f"    ⚠️  Error: {img_id}: {e}")
    
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list)
    print(f"  ✓ Loaded {len(X_train)} training images")
    
    # Load test images
    print("  Loading test images...")
    X_test_list = []
    y_test_list = []
    
    for idx, row in test_csv.iterrows():
        if idx % 500 == 0:
            print(f"    Processing {idx}/{len(test_csv)}...")
        
        img_id = row['id_code']
        img_path = test_images_dir / f"{img_id}.png"
        
        if not img_path.exists():
            continue
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (256, 256))
            features = img_resized.flatten()[:595]
            if len(features) < 595:
                features = np.pad(features, (0, 595 - len(features)))
            X_test_list.append(features.astype(np.float32))
            y_test_list.append(-1)  # Test set is unlabeled
        except Exception as e:
            print(f"    ⚠️  Error: {img_id}: {e}")
    
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list)
    print(f"  ✓ Loaded {len(X_test)} test images")
    
    # Scale features
    print("  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save for future use
    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)
    np.save("data/X_train_scaled.npy", X_train_scaled)
    np.save("data/X_test_scaled.npy", X_test_scaled)
    joblib.dump(scaler, "data/scaler.pkl")
    print("  ✓ Features extracted and saved")

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/updated", exist_ok=True)

CLASS_NAMES = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']

# Train models
classifiers = {
    "RandomForest": (RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1),
                    X_train, X_train_scaled),
    "SVM": (SVC(kernel="linear", probability=True, cache_size=2000, random_state=42, class_weight="balanced"),
           X_train, X_train_scaled),
    "GradientBoosting": (HistGradientBoostingClassifier(random_state=42),
                        X_train, X_train_scaled),
}

print("\nTraining models on training set...")
trained_models = {}

for name, (clf, X_train_use, X_train_scaled_use) in classifiers.items():
    print(f"  {name}...", end=" ")
    
    # Use scaled data for SVM and GB, unscaled for RF
    if name == "RandomForest":
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train_scaled, y_train)
    
    trained_models[name] = clf
    joblib.dump(clf, f"models/{name.lower()}_model.pkl")
    
    # Generate report on training data predictions (for validation)
    if name == "RandomForest":
        y_pred_train = clf.predict(X_train)
    else:
        y_pred_train = clf.predict(X_train_scaled)
    
    accuracy = accuracy_score(y_train, y_pred_train)
    
    report = classification_report(y_train, y_pred_train, target_names=CLASS_NAMES, zero_division=0)
    cm = confusion_matrix(y_train, y_pred_train)
    
    with open(f"outputs/updated/{name.lower()}_report.txt", "w") as f:
        f.write(f"Classification Report for {name}:\n")
        f.write("="*60 + "\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
        f.write(f"Training Accuracy: {accuracy:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{np.array2string(cm)}\n")
    
    print(f"✓ (Accuracy: {accuracy:.4f})")

# Train VotingClassifier
print(f"  VotingClassifier...", end=" ")

voting_clf = VotingClassifier(
    estimators=[('rf', trained_models['RandomForest']), 
                ('svm', trained_models['SVM']), 
                ('gb', trained_models['GradientBoosting'])],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_train_scaled)

joblib.dump(voting_clf, "models/votingclassifier_model.pkl")

accuracy_voting = accuracy_score(y_train, y_pred_voting)
report_voting = classification_report(y_train, y_pred_voting, target_names=CLASS_NAMES, zero_division=0)
cm_voting = confusion_matrix(y_train, y_pred_voting)

with open("outputs/updated/votingclassifier_report.txt", "w") as f:
    f.write(f"Classification Report for VotingClassifier:\n")
    f.write("="*60 + "\n")
    f.write(report_voting)
    f.write("\n" + "="*60 + "\n")
    f.write(f"Training Accuracy: {accuracy_voting:.4f}\n")
    f.write(f"\nConfusion Matrix:\n{np.array2string(cm_voting)}\n")

print(f"✓ (Accuracy: {accuracy_voting:.4f})")

print("\n" + "="*60)
print("✓ Training complete!")
print("  Check: models/ and outputs/updated/")
print("="*60 + "\n")
