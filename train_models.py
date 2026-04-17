"""
Quick Model Retraining Script

Loads real images, extracts features, and trains models.
Works with real data in data/train_images/ and data/test_images/

NOTE: For first-time setup, use the Jupyter notebook:
  notebooks/hybrid_dr_detection.ipynb
"""

import numpy as np
import joblib
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Add scripts to path
sys.path.insert(0, "scripts")
from preprocessing import preprocess_image
from feature_extraction import extract_features

print("\n" + "="*60)
print("REAL DATA MODEL TRAINING")
print("="*60)

# Define paths
train_images_dir = Path("data/train_images")
test_images_dir = Path("data/test_images")

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
else:
    print("\n✓ Extracting features from real images...")
    
    if not train_images_dir.exists():
        print(f"\n✗ ERROR: {train_images_dir} not found!")
        print("  Please organize your data as:")
        print("    data/train_images/0/")
        print("    data/train_images/1/")
        print("    data/train_images/2/")
        print("    data/train_images/3/")
        print("    data/train_images/4/")
        exit(1)
    
    # Load training images
    print("  Loading training images...")
    X_train = []
    y_train = []
    
    for class_idx in range(5):
        class_dir = train_images_dir / str(class_idx)
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        
        for img_path in image_files:
            try:
                img, img_gray = preprocess_image(str(img_path))
                features = extract_features(img, img_gray)
                X_train.append(features)
                y_train.append(class_idx)
            except Exception as e:
                print(f"    ⚠️  Error: {img_path.name}: {e}")
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    
    # Load test images
    print("  Loading test images...")
    X_test = []
    y_test = []
    
    for class_idx in range(5):
        class_dir = test_images_dir / str(class_idx)
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        
        for img_path in image_files:
            try:
                img, img_gray = preprocess_image(str(img_path))
                features = extract_features(img, img_gray)
                X_test.append(features)
                y_test.append(class_idx)
            except Exception as e:
                print(f"    ⚠️  Error: {img_path.name}: {e}")
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)
    
    # Scale features
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

# Train models
classifiers = {
    "RandomForest": (RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
                    X_train, X_test),
    "SVM": (SVC(kernel="linear", probability=True, cache_size=2000, random_state=42, class_weight="balanced"),
           X_train_scaled, X_test_scaled),
    "GradientBoosting": (HistGradientBoostingClassifier(random_state=42),
                        X_train_scaled, X_test_scaled),
}

print("\nTraining models...")
for name, (clf, X_tr, X_te) in classifiers.items():
    print(f"  {name}...", end=" ")
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    
    joblib.dump(clf, f"models/{name.lower()}_model.pkl")
    
    with open(f"outputs/updated/{name.lower()}_report.txt", "w") as f:
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        f.write(f"Classification Report for {name}:\n{report}")
        f.write(f"\nConfusion Matrix:\n{np.array2string(cm)}\n")
    
    print("✓")

# Train VotingClassifier
print("  VotingClassifier...", end=" ")
clf1 = joblib.load("models/randomforest_model.pkl")
clf2 = joblib.load("models/svm_model.pkl")
clf3 = joblib.load("models/gradientboosting_model.pkl")

voting_clf = VotingClassifier(
    estimators=[('rf', clf1), ('svm', clf2), ('gb', clf3)],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_test_scaled)

joblib.dump(voting_clf, "models/votingclassifier_model.pkl")

with open("outputs/updated/votingclassifier_report.txt", "w") as f:
    report = classification_report(y_test, y_pred_voting)
    cm = confusion_matrix(y_test, y_pred_voting)
    f.write(f"Classification Report for VotingClassifier:\n{report}")
    f.write(f"\nConfusion Matrix:\n{np.array2string(cm)}\n")

print("✓")

print("\n" + "="*60)
print("✓ Training complete!")
print("  Check: models/ and outputs/updated/")
print("="*60 + "\n")
