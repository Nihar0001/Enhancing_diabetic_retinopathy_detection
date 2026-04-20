"""
Model Retraining Script (Simple Pixel Feature Pipeline)

Loads real images, extracts simple 595-dimensional grayscale pixel features,
and trains ensemble models.

Data structure:
  data/train_images/ - flat directory with PNG files
  data/train.csv - maps id_code to diagnosis (0-4)
  data/test_images/ - flat directory with PNG files
  data/test.csv - contains id_code (unlabeled)

Features extracted: 595-dimensional (grayscale pixel vector)
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

print("\n" + "="*60)
print("MODEL TRAINING - SIMPLE PIXEL FEATURES")
print("(Deep learning feature extraction is in frontend)")
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
    print("\n✓ Extracting SIMPLE pixel features from real images...")
    print("  (Deep learning features are extracted in frontend)")
    
    if not train_images_dir.exists():
        print(f"\n❌ ERROR: {train_images_dir} not found!")
        print("   Please download data from:")
        print("   https://www.kaggle.com/competitions/diabetic-retinopathy-detection")
        exit(1)
    
    # Load training images
    print("  📷 Loading training images and extracting 595 pixel features...")
    X_train_list = []
    y_train_list = []
    
    for idx, row in train_csv.iterrows():
        if idx % 100 == 0:
            print(f"    Processing {idx}/{len(train_csv)}...")
        
        img_id = row['id_code']
        diagnosis = row['diagnosis']
        img_path = train_images_dir / f"{img_id}.png"
        
        if not img_path.exists():
            continue
        
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Resize to 256×256
            img_resized = cv2.resize(img, (256, 256))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Extract SIMPLE pixel features (595 pixels)
            features = img_gray.flatten()[:595]
            if len(features) < 595:
                features = np.pad(features, (0, 595 - len(features)))
            
            X_train_list.append(features.astype(np.float32))
            y_train_list.append(diagnosis)
        
        except Exception as e:
            print(f"    ⚠️  Error processing {img_id}: {e}")
    
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list)
    print(f"  ✓ Loaded {len(X_train)} training images with simple 595-dim pixel features")
    
    # Load test images
    print("  📷 Loading test images and extracting 595 pixel features...")
    X_test_list = []
    y_test_list = []
    
    for idx, row in test_csv.iterrows():
        if idx % 100 == 0:
            print(f"    Processing {idx}/{len(test_csv)}...")
        
        img_id = row['id_code']
        img_path = test_images_dir / f"{img_id}.png"
        
        if not img_path.exists():
            continue
        
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Resize to 256×256
            img_resized = cv2.resize(img, (256, 256))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Extract SIMPLE pixel features (595 pixels)
            features = img_gray.flatten()[:595]
            if len(features) < 595:
                features = np.pad(features, (0, 595 - len(features)))
            
            X_test_list.append(features.astype(np.float32))
            y_test_list.append(-1)  # Test set is unlabeled
        
        except Exception as e:
            print(f"    ⚠️  Error processing {img_id}: {e}")
    
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list)
    print(f"  ✓ Loaded {len(X_test)} test images with simple 595-dim pixel features")
    
    # Scale features
    print("\n  Scaling features using StandardScaler...")
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
    
    # Save scaler in models directory (for inference consistency)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("  ✓ Features extracted, scaled, and saved")

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/updated", exist_ok=True)

CLASS_NAMES = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']

print("\n" + "="*70)
print("TRAINING MODELS ON 595-DIMENSIONAL SIMPLE PIXEL FEATURES")
print("="*70)

# Use scaled data for consistent model behavior across all estimators.

classifiers = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "SVM": SVC(
        kernel="linear",
        probability=True,
        cache_size=2000,
        random_state=42,
        class_weight="balanced"
    ),
    "GradientBoosting": HistGradientBoostingClassifier(
        random_state=42
    ),
}

print("\n📊 Training individual models...")
trained_models = {}
test_results = {}

for name, clf in classifiers.items():
    print(f"\n  [{name}]")
    
    # Train all models on scaled features
    print(f"    Training...", end=" ", flush=True)
    clf.fit(X_train_scaled, y_train)
    trained_models[name] = clf
    joblib.dump(clf, f"models/{name.lower()}_model.pkl")
    print("✓")
    
    # Evaluate on TRAINING set
    y_pred_train = clf.predict(X_train_scaled)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    # Evaluate on TEST set (important!)
    y_pred_test = clf.predict(X_test_scaled)
    accuracy_test = accuracy_score(y_test[y_test != -1], y_pred_test[y_test != -1]) if np.any(y_test != -1) else 0
    
    print(f"    Train Accuracy: {accuracy_train:.4f}")
    if accuracy_test > 0:
        print(f"    Test Accuracy:  {accuracy_test:.4f}")
    
    # Generate detailed report
    report = classification_report(y_train, y_pred_train, target_names=CLASS_NAMES, zero_division=0)
    cm = confusion_matrix(y_train, y_pred_train)
    
    with open(f"outputs/updated/{name.lower()}_report.txt", "w") as f:
        f.write(f"Classification Report for {name}\n")
        f.write("="*70 + "\n")
        f.write("Features: 595-dimensional (simple grayscale pixel vector)\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Training Accuracy: {accuracy_train:.4f}\n\n")
        f.write(report)
        f.write("\n" + "="*70 + "\n")
        f.write(f"Confusion Matrix:\n{np.array2string(cm)}\n")
    
    test_results[name] = (accuracy_train, accuracy_test if accuracy_test > 0 else None)

# Train VotingClassifier (ensemble of all 3 models)
print(f"\n  [VotingClassifier - Ensemble]")
print(f"    Creating ensemble...", end=" ", flush=True)

voting_clf = VotingClassifier(
    estimators=[
        ('rf', trained_models['RandomForest']), 
        ('svm', trained_models['SVM']), 
        ('gb', trained_models['GradientBoosting'])
    ],
    voting='soft',      # Soft voting uses probability averages
    n_jobs=-1
)

voting_clf.fit(X_train_scaled, y_train)
joblib.dump(voting_clf, "models/votingclassifier_model.pkl")
print("✓")

# Evaluate VotingClassifier
y_pred_voting_train = voting_clf.predict(X_train_scaled)
accuracy_voting_train = accuracy_score(y_train, y_pred_voting_train)

y_pred_voting_test = voting_clf.predict(X_test_scaled)
accuracy_voting_test = accuracy_score(y_test[y_test != -1], y_pred_voting_test[y_test != -1]) if np.any(y_test != -1) else 0

print(f"    Train Accuracy: {accuracy_voting_train:.4f}")
if accuracy_voting_test > 0:
    print(f"    Test Accuracy:  {accuracy_voting_test:.4f}")

# Generate VotingClassifier report
report_voting = classification_report(y_train, y_pred_voting_train, target_names=CLASS_NAMES, zero_division=0)
cm_voting = confusion_matrix(y_train, y_pred_voting_train)

with open("outputs/updated/votingclassifier_report.txt", "w") as f:
    f.write(f"Classification Report for VotingClassifier (Ensemble)\n")
    f.write("="*70 + "\n")
    f.write("Features: 595-dimensional (simple grayscale pixel vector)\n")
    f.write(f"Ensemble Method: Soft voting (probability averaging)\n")
    f.write("Base Models: RandomForest, SVM (Linear), HistGradientBoosting\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Training Accuracy: {accuracy_voting_train:.4f}\n\n")
    f.write(report_voting)
    f.write("\n" + "="*70 + "\n")
    f.write(f"Confusion Matrix:\n{np.array2string(cm_voting)}\n")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# Summary
print("\n📊 MODEL SUMMARY:")
print(f"  Training samples: {len(X_train)}")
print("  Feature dimensions: 595 (simple grayscale pixel vector)")
print(f"  Classes: {len(CLASS_NAMES)} (0=Normal, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)")

print("\n📈 ACCURACY RESULTS:")
for name, (acc_train, acc_test) in test_results.items():
    print(f"  {name}:")
    print(f"    Training: {acc_train:.4f}")
    if acc_test is not None:
        print(f"    Testing:  {acc_test:.4f}")

print(f"  VotingClassifier (Ensemble):")
print(f"    Training: {accuracy_voting_train:.4f}")
if accuracy_voting_test > 0:
    print(f"    Testing:  {accuracy_voting_test:.4f}")

print("\n📁 SAVED FILES:")
print(f"  Models:  models/[randomforest_model.pkl, svm_model.pkl, gradientboosting_model.pkl, votingclassifier_model.pkl]")
print(f"  Scaler:  models/scaler.pkl")
print(f"  Reports: outputs/updated/[*_report.txt]")
print(f"  Features: data/[X_train.npy, X_test.npy, y_train.npy, y_test.npy, X_train_scaled.npy, X_test_scaled.npy]")

print("\n✅ Ready for inference! Use inference.py to make predictions on new images.")
print("="*70 + "\n")
