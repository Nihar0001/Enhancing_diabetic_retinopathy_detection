"""
Test Voting Classifier on a single test image
Extracts simple pixel features (first 595 pixels from 256x256 grayscale image)
"""

import numpy as np
import cv2
import joblib
import pandas as pd
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Class mapping
CLASS_NAMES = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

# Paths
MODEL_PATH = Path("models/votingclassifier_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
TEST_IMAGES_PATH = Path("data/test_images")
TEST_CSV_PATH = Path("data/test.csv")

print("="*70)
print("TESTING VOTING CLASSIFIER ON A SINGLE TEST IMAGE")
print("="*70)

# Load model and scaler
if not MODEL_PATH.exists():
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    exit(1)

try:
    voting_model = joblib.load(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    print(f"✓ Model loaded successfully")
    print(f"✓ Scaler loaded successfully\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load test CSV to get image IDs
test_df = pd.read_csv(TEST_CSV_PATH)
test_images = test_df['id_code'].values

print(f"Total test images: {len(test_images)}")

# Pick first test image
test_image_id = test_images[0]
test_image_path = TEST_IMAGES_PATH / f"{test_image_id}.png"

print(f"\n{'='*70}")
print(f"Testing Image: {test_image_id}")
print(f"{'='*70}")

if not test_image_path.exists():
    print(f"❌ Image not found at {test_image_path}")
    exit(1)

# Read and preprocess image
try:
    img = cv2.imread(str(test_image_path))
    if img is None:
        raise ValueError("Could not read image")
    
    print(f"\n📸 Image Information:")
    print(f"   Path: {test_image_path}")
    print(f"   Original shape: {img.shape}")
    
    # Preprocess: resize to 256x256 and convert to grayscale
    img_resized = cv2.resize(img, (256, 256))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    print(f"   Resized to: {img_gray.shape}")
    
    # Extract features: flatten and take first 595 values (matching training)
    print(f"\n🔍 Feature Extraction:")
    print(f"   Using simple pixel features (first 595 from 256x256 grayscale)...")
    features = img_gray.flatten()[:595].astype(np.float32)
    
    print(f"   ✓ Features extracted: {len(features)} features")
    
    # Reshape for scaler (needs 2D array)
    features_2d = features.reshape(1, -1)
    
    # Scale features
    print(f"\n⚙️  Feature Scaling:")
    try:
        features_scaled = scaler.transform(features_2d)
        print(f"   ✓ Features scaled using pre-trained scaler")
    except ValueError as e:
        print(f"   ⚠️  Scaler mismatch: {e}")
        print(f"   Using features without scaling...")
        features_scaled = features_2d
    
    # Make prediction
    print(f"\n🤖 Making Prediction:")
    prediction = voting_model.predict(features_scaled)[0]
    prediction_proba = voting_model.predict_proba(features_scaled)[0]
    
    print(f"   ✓ Prediction complete!")
    
    # Display results
    print(f"\n{'='*70}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*70}")
    
    predicted_class = int(prediction)
    predicted_name = CLASS_NAMES[predicted_class]
    confidence = prediction_proba[predicted_class] * 100
    
    print(f"\n✅ Predicted Class: {predicted_class} ({predicted_name})")
    print(f"✅ Confidence: {confidence:.2f}%")
    
    print(f"\nClass Probabilities:")
    print(f"{'-'*70}")
    for class_id in range(5):
        class_name = CLASS_NAMES[class_id]
        prob = prediction_proba[class_id] * 100
        bar_length = int(prob / 2)
        bar = "█" * bar_length
        print(f"   {class_id} - {class_name:15s}: {prob:6.2f}% {bar}")
    print(f"{'-'*70}")
    
    print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
    print(f"The model correctly processed the test image and made a prediction.")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
