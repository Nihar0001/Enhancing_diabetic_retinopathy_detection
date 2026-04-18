"""
Inference Script for Diabetic Retinopathy Detection
Properly extracts features and makes predictions using the trained VotingClassifier model
"""

import numpy as np
import cv2
import joblib
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from feature_extraction import extract_features
from preprocessing import preprocess_image

# Class mapping
CLASS_NAMES = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

# Load model and scaler
MODEL_PATH = Path("models/votingclassifier_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

print("="*60)
print("LOADING DIABETIC RETINOPATHY DETECTION MODEL")
print("="*60)

if not MODEL_PATH.exists():
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    print("   Run train_models.py first to train the model")
    exit(1)

try:
    voting_model = joblib.load(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    print(f"✓ Model loaded: {MODEL_PATH.name}")
    print(f"✓ Scaler loaded: {SCALER_PATH.name}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print("✓ Ready for inference!\n")


def predict_image(image_path):
    """
    Predict diabetic retinopathy class from an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Prediction results with class, confidence, and probabilities
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "success": False,
                "error": f"Could not read image: {image_path}"
            }
        
        print(f"\n📸 Processing: {image_path}")
        print(f"   Original shape: {img.shape}")
        
        # Preprocess: resize to 256x256
        img_resized = cv2.resize(img, (256, 256))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        print(f"   Resized to: {img_resized.shape}")
        
        # ✅ IMPORTANT: Extract features properly (512 + 59 + 24 = 595)
        print("   Extracting 595 features (VGG16 + LBP + Haralick)...")
        features = extract_features(img_resized, img_gray)
        
        if features is None:
            return {
                "success": False,
                "error": "Failed to extract features"
            }
        
        print(f"   Features shape: {features.shape}")
        print(f"   Features count: {features.shape[0]}")
        
        # Ensure correct shape: should be 595
        if features.shape[0] != 595:
            return {
                "success": False,
                "error": f"Expected 595 features, got {features.shape[0]}"
            }
        
        # Reshape for model input
        features = features.reshape(1, -1).astype(np.float32)
        
        # Scale features using the same scaler from training
        features_scaled = scaler.transform(features)
        
        # Make prediction
        print("   Making prediction...")
        prediction = voting_model.predict(features_scaled)[0]
        
        # Get probabilities
        try:
            probabilities = voting_model.predict_proba(features_scaled)[0]
            prob_dict = {CLASS_NAMES[i]: float(f"{prob:.4f}") for i, prob in enumerate(probabilities)}
            confidence = float(f"{max(probabilities):.4f}")
        except:
            prob_dict = {}
            confidence = 0.0
        
        result = {
            "success": True,
            "prediction": int(prediction),
            "class": CLASS_NAMES[prediction],
            "probabilities": prob_dict,
            "confidence": confidence
        }
        
        print(f"\n✅ PREDICTION RESULT:")
        print(f"   Class: {result['class']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Probabilities: {result['probabilities']}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n📝 Usage:")
        print("   python inference.py <image_path>")
        print("\n📚 Example:")
        print("   python inference.py data/test_images/sample.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = predict_image(image_path)
    
    if not result['success']:
        print(f"❌ {result['error']}")
        sys.exit(1)
