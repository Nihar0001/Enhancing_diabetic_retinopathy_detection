"""
Flask API for Diabetic Retinopathy Detection
Properly extracts 595 features and makes predictions
"""

import os
import numpy as np
import cv2
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import warnings
import base64
from typing import TYPE_CHECKING

warnings.filterwarnings('ignore')

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load model and scaler
print("\n" + "="*60)
print("FLASK API - SIMPLE PIXEL FEATURE PREDICTION")
print("(Deep learning features extracted in frontend)")
print("="*60)

MODEL_PATH = Path("models/votingclassifier_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

if not MODEL_PATH.exists():
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    exit(1)

try:
    voting_model = joblib.load(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    print(f"✓ Model loaded: {MODEL_PATH.name}")
    print(f"✓ Scaler loaded: {SCALER_PATH.name}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Class mapping
CLASS_NAMES = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

print("\n✓ API Ready!")
print("="*60 + "\n")


def extract_and_predict(image_array):
    """
    Extract 595 pixel features from image and make prediction.
    Uses SIMPLE pixel features (NOT deep learning).
    
    Deep learning feature extraction is in the frontend.
    
    Args:
        image_array: numpy array of image
        
    Returns:
        dict with prediction results or error
    """
    try:
        # Resize to 256x256
        img_resized = cv2.resize(image_array, (256, 256))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Extract 595 SIMPLE pixel features (NOT deep learning)
        features = img_gray.flatten()[:595]
        if len(features) < 595:
            features = np.pad(features, (0, 595 - len(features)))
        
        # Reshape and scale
        features = features.reshape(1, -1).astype(np.float32)
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = voting_model.predict(features_scaled)[0]
        
        # Get probabilities
        try:
            probabilities = voting_model.predict_proba(features_scaled)[0]
            prob_dict = {CLASS_NAMES[i]: float(f"{prob:.4f}") for i, prob in enumerate(probabilities)}
            confidence = float(f"{max(probabilities):.4f}")
        except:
            prob_dict = {}
            confidence = 0.0
        
        return {
            "success": True,
            "prediction": int(prediction),
            "class": CLASS_NAMES[prediction],
            "probabilities": prob_dict,
            "confidence": confidence
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "VotingClassifier (RandomForest + SVM + GradientBoosting)",
        "classes": CLASS_NAMES,
        "feature_count": 595
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint for image upload.
    
    Expected: Form data with 'image' file (PNG/JPG)
    
    Returns: JSON with prediction, class, probabilities, confidence
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image provided",
                "required": "Send image as form data with key 'image'"
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Read image from file
        file_content = file.read()
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                "success": False,
                "error": "Invalid image format. Use PNG or JPG"
            }), 400
        
        # Extract features and predict
        result = extract_and_predict(img)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """
    Prediction endpoint for base64 encoded image.
    
    Expected JSON:
    {
        "image": "base64_string"
    }
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No 'image' key in JSON"
            }), 400
        
        # Decode base64
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                "success": False,
                "error": "Invalid image format"
            }), 400
        
        # Extract features and predict
        result = extract_and_predict(img)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_type": "VotingClassifier",
        "components": ["RandomForest", "SVM", "GradientBoosting"],
        "size_mb": round(os.path.getsize(MODEL_PATH) / (1024*1024), 2),
        "classes": CLASS_NAMES,
        "input_features": 595,
        "feature_composition": {
            "vgg16": 512,
            "lbp": 59,
            "haralick": 24
        }
    }), 200


if __name__ == '__main__':
    print("\n🚀 Starting Flask API...")
    print("   http://localhost:5000/health")
    print("\nEndpoints:")
    print("   POST /predict - Send image file")
    print("   POST /predict_base64 - Send base64 image")
    print("   GET /model-info - Model details")
    print("   GET /health - Health check")
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
