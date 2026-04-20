# RETINASCAN - Backend API 🏥🔬

**Diabetic Retinopathy Detection System - Machine Learning Backend**

An intelligent medical imaging system that detects and classifies diabetic retinopathy severity from retinal images using ensemble machine learning models.

---

## 📋 Project Overview

**RETINASCAN** is a hybrid system with two parts:

### This Repository (Backend - ML Models & API)
- ✅ Preprocesses retinal images
- ✅ Extracts simple pixel features (595 dimensions)
- ✅ Runs ensemble prediction (VotingClassifier)
- ✅ Provides REST API endpoints
- ✅ Returns predictions with confidence scores

### Frontend Project (Separate Repository)
- 🧠 Deep learning feature extraction (VGG16)
- 🧠 Texture features (LBP, Haralick GLCM)
- 📊 Web UI for patient management
- 📊 Grad-CAM visualization for explainability
- 📊 Patient history & PDF reports

---

## 🎯 Key Features

| Feature | Details |
|---------|---------|
| **Preprocessing** | Grayscale conversion, 256×256 resize |
| **Features** | 595 simple pixel features |
| **Models** | RandomForest + SVM + GradientBoosting |
| **Ensemble** | Voting Classifier (Soft Voting) |
| **Output** | 5-class prediction: Normal → Proliferative |
| **Confidence** | Probability scores for each class |
| **API** | Flask REST with CORS support |

---

## 🔧 Installation

### 1. Clone the Repository
\\\ash
git clone <backend-repo-url>
cd Enhancing-diabetic-retinopathy-detection
\\\

### 2. Create Virtual Environment
\\\ash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
\\\

### 3. Install Dependencies
\\\ash
pip install -r requirements.txt
\\\

---

## 🚀 Quick Start

### Option 1: Run Flask API (Production)
```bash
python app.py
```
API runs on: http://localhost:5000

Then test with:
```bash
curl -X POST http://localhost:5000/predict -F "file=@test_image.jpg"
```

### Option 2: Direct Prediction (Development)
```bash
python inference.py
```

### Option 3: Load Models Programmatically
```python
import joblib
import cv2
import numpy as np

# Load trained models
voting_model = joblib.load('models/votingclassifier_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load and preprocess image
img = cv2.imread('test_image.jpg')
img_resized = cv2.resize(img, (256, 256))
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Extract features (595 simple pixels)
features = img_gray.flatten()[:595]
features = features.reshape(1, -1).astype(np.float32)

# Scale features
features_scaled = scaler.transform(features)

# Make prediction
prediction = voting_model.predict(features_scaled)[0]
probabilities = voting_model.predict_proba(features_scaled)[0]

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
```

---

## 📡 API Endpoints

### 1. Health Check
\\\
GET /health
\\\

### 2. Predict Image
\\\
POST /predict
Content-Type: multipart/form-data
Body: file (image.jpg or image.png)
\\\

**Response Example:**
\\\json
{
  "success": true,
  "prediction": 2,
  "class": "Moderate",
  "probabilities": {
    "Normal": 0.0234,
    "Mild": 0.1245,
    "Moderate": 0.7234,
    "Severe": 0.1198,
    "Proliferative": 0.0089
  },
  "confidence": 0.7234
}
\\\

---

## 🧠 Model Architecture

### Feature Extraction
- **Input:** Grayscale 256×256 image
- **Output:** 595 features (simple pixels)
- **No deep learning** in backend

### Classification
\\\
595-dim Feature Vector
    ↓
Random Forest (100 trees)
SVM (RBF kernel)
Gradient Boosting (HGB)
    ↓
Voting Classifier (Soft Vote)
    ↓
Final Prediction + Confidence
\\\

### Output Classes
| ID | Class | Severity |
|----|-------|----------|
| 0 | Normal | No DR |
| 1 | Mild | Early stage |
| 2 | Moderate | Progressing |
| 3 | Severe | Advanced |
| 4 | Proliferative | Critical |

---

## 📁 Project Structure

\\\
Enhancing-diabetic-retinopathy-detection/
├── app.py                              # Flask API (main)
├── inference.py                        # Prediction script
├── train_models.py                     # Training code
├── README.md                           # This file
│
├── models/
│   ├── votingclassifier_model.pkl      # ✅ Main model
│   ├── scaler.pkl                      # ✅ Feature scaler
│   ├── randomforest_model.pkl          # Base model
│   ├── svm_model.pkl                   # Base model
│   └── gradientboosting_model.pkl      # Base model
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_images/
│   └── test_images/
│
├── scripts/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── visualize.py
│
└── ARCHITECTURE_VERIFICATION.md        # Detailed architecture
\\\

---

## 🔗 Frontend Integration

### How to Use Models in Frontend Code

#### Method 1: Call Backend API (Recommended)
```python
import requests

def predict_image(image_path):
    """Send image to backend and get prediction"""
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:5000/predict', files=files)
        result = response.json()
    
    if result['success']:
        return {
            'class': result['class'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        }
    else:
        print(f"Error: {result['error']}")
        return None

# Usage
result = predict_image('retinal_image.jpg')
print(f"Prediction: {result['class']} (confidence: {result['confidence']})")
```

#### Method 2: Load Models Directly (for advanced users)
```python
import joblib
import cv2
import numpy as np

# Load models
voting_model = joblib.load('models/votingclassifier_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def preprocess_image(image_path):
    """Preprocess image to 595 simple pixel features"""
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features = img_gray.flatten()[:595]
    
    if len(features) < 595:
        features = np.pad(features, (0, 595 - len(features)))
    
    return features.reshape(1, -1).astype(np.float32)

def predict(image_path):
    """Get prediction for image"""
    features = preprocess_image(image_path)
    features_scaled = scaler.transform(features)
    
    prediction = voting_model.predict(features_scaled)[0]
    probabilities = voting_model.predict_proba(features_scaled)[0]
    
    class_names = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
    
    return {
        'prediction': prediction,
        'class': class_names[prediction],
        'probabilities': dict(zip(class_names.values(), probabilities.tolist()))
    }

# Usage
result = predict('retinal_image.jpg')
print(f"Class: {result['class']}")
print(f"Probabilities: {result['probabilities']}")
```

### JavaScript/Frontend Integration
```javascript
// Call backend API from JavaScript
async function predictImage(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            return {
                class: result.class,
                confidence: result.confidence,
                probabilities: result.probabilities
            };
        } else {
            console.error('Error:', result.error);
        }
    } catch (error) {
        console.error('Request failed:', error);
    }
}

// Usage
const imageInput = document.querySelector('input[type="file"]');
imageInput.addEventListener('change', async (e) => {
    const result = await predictImage(e.target.files[0]);
    console.log(`Prediction: ${result.class}`);
    console.log(`Confidence: ${result.confidence}`);
});
```

### What Frontend Should NOT Do
- ❌ Don't extract VGG16/LBP/Haralick features - let backend handle
- ❌ Don't reimplement feature extraction - use API
- ❌ Don't load models directly unless you have specific reasons
- ✅ Do send images to `/predict` endpoint
- ✅ Do use probabilities for confidence display
- ✅ Do implement Grad-CAM visualization (separate from models)

---

## ✅ Model Verification

✅ **VotingClassifier:** 3 estimators (RF + SVM + GB)
✅ **StandardScaler:** 595 features
✅ **Training/Inference:** Identical feature extraction
✅ **Classes:** 5 (Normal → Proliferative)
✅ **API:** Ready for frontend integration

---

## 🧪 Testing

### 1. Verify Models Exist and Load
```bash
python -c "import joblib; m=joblib.load('models/votingclassifier_model.pkl'); s=joblib.load('models/scaler.pkl'); print(f'✅ VotingClassifier: {len(m.estimators_)} estimators'); print(f'✅ Scaler: {s.n_features_in_} features')"
```

**Expected Output:**
```
✅ VotingClassifier: 3 estimators
✅ Scaler: 595 features
```

### 2. Test API Health
```bash
curl http://localhost:5000/health
```

**Expected Response:**
```json
{
  "status": "running",
  "model": "votingclassifier",
  "features": 595
}
```

### 3. Test Prediction Endpoint
```bash
curl -X POST http://localhost:5000/predict -F "file=@data/test_images/image.jpg"
```

**Expected Response:**
```json
{
  "success": true,
  "prediction": 2,
  "class": "Moderate",
  "probabilities": {
    "Normal": 0.0234,
    "Mild": 0.1245,
    "Moderate": 0.7234,
    "Severe": 0.1198,
    "Proliferative": 0.0089
  },
  "confidence": 0.7234
}
```

### 4. Test with Python Script
```python
import requests

# Test health
response = requests.get('http://localhost:5000/health')
print(response.json())

# Test prediction
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    print(response.json())
```

---

## 📚 Key Points for Presentation

**What Backend Does:**
- ✅ Receives image from frontend
- ✅ Extracts 595 simple pixel features
- ✅ Runs through trained VotingClassifier
- ✅ Returns prediction + probabilities

**Why Simple Pixels:**
- ✅ Fast and reliable
- ✅ No deep learning overhead
- ✅ Works with ensemble models
- ✅ Frontend provides advanced features

**Model Architecture:**
- ✅ 3 base models (RF, SVM, GB)
- ✅ Soft voting ensemble
- ✅ 5-class output
- ✅ Confidence scoring

---

**Backend Status:** ✅ Ready for Frontend Integration

---
