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

### Run Flask API
\\\ash
python app.py
\\\

API runs on: http://localhost:5000

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

The **frontend project** should:

1. **Send images to /predict endpoint**
   - Upload retinal image
   - Get prediction with probabilities

2. **Extract deep learning features**
   - VGG16: 512 features
   - LBP: 59 features
   - Haralick: 24 features
   - (This is frontend responsibility, not backend)

3. **Display results**
   - Show predicted class
   - Display confidence score
   - Show Grad-CAM heatmap
   - Store in patient history

---

## ✅ Model Verification

✅ **VotingClassifier:** 3 estimators (RF + SVM + GB)
✅ **StandardScaler:** 595 features
✅ **Training/Inference:** Identical feature extraction
✅ **Classes:** 5 (Normal → Proliferative)
✅ **API:** Ready for frontend integration

---

## 🧪 Testing

### Test API
\\\ash
curl -X POST http://localhost:5000/predict -F "file=@test_image.jpg"
\\\

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
