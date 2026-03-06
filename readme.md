# Diabetic Retinopathy Detection - Hybrid Deep Learning Model

## 📋 Project Overview

A comprehensive **hybrid diabetic retinopathy detection system** that combines three complementary feature extraction techniques:
- **Deep Learning Features**: VGG16 pre-trained neural network
- **Texture Features**: Local Binary Pattern (LBP) analysis
- **GLCM Features**: Haralick texture descriptors

The system uses an ensemble voting classifier combining RandomForest, SVM, and KNN classifiers to achieve robust predictions across 5 DR severity classes (0-4).

---

## 🚀 CHANGE LOG & IMPROVEMENTS (UPDATED: March 6, 2026)

### ✅ **COMPLETED CHANGES**

#### 1. **Version Control Setup** ✓
- ✅ Initialized Git repository (`git init`)
- ✅ Created comprehensive `.gitignore` file

#### 2. **Code Quality & Error Handling** ✓
- ✅ Fixed TensorFlow deprecation warnings in `feature_extraction.py`
- ✅ Enhanced `preprocessing.py` with validation and better docs
- ✅ Improved error handling in all feature extraction functions

#### 3. **Project Structure & Configuration** ✓
- ✅ Created `scripts/__init__.py` for proper package structure
- ✅ Created `config.py` (Central Configuration File) with:
  - Path Management using `pathlib.Path` for cross-platform compatibility
  - All hyperparameters in one place
  - Utility functions for validation

#### 4. **Documentation & Tracking** ✓
- ✅ Created this comprehensive change-tracked README
- ✅ Added detailed docstrings to all functions

### 🔄 **IN-PROGRESS CHANGES**

#### 1. **Notebook Path Fixes** (High Priority)
Will update relative paths to use `config.py`

#### 2. **Model Evaluation Enhancements**
Adding improved metrics and visualization

### ⏳ **UPCOMING CHANGES**

#### 1. **Requirements.txt with Version Pinning**
#### 2. **Enhanced Model Training Pipeline**
#### 3. **Improved Evaluation Metrics**
#### 4. **Better Visualization Suite**

---

## 📁 Project Structure

```
Enhancing-diabetic-retinopathy-detection/
├── .git/                      # Git repository (NEW)
├── .gitignore                 # Git ignore rules (NEW)
├── config.py                  # Central configuration (NEW)
├── readme.md                  # This file (UPDATED)
├── scripts/
│   ├── __init__.py           # Package init (NEW)
│   ├── preprocessing.py      # Image preprocessing (IMPROVED)
│   ├── feature_extraction.py # Feature extraction (IMPROVED)
│   └── visualize.py          # Visualization functions
├── notebooks/
│   └── hybrid_dr_detection.ipynb
├── models/
│   ├── randomforest_model.pkl
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── votingclassifier_model.pkl
│   └── scaler.pkl
└── outputs/
    └── [reports and visualizations]
```

---

## 🔧 How to Use

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Running the Pipeline
```bash
jupyter notebook notebooks/hybrid_dr_detection.ipynb
```

### 3. Key Functions
```python
from scripts.preprocessing import preprocess_image
from scripts.feature_extraction import extract_deep_features, extract_lbp, extract_haralick
from config import PROJECT_ROOT, MODELS_DIR, DATA_DIR

# Use configuration
model_path = MODELS_DIR / "votingclassifier_model.pkl"
```

---

## 📊 Model Information

### Architecture
- **Ensemble Voting Classifier** (Soft Voting)
  - Base Learner 1: RandomForest (100 estimators)
  - Base Learner 2: SVM (Linear kernel)
  - Base Learner 3: KNN (k=5)

### Classes
- Class 0: No Diabetic Retinopathy
- Class 1: Mild DR
- Class 2: Moderate DR
- Class 3: Severe DR
- Class 4: Proliferative DR

---

## 📝 Code Quality Improvements

**Key Enhancements:**
- Enhanced error handling with try-except blocks
- Added comprehensive docstrings (Args/Returns/Raises)
- Suppressed TensorFlow verbose logging
- Added input validation across all functions
- Cross-platform path handling with `pathlib`

---

## 🎯 Next Steps for Team

1. Update notebook paths to use `config.py`
2. Test entire pipeline end-to-end
3. Pin package versions in requirements.txt
4. Add more evaluation metrics (ROC-AUC, sensitivity, specificity)

---

## 📞 Quick Reference

### Important Files Modified
- ✅ `scripts/feature_extraction.py` - TensorFlow fixes, error handling
- ✅ `scripts/preprocessing.py` - Input validation, better docs
- ✅ `scripts/__init__.py` - Created
- ✅ `config.py` - Created (Central configuration)
- ✅ `.gitignore` - Created

### Configuration Usage
```python
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, 
    CLASS_NAMES, IMAGE_TARGET_SIZE
)
```

---

**Status**: 🟡 In Progress (60% complete)  
**Last Updated**: March 6, 2026  
*This README tracks all changes made to the project. Update it as you make new changes!*