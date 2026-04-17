# Diabetic Retinopathy Detection - Complete Setup Guide

## 📋 Project Overview

**Enhancing Diabetic Retinopathy Detection** is a machine learning project that uses multiple classification algorithms to detect and classify diabetic retinopathy from medical images.

### Key Features
- **Multiple Classifiers**: RandomForest, SVM, GradientBoosting, VotingClassifier
- **Complete Pipeline**: Data generation → Feature extraction → Model training → Visualization
- **Two Workflows**:
  - **Notebook**: Full end-to-end pipeline (first-time setup)
  - **Script**: Quick retraining (fast iteration)

---

## 🏗️ Project Architecture

### Folder Structure
```
Enhancing-diabetic-retinopathy-detection/
├── notebooks/
│   └── hybrid_dr_detection.ipynb      ← Master pipeline (run this first!)
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_images/
│   ├── test_images/
│   └── *.npy files (auto-generated)    ← Feature arrays
├── models/
│   └── *.pkl files (auto-generated)    ← Trained models
├── outputs/
│   └── updated/
│       └── *.txt files (auto-generated) ← Reports
├── scripts/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── visualize.py
├── utils/
│   ├── generate_demo_data.py
│   └── test_project.py
├── train_models.py                     ← Quick retrain script
├── config.py
└── requirements.txt
```

### Workflow Architecture

```
FIRST TIME (Use Notebook)
├─ Cell 1: Setup & Imports
│  ├─ Install packages
│  ├─ Create directories
│  └─ Import libraries
│
├─ Cell 2: Load & Extract Features from REAL IMAGES
│  ├─ Read images from data/train_images/ and data/test_images/
│  ├─ Extract 595-dimensional features (VGG16 + LBP + Haralick)
│  ├─ Scale features with StandardScaler
│  └─ Save to data/*.npy for reuse
│
├─ Cell 3: Train Models
│  ├─ Load features (pre-extracted)
│  ├─ Train 4 models (RF, SVM, GB, Voting)
│  ├─ Save models to models/
│  └─ Save reports to outputs/updated/
│
└─ Cell 4: Visualize Results
   ├─ Read reports
   ├─ Plot confusion matrices
   └─ Display visualizations

QUICK RETRAIN (Use Script)
├─ Load pre-extracted features OR extract from images
├─ Train models
├─ Save models
└─ Fast! (seconds if features cached, minutes if extracting)
```

---

## 🚀 Quick Start (From Scratch)

### Prerequisites
- Python 3.8+
- Windows/Mac/Linux
- ~500MB disk space

### Step 1: Clone & Setup
```bash
git clone https://github.com/Nihar0001/Enhancing_diabetic_retinopathy_detection.git
cd Enhancing_diabetic_retinopathy_detection

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1           # Windows
source .venv/bin/activate              # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Your Data (IMPORTANT!)

**Option A: Real Data** (Recommended)
Place your retinopathy images in:
```
data/
├── train_images/
│   ├── 0/ (Normal)
│   ├── 1/ (Mild)
│   ├── 2/ (Moderate)
│   ├── 3/ (Severe)
│   └── 4/ (Proliferative)
├── test_images/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
```

**Option B: Demo Data** (For Testing)
If you don't have real data yet:
```bash
python utils/generate_demo_data.py
# Creates 100 training + 20 test samples (synthetic)
```

### Step 3: Run the Notebook (Complete Pipeline)
```bash
# Open notebook
jupyter notebook notebooks/hybrid_dr_detection.ipynb

# Run cells in order:
# 1. Cell 1: Setup & Imports
# 2. Cell 2: Load & Extract Features (from real images OR demo data)
# 3. Cell 3: Train Models (RandomForest, SVM, GradientBoosting, VotingClassifier)
# 4. Cell 4: Visualize Results (Confusion matrices & reports)
```

✅ **Done!** All models trained and visualized.

---

## ⚡ Quick Retraining (After First Run)

Once you've run the notebook and extracted features:

```bash
# Option 1: Fast retrain (loads cached features)
python train_models.py

# Option 2: Retrain with new images
# (Script auto-detects changes in data/train_images/ and re-extracts)
python train_models.py
```

**Use this when you want to:**
- Change model hyperparameters
- Re-train with updated images
- Compare different model versions
- Quick iteration without re-extracting features

---

## 📊 What Gets Created

### Input Data (You Provide)
```
data/
├── train_images/0/, 1/, 2/, 3/, 4/    ← Your image files
├── test_images/0/, 1/, 2/, 3/, 4/     ← Your image files
└── train.csv, test.csv (optional)
```

### After Cell 2 (Feature Extraction)
- `data/X_train.npy` - Extracted training features (n × 595)
- `data/X_test.npy` - Extracted test features (m × 595)
- `data/y_train.npy` - Training labels (0-4)
- `data/y_test.npy` - Test labels (0-4)
- `data/X_train_scaled.npy` - Scaled training features
- `data/X_test_scaled.npy` - Scaled test features
- `data/scaler.pkl` - StandardScaler object (for predictions)

### After Cell 3 (Model Training)
- `models/randomforest_model.pkl`
- `models/svm_model.pkl`
- `models/gradientboosting_model.pkl`
- `models/votingclassifier_model.pkl`
- `models/scaler.pkl`

### After Cell 3 (Reports)
- `outputs/updated/randomforest_report.txt`
- `outputs/updated/svm_report.txt`
- `outputs/updated/gradientboosting_report.txt`
- `outputs/updated/votingclassifier_report.txt`

### After Cell 4 (Visualizations)
- `outputs/updated/confusion_matrices.png`
- `outputs/updated/model_accuracy_bar_chart.png`
- `outputs/updated/model_comparison.png`
- `outputs/updated/model_radar_chart.png`

- `outputs/updated/svm_report.txt`
- `outputs/updated/gradientboosting_report.txt`
- `outputs/updated/votingclassifier_report.txt`

### After Cell 4 (Visualizations)
- Confusion matrix heatmaps for all 4 models
- Printed classification reports

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Dataset size (for demo data)
n_train = 100        # Training samples
n_test = 20          # Test samples
n_features = 595     # Feature vector size
n_classes = 5        # DR classification (0-4)

# Model hyperparameters
RANDOM_FOREST_N_ESTIMATORS = 100
SVM_KERNEL = 'linear'
RANDOM_STATE = 42
```

---

## 📚 Detailed Workflow

### Workflow A: Full Pipeline (Notebook)

**When to use:** First time, exploring data, feature engineering changes

```
1. Open: notebooks/hybrid_dr_detection.ipynb
2. Run Cell 1  → Dependencies installed
3. Run Cell 2  → Demo data created
4. Run Cell 3  → Models trained
5. Run Cell 4  → Visualizations displayed
```

**Time:** ~5-7 minutes total

### Workflow B: Quick Retrain (Script)

**When to use:** Testing hyperparameters, iterating on models

```
1. python train_models.py
2. Check outputs/updated/ for reports
```

**Time:** ~30-60 seconds

### Workflow C: Real Data (Manual)

**When to use:** Using actual retinopathy images

```
1. Place images in data/train_images/ and data/test_images/
2. Run feature extraction: python scripts/feature_extraction.py
3. Run train_models.py
```

---

## 🧪 Testing

Verify everything works:

```bash
# Quick test
python utils/test_project.py

# Run full pipeline
jupyter notebook notebooks/hybrid_dr_detection.ipynb
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Module not found"** | `pip install -r requirements.txt` |
| **"data files not found"** | Run Cell 2 of notebook first |
| **"models folder empty"** | Run Cell 3 of notebook |
| **Jupyter not found** | `pip install jupyter` |
| **Python version error** | Use Python 3.8+ |

---

## 📈 Understanding Results

### Classification Report
```
              precision    recall  f1-score   support
           0       0.50      1.00      0.67        1
           1       0.50      0.50      0.50        2
           2       0.00      0.00      0.00        2
           3       1.00      0.67      0.80        3
           4       0.33      0.50      0.40        2

    accuracy                           0.55       10
   macro avg       0.47      0.53      0.47       10
weighted avg       0.51      0.55      0.50       10
```

- **Precision**: Of predicted class X, how many were correct?
- **Recall**: Of actual class X, how many did we find?
- **F1-Score**: Harmonic mean of precision & recall
- **Support**: Number of actual samples in each class

### Confusion Matrix
```
[[1 0 0 0 0]    ← Predicted class 0
 [1 1 0 0 0]      (rows = actual, cols = predicted)
 [2 0 0 0 0]
 [0 1 0 2 0]
 [0 1 0 0 1]]
```

---

## 📖 File Reference

### Main Files

| File | Purpose |
|------|---------|
| `hybrid_dr_detection.ipynb` | Master pipeline notebook |
| `train_models.py` | Quick retraining script |
| `config.py` | Project configuration |
| `requirements.txt` | Python dependencies |

### Supporting Scripts

| File | Purpose |
|------|---------|
| `scripts/feature_extraction.py` | Extract features from images |
| `scripts/preprocessing.py` | Image preprocessing |
| `scripts/visualize.py` | Visualization utilities |
| `utils/generate_demo_data.py` | Generate synthetic data |
| `utils/test_project.py` | Test suite |

---

## 🎯 Common Tasks

### Task 1: Run from Complete Scratch
```bash
# Setup
git clone <repo>
cd <project>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run notebook end-to-end
jupyter notebook notebooks/hybrid_dr_detection.ipynb
```

### Task 2: Train on Your Data
```bash
# 1. Place images in data/train_images and data/test_images
# 2. Extract features
python scripts/feature_extraction.py
# 3. Train models
python train_models.py
```

### Task 3: Retrain with Different Hyperparameters
```bash
# Edit config.py
# Update RANDOM_FOREST_N_ESTIMATORS, etc.

# Retrain
python train_models.py
```

### Task 4: Compare Models
```bash
# Run notebook Cell 4 for visualizations
# or check outputs/updated/*.txt for reports
```

---

## ✅ Validation Checklist

After running the full workflow, verify:

- [ ] All 4 models present in `models/`
- [ ] All 4 reports in `outputs/updated/`
- [ ] Classification reports show metrics (precision, recall, F1)
- [ ] Confusion matrices visualized
- [ ] No errors in cell outputs

---

## 🔄 Model Details

| Model | Type | Use Case |
|-------|------|----------|
| **RandomForest** | Ensemble | Fast, good baseline |
| **SVM** | Kernel | High-dimensional data |
| **GradientBoosting** | Ensemble | Best accuracy potential |
| **VotingClassifier** | Ensemble | Combines all 3 models |

All models use:
- **Random State**: 42 (reproducible)
- **Class Weight**: Balanced (handles imbalance)

---

## 📞 Support

**Reading Reports:**
- Check `outputs/updated/*.txt`

**Debugging:**
- Run `utils/test_project.py`
- Check console output for error messages
- Read `docs/WORKING_WITHOUT_DATA.md`

**Dependencies:**
- See `requirements.txt`
- Full list in `docs/README.md`

---

## 📝 Notes

- **Demo Data**: 100 training + 20 test samples (synthetic)
- **Real Data**: Replace with your retinopathy images
- **Feature Size**: 595-dimensional (fixed)
- **Classes**: 0-4 (5 severity levels)
- **Train/Test Split**: Auto-generated

---

## 🎓 Project Status

✅ **Complete & Tested**
- All 4 models implemented
- Demo data generation working
- Notebook pipeline functional
- Quick retraining script ready
- Visualization tools included

---

## 📄 License & Attribution

See individual documentation files for details.

---

**Last Updated**: April 2026
**Version**: 1.0
**Status**: Production Ready

---

## 🚀 Next Steps

1. ✅ Clone repository
2. ✅ Run Setup (Cell 1)
3. ✅ Generate Data (Cell 2)
4. ✅ Train Models (Cell 3)
5. ✅ Visualize (Cell 4)
6. 🔄 Iterate with `train_models.py`

**Enjoy exploring Diabetic Retinopathy Detection!**
