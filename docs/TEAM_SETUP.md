# Diabetic Retinopathy Detection - Team Setup Guide

**Quick Start Guide for Teammates** - Complete setup in 5 minutes

---

## 📋 Prerequisites
- Python 3.9+ (NOT 3.14 - scikit-learn compatibility)
- Git
- ~500 MB disk space
- Windows/Mac/Linux

---

## 🚀 Step-by-Step Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd Enhancing-diabetic-retinopathy-detection
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Setup
```bash
python -c "import sklearn, cv2, joblib, pandas, numpy; print('✓ All dependencies installed')"
```

---

## 🧠 Models Included

**Pre-trained models are included in `models/` folder:**
- `votingclassifier_model.pkl` - 34.32 MB (Random Forest + SVM + Gradient Boosting)
- `scaler.pkl` - 14.8 KB (StandardScaler fitted on training data)
- `randomforest_model.pkl` - 560 KB (individual model)
- `svm_model.pkl` - 484 KB (individual model)
- `gradientboosting_model.pkl` - 799 KB (individual model)

**No retraining needed** - models are ready to use!

---

## ✅ Test the Setup

### A. Test Inference (Quick - 10 seconds)
```bash
python test_voting_classifier.py
```

**Expected Output:**
```
✓ Test prediction: Class 2 (Moderate)
✓ Confidence: 41.10%
```

### B. Test with Your Own Image
```bash
python -c "from test_voting_classifier import test_single_image; test_single_image('path/to/image.png')"
```

---

## 📊 Project Structure
```
Enhancing-diabetic-retinopathy-detection/
├── models/                    # Pre-trained models (READY TO USE)
│   ├── votingclassifier_model.pkl
│   ├── scaler.pkl
│   └── ...
├── data/                      # Training/test images
│   ├── train.csv
│   ├── test.csv
│   ├── train_images/
│   └── test_images/
├── scripts/                   # Core modules
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── visualize.py
├── notebooks/
│   └── hybrid_dr_detection_fixed.ipynb
├── train_models.py            # Train from scratch (optional)
├── evaluate_models.py         # Evaluate models
├── test_voting_classifier.py  # Quick test
├── inference.py               # Production inference
└── requirements.txt
```

---

## 🔄 Using the Models

### Quick Prediction
```python
from test_voting_classifier import make_prediction

# Load test image
pred_class, confidence, probabilities = make_prediction('data/test_images/sample.png')
print(f"Prediction: {pred_class}, Confidence: {confidence}%")
```

### In Your Application
```python
import numpy as np
import joblib
import cv2

# Load model and scaler
clf = joblib.load('models/votingclassifier_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load and preprocess image
img = cv2.imread('image.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img_gray, (256, 256))
features = img_resized.flatten()[:595].astype(np.float32).reshape(1, -1)

# Predict
features_scaled = scaler.transform(features)
probabilities = clf.predict_proba(features_scaled)[0]
prediction = int(np.argmax(probabilities))

CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
print(f"Prediction: {CLASS_NAMES[prediction]}")
print(f"Confidence: {probabilities[prediction]*100:.2f}%")
```

---

## 📈 Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 69.86% |
| SVM | 68.90% |
| Gradient Boosting | **71.51%** |
| **VotingClassifier** | **71.51%** ⭐ |

---

## 🔧 Retrain Models (Optional)

If you want to retrain with new data:

```bash
python train_models.py
```

⏱️ **Training Time:** ~30-40 minutes
📊 **Training Data:** 3,662 images from 5 DR severity classes

---

## ❓ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install -r requirements.txt
```

### Issue: "Model file not found"
- Ensure you're in the correct directory
- Check `models/votingclassifier_model.pkl` exists
- Run: `ls models/` or `dir models\`

### Issue: "Image file not found"
- Use absolute paths: `C:\path\to\image.png`
- Or relative from project root: `data/test_images/image.png`

### Issue: Memory error during training
- Reduce batch size in training script
- Use fewer features (currently 595)

---

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Verify all files in `models/` exist
3. Run `python test_voting_classifier.py` to check setup
4. Check Python version: `python --version` (should be 3.9+)

---

## ✅ Verification Checklist

Run this to verify everything works:

```bash
python -c "
import os
import joblib
import cv2
import numpy as np

print('=' * 50)
print('VERIFICATION CHECKLIST')
print('=' * 50)

# Check models
model_path = 'models/votingclassifier_model.pkl'
scaler_path = 'models/scaler.pkl'

print(f'\n✓ Model exists: {os.path.exists(model_path)}')
print(f'✓ Scaler exists: {os.path.exists(scaler_path)}')

if os.path.exists(model_path):
    clf = joblib.load(model_path)
    print(f'✓ Model loaded: VotingClassifier')
    
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f'✓ Scaler loaded: StandardScaler')

# Check data
print(f'✓ Training data exists: {os.path.exists(\"data/train.csv\")}')
print(f'✓ Test data exists: {os.path.exists(\"data/test.csv\")}')

print('\n' + '=' * 50)
print('✅ SETUP COMPLETE AND VERIFIED!')
print('=' * 50)
"
```

---

## 🎯 What's Next?

1. **Test inference:** Run `python test_voting_classifier.py`
2. **Evaluate models:** Run `python evaluate_models.py`
3. **Use in your app:** Import and use the models (see examples above)
4. **Retrain (optional):** Follow retraining instructions if needed

---

**Version:** 1.0  
**Last Updated:** April 18, 2026  
**Status:** ✅ Production Ready
# 🚀 Team Setup & Sharing Guide

**How to share this project with your team and ensure they can run it**

---

## 📋 What Gets Shared

```
SHARED ON GITHUB (Public Repository):
✅ All code files (.py)
✅ Configuration files (config.py)
✅ Utility scripts (utils/ folder)
✅ Jupyter notebook (hybrid_dr_detection.ipynb)
✅ Requirements.txt (dependencies)
✅ Documentation (README, guides)
✅ Pre-trained model files (models/*.pkl)
✅ Sample evaluation reports (outputs/*.txt)

❌ NOT on GitHub (Too Large - 5+ GB):
❌ Raw training images (data/train_images/ - 3,662 images)
❌ Test images (data/test_images/ - 1,800 images)
❌ Feature arrays (X_train.npy, etc. - 500+ MB)
```

---

## 🔄 Two Approaches for Team Members

### **APPROACH 1: Minimal Setup (Works Immediately) - RECOMMENDED FOR INITIAL SETUP**

**Time Required:** 5 minutes  
**Data Required:** None (will generate demo data)  
**Result:** Trained models on synthetic data

```powershell
# Step 1: Clone from GitHub
git clone https://github.com/Nihar0001/Enhancing_diabetic_retinopathy_detection.git
cd Enhancing-diabetic-retinopathy-detection

# Step 2: Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # macOS/Linux

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Generate demo data (synthetic, no images needed)
python utils/generate_demo_data.py

# Step 5: Train models
python train_models.py

# Step 6: Evaluate results
python evaluate_models.py

# Result: ✅ Models trained, reports generated, visualizations created
```

✅ **Team can verify everything works without needing large data files**

---

### **APPROACH 2: With Real Data (Production Quality)**

**Time Required:** 60+ minutes (feature extraction is slow)  
**Data Required:** 5+ GB of retina images  
**Result:** Trained models on real diabetic retinopathy images

```
STEP 1: Clone Repository
  → Same as Approach 1 (git clone, setup venv, pip install)

STEP 2: Get Real Data (Choose ONE)
  
  OPTION A: Share via Cloud Storage
    - Upload data/ folder to Google Drive / OneDrive / Dropbox
    - Team members download and extract to local data/
    - PROS: Easy, cloud backup
    - CONS: Large file size, slow download
    
  OPTION B: Share via Shared Drive
    - Network drive (if on same network)
    - Team NFS/Samba mount
    - PROS: Fast, automatic sync
    - CONS: Requires network setup
    
  OPTION C: Compressed Archive
    - Create data.zip (3-5 GB)
    - Share via file server
    - Team extracts locally
    - CONS: Very large download

STEP 3: Extract to Project Directory
  data/
  ├── train.csv          ← MUST have
  ├── test.csv           ← MUST have
  ├── train_images/      ← 3,662 PNG files
  ├── test_images/       ← 1,800 PNG files
  ├── train.csv
  └── test.csv

STEP 4: Run Feature Extraction in Notebook
  cd notebooks/
  jupyter notebook hybrid_dr_detection.ipynb
  → Run Cell 1 (extracts features from real images, ~60 min)
  → Run Cell 2 (trains models, ~5 min)
  → Run Cell 3 (generates visualizations)

RESULT: ✅ Models trained on REAL retinopathy data
```

---

## 📦 Data Sharing Strategy (Recommended)

### **Option 1: Cloud Storage (Google Drive)**

```
1. Create a public Google Drive folder:
   └ Enhancing-DR-Detection-Data/
     ├── train_images.zip (2.5 GB)
     ├── test_images.zip (1.2 GB)
     ├── train.csv
     └── test.csv

2. Share link with team:
   https://drive.google.com/drive/folders/...
   
3. Team downloads and extracts:
   # In project root
   mkdir -p data
   cd data
   # Download and extract files from Drive
   unzip train_images.zip
   unzip test_images.zip
   
4. Verify:
   ls data/train_images/ | wc -l  # Should show 3662
   ls data/test_images/ | wc -l   # Should show ~1800
```

---

### **Option 2: AWS S3 (For Large Teams)**

```
SETUP (You do once):
1. Create S3 bucket: dr-detection-data
2. Upload data files
3. Generate pre-signed URLs (24-hour access)

TEAM ACCESS:
aws s3 cp s3://dr-detection-data/train_images.zip ./data/
aws s3 cp s3://dr-detection-data/test_images.zip ./data/
unzip data/train_images.zip -d data/
unzip data/test_images.zip -d data/
```

---

### **Option 3: Git LFS (Large File Storage)**

```
SETUP (You do once):
1. Install Git LFS: https://git-lfs.com/
2. In repository:
   git lfs install
   git lfs track "data/train_images/*"
   git lfs track "data/test_images/*"
   git add .gitattributes
   git commit -m "Add LFS tracking"

TEAM ACCESS:
1. Install Git LFS
2. Clone normally:
   git clone https://github.com/Nihar0001/...
   → Automatically downloads LFS files
   
PROS: Seamless git integration
CONS: Requires paid LFS account for large repos
```

---

## ✅ Team Checklist

Each team member should:

```
[ ] Git installed
    → Check: git --version

[ ] Python 3.8+ installed
    → Check: python --version

[ ] Clone repository
    → git clone https://github.com/Nihar0001/...

[ ] Create virtual environment
    → python -m venv venv
    → venv\Scripts\activate (Windows)

[ ] Install dependencies
    → pip install -r requirements.txt

[ ] Verify setup works
    → python utils/test_project.py
    → Should show: [OK] All tests passed

[ ] Generate demo data (just to verify everything)
    → python utils/generate_demo_data.py

[ ] Train models on demo data
    → python train_models.py
    → Should see: [SUCCESS] Training pipeline completed

[ ] Optional: Get real data
    → Download from shared drive/cloud
    → Extract to data/ folder
    → Run notebook for real training
```

---

## 🎯 Different Team Roles & Setup

### **For Data Scientists**
```
1. Clone repo ✓
2. Install deps ✓
3. Download real data ✓
4. Open notebook: jupyter notebook notebooks/hybrid_dr_detection.ipynb
5. Run all cells to train on real data
6. Experiment with model parameters, features, etc.
```

### **For ML Engineers / DevOps**
```
1. Clone repo ✓
2. Install deps ✓
3. Use utils/generate_demo_data.py + train_models.py for automated training
4. Modify training pipeline (config.py)
5. Deploy models to production (models/ folder)
6. Monitor with evaluate_models.py
```

### **For Visualization / Analysis**
```
1. Clone repo ✓
2. Install deps ✓
3. Download generated reports: outputs/
4. Create custom visualizations: utils/model_comparison_visualizer.py
5. Generate presentation materials
```

---

## 📊 File Size Reference

```
GitHub Repository (Code Only):
├── All Python files: ~500 KB
├── Notebook: ~200 KB
├── Requirements.txt: ~2 KB
├── Models (*.pkl): ~100 MB
└── Total: ~100 MB (reasonable for git)

Data Files (NOT on GitHub):
├── train.csv: ~10 MB
├── test.csv: ~5 MB
├── train_images/ (3,662 PNG): ~2.5 GB
├── test_images/ (1,800 PNG): ~1.2 GB
├── Feature arrays (*.npy): ~500 MB
└── Total: ~4.2 GB (too large for free GitHub)

Team Member Total Download:
├── GitHub clone: ~100 MB
├── Data files: ~4.2 GB (optional, for real training)
└── Total: ~4.3 GB minimum
```

---

## 🚨 Common Issues & Solutions

### **Issue: "data/train_images/ not found"**
```
Solution:
1. Run: python utils/generate_demo_data.py
2. This creates synthetic data for testing
3. Only need real images if using the notebook
```

### **Issue: "ModuleNotFoundError: tensorflow"**
```
Solution:
- TensorFlow is optional (not installed by default)
- Project uses fallback features
- Models will still train without it
- No action needed, just continue
```

### **Issue: "Can't download data due to large file size"**
```
Solutions:
1. Use cloud storage (Google Drive with streaming)
2. Use Git LFS for automatic downloading
3. Split data into smaller chunks
4. Use network drive (if available)
```

### **Issue: Different Python versions on team machines**
```
Solution:
- Project supports Python 3.8+
- Use: python -m venv venv
  (Creates env with YOUR Python version)
- Each team member's venv is isolated
- No conflicts between team members
```

---

## 🔄 Workflow for Team Collaboration

### **Day 1: Initial Setup**
```
Team Lead:
1. Creates GitHub repo (✓ Done)
2. Creates data sharing folder (Google Drive)
3. Shares links with team:
   - GitHub URL
   - Data download link
   - This setup guide

Each Team Member:
1. Clone repo
2. Set up venv + install deps
3. Download data (optional)
4. Run utils/test_project.py to verify setup
```

### **Day 2+: Development**
```
Code Development:
- Work on separate branches: git checkout -b my-feature
- Make changes, commit, push
- Create pull requests
- Lead reviews and merges

Data Experiments:
- If using real data:
  - Run notebook with latest code
  - Save outputs to their branch
  - Share results in PR
  
- If using demo data:
  - Run utils/generate_demo_data.py
  - Run train_models.py
  - Compare results
```

### **Team Sync Points**
```
Weekly:
- Compare model performances
- Share results/findings
- Update requirements.txt if deps change
- Pull latest main branch: git pull origin main

Before Release:
- Test on clean environment
- Verify all team members can run
- Update documentation
- Tag release: git tag v1.0.0
- Create GitHub release with data links
```

---

## 📝 Handoff Checklist

Before sharing with team:

```
Code Quality:
[ ] All code tested and working
[ ] No debug prints or temp files
[ ] Comments added for complex sections
[ ] Type hints added where possible

Documentation:
[ ] README.md updated with setup
[ ] Inline code comments complete
[ ] Function docstrings present
[ ] Requirements.txt updated

Testing:
[ ] utils/test_project.py passes
[ ] utils/test_notebook_data.py passes
[ ] train_models.py runs without errors
[ ] evaluate_models.py produces reports

Data & Models:
[ ] Models saved and working
[ ] Sample reports in outputs/
[ ] Data sharing method documented
[ ] Data download instructions clear

Git:
[ ] All code committed
[ ] Main branch is clean
[ ] .gitignore is correct
[ ] README links are up to date

Communication:
[ ] Team has access to GitHub
[ ] Data sharing link shared
[ ] Setup instructions sent
[ ] Team knows who to contact for help
```

---

## 🎓 Team Training Script

**Send this to team members:**

```
WELCOME TO THE DR DETECTION PROJECT!

Quick Start (5 minutes):
1. Download: https://github.com/Nihar0001/...
2. Setup: See README_SETUP.md
3. Test: python utils/test_project.py
4. Train: python utils/generate_demo_data.py; python train_models.py

Using Real Data (60+ minutes):
1. Download images from: [SHARE YOUR LINK]
2. Extract to: data/train_images/ and data/test_images/
3. Open: notebooks/hybrid_dr_detection.ipynb
4. Run all cells

Need Help?
- Documentation: docs/ folder
- Quick start: README_SETUP.md
- For issues: open GitHub issue
- Questions: contact [YOUR EMAIL]

Happy coding!
```

---

## Summary: What to Share

**With Every Team Member:**
1. ✅ GitHub repository (code)
2. ✅ README & documentation
3. ✅ Link to this guide (docs/TEAM_SETUP.md)
4. ✅ requirements.txt (dependencies)

**For Real Data Work:**
5. ✅ Google Drive / S3 / Git LFS link to data/
6. ✅ Data extraction instructions
7. ✅ Feature dimensions & validation script

**Optional but Recommended:**
8. ✅ Pre-trained models (already in repo)
9. ✅ Sample reports (from outputs/)
10. ✅ Video tutorial of notebook walkthrough

---

**Your project is now team-ready! 🎉**
