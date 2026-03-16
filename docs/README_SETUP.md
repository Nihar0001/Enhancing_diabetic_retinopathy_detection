# 🚀 Setup & Getting Started Guide

**Get the project running in 5 minutes!**

---

## 📋 Prerequisites

- **Python 3.8 or higher** (check with `python --version`)
- **pip** (should come with Python)
- **Git** (to clone the repository)

---

## ⚡ Quick Setup (5 minutes)

### **Step 1: Clone the Repository**
```bash
git clone <your-repo-url>
cd Enhancing-diabetic-retinopathy-detection
```

### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate    # Windows
# or
source venv/bin/activate # macOS/Linux
```

**You should see `(venv)` at the start of your terminal line**

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Expected output:** `Successfully installed sklearn tensorflow pandas...`

### **Step 4: Verify Setup Works**
```bash
python test_project.py
```

**Expected:** Most tests pass (remaining failures are just optional packages)

### **Step 5: Get Data**

#### Option A: Download from Google Drive (Recommended for real training)
```bash
# 1. Visit the data folder on Google Drive:
#    https://drive.google.com/drive/folders/1lKMGO2NrZ67wH5LJkvFxRpF75I89CgZt?usp=drive_link
# 2. Download and extract all files into the data/ folder
# 3. Your data/ folder should contain: train_images/, test_images/, *.csv, *.npy files
```
> 📖 See [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md) for a step-by-step guide.

#### Option B: Generate Demo Data
```bash
python utils/generate_demo_data.py
```

**Expected output:**
```
============================================================
GENERATING DEMO DATA
============================================================
Training samples: 100
Test samples: 20
...
✅ DEMO DATA GENERATED SUCCESSFULLY!
============================================================
```

### **Step 6: Test Training Pipeline**
```bash
python train_models.py
```

**Expected:** Models train successfully (accuracy will be low since it's synthetic data)

---

## 📁 Project Structure

```
Enhancing-diabetic-retinopathy-detection/
├── venv/                      # Virtual environment (created in step 2)
├── data/                      # Training data (created automatically)
│   ├── X_train.npy           # Training features
│   ├── X_test.npy            # Test features
│   ├── y_train.npy           # Training labels
│   └── y_test.npy            # Test labels
├── models/                     # Trained models saved here
│   ├── randomforest_model.pkl
│   ├── svm_model.pkl
│   └── voting_model.pkl
├── outputs/                    # Reports and results
│   ├── randomforest_report.txt
│   ├── svm_report.txt
│   └── evaluation_results.json
├── scripts/                    # Core code
│   ├── config.py              # Configuration
│   ├── train_models.py        # Training pipeline
│   ├── evaluate_models.py     # Evaluation
│   └── ...
├── notebooks/                  # Jupyter notebooks
├── requirements.txt           # Python dependencies
├── generate_demo_data.py      # Create synthetic data
│
└── README_SETUP.md           # This file!
    WORKING_WITHOUT_DATA.md   # Detailed guide for working without real data
```

---

## ✅ Verify Everything Works

Run the test suite:
```bash
python test_project.py
```

If you see:
```
Configuration test ........................... PASSED ✓
Directory creation test ...................... PASSED ✓
Demo data generation test .................... PASSED ✓
...
```

**Congratulations! Setup is complete! ✅**

---

## 🎯 Next Steps

### **If you have real data:**
1. Place it in the `data/` folder
2. Files must be named: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
3. Run training: `python train_models.py`

### **If you don't have data yet:**
1. ✅ You already have demo data (from step 5 above)
2. Run training: `python train_models.py`
3. Review results: `cat outputs/randomforest_report.txt`
4. See **WORKING_WITHOUT_DATA.md** for complete guide

### **To train models:**
```bash
python train_models.py
```

### **To evaluate models:**
```bash
python evaluate_models.py
```

---

## 🔧 Common Issues & Solutions

### **Issue: "python command not found"**
```
Solution: Python is not in your PATH
- Reinstall Python
- Check "Add Python to PATH" during installation
- Or use `python3` instead of `python`
```

### **Issue: "No module named 'sklearn'"**
```
Solution: Dependencies not installed
- Run: pip install -r requirements.txt
- Or: pip install scikit-learn
```

### **Issue: "Data files not found"**
```
Solution: Demo data not generated
- Run: python generate_demo_data.py
- Or place real data in data/ folder
- See: WORKING_WITHOUT_DATA.md
```

### **Issue: "Permission denied" (macOS/Linux)**
```
Solution: Virtual environment not activated
- Run: source venv/bin/activate
- Should see (venv) at start of terminal line
```

### **Issue: "ModuleNotFoundError: No module named 'tensorflow'"**
```
Solution: Optional package not installed (OK to skip for now)
- Run: pip install tensorflow
- Not required for basic functionality
- See requirements_optional.txt for advanced features
```

---

## 📚 Understanding the Workflow

### **Data Flow:**
```
Real Data (if available)
    ↓
generate_demo_data.py (if no real data)
    ↓
data/ folder (contains npy files)
    ↓
train_models.py (training)
    ↓
models/ folder (trained models)
    ↓
evaluate_models.py (evaluation)
    ↓
outputs/ folder (reports & results)
```

### **Training Folder Structure:**
```
[1] Load Data
    - X_train.npy, X_test.npy
    - y_train.npy, y_test.npy
    ↓
[2] Train Individual Models
    - RandomForest
    - SVM
    - KNN
    ↓
[3] Train Ensemble
    - Voting Classifier
    ↓
[4] Evaluate & Save
    - Generate reports
    - Save models
```

---

## 🐍 Python Environment Management

### **Check if virtual environment is activated:**
```bash
# Should show path to venv
which python    # macOS/Linux
where python    # Windows
```

### **Deactivate virtual environment:**
```bash
deactivate
```

### **Reactivate virtual environment:**
```bash
venv\Scripts\activate    # Windows
source venv/bin/activate # macOS/Linux
```

### **Update pip (if getting warnings):**
```bash
python -m pip install --upgrade pip
```

---

## 📊 View Training Results

### **After training, view reports:**
```bash
# View RandomForest report
cat outputs/randomforest_report.txt

# View SVM report
cat outputs/svm_report.txt

# View summary
cat outputs/voting_report.txt
```

### **Or on Windows (with PowerShell):**
```powershell
Get-Content outputs/randomforest_report.txt
```

---

## 🚀 Commands Cheat Sheet

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Generate data
python generate_demo_data.py

# Test
python test_project.py

# Train models
python train_models.py

# Evaluate
python evaluate_models.py

# View results
cat outputs/randomforest_report.txt
```

---

## 📖 For More Information

- **Working without real data?** → See [WORKING_WITHOUT_DATA.md](WORKING_WITHOUT_DATA.md)
- **Need feature extraction?** → Check `notebooks/feature_extraction.ipynb`
- **Configuration details?** → See `scripts/config.py`
- **Want to understand models?** → See `scripts/train_models.py`

---

## ✨ You're Ready!

**Your project is set up and ready to use! 🎉**

Next: 
1. Run training with demo data
2. Review the results
3. Add your real data when ready
4. Run training again with real data

**Happy coding! 💻**

---

## 📞 Troubleshooting Checklist

- [ ] Python 3.8+ installed? (`python --version`)
- [ ] Virtual environment created and activated? (`(venv)` visible)
- [ ] Dependencies installed? (`pip install -r requirements.txt`)
- [ ] Demo data generated? (`python generate_demo_data.py`)
- [ ] Training runs? (`python train_models.py`)
- [ ] Can see reports? (`cat outputs/randomforest_report.txt`)

If all checked, you're good to go! ✅

---

**Last updated:** 2024
