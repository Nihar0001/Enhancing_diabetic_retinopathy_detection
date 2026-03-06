# ⚡ QUICK START - 5 Minutes to Running

**Want to get started immediately?** Follow this guide.

---

## 🚀 Step 1: Setup (2 minutes)

```bash
# Navigate to project
cd Enhancing-diabetic-retinopathy-detection

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## ✅ Step 2: Verify (1 minute)

```bash
# Test environment setup
python setup_env.py

# Expected output:
# [INFO] Project root set to: ...
# [INFO] ✓ All required paths are valid
# [SUCCESS] Environment setup complete!
```

---

## 🎯 Step 3: Run the Pipelines (2 minutes)

### **Train Models** (requires preprocessed data)
```bash
python train_models.py
```

**Expected Output:**
```
[STEP 1] Loading data...
  ✓ X_train_scaled shape: (3512, 595)
  ✓ X_test_scaled shape: (878, 595)

[STEP 2] Training individual models...
  Training RandomForest...
    ✓ Accuracy: 0.8523
    ✓ F1-Score (weighted): 0.8401
    ✓ Model saved to randomforest_model.pkl
  ...

[STEP 3] Training Voting Classifier...
  ✓ Voting Classifier Accuracy: 0.8634
  ✓ Voting Classifier F1-Score: 0.8512
  ✓ Model saved to votingclassifier_model.pkl

TRAINING SUMMARY
1. VotingClassifier - Accuracy: 0.8634
2. RandomForest - Accuracy: 0.8523
3. SVM - Accuracy: 0.7845
4. KNN - Accuracy: 0.7234

[SUCCESS] Training pipeline completed successfully!
```

### **Evaluate Models**
```bash
python evaluate_models.py
```

**Expected Output:**
```
[STEP 1] Loading test data...
  ✓ Test data loaded

[STEP 2] Loading models...
  ✓ RandomForest loaded
  ✓ SVM loaded
  ✓ KNN loaded
  ✓ VotingClassifier loaded

[STEP 3] Evaluating models...
  RandomForest:
    Accuracy:  0.8523
    Precision: 0.8401
    Recall:    0.8523
    F1-Score:  0.8401

[STEP 4] Creating visualizations...
  ✓ Comparison plot saved
  ✓ Confusion matrices saved

🏆 Best Model: VotingClassifier
   Accuracy: 0.8634
   F1-Score: 0.8512

[SUCCESS] Evaluation completed successfully!
```

---

## 📊 Check Your Results

After running the pipelines, check:

```bash
# Training outputs
ls models/          # *.pkl model files
ls outputs/         # .txt reports
ls outputs/         # .png visualizations
```

**Generated Files:**
- `models/randomforest_model.pkl` - Trained model
- `models/svm_model.pkl` - Trained model
- `models/knn_model.pkl` - Trained model
- `models/votingclassifier_model.pkl` - Ensemble model
- `outputs/randomforest_report.txt` - Performance report
- `outputs/model_comparison.png` - Accuracy chart
- `outputs/confusion_matrices.png` - Confusion matrices

---

## 💡 Key Files to Know

| File | Purpose | Run? |
|------|---------|------|
| `setup_env.py` | Verify environment | `python setup_env.py` ✓ |
| `train_models.py` | Train models | `python train_models.py` ✓ |
| `evaluate_models.py` | Evaluate & compare | `python evaluate_models.py` ✓ |
| `config.py` | All settings | Reference only |
| `scripts/` | Core functions | Import as needed |
| `notebooks/` | Interactive analysis | Edit and run cells |

---

## 🔧 Common Commands

```bash
# Check project configuration
python config.py

# View recent changes
git log --oneline

# Check what's been modified
git status

# See all improvements made
cat CHANGELOG.md
```

---

## 📚 Learn More

Read these docs in order:

1. **readme.md** - High-level overview
2. **IMPLEMENTATION_GUIDE.md** - Detailed usage (20 min read)
3. **CHANGELOG.md** - What was changed (10 min read)
4. Code comments in `train_models.py` and `evaluate_models.py`

---

## 🆘 Something Went Wrong?

### Error: "Module not found"
```bash
# Make sure you're in correct directory
cd Enhancing-diabetic-retinopathy-detection

# Reinstall packages
pip install -r requirements.txt

# Run setup again
python setup_env.py
```

### Error: "Data files not found"
```bash
# Check what's missing
python config.py

# Ensure these exist:
# - data/train.csv
# - data/train_images/
# - data/X_train_scaled.npy
# - data/X_test_scaled.npy
```

### Error: "TensorFlow issues"
```bash
# Already handled! Warnings are suppressed.
# If still seeing issues, reinstall TensorFlow:
pip install --upgrade tensorflow==2.13.0
```

---

## ✨ What's New in This Version?

✅ **Fixed**: TensorFlow deprecation warnings  
✅ **Added**: Central configuration system  
✅ **Added**: Professional training pipeline  
✅ **Added**: Comprehensive evaluation suite  
✅ **Added**: Complete documentation  
✅ **Improved**: Error handling throughout  
✅ **Improved**: Code quality & organization  

---

## 🎯 Next Steps

1. **Verify everything works** → Run all commands above
2. **Read IMPLEMENTATION_GUIDE** → Understand the new structure
3. **Upload to GitHub** → Share with team
4. **Discuss results** → Check model performance

---

## 📞 Need Help?

- Check **IMPLEMENTATION_GUIDE.md** → Troubleshooting section
- View **Git history** → `git log --oneline`
- Read **Code comments** → Detailed documentation
- Check **readme.md** → Project overview

---

**Status**: ✅ Ready to use  
**Version**: 1.0.0  
**Last Updated**: March 6, 2026

---

*This is your starting point. As you work, you'll need IMPLEMENTATION_GUIDE.md for detailed operations.*
