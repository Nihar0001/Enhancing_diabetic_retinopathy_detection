# Enhancing Diabetic Retinopathy Detection

Hybrid diabetic retinopathy classification pipeline using engineered image features and an ensemble of classical ML models.

## What this project does
- Preprocesses retinal images and extracts features.
- Trains multiple classifiers: RandomForest, SVM, HistGradientBoosting.
- Trains a soft-voting ensemble.
- Generates reports and comparison visualizations.

## Project structure
- `config.py`: central paths and training settings
- `train_models.py`: training pipeline (saves models/reports)
- `evaluate_models.py`: evaluation and comparison plots
- `scripts/`: preprocessing, feature extraction, visualization helpers
- `utils/`: setup and validation utilities
- `notebooks/`: exploratory notebook workflow
- `data/`: input CSV/images and cached feature arrays
- `models/`: saved `.pkl` models
- `outputs/updated/`: latest reports and plots

## Run from scratch

### 1) Clone and set up environment
```bash
git clone <your-repo-url>
cd Enhancing-diabetic-retinopathy-detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Prepare data
Place files under `data/`:
- `train.csv`, `test.csv`
- `train_images/`, `test_images/`

Data setup references:
- `docs/GOOGLE_DRIVE_SETUP.md` (detailed)
- `docs/DOWNLOAD_DATA.md` (quick reference)

### 3) Quick verification (recommended)
```bash
python utils/test_project.py
```

### 4) If you do not have real feature arrays yet
Generate demo arrays:
```bash
python utils/generate_demo_data.py
```

### 5) Train models
```bash
python train_models.py
```

### 6) Evaluate models
```bash
python evaluate_models.py
```

Outputs are written to `outputs/updated/`.

## Notebook usage
Use `notebooks/hybrid_dr_detection.ipynb` for interactive experimentation. For reproducible/team workflows, prefer `train_models.py` and `evaluate_models.py`.

## Team handoff checklist
- Run `python utils/test_project.py`
- Run `python train_models.py`
- Run `python evaluate_models.py`
- Confirm models exist in `models/`
- Confirm reports/plots exist in `outputs/updated/`

## Notes
- TensorFlow is optional in this setup; fallback feature behavior is supported where applicable.
- If paths fail, run commands from the project root.
