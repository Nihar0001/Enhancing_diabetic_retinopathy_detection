"""
Configuration module for Diabetic Retinopathy Detection project.
Manages paths, hyperparameters, and project settings.
"""

from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# ===================== PATHS =====================
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "updated"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Feature cache files
X_FEATURES_FILE = DATA_DIR / "X_features.npy"
Y_LABELS_FILE = DATA_DIR / "y_labels.npy"
X_TRAIN_FILE = DATA_DIR / "X_train.npy"
X_TEST_FILE = DATA_DIR / "X_test.npy"
X_TRAIN_SCALED_FILE = DATA_DIR / "X_train_scaled.npy"
X_TEST_SCALED_FILE = DATA_DIR / "X_test_scaled.npy"
Y_TRAIN_FILE = DATA_DIR / "y_train.npy"
Y_TEST_FILE = DATA_DIR / "y_test.npy"

# Model files
RF_MODEL_FILE = MODELS_DIR / "randomforest_model.pkl"
SVM_MODEL_FILE = MODELS_DIR / "svm_model.pkl"
GB_MODEL_FILE = MODELS_DIR / "gradientboosting_model.pkl"
VOTING_MODEL_FILE = MODELS_DIR / "votingclassifier_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"

# ===================== HYPERPARAMETERS =====================
# Image preprocessing
IMAGE_TARGET_SIZE = (256, 256)
IMAGE_TARGET_SIZE_ADVANCED = (224, 224)

# Model hyperparameters
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_RANDOM_STATE = 42
RANDOM_FOREST_N_JOBS = -1

SVM_KERNEL = 'linear'
SVM_RANDOM_STATE = 42
SVM_PROBABILITY = True

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===================== CLASS LABELS =====================
CLASS_NAMES = ['0', '1', '2', '3', '4']
NUM_CLASSES = len(CLASS_NAMES)

# ===================== UTILITY FUNCTIONS =====================
def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def validate_paths():
    """Validate that required paths exist."""
    required_data_files = [TRAIN_CSV, TRAIN_IMAGES_DIR]
    
    missing_files = []
    for file_path in required_data_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print("[WARNING] Missing required data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    return True


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Outputs Directory: {OUTPUTS_DIR}")
    print(f"\nClass Names: {CLASS_NAMES}")
    print(f"Number of Classes: {NUM_CLASSES}")
