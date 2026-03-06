"""
Improved Model Training Pipeline for Diabetic Retinopathy Detection

This script handles the complete training pipeline with:
- Proper path management using config.py
- Enhanced error handling
- Better model evaluation
- Progress tracking
"""

import os
import sys
import numpy as np
import joblib
import warnings
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE,
    X_TRAIN_SCALED_FILE, X_TEST_SCALED_FILE,
    RF_MODEL_FILE, SVM_MODEL_FILE, KNN_MODEL_FILE, 
    VOTING_MODEL_FILE, SCALER_FILE,
    CLASS_NAMES, RANDOM_STATE, TEST_SIZE,
    RANDOM_FOREST_N_ESTIMATORS, SVM_KERNEL, KNN_N_NEIGHBORS,
    create_directories, validate_paths
)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class DRModelTrainer:
    """Handles the complete diabetic retinopathy model training pipeline."""
    
    def __init__(self):
        """Initialize the trainer with configuration."""
        self.project_root = PROJECT_ROOT
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.outputs_dir = OUTPUTS_DIR
        
        # Create necessary directories
        create_directories()
        
        # Validate paths
        if not validate_paths():
            print("[WARNING] Some data files might be missing")
        
        self.classifiers = {}
        self.results = {}
        
        print("\n" + "="*60)
        print("DIABETIC RETINOPATHY MODEL TRAINER")
        print("="*60)
        print(f"Project Root: {self.project_root}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Models Directory: {self.models_dir}")
        print("="*60 + "\n")
    
    def load_data(self):
        """Load training and testing data."""
        print("[STEP 1] Loading data...")
        try:
            # Check if data files exist
            missing_files = []
            required_files = {
                'X_train.npy': X_TRAIN_FILE,
                'X_test.npy': X_TEST_FILE,
                'y_train.npy': Y_TRAIN_FILE,
                'y_test.npy': Y_TEST_FILE,
                'X_train_scaled.npy': X_TRAIN_SCALED_FILE,
                'X_test_scaled.npy': X_TEST_SCALED_FILE,
            }
            
            for name, path in required_files.items():
                if not path.exists():
                    missing_files.append(name)
            
            if missing_files:
                print(f"\n  ✗ Missing data files:")
                for f in missing_files:
                    print(f"    - {f}")
                print(f"\n  📋 SOLUTION: Generate demo data first!")
                print(f"     Run: python generate_demo_data.py")
                print(f"\n     After that, run this script again.")
                print(f"\n     📖 See WORKING_WITHOUT_DATA.md for complete guide")
                return False
            
            # Load the data
            print(f"  Loading scaled data...")
            self.X_train_scaled = np.load(str(X_TRAIN_SCALED_FILE))
            self.X_test_scaled = np.load(str(X_TEST_SCALED_FILE))
            self.X_train = np.load(str(X_TRAIN_FILE))
            self.X_test = np.load(str(X_TEST_FILE))
            self.y_train = np.load(str(Y_TRAIN_FILE))
            self.y_test = np.load(str(Y_TEST_FILE))
            
            print(f"  [OK] X_train_scaled shape: {self.X_train_scaled.shape}")
            print(f"  [OK] X_test_scaled shape: {self.X_test_scaled.shape}")
            print(f"  [OK] y_train shape: {self.y_train.shape}")
            print(f"  [OK] y_test shape: {self.y_test.shape}")
            print(f"  [OK] Unique classes in y_test: {np.unique(self.y_test)}")
            
            return True
        
        except Exception as e:
            print(f"  [ERROR] Error loading data: {e}")
            print(f"\n  📋 TROUBLESHOOTING:")
            print(f"     1. Generate demo data: python generate_demo_data.py")
            print(f"     2. Check file shapes are correct (n_samples, 595)")
            print(f"     3. Check labels are in range 0-4")
            print(f"\n     📖 See WORKING_WITHOUT_DATA.md for help")
            return False
    
    def train_individual_models(self):
        """Train individual classifiers."""
        print("\n[STEP 2] Training individual models...")
        
        if not hasattr(self, 'X_train_scaled'):
            print("  [ERROR] Data not loaded. Skipping model training.")
            return
        
        # Define classifiers
        self.classifiers = {
            "RandomForest": RandomForestClassifier(
                n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0  # Changed to 0 to reduce output
            ),
            "SVM": SVC(
                kernel=SVM_KERNEL,
                probability=True,
                random_state=RANDOM_STATE,
                verbose=0  # Changed to 0
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=KNN_N_NEIGHBORS
            ),
        }
        
        for name, clf in self.classifiers.items():
            print(f"\n  Training {name}...")
            
            try:
                # Use scaled data for SVM and KNN, unscaled for RandomForest
                if name in ["SVM", "KNN"]:
                    X_train = self.X_train_scaled
                    X_test = self.X_test_scaled
                else:
                    X_train = self.X_train
                    X_test = self.X_test
                
                # Train model
                clf.fit(X_train, self.y_train)
                
                # Predict
                y_pred = clf.predict(X_test)
                
                # Evaluate
                accuracy = accuracy_score(self.y_test, y_pred)
                f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
                
                print(f"    [OK] Accuracy: {accuracy:.4f}")
                print(f"    [OK] F1-Score (weighted): {f1_weighted:.4f}")
                
                # Store results
                self.results[name] = {
                    'model': clf,
                    'accuracy': accuracy,
                    'f1_score': f1_weighted,
                    'predictions': y_pred,
                    'cm': confusion_matrix(self.y_test, y_pred)
                }
                
                # Save model
                model_path = {
                    'RandomForest': RF_MODEL_FILE,
                    'SVM': SVM_MODEL_FILE,
                    'KNN': KNN_MODEL_FILE
                }[name]
                
                joblib.dump(clf, str(model_path))
                print(f"    [OK] Model saved to {model_path.name}")
                
            except Exception as e:
                print(f"    [ERROR] Error training {name}: {e}")
                print(f"       Continuing with other models...")
                continue
    
    def train_voting_classifier(self):
        """Train the ensemble voting classifier."""
        print("\n[STEP 3] Training Voting Classifier...")
        
        try:
            voting_clf = VotingClassifier(
                estimators=[
                    ('rf', self.classifiers['RandomForest']),
                    ('svm', self.classifiers['SVM']),
                    ('knn', self.classifiers['KNN'])
                ],
                voting='soft'
            )
            
            # Train on scaled data (SVM and KNN need it)
            voting_clf.fit(self.X_train_scaled, self.y_train)
            
            # Predict
            y_pred = voting_clf.predict(self.X_test_scaled)
            
            # Evaluate
            accuracy = accuracy_score(self.y_test, y_pred)
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
            
            print(f"  [OK] Voting Classifier Accuracy: {accuracy:.4f}")
            print(f"  [OK] Voting Classifier F1-Score: {f1_weighted:.4f}")
            
            # Store results
            self.results['VotingClassifier'] = {
                'model': voting_clf,
                'accuracy': accuracy,
                'f1_score': f1_weighted,
                'predictions': y_pred,
                'cm': confusion_matrix(self.y_test, y_pred)
            }
            
            # Save model
            joblib.dump(voting_clf, str(VOTING_MODEL_FILE))
            print(f"  [OK] Model saved to {VOTING_MODEL_FILE.name}")
            
        except Exception as e:
            print(f"  [ERROR] Error training voting classifier: {e}")
    
    def generate_reports(self):
        """Generate classification reports."""
        print("\n[STEP 4] Generating reports...")
        
        for name, result in self.results.items():
            try:
                y_pred = result['predictions']
                
                # Classification report
                report = classification_report(
                    self.y_test, y_pred,
                    target_names=CLASS_NAMES,
                    output_dict=False
                )
                
                print(f"\n  {name} Classification Report:")
                print(report)
                
                # Save report
                report_file = self.outputs_dir / f"{name.lower()}_report.txt"
                with open(str(report_file), 'w') as f:
                    f.write(f"Classification Report for {name}\n")
                    f.write("="*60 + "\n")
                    f.write(report)
                    f.write("\n" + "="*60 + "\n")
                    f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                    f.write(f"F1-Score (weighted): {result['f1_score']:.4f}\n")
                
                print(f"  [OK] Report saved to {report_file.name}")
                
            except Exception as e:
                print(f"  [ERROR] Error generating report for {name}: {e}")
    
    def print_summary(self):
        """Print training summary."""
        if not self.results:
            print("\n" + "="*60)
            print("NO TRAINING RESULTS")
            print("="*60)
            print("No models were successfully trained.")
            print("="*60)
            return
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        # Sort models by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            print(f"{rank}. {name}")
            print(f"   Accuracy: {result['accuracy']:.4f}")
            print(f"   F1-Score: {result['f1_score']:.4f}")
        
        print("="*60)
    
    def run(self):
        """Run the complete training pipeline."""
        try:
            if not self.load_data():
                print("\n" + "="*60)
                print("DATA LOADING FAILED")
                print("="*60)
                print("\n🚀 NEXT STEPS:")
                print("   1. Generate demo data:")
                print("      python generate_demo_data.py")
                print("\n   2. Run training again:")
                print("      python train_models.py")
                print("\n   3. Read the complete guide:")
                print("      See WORKING_WITHOUT_DATA.md")
                print("\n" + "="*60)
                return False
            
            self.train_individual_models()
            self.train_voting_classifier()
            self.generate_reports()
            self.print_summary()
            
            print("\n[SUCCESS] Training pipeline completed successfully!")
            print(f"Models saved in: {self.models_dir}")
            print(f"Reports saved in: {self.outputs_dir}")
            print("\n📖 View results with: cat outputs/randomforest_report.txt")
            
            return True
        
        except Exception as e:
            print(f"\n[ERROR] Training pipeline failed: {e}")
            print(f"\n[TIP] Try running: python generate_demo_data.py")
            return False


if __name__ == "__main__":
    trainer = DRModelTrainer()
    success = trainer.run()
    sys.exit(0 if success else 1)
