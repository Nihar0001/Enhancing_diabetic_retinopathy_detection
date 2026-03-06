"""
Model Evaluation and Comparison Script
Loads trained models and evaluates them against test data.
"""

import os
import sys
import numpy as np
import joblib
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from config import (
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, CLASS_NAMES,
    X_TEST_SCALED_FILE, Y_TEST_FILE,
    RF_MODEL_FILE, SVM_MODEL_FILE, KNN_MODEL_FILE,
    VOTING_MODEL_FILE
)

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class DRModelEvaluator:
    """Evaluates and compares trained diabetic retinopathy models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        self.models_to_load = {
            'RandomForest': RF_MODEL_FILE,
            'SVM': SVM_MODEL_FILE,
            'KNN': KNN_MODEL_FILE,
            'VotingClassifier': VOTING_MODEL_FILE
        }
        
        print("\n" + "="*60)
        print("MODEL EVALUATION & COMPARISON")
        print("="*60 + "\n")
    
    def load_test_data(self):
        """Load test data."""
        print("[STEP 1] Loading test data...")
        try:
            self.X_test_scaled = np.load(str(X_TEST_SCALED_FILE))
            self.y_test = np.load(str(Y_TEST_FILE))
            
            print(f"  ✓ Test data loaded")
            print(f"    X_test_scaled shape: {self.X_test_scaled.shape}")
            print(f"    y_test shape: {self.y_test.shape}")
            
            return True
        
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            return False
    
    def load_models(self):
        """Load all trained models."""
        print("\n[STEP 2] Loading models...")
        
        for name, model_path in self.models_to_load.items():
            try:
                if model_path.exists():
                    model = joblib.load(str(model_path))
                    self.results[name] = {'model': model}
                    print(f"  ✓ {name} loaded from {model_path.name}")
                else:
                    print(f"  ⚠ {name} not found at {model_path}")
            
            except Exception as e:
                print(f"  ✗ Error loading {name}: {e}")
    
    def evaluate_models(self):
        """Evaluate all loaded models."""
        print("\n[STEP 3] Evaluating models...")
        
        for name, data in self.results.items():
            try:
                model = data['model']
                
                # Make predictions
                y_pred = model.predict(self.X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(
                    self.y_test, y_pred,
                    average='weighted',
                    zero_division=0
                )
                recall = recall_score(
                    self.y_test, y_pred,
                    average='weighted',
                    zero_division=0
                )
                f1 = f1_score(
                    self.y_test, y_pred,
                    average='weighted',
                    zero_division=0
                )
                cm = confusion_matrix(self.y_test, y_pred)
                
                # Store results
                data['predictions'] = y_pred
                data['accuracy'] = accuracy
                data['precision'] = precision
                data['recall'] = recall
                data['f1_score'] = f1
                data['confusion_matrix'] = cm
                data['classification_report'] = classification_report(
                    self.y_test, y_pred,
                    target_names=CLASS_NAMES,
                    zero_division=0
                )
                
                print(f"\n  {name}:")
                print(f"    Accuracy:  {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall:    {recall:.4f}")
                print(f"    F1-Score:  {f1:.4f}")
            
            except Exception as e:
                print(f"  ✗ Error evaluating {name}: {e}")
    
    def plot_comparison(self):
        """Create comparison visualizations."""
        print("\n[STEP 4] Creating visualizations...")
        
        try:
            # Prepare data
            names = []
            accuracies = []
            f1_scores = []
            
            for name, data in self.results.items():
                if 'accuracy' in data:
                    names.append(name)
                    accuracies.append(data['accuracy'])
                    f1_scores.append(data['f1_score'])
            
            if not names:
                print("  ⚠ No models to visualize")
                return
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy comparison
            axes[0].bar(names, accuracies, color='steelblue', alpha=0.7)
            axes[0].set_ylabel('Accuracy', fontsize=12)
            axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            axes[0].set_ylim([0, 1.0])
            axes[0].grid(axis='y', alpha=0.3)
            for i, v in enumerate(accuracies):
                axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
            
            # F1-Score comparison
            axes[1].bar(names, f1_scores, color='lightcoral', alpha=0.7)
            axes[1].set_ylabel('F1-Score', fontsize=12)
            axes[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
            axes[1].set_ylim([0, 1.0])
            axes[1].grid(axis='y', alpha=0.3)
            for i, v in enumerate(f1_scores):
                axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
            
            plt.tight_layout()
            
            # Save figure
            comparison_path = OUTPUTS_DIR / 'model_comparison.png'
            plt.savefig(str(comparison_path), dpi=300, bbox_inches='tight')
            print(f"  ✓ Comparison plot saved to {comparison_path.name}")
            plt.close()
        
        except Exception as e:
            print(f"  ✗ Error creating visualizations: {e}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        print("\n[STEP 5] Creating confusion matrices...")
        
        try:
            n_models = len([d for d in self.results.values() if 'confusion_matrix' in d])
            
            if n_models == 0:
                print("  ⚠ No models to visualize")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()
            
            for idx, (name, data) in enumerate(self.results.items()):
                if 'confusion_matrix' not in data:
                    continue
                
                cm = data['confusion_matrix']
                
                # Normalize confusion matrix
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Plot
                sns.heatmap(
                    cm_normalized,
                    annot=True,
                    fmt='.2f',
                    cmap='Blues',
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,
                    ax=axes[idx],
                    cbar_kws={'label': 'Normalized Count'}
                )
                
                axes[idx].set_title(f'{name} Confusion Matrix', fontweight='bold')
                axes[idx].set_ylabel('True Label', fontsize=11)
                axes[idx].set_xlabel('Predicted Label', fontsize=11)
            
            # Hide unused subplots
            for idx in range(len(self.results), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            cm_path = OUTPUTS_DIR / 'confusion_matrices.png'
            plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
            print(f"  ✓ Confusion matrices saved to {cm_path.name}")
            plt.close()
        
        except Exception as e:
            print(f"  ✗ Error creating confusion matrices: {e}")
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        print(f"\n{'Rank':<6}{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
        print("-" * 60)
        
        for rank, (name, data) in enumerate(sorted_results, 1):
            if 'accuracy' in data:
                print(f"{rank:<6}{name:<20}{data['accuracy']:<12.4f}"
                      f"{data['precision']:<12.4f}{data['recall']:<12.4f}"
                      f"{data['f1_score']:<12.4f}")
        
        print("="*60)
        
        # Print best model
        if sorted_results:
            best_name, best_data = sorted_results[0]
            print(f"\n🏆 Best Model: {best_name}")
            print(f"   Accuracy: {best_data['accuracy']:.4f}")
            print(f"   F1-Score: {best_data['f1_score']:.4f}")
    
    def run(self):
        """Run the complete evaluation pipeline."""
        try:
            if not self.load_test_data():
                return False
            
            self.load_models()
            
            if not self.results:
                print("\n[ERROR] No models were loaded")
                return False
            
            self.evaluate_models()
            self.plot_comparison()
            self.plot_confusion_matrices()
            self.print_summary()
            
            print("\n[SUCCESS] Evaluation completed successfully!")
            print(f"Results saved in: {OUTPUTS_DIR}")
            
            return True
        
        except Exception as e:
            print(f"\n[ERROR] Evaluation failed: {e}")
            return False


if __name__ == "__main__":
    evaluator = DRModelEvaluator()
    success = evaluator.run()
    sys.exit(0 if success else 1)
