"""
Demo Data Generator for Testing Without Real Data
Generates synthetic training/test data for demonstration and testing purposes.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, create_directories

def generate_demo_data(n_train=100, n_test=20, n_features=595, n_classes=5):
    """
    Generate synthetic data for testing without real data.
    
    This is useful for:
    - Testing the pipeline without real data
    - Demo purposes
    - Verifying code works before using real data
    
    Args:
        n_train (int): Number of training samples (default: 100)
        n_test (int): Number of test samples (default: 20)
        n_features (int): Number of features per sample (default: 595)
        n_classes (int): Number of classes (default: 5)
    
    Returns:
        dict: Dictionary with generated data arrays
    """
    
    print("\n" + "="*60)
    print("GENERATING DEMO DATA")
    print("="*60)
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Features: {n_features}")
    print(f"Classes: {n_classes}")
    
    # Create directories
    create_directories()
    
    # Generate synthetic features
    print("\n[1] Generating training features...")
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.randint(0, n_classes, n_train)
    
    print(f"     X_train shape: {X_train.shape}")
    print(f"     y_train shape: {y_train.shape}")
    print(f"     Class distribution: {np.bincount(y_train)}")
    
    # Generate test features
    print("\n[2] Generating test features...")
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_test = np.random.randint(0, n_classes, n_test)
    
    print(f"     X_test shape: {X_test.shape}")
    print(f"     y_test shape: {y_test.shape}")
    print(f"     Class distribution: {np.bincount(y_test)}")
    
    # Scale features (mimic real preprocessing)
    print("\n[3] Scaling features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"     X_train_scaled shape: {X_train_scaled.shape}")
    print(f"     X_test_scaled shape: {X_test_scaled.shape}")
    
    # Save data files
    print("\n[4] Saving data files...")
    
    data_files = {
        'X_train.npy': X_train,
        'X_test.npy': X_test,
        'y_train.npy': y_train,
        'y_test.npy': y_test,
        'X_train_scaled.npy': X_train_scaled,
        'X_test_scaled.npy': X_test_scaled
    }
    
    for filename, data in data_files.items():
        filepath = DATA_DIR / filename
        np.save(str(filepath), data)
        print(f"     [OK] Saved: {filename}")
    
    print("\n" + "="*60)
    print("[SUCCESS] DEMO DATA GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nData files saved to: {DATA_DIR}")
    print("\nYou can now run:")
    print("  python train_models.py")
    print("  python evaluate_models.py")
    print("\nNote: This is synthetic data for TESTING ONLY!")
    print("      Replace with real data for actual training.")
    print("="*60 + "\n")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }


if __name__ == "__main__":
    try:
        data = generate_demo_data(
            n_train=100,      # Small demo dataset
            n_test=20,
            n_features=595,   # Matches real feature vector size
            n_classes=5       # 5 DR classes
        )
        print("[SUCCESS] Demo data ready for testing!")
    except Exception as e:
        print(f"[ERROR] Failed to generate demo data: {e}")
        import traceback
        traceback.print_exc()
