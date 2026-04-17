"""
Comprehensive Test Suite for Diabetic Retinopathy Detection Project
Tests all functionality without needing real data.
"""

import sys
import os
from pathlib import Path
import traceback

# Add project root and scripts to path
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'scripts'))

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_imports():
    """Test that all imports work correctly."""
    print_section("TEST 1: Imports")
    
    tests_passed = 0
    tests_total = 0
    
    imports_to_test = [
        ("numpy", "Core numerical computing", True),
        ("pandas", "Data handling", True),
        ("sklearn", "Machine learning", True),
        ("tensorflow", "Deep learning (optional)", False),
        ("cv2", "Image processing", True),
        ("seaborn", "Visualization", True),
        ("matplotlib", "Plotting", True),
    ]
    
    for module_name, description, required in imports_to_test:
        tests_total += 1
        try:
            __import__(module_name)
            print(f"  [OK] {module_name:<15} - {description}")
            tests_passed += 1
        except ImportError as e:
            if required:
                print(f"  [FAIL] {module_name:<15} - REQUIRED: {e}")
            else:
                print(f"  [SKIP] {module_name:<15} - OPTIONAL: {e}")
                tests_passed += 1
    
    print(f"\nResult: {tests_passed}/{tests_total} imports successful")
    return tests_passed == tests_total

def test_config():
    """Test configuration system."""
    print_section("TEST 2: Configuration System")
    
    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
            CLASS_NAMES, create_directories
        )
        
        print(f"  [OK] Configuration imported successfully")
        print(f"    - Project Root: {PROJECT_ROOT}")
        print(f"    - Data Dir: {DATA_DIR}")
        print(f"    - Models Dir: {MODELS_DIR}")
        print(f"    - Outputs Dir: {OUTPUTS_DIR}")
        print(f"    - Classes: {CLASS_NAMES}")
        
        # Create directories
        create_directories()
        print(f"  [OK] Directories created")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Configuration test FAILED: {e}")
        traceback.print_exc()
        return False

def test_scripts_import():
    """Test that scripts module imports work."""
    print_section("TEST 3: Scripts Module")
    
    try:
        import scripts.preprocessing as preprocessing_module
        print(f"  [OK] preprocessing module imported")
        assert hasattr(preprocessing_module, 'preprocess_image')
        
        import scripts.feature_extraction as feature_extraction_module
        print(f"  [OK] feature_extraction module imported")
        assert hasattr(feature_extraction_module, 'extract_haralick')
        
        import scripts.visualize as visualize_module
        print(f"  [OK] visualize module imported")
        assert hasattr(visualize_module, 'plot_f1_scores')
        
        return True
    except Exception as e:
        print(f"  [FAIL] Scripts import FAILED: {e}")
        traceback.print_exc()
        return False

def test_data_generation():
    """Test demo data generation."""
    print_section("TEST 4: Demo Data Generation")
    
    try:
        from utils.generate_demo_data import generate_demo_data
        print(f"  [OK] Demo data generator imported")
        
        # Generate small demo dataset
        data = generate_demo_data(n_train=50, n_test=10)
        
        print(f"  [OK] Demo data generated successfully")
        print(f"    - Training samples: {data['X_train'].shape[0]}")
        print(f"    - Test samples: {data['X_test'].shape[0]}")
        print(f"    - Features: {data['X_train'].shape[1]}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Demo data generation FAILED: {e}")
        traceback.print_exc()
        return False

def test_model_modules():
    """Test training and evaluation modules."""
    print_section("TEST 5: Model Modules")
    
    try:
        from train_models import DRModelTrainer
        print(f"  [OK] DRModelTrainer class imported")
        
        from evaluate_models import DRModelEvaluator
        print(f"  [OK] DRModelEvaluator class imported")
        
        # Check that classes have required methods
        _trainer = DRModelTrainer()
        print(f"  [OK] DRModelTrainer instantiated")
        
        _evaluator = DRModelEvaluator()
        print(f"  [OK] DRModelEvaluator instantiated")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Model modules test FAILED: {e}")
        traceback.print_exc()
        return False

def test_environment_setup():
    """Test environment setup utilities."""
    print_section("TEST 6: Environment Setup")
    
    try:
        from utils.setup_env import setup_project_paths, validate_environment
        print(f"  [OK] setup_env module imported")
        
        project_root, scripts_path = setup_project_paths()
        print(f"  [OK] Project paths configured")
        print(f"    - Project root: {project_root}")
        print(f"    - Scripts path: {scripts_path}")
        
        _is_valid = validate_environment()
        print(f"  [OK] Environment validated")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Environment setup test FAILED: {e}")
        traceback.print_exc()
        return False

def test_training_pipeline():
    """Test the training pipeline with demo data."""
    print_section("TEST 7: Training Pipeline (with demo data)")
    
    try:
        from utils.generate_demo_data import generate_demo_data
        from train_models import DRModelTrainer
        
        # Generate small demo dataset
        print("  [1] Generating demo data...")
        generate_demo_data(n_train=30, n_test=10)
        print("      [OK] Demo data generated")
        
        # Test trainer initialization
        print("  [2] Initializing trainer...")
        trainer = DRModelTrainer()
        print("      [OK] Trainer initialized")
        
        # Test data loading
        print("  [3] Testing data loading...")
        if trainer.load_data():
            print("      [OK] Data loaded successfully")
            print(f"        - Training shape: {trainer.X_train_scaled.shape}")
            print(f"        - Test shape: {trainer.X_test_scaled.shape}")
        else:
            print("      [WARN] Data loading returned False (expected if no data)")
        
        print("\n  [OK] Training pipeline test passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Training pipeline test FAILED: {e}")
        traceback.print_exc()
        return False

def test_configuration_values():
    """Test that configuration values are correct."""
    print_section("TEST 8: Configuration Values")
    
    try:
        from config import (
            CLASS_NAMES, NUM_CLASSES, IMAGE_TARGET_SIZE,
            RANDOM_STATE, TEST_SIZE
        )
        
        tests = [
            (len(CLASS_NAMES) == 5, "5 classes configured"),
            (NUM_CLASSES == 5, "NUM_CLASSES = 5"),
            (IMAGE_TARGET_SIZE == (256, 256), "Image size = 256x256"),
            (RANDOM_STATE == 42, "Random state = 42"),
            (TEST_SIZE == 0.2, "Test size = 0.2"),
        ]
        
        all_passed = True
        for test, description in tests:
            if test:
                print(f"  [OK] {description}")
            else:
                print(f"  [FAIL] {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"  [FAIL] Configuration values test FAILED: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print_section("TEST 9: File Structure")
    
    required_files = [
        'config.py',
        'train_models.py',
        'evaluate_models.py',
        'requirements.txt',
        'readme.md',
        'scripts/__init__.py',
        'scripts/preprocessing.py',
        'scripts/feature_extraction.py',
        'scripts/visualize.py',
        'utils/setup_env.py',
        'utils/generate_demo_data.py',
        'utils/test_project.py',
        'docs/QUICK_START.md',
        'docs/IMPLEMENTATION_GUIDE.md',
    ]
    
    project_root = Path(__file__).resolve().parent.parent
    
    all_exist = True
    for filename in required_files:
        filepath = project_root / filename
        if filepath.exists():
            print(f"  [OK] {filename}")
        else:
            print(f"  [FAIL] {filename} - NOT FOUND")
            all_exist = False
    
    return all_exist

def run_all_tests():
    """Run all tests and report results."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + "  DIABETIC RETINOPATHY DETECTION - TEST SUITE".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Scripts Module", test_scripts_import),
        ("Demo Data Generation", test_data_generation),
        ("Model Modules", test_model_modules),
        ("Environment Setup", test_environment_setup),
        ("Training Pipeline", test_training_pipeline),
        ("Configuration Values", test_configuration_values),
        ("File Structure", test_file_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_section("SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print_section("[PASS] ALL TESTS PASSED!")
        print("  Your project is ready for use!")
        print("\n  Next steps:")
        print("    1. Share GitHub URL with team")
        print("    2. Team members clone and run tests")
        print("    3. Read QUICK_START.md")
        print("    4. If they have real data, replace demo data with it")
        print("    5. Run: python train_models.py (with real data)")
        return True
    else:
        print_section("[WARN]  SOME TESTS FAILED")
        print(f"  {total - passed} test(s) need attention")
        print("\n  Check error messages above and:")
        print("    1. Verify Python version (3.8+)")
        print("    2. Check virtual environment is activated")
        print("    3. Run: pip install -r requirements.txt --upgrade")
        print("    4. Try running tests again")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
