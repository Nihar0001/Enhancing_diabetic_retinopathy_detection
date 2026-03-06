"""
Path and Environment Setup Utility
Ensures correct imports and paths regardless of where the script is run from.
"""

import sys
import os
from pathlib import Path

def setup_project_paths():
    """
    Sets up the project paths correctly.
    This should be called at the beginning of any script/notebook.
    
    Returns:
        tuple: (PROJECT_ROOT, scripts_path)
    """
    # Method 1: Try to get project root from config
    try:
        from config import PROJECT_ROOT
        project_root = PROJECT_ROOT
    except ImportError:
        # Method 2: If config not available, find it from file location
        current_file = Path(__file__).resolve()
        
        # If this file is in root, that's our project root
        if current_file.parent.name in ['Enhancing-diabetic-retinopathy-detection', 'project']:
            project_root = current_file.parent
        else:
            # Otherwise, go up until we find a directory with 'data' and 'models' folders
            project_root = current_file.parent
            while project_root != project_root.parent:
                if (project_root / 'data').exists() and (project_root / 'models').exists():
                    break
                project_root = project_root.parent
    
    # Add scripts directory to path if not already there
    scripts_path = project_root / 'scripts'
    scripts_str = str(scripts_path)
    
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
        print(f"[INFO] Added {scripts_path} to Python path")
    
    print(f"[INFO] Project root set to: {project_root}")
    
    return project_root, scripts_path


def validate_environment():
    """
    Validates that the environment is correctly set up.
    Checks for required files and packages.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    try:
        from config import validate_paths
        is_valid = validate_paths()
        if is_valid:
            print("[INFO] ✓ All required paths are valid")
        else:
            print("[WARNING] Some required data files are missing")
        return is_valid
    except Exception as e:
        print(f"[ERROR] Environment validation failed: {e}")
        return False


def print_project_info():
    """Prints project information for debugging."""
    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, MODELS_DIR, 
            OUTPUTS_DIR, CLASS_NAMES, NUM_CLASSES
        )
        
        print("\n" + "="*60)
        print("PROJECT CONFIGURATION")
        print("="*60)
        print(f"Project Root: {PROJECT_ROOT}")
        print(f"Data Directory: {DATA_DIR}")
        print(f"Models Directory: {MODELS_DIR}")
        print(f"Outputs Directory: {OUTPUTS_DIR}")
        print(f"Number of Classes: {NUM_CLASSES}")
        print(f"Class Names: {CLASS_NAMES}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"[ERROR] Could not print project info: {e}")


if __name__ == "__main__":
    print("Setting up project environment...")
    setup_project_paths()
    validate_environment()
    print_project_info()
    print("[SUCCESS] Environment setup complete!")
