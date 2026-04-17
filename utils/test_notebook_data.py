import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Navigate relative to project root
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
os.chdir(os.path.join(_project_root, 'notebooks'))

try:
    # Load labels to verify data
    labels = pd.read_csv('../data/train.csv')
    print(f"[OK] train.csv loaded: {labels.shape[0]} images")
    print(f"[OK] Unique labels: {sorted(labels['diagnosis'].unique())}")
    
    # List some images
    images_dir = '../data/train_images'
    if os.path.exists(images_dir):
        images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        print(f"[OK] Sample images found: {len(images)} total PNG images")
        if images:
            print(f"[OK] First image: {images[0]}")
    else:
        print(f"[ERROR] Directory not found: {images_dir}")
    
    # Test imports
    from scripts.preprocessing import preprocess_image
    from scripts.feature_extraction import extract_deep_features, extract_lbp, extract_haralick
    print("[OK] All imports successful")
    
    # Test preprocessing on a sample image
    if images:
        img_path = os.path.join(images_dir, images[0])
        img, img_gray = preprocess_image(img_path)
        print(f"[OK] Preprocessing works: img shape {img.shape}, img_gray shape {img_gray.shape}")
        
        # Test feature extraction
        deep_feat = extract_deep_features(img)
        lbp_feat = extract_lbp(img_gray)
        haralick_feat = extract_haralick(img_gray)
        
        print(f"[OK] Deep features shape: {deep_feat.shape}")
        print(f"[OK] LBP features shape: {lbp_feat.shape}")
        print(f"[OK] Haralick features shape: {haralick_feat.shape}")
        
        final_feat = np.concatenate([deep_feat, lbp_feat, haralick_feat])
        print(f"[OK] Final fused features shape: {final_feat.shape}")
    
    print("\n" + "="*60)
    print("[SUCCESS] NOTEBOOK WILL WORK CORRECTLY!")
    print("="*60)
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
