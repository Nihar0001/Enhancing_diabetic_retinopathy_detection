# scripts/preprocessing.py
"""
Preprocessing module for Diabetic Retinopathy detection.
Handles image loading, resizing, normalization, and enhancement.
"""

import cv2
import numpy as np


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocesses a single image for feature extraction.
    Loads, resizes, and converts the image to a format suitable for models.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (default: 256x256)
        
    Returns:
        tuple: (preprocessed RGB/BGR image, grayscale image)
        
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image cannot be loaded
    """
    if not isinstance(image_path, str):
        raise ValueError(f"image_path must be a string, got {type(image_path)}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")

    # Resize image to target size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale for LBP and Haralick texture features
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, img_gray
def advanced_preprocess_image(image_data, target_size=(224, 224), from_numpy=False):
    """
    Preprocesses an image with CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Accepts either an image path or a NumPy array.
    
    Args:
        image_data (str or np.ndarray): Path to the image file or NumPy array
        target_size (tuple): Target size for resizing (default: 224x224)
        from_numpy (bool): True if image_data is a NumPy array, False if a path (default: False)
        
    Returns:
        tuple: (Original preprocessed image, CLAHE-enhanced grayscale image)
        
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image data is invalid
    """
    try:
        if from_numpy:
            if not isinstance(image_data, np.ndarray):
                raise ValueError("Expected numpy array when from_numpy=True")
            img_bgr = image_data
        else:
            if not isinstance(image_data, str):
                raise ValueError("Expected image path as string when from_numpy=False")
            img_bgr = cv2.imread(image_data)
        
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found or could not be read: {image_data}")
            
        # Resize image
        img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        
        return img_bgr, img_clahe
        
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error during preprocessing: {e}")
        raise
