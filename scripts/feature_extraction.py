# scripts/feature_extraction.py
"""
Feature extraction module for Diabetic Retinopathy detection.
Extracts deep learning features (VGG16), Local Binary Pattern (LBP), and Haralick features.
"""

import os
import numpy as np
import warnings
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# Load a pre-trained deep learning model for feature extraction
# Using weights='imagenet' with proper TensorFlow 2.x compatibility
try:
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )
    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('block5_pool').output
    )
    print("[INFO] VGG16 model loaded successfully for deep feature extraction.")
except Exception as e:
    print(f"[ERROR] Failed to load VGG16 model: {e}")
    raise

def extract_deep_features(img):
    """
    Extracts deep features from an image using a pre-trained VGG16 model.
    
    Args:
        img (np.ndarray): Input image array
        
    Returns:
        np.ndarray: Flattened feature vector from VGG16 block5_pool layer
        
    Raises:
        ValueError: If image shape is invalid
    """
    if img is None or img.size == 0:
        raise ValueError("Image cannot be empty")
        
    try:
        # Ensure image has correct shape
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # Suppress TensorFlow output
        with tf.device('/CPU:0'):
            features = model.predict(img, verbose=0)
        
        return features.flatten()
    except Exception as e:
        print(f"[ERROR] Failed to extract deep features: {e}")
        raise

def extract_lbp(img_gray):
    """
    Extracts Local Binary Pattern (LBP) texture features from a grayscale image.
    
    Args:
        img_gray (np.ndarray): Grayscale image array
        
    Returns:
        np.ndarray: Normalized LBP histogram (normalized feature vector)
        
    Raises:
        ValueError: If image is invalid or empty
    """
    if img_gray is None or img_gray.size == 0:
        raise ValueError("Image cannot be empty")
        
    try:
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist
    except Exception as e:
        print(f"[ERROR] Failed to extract LBP features: {e}")
        raise

def extract_haralick(img_gray):
    """
    Extracts Haralick texture features (GLCM) from a grayscale image.
    
    Args:
        img_gray (np.ndarray): Grayscale image array
        
    Returns:
        np.ndarray: Haralick feature vector containing contrast, dissimilarity,
                   homogeneity, energy, correlation, and ASM features
                   
    Raises:
        ValueError: If image is invalid or empty
    """
    if img_gray is None or img_gray.size == 0:
        raise ValueError("Image cannot be empty")
        
    try:
        glcm = graycomatrix(
            img_gray,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        features = np.hstack([
            graycoprops(glcm, 'contrast').ravel(),
            graycoprops(glcm, 'dissimilarity').ravel(),
            graycoprops(glcm, 'homogeneity').ravel(),
            graycoprops(glcm, 'energy').ravel(),
            graycoprops(glcm, 'correlation').ravel(),
            graycoprops(glcm, 'ASM').ravel()
        ])
        return features
    except Exception as e:
        print(f"[ERROR] Failed to extract Haralick features: {e}")
        raise