"""
Utility functions for the frame interpolation system.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return img


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to file path.
    
    Args:
        image: Image as numpy array
        output_path: Path to save the image
    """
    ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(output_path, image)
    logger.info(f"Saved image: {output_path}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad if necessary to match exact target size
    if new_w != target_w or new_h != target_h:
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        resized = cv2.copyMakeBorder(
            resized, pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image (0-255)
        
    Returns:
        Normalized image (0-1)
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255] range.
    
    Args:
        image: Normalized image (0-1)
        
    Returns:
        Denormalized image (0-255)
    """
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def validate_resolution(resolution: Tuple[int, int]) -> bool:
    """
    Validate resolution values.
    
    Args:
        resolution: (width, height) tuple
        
    Returns:
        True if valid, False otherwise
    """
    width, height = resolution
    return (width > 0 and height > 0 and 
            width % 2 == 0 and height % 2 == 0 and
            width <= 7680 and height <= 4320)  # 8K max


def get_project_root() -> str:
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


