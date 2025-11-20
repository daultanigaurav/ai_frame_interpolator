"""
Evaluation metrics for frame interpolation quality.
Includes SSIM, PSNR, and runtime evaluation.
"""

import numpy as np
import cv2
from typing import Tuple, Dict
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import logging

logger = logging.getLogger(__name__)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image (H, W, 3) or (H, W)
        img2: Second image (H, W, 3) or (H, W)
        
    Returns:
        SSIM score (0-1, higher is better)
    """
    # Ensure images are the same size
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Normalize to [0, 1]
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    
    # Calculate SSIM
    score = ssim(img1, img2, data_range=1.0)
    
    return float(score)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: First image (H, W, 3) or (H, W)
        img2: Second image (H, W, 3) or (H, W)
        
    Returns:
        PSNR score in dB (higher is better)
    """
    # Ensure images are the same size
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return float(psnr_value)


def evaluate_interpolation_quality(
    original_frame1: np.ndarray,
    original_frame2: np.ndarray,
    interpolated_frame: np.ndarray,
    alpha: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate the quality of an interpolated frame.
    
    Args:
        original_frame1: First original frame
        original_frame2: Second original frame
        interpolated_frame: Interpolated frame to evaluate
        alpha: Interpolation factor used
        
    Returns:
        Dictionary with SSIM, PSNR, and other metrics
    """
    # Calculate SSIM with both original frames
    ssim1 = calculate_ssim(original_frame1, interpolated_frame)
    ssim2 = calculate_ssim(original_frame2, interpolated_frame)
    avg_ssim = (ssim1 + ssim2) / 2
    
    # Calculate PSNR with both original frames
    psnr1 = calculate_psnr(original_frame1, interpolated_frame)
    psnr2 = calculate_psnr(original_frame2, interpolated_frame)
    avg_psnr = (psnr1 + psnr2) / 2
    
    # Calculate ground truth interpolation (linear blend)
    ground_truth = (1 - alpha) * original_frame1.astype(np.float32) + alpha * original_frame2.astype(np.float32)
    ground_truth = ground_truth.astype(np.uint8)
    
    # Compare with ground truth
    ssim_gt = calculate_ssim(ground_truth, interpolated_frame)
    psnr_gt = calculate_psnr(ground_truth, interpolated_frame)
    
    return {
        "ssim_avg": avg_ssim,
        "ssim_frame1": ssim1,
        "ssim_frame2": ssim2,
        "ssim_ground_truth": ssim_gt,
        "psnr_avg": avg_psnr,
        "psnr_frame1": psnr1,
        "psnr_frame2": psnr2,
        "psnr_ground_truth": psnr_gt,
        "alpha": alpha
    }


def evaluate_frame_sequence(
    frames: list,
    interpolation_times: list = None
) -> Dict[str, float]:
    """
    Evaluate a sequence of interpolated frames.
    
    Args:
        frames: List of frames (including originals and interpolated)
        interpolation_times: List of times taken for each interpolation
        
    Returns:
        Dictionary with aggregate metrics
    """
    metrics = {
        "total_frames": len(frames),
        "avg_ssim": 0.0,
        "avg_psnr": 0.0,
        "avg_interpolation_time": 0.0
    }
    
    if len(frames) < 3:
        return metrics
    
    # Calculate pairwise SSIM and PSNR
    ssim_scores = []
    psnr_scores = []
    
    for i in range(len(frames) - 1):
        ssim_val = calculate_ssim(frames[i], frames[i + 1])
        psnr_val = calculate_psnr(frames[i], frames[i + 1])
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
    
    metrics["avg_ssim"] = float(np.mean(ssim_scores))
    metrics["avg_psnr"] = float(np.mean(psnr_scores))
    
    if interpolation_times:
        metrics["avg_interpolation_time"] = float(np.mean(interpolation_times))
        metrics["total_interpolation_time"] = float(np.sum(interpolation_times))
    
    return metrics


def calculate_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    Calculate optical flow between two frames using Farneback method.
    
    Args:
        frame1: First frame
        frame2: Second frame
        
    Returns:
        Optical flow visualization
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Convert flow to visualization
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to BGR
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flow_vis

