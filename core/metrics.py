import numpy as np
import cv2
from typing import Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging

logger = logging.getLogger(__name__)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    score = ssim(img1, img2, data_range=1.0)
    return float(score)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr_value)


def evaluate_interpolation_quality(
    original_frame1: np.ndarray,
    original_frame2: np.ndarray,
    interpolated_frame: np.ndarray,
    alpha: float = 0.5
) -> Dict[str, float]:
    ssim1 = calculate_ssim(original_frame1, interpolated_frame)
    ssim2 = calculate_ssim(original_frame2, interpolated_frame)
    avg_ssim = (ssim1 + ssim2) / 2

    psnr1 = calculate_psnr(original_frame1, interpolated_frame)
    psnr2 = calculate_psnr(original_frame2, interpolated_frame)
    avg_psnr = (psnr1 + psnr2) / 2

    ground_truth = (1 - alpha) * original_frame1.astype(np.float32) + alpha * original_frame2.astype(np.float32)
    ground_truth = ground_truth.astype(np.uint8)

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


def evaluate_frame_sequence(frames: list, interpolation_times: list = None) -> Dict[str, float]:
    metrics = {
        "total_frames": len(frames),
        "avg_ssim": 0.0,
        "avg_psnr": 0.0,
        "avg_interpolation_time": 0.0
    }

    if len(frames) < 3:
        return metrics

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
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_vis

