import os
import cv2
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def load_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return img


def save_image(image: np.ndarray, output_path: str) -> None:
    ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(output_path, image)
    logger.info(f"Saved image: {output_path}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def validate_resolution(resolution: Tuple[int, int]) -> bool:
    width, height = resolution
    return (
        width > 0 and height > 0 and
        width % 2 == 0 and height % 2 == 0 and
        width <= 7680 and height <= 4320
    )


def get_project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


